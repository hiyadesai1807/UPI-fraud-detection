"""End-to-end pipeline: audit -> EDA -> 6 experiments -> explainability.

    python run_all.py            # full run (~10-15 min on a laptop CPU)
    python run_all.py --fast     # fewer tuning iterations / 1 DL seed (~3 min)

Outputs: reports/tables/*.csv, reports/figures/*.png, models/*, reports/summary.json
"""
import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import config as C
from src import plots as P
from src.evaluation import (cluster_bootstrap_ci, evaluate, expected_cost, mcnemar,
                            metrics_at)
from src.explain import (global_importance, integrated_gradients,
                         lstm_permutation_importance, rank_agreement, shap_tree)
from src.features import feature_sets, prepare_dataset
from src.leakage_audit import rule_baseline, run_full_audit
from src.models_classical import STRATEGIES, build, tune
from src.models_deep import TorchClassifier, build_sequences
from src.splits import get_split, make_random_split, split_summary
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

NAMES = {"logreg": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
plt = P.plt


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_table(df, name):
    df.to_csv(C.TABLE_DIR / f"{name}.csv", index=False)
    return df


def row(model, setting, m, **extra):
    return {"model": model, "setting": setting, **extra,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}}


# ===========================================================================
# Stage 1-2: audit + EDA
# ===========================================================================
def stage_audit_eda(df, split):
    log("Stage 1: data-quality & leakage audit")
    res = run_full_audit(df)
    auc = res["single_feature_auc"].head(20).set_index("feature")["auc"]
    P.hbar(auc, "Single-feature ROC-AUC (top 20)", "ROC-AUC using that column alone",
           "01_single_feature_auc", highlight=lambda k, v: v >= 0.9)
    lc = res["legit_constant"].set_index("feature")["fraud_rows_with_other_value_pct"].head(25)
    P.hbar(lc, "Scenario-signature columns: constant for every legitimate row",
           "% of fraud rows carrying a non-default value", "02_legit_constant_columns",
           color=P.SERIES[1], fmt="{:.1f}%")
    oor = res["out_of_range"].set_index("feature")["fraud_outside_pct"]
    pe = res["pairwise_equality"]
    if len(pe):
        geo = df[C.GEO_PAIR_COLS[0]].values - df[C.GEO_PAIR_COLS[1]].values
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        eq = (np.isclose(geo, 0)).astype(int)
        tab = pd.crosstab(eq, df[C.TARGET]).reindex(index=[1, 0], fill_value=0)
        x = np.arange(2)
        ax.bar(x - 0.2, tab[0].values, 0.38, color=P.CLASS_COLORS[0], label="Legitimate")
        ax.bar(x + 0.2, tab[1].values, 0.38, color=P.CLASS_COLORS[1], label="Fraud")
        for xi, (a, b) in enumerate(zip(tab[0].values, tab[1].values)):
            ax.text(xi - 0.2, a, f"{a:,}", ha="center", va="bottom", fontsize=8)
            ax.text(xi + 0.2, b, f"{b:,}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, ["geo values equal", "geo values differ"]); ax.grid(axis="x", visible=False)
        ax.set_ylabel("Transactions"); ax.legend()
        ax.set_title("Interaction leak: geographic_disparity vs geographic_location_vs_ip", loc="left", fontsize=10)
        P.save(fig, "03b_geo_pair_interaction_leak")
    P.hbar(oor, "Behavioural features: fraud outside the legitimate range",
           "% of fraud rows outside [min, max] of legitimate rows", "03_out_of_range",
           fmt="{:.1f}%")

    log("Stage 2: EDA")
    y = df[C.TARGET]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    counts = y.value_counts().sort_index()
    ax.bar(["Legitimate", "Fraud"], counts.values, color=[P.CLASS_COLORS[0], P.CLASS_COLORS[1]], width=0.55)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(y):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title("Class distribution", loc="left"); ax.set_ylabel("Transactions")
    ax.set_ylim(0, counts.max() * 1.2); ax.grid(axis="x", visible=False)
    P.save(fig, "04_class_balance")

    cols = C.BEHAVIOURAL_COLS
    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    for ax, c in zip(axes.flat, cols):
        lo, hi = df[c].quantile([0.005, 0.995])
        bins = np.linspace(lo, hi, 40)
        for cls in (0, 1):
            ax.hist(df.loc[y == cls, c].clip(lo, hi), bins=bins, density=True, alpha=0.55,
                    color=P.CLASS_COLORS[cls], label=["Legitimate", "Fraud"][cls])
        ax.set_title(c, fontsize=9, loc="left"); ax.tick_params(labelsize=7)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    axes.flat[0].legend()
    fig.suptitle("Behavioural feature distributions by class (density)", x=0.01, ha="left", fontweight="bold")
    P.save(fig, "05_behavioural_distributions")

    hr = df.groupby("transaction_time_of_day")[C.TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(hr.index, hr.values, color=P.SERIES[1], marker="o", markersize=4)
    ax.set_title("Fraud rate by hour of day", loc="left"); ax.set_xlabel("Hour"); ax.set_ylabel("Fraud rate (%)")
    P.save(fig, "06_fraud_rate_by_hour")

    corr = df[C.BEHAVIOURAL_COLS + C.HISTORY_COLS + [C.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    ax.grid(False); fig.colorbar(im, shrink=0.7)
    ax.set_title("Correlation matrix: behavioural + history features", loc="left")
    P.save(fig, "07_correlation_matrix")

    ss = split_summary(df, split)
    save_table(ss, "split_summary")
    eda_stats = df.groupby(C.TARGET)[C.BEHAVIOURAL_COLS + C.HISTORY_COLS].median().T
    eda_stats.columns = ["median_legit", "median_fraud"]
    eda_stats.reset_index(names="feature").to_csv(C.TABLE_DIR / "eda_medians_by_class.csv", index=False)
    return res, ss


# ===========================================================================
# Helpers
# ===========================================================================
def fit_classical(name, df, cols, split, strategy, pos_w):
    tr, va, te = split["train"], split["val"], split["test"]
    y = df[C.TARGET].values
    pipe = build(name, df, cols, strategy, pos_w)
    pipe.fit(df.iloc[tr][cols], y[tr])
    pv = pipe.predict_proba(df.iloc[va][cols])[:, 1]
    pt = pipe.predict_proba(df.iloc[te][cols])[:, 1]
    return pipe, pv, pt


def val_ap(y, p):
    return round(average_precision_score(y, p), 4)


def train_mlp(df, cols, split, strategy, seed):
    y = df[C.TARGET].values
    if all(pd.api.types.is_numeric_dtype(df[c]) for c in cols):
        X = df[cols].values.astype(np.float32)
    else:  # one-hot categoricals, encoder fitted on training rows only
        from src.features import make_preprocessor
        prep = make_preprocessor(df, cols, scale=False).fit(df.iloc[split["train"]][cols])
        X = prep.transform(df[cols]).astype(np.float32)
    m = TorchClassifier("mlp", strategy=strategy, seed=seed)
    m.fit(X[split["train"]], y[split["train"]], X[split["val"]], y[split["val"]])
    return m, m.predict_proba(X[split["val"]]), m.predict_proba(X[split["test"]])


def lstm_inputs(df, split, shuffle=False, seq_len=C.SEQ_LEN, seed=C.SEED):
    X = df[C.BEHAVIOURAL_COLS].values.astype(np.float32)
    sc = StandardScaler().fit(X[split["train"]])
    S, L = build_sequences(df, sc.transform(X).astype(np.float32), seq_len=seq_len,
                           shuffle_history=shuffle, seed=seed)
    return S, L, sc


def train_lstm(df, split, S, L, seed):
    y = df[C.TARGET].values
    tr, va, te = split["train"], split["val"], split["test"]
    m = TorchClassifier("lstm", strategy="class_weight", seed=seed, epochs=60)
    m.fit(S[tr], y[tr], S[va], y[va], L[tr], L[va])
    return m, m.predict_proba(S[va], L[va]), m.predict_proba(S[te], L[te])


# ===========================================================================
# Experiments
# ===========================================================================
def exp1_leakage(df, split, fs, pos_w):
    log("Experiment 1: leakage tiers")
    y = df[C.TARGET].values
    va, te = split["val"], split["test"]
    rows = []
    for tier in ["T0_all_features", "T1_proxies_removed", "T2a_behavioural_with_geo_pair", "T2_behavioural"]:
        for name in ["logreg", "xgboost"]:
            _, pv, pt = fit_classical(name, df, fs[tier], split, "class_weight", pos_w)
            m = evaluate(y[va], pv, y[te], pt)
            rows.append(row(NAMES[name], tier, m, n_features=len(fs[tier]), val_pr_auc=val_ap(y[va], pv)))
        _, pv, pt = train_mlp(df, fs[tier], split, "class_weight", C.SEED)
        rows.append(row("MLP", tier, evaluate(y[va], pv, y[te], pt), n_features=len(fs[tier]),
                        val_pr_auc=val_ap(y[va], pv)))
    train_mask = np.zeros(len(df), bool); train_mask[split["train"]] = True
    rule = rule_baseline(df, train_mask)
    m = metrics_at(y[te], rule[te].astype(float), 0.5)
    rows.append(row("Rule baseline", "T2_behavioural (range rule)", m, n_features=len(C.BEHAVIOURAL_COLS)))
    out = save_table(pd.DataFrame(rows), "exp1_leakage_tiers")
    plot = out[out.model != "Rule baseline"].copy()
    plot["setting"] = plot["setting"].map({"T0_all_features": "T0 all features",
                                           "T1_proxies_removed": "T1 proxies removed",
                                           "T2a_behavioural_with_geo_pair": "T2a + geo pair",
                                           "T2_behavioural": "T2 strict behavioural"})
    P.grouped_bar(plot, "setting", "model", "pr_auc", "Experiment 1: how leakage inflates performance",
                  "Test PR-AUC", "10_exp1_leakage", ylim=(0, 1.08), colors=P.MODEL_COLORS)
    return out, rule


def exp3_imbalance(df, split, fs, pos_w, fast):
    log("Experiment 3: class-imbalance strategies (T2 behavioural)")
    y = df[C.TARGET].values
    va, te = split["val"], split["test"]
    cols = fs["T2_behavioural"]
    rows = []
    for name in ["logreg", "random_forest", "xgboost"]:
        for st in STRATEGIES:
            _, pv, pt = fit_classical(name, df, cols, split, st, pos_w)
            rows.append(row(NAMES[name], st, evaluate(y[va], pv, y[te], pt), val_pr_auc=val_ap(y[va], pv)))
    for st in STRATEGIES:
        _, pv, pt = train_mlp(df, cols, split, st, C.SEED)
        rows.append(row("MLP", st, evaluate(y[va], pv, y[te], pt), val_pr_auc=val_ap(y[va], pv)))
    out = save_table(pd.DataFrame(rows), "exp3_imbalance")
    P.grouped_bar(out, "model", "setting", "recall", "Experiment 3: fraud recall by imbalance strategy",
                  "Test recall (threshold tuned on validation)", "12_exp3_imbalance_recall", ylim=(0, 1.08))
    P.grouped_bar(out, "model", "setting", "pr_auc", "Experiment 3: PR-AUC by imbalance strategy",
                  "Test PR-AUC", "12_exp3_imbalance_prauc", ylim=(0, 1.08))
    best = out.loc[out.groupby("model")["val_pr_auc"].idxmax(), ["model", "setting"]]
    return out, dict(zip(best.model, best.setting))


def exp4_features(df, split, fs, pos_w, best_strat):
    log("Experiment 4: behavioural vs behavioural + history features")
    y = df[C.TARGET].values
    va, te = split["val"], split["test"]
    rows = []
    for tier in ["T2_behavioural", "T2H_behavioural_history"]:
        for name in ["logreg", "random_forest", "xgboost"]:
            _, pv, pt = fit_classical(name, df, fs[tier], split, best_strat[NAMES[name]], pos_w)
            rows.append(row(NAMES[name], tier, evaluate(y[va], pv, y[te], pt), val_pr_auc=val_ap(y[va], pv)))
        _, pv, pt = train_mlp(df, fs[tier], split, best_strat["MLP"], C.SEED)
        rows.append(row("MLP", tier, evaluate(y[va], pv, y[te], pt), val_pr_auc=val_ap(y[va], pv)))
    out = save_table(pd.DataFrame(rows), "exp4_feature_engineering")
    P.grouped_bar(out, "model", "setting", "pr_auc", "Experiment 4: value of user-history features",
                  "Test PR-AUC", "13_exp4_features", ylim=(0.8, 1.0))
    return out


def exp5_split(df, split, fs, pos_w, best_strat):
    log("Experiment 5: user-grouped vs random row split")
    y = df[C.TARGET].values
    rsplit = {k: np.array(v) for k, v in make_random_split(df).items()}
    cols = fs["T2H_behavioural_history"]
    rows = []
    for label, sp in [("group split (by user)", split), ("random row split", rsplit)]:
        for name in ["logreg", "xgboost"]:
            _, pv, pt = fit_classical(name, df, cols, sp, best_strat[NAMES[name]], pos_w)
            rows.append(row(NAMES[name], label, evaluate(y[sp["val"]], pv, y[sp["test"]], pt)))
        _, pv, pt = train_mlp(df, cols, sp, best_strat["MLP"], C.SEED)
        rows.append(row("MLP", label, evaluate(y[sp["val"]], pv, y[sp["test"]], pt)))
    users_tr = set(df.iloc[rsplit["train"]][C.GROUP])
    shared = df.iloc[rsplit["test"]][C.GROUP].isin(users_tr).mean()
    out = save_table(pd.DataFrame(rows), "exp5_split_strategy")
    out.attrs["random_split_test_rows_with_seen_user"] = float(shared)
    return out, float(shared)


def exp6_sequence(df, split, fs, best_strat, seeds):
    log("Experiment 6: sequence modelling (MLP vs LSTM, ordered vs shuffled)")
    y = df[C.TARGET].values
    va, te = split["val"], split["test"]
    S, L, sc = lstm_inputs(df, split)
    S_sh, L_sh, _ = lstm_inputs(df, split, shuffle=True)
    S1, L1, _ = lstm_inputs(df, split, seq_len=1)
    rows, models, preds = [], {}, {}
    configs = [
        ("MLP", "current transaction only (T2)", lambda s: train_mlp(df, fs["T2_behavioural"], split, best_strat["MLP"], s)),
        ("MLP", "current + history aggregates (T2H)", lambda s: train_mlp(df, fs["T2H_behavioural_history"], split, best_strat["MLP"], s)),
        ("LSTM", "sequence length 1 (no history)", lambda s: train_lstm(df, split, S1, L1, s)),
        ("LSTM", "last 5 transactions, ordered", lambda s: train_lstm(df, split, S, L, s)),
        ("LSTM", "last 5 transactions, history shuffled", lambda s: train_lstm(df, split, S_sh, L_sh, s)),
    ]
    for model, setting, fn in configs:
        best = None
        for s in seeds:
            m, pv, pt = fn(s)
            met = evaluate(y[va], pv, y[te], pt)
            rows.append(row(model, setting, met, seed=s, val_pr_auc=val_ap(y[va], pv), epochs=len(m.history)))
            if best is None or val_ap(y[va], pv) > best[0]:
                best = (val_ap(y[va], pv), m, pv, pt)
        models[(model, setting)] = best[1]
        preds[(model, setting)] = (best[2], best[3])
        log(f"   {model:5s} | {setting:40s} done")
    raw = save_table(pd.DataFrame(rows), "exp6_sequence_runs")
    agg = raw.groupby(["model", "setting"], sort=False).agg(
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"), f1_mean=("f1", "mean"), recall_mean=("recall", "mean"),
        precision_mean=("precision", "mean")).round(4).reset_index()
    save_table(agg, "exp6_sequence_summary")

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    lab = agg["model"] + " | " + agg["setting"]
    cols = [P.MODEL_COLORS[m] for m in agg["model"]]
    ax.barh(lab[::-1], agg["pr_auc_mean"][::-1], xerr=agg["pr_auc_std"].fillna(0)[::-1],
            color=cols[::-1], height=0.6, ecolor=P.INK2, capsize=3)
    for i, (v, sd) in enumerate(zip(agg["pr_auc_mean"][::-1], agg["pr_auc_std"].fillna(0)[::-1])):
        ax.text(v + sd + 0.002, i, f"{v:.3f}", va="center", fontsize=8, color=P.INK2)
    lo = max(0.0, agg["pr_auc_mean"].min() - 0.05)
    ax.set_xlim(lo, 1.0); ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Test PR-AUC (mean ± sd over {len(seeds)} seeds)")
    ax.set_title("Experiment 6: does transaction order carry signal?", loc="left")
    P.save(fig, "14_exp6_sequence")

    # training curves of the representative MLP / LSTM
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key in zip(axes, [("MLP", "current + history aggregates (T2H)"), ("LSTM", "last 5 transactions, ordered")]):
        h = pd.DataFrame(models[key].history)
        ax.plot(h.epoch, h.train_loss, color=P.SERIES[0], label="train loss")
        ax2 = ax.twinx(); ax2.grid(False)
        ax2.plot(h.epoch, h.val_pr_auc, color=P.SERIES[1], label="val PR-AUC")
        ax.set_title(f"{key[0]} training", loc="left"); ax.set_xlabel("epoch")
        ax.set_ylabel("train loss", color=P.SERIES[0]); ax2.set_ylabel("val PR-AUC", color=P.SERIES[1])
    P.save(fig, "15_dl_training_curves")
    return agg, models, preds, sc, (S, L)


MLP_KEY = {"T2_behavioural": ("MLP", "current transaction only (T2)"),
           "T2H_behavioural_history": ("MLP", "current + history aggregates (T2H)")}
TIER_SHORT = {"T2_behavioural": "T2", "T2H_behavioural_history": "T2H"}


def exp2_main(df, split, fs, pos_w, best_strat, dl_preds, fast, rule, tier):
    log(f"Experiment 2: tuned model comparison (main result) on {tier}")
    y = df[C.TARGET].values
    tr, va, te = split["train"], split["val"], split["test"]
    cols = fs[tier]
    probs, models, params = {}, {}, {}
    for name in ["logreg", "random_forest", "xgboost"]:
        est, bp, cv = tune(name, df, cols, df.iloc[tr][cols], y[tr], df.iloc[tr][C.GROUP].values,
                           strategy=best_strat[NAMES[name]], pos_weight=pos_w, n_iter=4 if fast else 12)
        probs[NAMES[name]] = (est.predict_proba(df.iloc[va][cols])[:, 1], est.predict_proba(df.iloc[te][cols])[:, 1])
        models[NAMES[name]] = est
        params[NAMES[name]] = {"strategy": best_strat[NAMES[name]], "cv_pr_auc": round(cv, 4),
                               **{k: (v if isinstance(v, (int, float, str)) or v is None else float(v)) for k, v in bp.items()}}
        joblib.dump(est, C.MODELS_DIR / f"{name}.joblib")
        log(f"   tuned {name}: CV PR-AUC {cv:.4f}")
    probs["MLP"] = dl_preds[MLP_KEY[tier]]
    probs["LSTM"] = dl_preds[("LSTM", "last 5 transactions, ordered")]

    groups_te = df.iloc[te][C.GROUP].values
    hard = rule[te] == 0              # test rows where NO feature is out of legit range
    rows, thresholds, decisions = [], {}, {}
    for mname, (pv, pt) in probs.items():
        m = evaluate(y[va], pv, y[te], pt)
        lo, hi = cluster_bootstrap_ci(y[te], pt, groups_te, n_boot=200 if fast else 500)
        hard_ap = average_precision_score(y[te][hard], pt[hard])
        rows.append(row(mname, f"{TIER_SHORT[tier]} tuned" if mname not in ("MLP", "LSTM") else
                        (TIER_SHORT[tier] if mname == "MLP" else "T2 sequences (k=5)"), m,
                        val_pr_auc=val_ap(y[va], pv), pr_auc_ci_low=round(lo, 4), pr_auc_ci_high=round(hi, 4),
                        hard_subset_pr_auc=round(hard_ap, 4)))
        thresholds[mname] = m["threshold"]
        decisions[mname] = (pt >= m["threshold"]).astype(int)
    m_rule = metrics_at(y[te], rule[te].astype(float), 0.5)
    rows.append(row("Rule baseline", "range rule", m_rule))
    out = pd.DataFrame(rows)
    champion = out[out.model != "Rule baseline"].sort_values("val_pr_auc", ascending=False).iloc[0]["model"]
    out["champion"] = out.model == champion
    save_table(out, "exp2_model_comparison")

    mc = []
    for mname in probs:
        if mname == champion:
            continue
        r = mcnemar(y[te], decisions[champion], decisions[mname])
        mc.append({"champion": champion, "versus": mname, **{k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()}})
    save_table(pd.DataFrame(mc), "exp2_mcnemar_tests")

    # PR curves
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for mname, (_, pt) in probs.items():
        p, r, _ = precision_recall_curve(y[te], pt)
        ax.plot(r, p, color=P.MODEL_COLORS[mname], label=f"{mname} (AP {average_precision_score(y[te], pt):.3f})")
    ax.scatter([m_rule["recall"]], [m_rule["precision"]], color=P.MODEL_COLORS["Rule baseline"], s=60,
               zorder=5, label="Rule baseline (single point)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0.4, 1.01)
    ax.set_title("Precision-recall curves (test set)", loc="left"); ax.legend(loc="lower left")
    P.save(fig, "11_pr_curves")

    # confusion matrix for champion
    mrow = out[out.model == champion].iloc[0]
    cm = np.array([[mrow.tn, mrow.fp], [mrow.fn, mrow.tp]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap="Blues"); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() / 2 else P.INK)
    ax.set_xticks([0, 1], ["Pred legit", "Pred fraud"]); ax.set_yticks([0, 1], ["Legit", "Fraud"])
    ax.set_title(f"Confusion matrix: {champion}", loc="left")
    P.save(fig, "16_confusion_matrix_champion")

    # threshold + cost analysis for champion
    pt = probs[champion][1]
    ths = np.linspace(0.01, 0.99, 99)
    rec = [metrics_at(y[te], pt, t) for t in ths]
    amounts = df.iloc[te]["amount"].values
    costs = [expected_cost(y[te], pt, t, amounts)[0] for t in ths]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(ths, [r["precision"] for r in rec], color=P.SERIES[0], label="precision")
    axes[0].plot(ths, [r["recall"] for r in rec], color=P.SERIES[1], label="recall")
    axes[0].plot(ths, [r["f1"] for r in rec], color=P.SERIES[2], label="F1")
    axes[0].axvline(thresholds[champion], color=P.INK2, ls="--", lw=1)
    axes[0].text(thresholds[champion], 0.62, " threshold chosen\n on validation", fontsize=8, color=P.INK2)
    axes[0].set_xlabel("decision threshold"); axes[0].set_ylim(0.4, 1.01)
    axes[0].legend(loc="lower right", ncol=3)
    axes[0].set_title(f"Threshold trade-off: {champion}", loc="left")
    axes[1].plot(ths, np.array(costs) / 1e5, color=P.SERIES[1])
    best_t = ths[int(np.argmin(costs))]
    axes[1].axvline(best_t, color=P.INK2, ls="--", lw=1)
    axes[1].text(best_t, np.max(costs) / 1e5 * 0.9, f" cost-optimal\n threshold = {best_t:.2f}", fontsize=8, color=P.INK2)
    axes[1].set_xlabel("decision threshold"); axes[1].set_ylabel("expected cost (₹ lakh)")
    axes[1].set_title("Illustrative cost (assumed: ₹100/alert + ₹1,000/false alarm)", loc="left", fontsize=10)
    P.save(fig, "17_threshold_and_cost")
    save_table(pd.DataFrame({"threshold": ths, "precision": [r["precision"] for r in rec],
                             "recall": [r["recall"] for r in rec], "f1": [r["f1"] for r in rec],
                             "expected_cost_inr": costs}), "threshold_curve_champion")

    pred_df = df.iloc[te][["transaction_id", C.GROUP, "amount", "transaction_time_of_day", C.TARGET]].copy()
    pred_df["all_features_in_legit_range"] = hard.astype(int)
    for mname, (_, p) in probs.items():
        pred_df[f"p_{mname.replace(' ', '_').lower()}"] = np.round(p, 6)
    pred_df.to_csv(C.TABLE_DIR / "test_predictions.csv", index=False)
    json.dump({"thresholds": thresholds, "champion": champion, "best_params": params,
               "feature_set": tier, "feature_columns": cols,
               "cost_optimal_threshold_champion": float(best_t)},
              open(C.MODELS_DIR / "model_meta.json", "w"), indent=2, default=str)
    return out, models, champion, params


def exp7_explain(df, split, fs, models, dl_models, S_L, tier):
    log("Experiment 7: explainability")
    y = df[C.TARGET].values
    te = split["test"]
    cols = fs[tier]
    Xte = df.iloc[te][cols]
    xgb = models["XGBoost"]
    sv, Xt, base = shap_tree(xgb, Xte, max_rows=len(Xte))
    names = list(Xt.columns)
    shap_imp = global_importance(sv, names)
    pd.DataFrame(sv, columns=names, index=df.iloc[Xt.index]["transaction_id"].values) \
        .round(5).reset_index(names="transaction_id").to_csv(C.TABLE_DIR / "shap_values_test_xgboost.csv", index=False)
    json.dump({"base_value_logodds": base}, open(C.MODELS_DIR / "shap_base.json", "w"))
    P.hbar(shap_imp, "XGBoost: global SHAP importance", "mean |SHAP value| (log-odds)", "20_shap_global_xgboost")

    import shap
    fig = plt.figure()
    shap.summary_plot(sv, Xt, show=False, max_display=15, plot_size=(8, 6))
    plt.title("XGBoost: SHAP beeswarm (test set)", loc="left", fontweight="bold")
    P.save(plt.gcf(), "21_shap_beeswarm_xgboost")

    # local explanation: highest-risk true fraud
    p = xgb.predict_proba(Xte)[:, 1]
    i = int(np.argmax(np.where(y[te] == 1, p, -1)))
    loc = pd.Series(sv[i], index=names).sort_values(key=np.abs, ascending=False).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(loc.index, loc.values, color=[P.SERIES[1] if v > 0 else P.SERIES[0] for v in loc.values], height=0.6)
    ax.axvline(0, color=P.INK2, lw=0.8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("SHAP contribution (log-odds); orange pushes towards fraud")
    ax.set_title(f"Why was transaction {df.iloc[te[i]]['transaction_id'][:8]}… flagged? (p = {p[i]:.2f})", loc="left")
    P.save(fig, "22_shap_local_example")

    mlp = dl_models[MLP_KEY[tier]]
    ig = integrated_gradients(mlp, Xte.values.astype(np.float32))
    ig_imp = global_importance(ig, cols)
    P.hbar(ig_imp, "MLP: global Integrated Gradients importance", "mean |attribution|", "23_ig_global_mlp",
           color=P.SERIES[3])
    rho = rank_agreement(shap_imp, ig_imp)

    lstm = dl_models[("LSTM", "last 5 transactions, ordered")]
    S, L = S_L
    perm = lstm_permutation_importance(lstm, S[te], L[te], y[te], C.BEHAVIOURAL_COLS)
    P.hbar(perm.set_index("feature")["pr_auc_drop"], "LSTM: permutation importance",
           "drop in test PR-AUC when the feature is shuffled", "24_lstm_permutation_importance",
           color=P.SERIES[4], fmt="{:.4f}")

    comp = pd.DataFrame({"shap_xgboost": shap_imp, "ig_mlp": ig_imp})
    comp["shap_rank"] = comp.shap_xgboost.rank(ascending=False).astype(int)
    comp["ig_rank"] = comp.ig_mlp.rank(ascending=False).astype(int)
    comp = comp.sort_values("shap_rank").round(5).reset_index(names="feature")
    save_table(comp, "exp7_explainability_comparison")
    save_table(perm.round(5), "exp7_lstm_permutation_importance")
    return {"spearman_shap_vs_ig": round(rho, 3), "top5_shap": list(shap_imp.index[:5]),
            "top5_ig": list(ig_imp.index[:5]), "top5_lstm": list(perm.feature[:5])}


def exp7b_explanation_agreement_vs_leakage(df, split, fs, pos_w):
    """Do SHAP (XGBoost) and IG (MLP) agree more once the interaction leak is removed?"""
    log("Experiment 7b: explanation agreement with vs without the geographic pair")
    te = split["test"]
    out = {}
    for tier in ["T2a_behavioural_with_geo_pair", "T2_behavioural"]:
        cols = fs[tier]
        xgb, _, _ = fit_classical("xgboost", df, cols, split, "class_weight", pos_w)
        sv, Xt, _ = shap_tree(xgb, df.iloc[te][cols], max_rows=len(te))
        mlp, _, _ = train_mlp(df, cols, split, "class_weight", C.SEED)
        ig = integrated_gradients(mlp, df.iloc[te][cols].values.astype(np.float32))
        s_imp, i_imp = global_importance(sv, list(Xt.columns)), global_importance(ig, cols)
        out[tier] = {"spearman": round(rank_agreement(s_imp, i_imp), 3),
                     "top3_shap": list(s_imp.index[:3]), "top3_ig": list(i_imp.index[:3])}
    save_table(pd.DataFrame([{"feature_set": k, **v} for k, v in out.items()]), "exp7b_agreement_vs_leakage")
    return out


# ===========================================================================
def main(fast=False):
    t0 = time.time()
    df = prepare_dataset()
    split = get_split(df)
    fs = feature_sets(df)
    y = df[C.TARGET].values
    pos_w = float((1 - y[split["train"]].mean()) / y[split["train"]].mean())
    seeds = C.DL_SEEDS[:1] if fast else C.DL_SEEDS

    audit, ss = stage_audit_eda(df, split)
    e1, rule = exp1_leakage(df, split, fs, pos_w)
    e3, best_strat = exp3_imbalance(df, split, fs, pos_w, fast)
    log(f"   imbalance strategy chosen on validation PR-AUC: {best_strat}")
    e4 = exp4_features(df, split, fs, pos_w, best_strat)
    e5, shared = exp5_split(df, split, fs, pos_w, best_strat)
    e6, dl_models, dl_preds, lstm_scaler, S_L = exp6_sequence(df, split, fs, best_strat, seeds)
    # feature set for the main comparison: chosen on mean VALIDATION PR-AUC in Exp 4
    tier = e4.groupby("setting")["val_pr_auc"].mean().idxmax()
    log(f"   feature set chosen on validation: {tier}")
    dl_models[MLP_KEY[tier]].save(C.MODELS_DIR / "mlp.pt")
    dl_models[("LSTM", "last 5 transactions, ordered")].save(C.MODELS_DIR / "lstm.pt")
    joblib.dump(lstm_scaler, C.MODELS_DIR / "lstm_scaler.joblib")
    e2, models, champion, params = exp2_main(df, split, fs, pos_w, best_strat, dl_preds, fast, rule, tier)
    e7 = exp7_explain(df, split, fs, models, dl_models, S_L, tier)
    e7["agreement_vs_leakage"] = exp7b_explanation_agreement_vs_leakage(df, split, fs, pos_w)

    summary = {
        "dataset": {"rows": len(df), "users": int(df[C.GROUP].nunique()), "fraud_rate": round(float(y.mean()), 4),
                    "columns_raw": 65},
        "split": ss.to_dict("records"),
        "feature_set_sizes": {k: len(v) for k, v in fs.items()},
        "n_legit_constant_columns": int(len(audit["legit_constant"])),
        "imbalance_strategy_selected": best_strat,
        "random_split_test_rows_with_seen_user": round(shared, 3),
        "main_feature_set": tier,
        "champion": champion,
        "hard_subset_test": {"rows": int((rule[split["test"]] == 0).sum()),
                             "fraud_rows": int(y[split["test"]][rule[split["test"]] == 0].sum())},
        "explainability": e7,
        "runtime_minutes": round((time.time() - t0) / 60, 1),
        "fast_mode": fast,
    }
    json.dump(summary, open(C.ROOT / "reports" / "summary.json", "w"), indent=2, default=str)
    log(f"Done in {summary['runtime_minutes']} min. Champion: {champion}")
    print(e1.to_string()); print(e2.to_string()); print(e6.to_string())


def explain_only():
    """Re-run only Experiment 7 (explainability figures/tables) from the saved models (~1-2 min)."""
    df = prepare_dataset()
    split = get_split(df)
    fs = feature_sets(df)
    meta = json.load(open(C.MODELS_DIR / "model_meta.json"))
    tier = meta["feature_set"]
    models = {"XGBoost": joblib.load(C.MODELS_DIR / "xgboost.joblib")}
    sc = joblib.load(C.MODELS_DIR / "lstm_scaler.joblib")
    F = sc.transform(df[C.BEHAVIOURAL_COLS].values.astype(np.float32)).astype(np.float32)
    S, L = build_sequences(df, F)
    dl_models = {MLP_KEY[tier]: TorchClassifier.load(C.MODELS_DIR / "mlp.pt"),
                 ("LSTM", "last 5 transactions, ordered"): TorchClassifier.load(C.MODELS_DIR / "lstm.pt")}
    e7 = exp7_explain(df, split, fs, models, dl_models, (S, L), tier)
    p = C.ROOT / "reports" / "summary.json"
    summ = json.load(open(p))
    summ["explainability"].update(e7)
    json.dump(summ, open(p, "w"), indent=2, default=str)
    log("Explainability outputs regenerated.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--explain-only", action="store_true",
                    help="only regenerate Experiment 7 outputs from the saved models")
    a = ap.parse_args()
    explain_only() if a.explain_only else main(a.fast)
