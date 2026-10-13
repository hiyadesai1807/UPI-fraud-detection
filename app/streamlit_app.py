"""Member 5 — UPI Fraud Risk Monitor (Streamlit research prototype).

Run from the project root:
    streamlit run app/streamlit_app.py
Requires `python run_all.py` to have been run first (models + reports).
"""
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.evaluation import expected_cost, metrics_at  # noqa: E402

TABLES, FIGS = C.TABLE_DIR, C.FIG_DIR
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
MODEL_COLS = {"Logistic Regression": "p_logistic_regression", "Random Forest": "p_random_forest",
              "XGBoost": "p_xgboost", "MLP": "p_mlp", "LSTM": "p_lstm"}

st.set_page_config(page_title="UPI Fraud Risk Monitor", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------------------
@st.cache_data
def table(name):
    return pd.read_csv(TABLES / f"{name}.csv")


@st.cache_data
def meta():
    return json.loads((C.MODELS_DIR / "model_meta.json").read_text())


@st.cache_data
def summary():
    return json.loads((ROOT / "reports" / "summary.json").read_text())


@st.cache_data
def features_df():
    from src.features import prepare_dataset
    return prepare_dataset()


@st.cache_resource
def xgb_model():
    return joblib.load(C.MODELS_DIR / "xgboost.joblib")


def fig(name, caption=None):
    p = FIGS / f"{name}.png"
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


def risk_band(p, thr):
    if p >= max(thr, 0.8):
        return "🔴 High risk"
    if p >= thr:
        return "🟠 Flag for review"
    if p >= thr / 2:
        return "🟡 Watch"
    return "🟢 Low risk"


def contrib_chart(contrib: pd.Series):
    d = contrib.reset_index()
    d.columns = ["feature", "contribution"]
    d["direction"] = np.where(d.contribution > 0, "towards fraud", "towards legitimate")
    return alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("contribution:Q", title="SHAP contribution (log-odds)"),
        y=alt.Y("feature:N", sort=alt.EncodingSortField("contribution", op="sum", order="descending"), title=None),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["towards fraud", "towards legitimate"],
                                                        range=[ORANGE, BLUE]), legend=alt.Legend(orient="bottom")),
        tooltip=["feature", alt.Tooltip("contribution:Q", format=".3f")],
    ).properties(height=320)


# ---------------------------------------------------------------------------
if not (C.MODELS_DIR / "model_meta.json").exists():
    st.error("Models not found. Run `python run_all.py` from the project root first.")
    st.stop()

M, S = meta(), summary()
preds = table("test_predictions")
e2 = table("exp2_model_comparison")
champion = M["champion"]

st.sidebar.title("🛡️ UPI Fraud Risk Monitor")
st.sidebar.caption("Leakage-aware ML + DL fraud detection · research prototype")
page = st.sidebar.radio("Go to", ["Overview", "Leakage audit", "Model comparison",
                                  "Deep learning & sequences", "Transaction monitor",
                                  "Threshold & cost", "Score a transaction", "Explainability"])
st.sidebar.divider()
st.sidebar.caption("Synthetic Kaggle dataset — results are experimental and do not "
                   "represent real UPI performance. Not connected to any bank or NPCI system.")

# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Leakage-aware & explainable UPI fraud detection")
    champ = e2[e2.model == champion].iloc[0]
    c = st.columns(3)
    c[0].metric("Transactions", f"{S['dataset']['rows']:,}")
    c[1].metric("Users", f"{S['dataset']['users']:,}")
    c[2].metric("Fraud rate", f"{S['dataset']['fraud_rate']:.1%}")
    c = st.columns(4)
    c[0].metric("Champion model", champion)
    c[1].metric("Test PR-AUC", f"{champ.pr_auc:.3f}")
    c[2].metric("Recall", f"{champ.recall:.1%}")
    c[3].metric("Precision", f"{champ.precision:.1%}")
    st.markdown(f"""
**Pipeline:** data audit → leakage audit → EDA → feature tiers & user-history features →
user-grouped train/val/test split → imbalance strategies → Logistic Regression, Random Forest,
XGBoost, **MLP**, **LSTM** → evaluation (PR-AUC, bootstrap CI, McNemar) → SHAP / Integrated
Gradients → this dashboard.

**Headline findings**
- With all features, models score PR-AUC ≈ 1.000: the dataset contains label proxies
  (`handle_verification_status` alone separates the classes perfectly) and
  **{S['n_legit_constant_columns']} scenario-signature columns** that are constant for every legitimate row,
  plus a geographic column pair that leaks only through its interaction.
- On the leakage-safe behavioural feature set ({S['main_feature_set']}), the champion
  **{champion}** reaches PR-AUC {champ.pr_auc:.3f} (95 % user-bootstrap CI {champ.pr_auc_ci_low:.3f}–{champ.pr_auc_ci_high:.3f}).
- A transparent *range rule* already catches much of the fraud; on the **hard subset**
  ({S['hard_subset_test']['rows']:,} test rows, {S['hard_subset_test']['fraud_rows']} frauds) where no feature is
  out of range, the best PR-AUC drops to {e2.hard_subset_pr_auc.max():.3f}.
""")
    fig("10_exp1_leakage")

elif page == "Leakage audit":
    st.title("Data-quality & leakage audit")
    t1, t2, t5, t3, t4 = st.tabs(["Single-feature AUC", "Scenario signatures", "Interaction leak",
                                  "Out-of-range", "Tier experiment"])
    with t1:
        st.markdown("A single column predicting the label with AUC ≥ 0.99 is a **label proxy**, not a signal.")
        fig("01_single_feature_auc")
        st.dataframe(table("audit_single_feature_auc"), width="stretch", hide_index=True)
    with t2:
        st.markdown("Columns where **every legitimate row has the same value** — any other value means fraud. "
                    "These are artefacts of how the synthetic generator injected fraud scenarios.")
        fig("02_legit_constant_columns")
        st.dataframe(table("audit_legit_constant"), width="stretch", hide_index=True)
    with t5:
        st.markdown("Two geographic columns are each useless alone (AUC ≈ 0.50) but are **identical in every "
                    "legitimate row**. Univariate checks miss this; it was spotted through the MLP's "
                    "Integrated-Gradients ranking and confirmed by a pairwise-equality check.")
        fig("03b_geo_pair_interaction_leak")
        st.dataframe(table("audit_pairwise_equality"), width="stretch", hide_index=True)
    with t3:
        fig("03_out_of_range")
        st.dataframe(table("audit_out_of_range"), width="stretch", hide_index=True)
    with t4:
        st.dataframe(table("exp1_leakage_tiers"), width="stretch", hide_index=True)
        fig("10_exp1_leakage")

elif page == "Model comparison":
    st.title("Model comparison (test set)")
    show = ["model", "setting", "val_pr_auc", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "roc_auc",
            "precision", "recall", "f1", "fpr", "fp", "fn", "hard_subset_pr_auc", "threshold", "champion"]
    st.dataframe(e2[[c for c in show if c in e2.columns]], width="stretch", hide_index=True)
    st.caption("Threshold chosen on validation (max F1). Champion = best validation PR-AUC. "
               "CI = 95 % bootstrap over users. Hard subset = test rows with every feature inside the legitimate range.")
    a, b = st.columns(2)
    with a:
        fig("11_pr_curves")
    with b:
        fig("16_confusion_matrix_champion")
    st.subheader("McNemar tests vs champion")
    st.dataframe(table("exp2_mcnemar_tests"), width="stretch", hide_index=True)
    st.subheader("Class-imbalance strategies (Exp 3)")
    a, b = st.columns(2)
    with a:
        fig("12_exp3_imbalance_prauc")
    with b:
        fig("12_exp3_imbalance_recall")
    st.subheader("Feature engineering (Exp 4) and split strategy (Exp 5)")
    a, b = st.columns(2)
    with a:
        fig("13_exp4_features")
    with b:
        st.dataframe(table("exp5_split_strategy")[["model", "setting", "pr_auc", "recall", "precision", "f1"]],
                     width="stretch", hide_index=True)

elif page == "Deep learning & sequences":
    st.title("MLP vs LSTM — does transaction order carry signal?")
    st.markdown("""
The LSTM reads a user's last 5 transactions (oldest → current) and scores the current one.
Control conditions: **sequence length 1** (no history) and **history shuffled** (same transactions,
random order). If ordered ≈ shuffled, the LSTM's gain (if any) comes from *which* transactions it sees,
not from their *order*.
> The raw timestamp only contains minutes:seconds, so ordering uses hour-of-day + mm:ss — an
> intra-day pseudo-chronology. This is a documented limitation.
""")
    fig("14_exp6_sequence")
    st.dataframe(table("exp6_sequence_summary"), width="stretch", hide_index=True)
    fig("15_dl_training_curves")

elif page == "Transaction monitor":
    st.title("Transaction monitor (held-out test transactions)")
    c1, c2 = st.columns([1, 2])
    model = c1.selectbox("Model", list(MODEL_COLS), index=list(MODEL_COLS).index(champion))
    thr = c2.slider("Alert threshold", 0.01, 0.99, float(round(M["thresholds"][model], 2)), 0.01)
    p = preds[MODEL_COLS[model]]
    df = preds.assign(risk=p, alert=(p >= thr).astype(int),
                      band=[risk_band(v, thr) for v in p])
    m = metrics_at(df.is_fraud.values, p.values, thr)
    k = st.columns(5)
    k[0].metric("Alerts raised", f"{df.alert.sum():,}")
    k[1].metric("Fraud caught", f"{m['tp']:,} / {m['tp'] + m['fn']:,}")
    k[2].metric("Precision", f"{m['precision']:.1%}")
    k[3].metric("Recall", f"{m['recall']:.1%}")
    k[4].metric("False alarms", f"{m['fp']:,}")
    only = st.toggle("Show only alerted transactions", value=True)
    view = df[df.alert == 1] if only else df
    view = view.sort_values("risk", ascending=False)
    st.dataframe(view[["transaction_id", "user_id", "amount", "transaction_time_of_day", "risk", "band", "is_fraud"]]
                 .rename(columns={"is_fraud": "true_label"}).head(500),
                 width="stretch", hide_index=True,
                 column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="%.3f"),
                                "amount": st.column_config.NumberColumn(format="₹%.0f")})
    st.subheader("Explain one transaction (XGBoost SHAP)")
    tid = st.selectbox("Transaction", view.transaction_id.head(200))
    shap_df = table("shap_values_test_xgboost").set_index("transaction_id")
    if tid in shap_df.index:
        contrib = shap_df.loc[tid].sort_values(key=np.abs, ascending=False).head(10)
        r = df[df.transaction_id == tid].iloc[0]
        a, b = st.columns([1, 2])
        a.metric("Risk score", f"{r.risk:.3f}", help=f"model: {model}")
        a.write(r.band)
        a.write(f"True label: **{'Fraud' if r.is_fraud else 'Legitimate'}**")
        feats = features_df().set_index("transaction_id").loc[tid, M["feature_columns"]]
        a.dataframe(feats.rename("value").to_frame(), width="stretch")
        b.altair_chart(contrib_chart(contrib), width="stretch")
        b.caption("Suggested action: route to a fraud analyst for review — the model output is a risk score, "
                  "not proof of fraud.")

elif page == "Threshold & cost":
    st.title("Threshold & cost analysis")
    st.caption("All costs are illustrative assumptions you can change — not real UPI economics.")
    c = st.columns(3)
    model = c[0].selectbox("Model", list(MODEL_COLS), index=list(MODEL_COLS).index(champion))
    review = c[1].number_input("Review cost per alert (₹)", 0, 5000, 100, 50)
    friction = c[2].number_input("Extra cost per false alarm (₹)", 0, 20000, 1000, 100)
    y, p, amt = preds.is_fraud.values, preds[MODEL_COLS[model]].values, preds.amount.values
    ths = np.round(np.linspace(0.01, 0.99, 99), 2)
    rows = []
    for t in ths:
        m = metrics_at(y, p, t)
        tot, fn_c, rv = expected_cost(y, p, t, amt, review, friction)
        rows.append({"threshold": t, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                     "missed fraud (₹)": fn_c, "review + friction (₹)": rv, "total cost (₹)": tot})
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["total cost (₹)"].idxmin()]
    k = st.columns(4)
    k[0].metric("Cost-optimal threshold", f"{best.threshold:.2f}")
    k[1].metric("Total cost", f"₹{best['total cost (₹)']:,.0f}")
    k[2].metric("Recall at optimum", f"{best.recall:.1%}")
    k[3].metric("Precision at optimum", f"{best.precision:.1%}")
    long = curve.melt("threshold", ["precision", "recall", "f1"], var_name="metric")
    st.altair_chart(alt.Chart(long).mark_line(strokeWidth=2).encode(
        x="threshold:Q", y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("metric:N", scale=alt.Scale(range=[BLUE, ORANGE, "#1baf7a"])),
        tooltip=["threshold", "metric", alt.Tooltip("value:Q", format=".3f")]).properties(height=280),
        width="stretch")
    st.altair_chart(alt.Chart(curve).mark_line(strokeWidth=2, color=ORANGE).encode(
        x="threshold:Q", y=alt.Y("total cost (₹):Q"),
        tooltip=["threshold", alt.Tooltip("total cost (₹):Q", format=",.0f")]).properties(height=240),
        width="stretch")

elif page == "Score a transaction":
    st.title("Score a hypothetical transaction (XGBoost)")
    st.caption("Defaults are the median legitimate transaction. Change values to see how the risk responds.")
    data = features_df()
    cols = M["feature_columns"]
    legit = data[data.is_fraud == 0][cols]
    vals = {}
    grid = st.columns(3)
    for i, ccol in enumerate(cols):
        vals[ccol] = grid[i % 3].number_input(ccol, value=float(legit[ccol].median()), format="%.4f",
                                              help=f"legitimate range: {legit[ccol].min():.3g} – {legit[ccol].max():.3g}")
    x = pd.DataFrame([vals])[cols]
    pipe = xgb_model()
    prob = float(pipe.predict_proba(x)[:, 1][0])
    thr = M["thresholds"]["XGBoost"]
    st.metric("Fraud probability", f"{prob:.3f}", help=f"alert threshold {thr:.2f}")
    st.write(risk_band(prob, thr))
    import shap
    expl = shap.TreeExplainer(pipe.named_steps["model"])
    xt = pipe.named_steps["prep"].transform(x)
    sv = np.asarray(expl.shap_values(xt)).reshape(-1)[: len(cols)]
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    st.altair_chart(contrib_chart(pd.Series(sv, index=names).sort_values(key=np.abs, ascending=False).head(10)),
                    width="stretch")
    out_rng = [c for c in cols if vals[c] < legit[c].min() or vals[c] > legit[c].max()]
    if out_rng:
        st.warning("Outside the range of any legitimate training transaction: " + ", ".join(out_rng))

elif page == "Explainability":
    st.title("Explainability")
    st.markdown(f"Spearman rank agreement between XGBoost SHAP and MLP Integrated Gradients: "
                f"**ρ = {S['explainability']['spearman_shap_vs_ig']}**. Importance shows what a model "
                f"*uses*, not what *causes* fraud.")
    a, b = st.columns(2)
    with a:
        fig("20_shap_global_xgboost")
        fig("22_shap_local_example")
    with b:
        fig("21_shap_beeswarm_xgboost")
        fig("23_ig_global_mlp")
    fig("24_lstm_permutation_importance")
    st.dataframe(table("exp7_explainability_comparison"), width="stretch", hide_index=True)
    st.subheader("Explanations as a leakage detector")
    st.markdown("SHAP (XGBoost) vs Integrated Gradients (MLP) rank agreement **with** and **without** the "
                "jointly-leaky geographic pair. Disagreement between explanation methods was the first "
                "symptom of the interaction leak.")
    st.dataframe(table("exp7b_agreement_vs_leakage"), width="stretch", hide_index=True)
