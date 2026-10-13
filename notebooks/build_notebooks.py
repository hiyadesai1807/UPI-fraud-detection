"""Generates the five member notebooks. Run from the project root:
    python notebooks/build_notebooks.py
then execute them with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SETUP = """import sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from IPython.display import Image, display
from src import config as C
pd.set_option('display.max_columns', 40); pd.set_option('display.width', 200)
def show(name): display(Image(filename=str(C.FIG_DIR / f'{name}.png')))
def tbl(name): return pd.read_csv(C.TABLE_DIR / f'{name}.csv')"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


NOTEBOOKS = {
"01_data_audit_and_eda.ipynb": [
"md:# Member 1 — Data audit, leakage audit & EDA\n**Owner:** Member 1  \n**Goal:** establish whether the data can be trusted before any model is trained.\n\nModules: `src/data_loader.py`, `src/leakage_audit.py`",
SETUP,
"md:## 1. Load and profile",
"""from src.data_loader import load_raw, load_clean, basic_profile, empty_or_constant_columns
raw = load_raw()
print('shape:', raw.shape, '| duplicates:', raw.duplicated().sum(), '| fraud rate:', round(raw.is_fraud.mean(), 4))
basic_profile(raw).head(65)""",
"""print('Empty (>=95% missing) or single-valued columns:')
empty_or_constant_columns(raw)""",
"md:## 2. The timestamp problem\nThe `timestamp` column only holds `MM:SS.f` — the date and hour were lost (likely a spreadsheet export). A true chronological split is therefore impossible; we use a **user-grouped split** instead and build an *intra-day pseudo-order* (hour + mm:ss) for sequences.",
"""print(raw.timestamp.sample(8, random_state=0).tolist())
df = load_clean()
df[['user_id','transaction_time_of_day','timestamp','mmss_seconds','order_key']].head(8)""",
"md:## 3. Leakage audit\n### 3a. Single-feature AUC — can one column predict the label on its own?",
"""from src.leakage_audit import single_feature_auc, legit_constant_columns, pure_levels, out_of_range
sfa = single_feature_auc(df); sfa.head(15)""",
"""pd.crosstab(df.handle_verification_status, df.is_fraud)   # a perfect label proxy""",
"md:### 3b. Scenario signatures — columns that are constant for every legitimate row",
"""lc = legit_constant_columns(df)
print(len(lc), 'columns have one single value across all 21,848 legitimate rows')
lc.head(20)""",
"md:### 3c. Interaction leakage — a pair of columns that is only *jointly* leaky",
"""from src.leakage_audit import pairwise_equality
display(pairwise_equality(df))
for c in C.GEO_PAIR_COLS:
    from sklearn.metrics import roc_auc_score
    print(c, 'single-feature AUC =', round(roc_auc_score(df.is_fraud, df[c]), 3))
show('03b_geo_pair_interaction_leak')""",
"md:Each geographic column alone is useless (AUC ≈ 0.50), yet the two are **identical in every legitimate row**. This leak was first spotted when the MLP's Integrated-Gradients ranking put these columns at the top (Member 5); it is excluded from the strict Tier 2.",
"md:### 3d. Behavioural features — fraud outside the legitimate range",
"""out_of_range(df)""",
"""show('01_single_feature_auc'); show('02_legit_constant_columns'); show('03_out_of_range')""",
"md:**Audit conclusion.** Three columns are label proxies (Tier 0 → Tier 1). A further 39 columns are scenario-injection artefacts, and one column pair leaks through its interaction. 12 behavioural / contextual columns remain; these form the strict Tier 2 used for the main comparison. Even these have hard range boundaries in the synthetic generator, which we quantify with a rule baseline and a *hard subset* evaluation.",
"md:## 4. Exploratory data analysis",
"""show('04_class_balance'); show('05_behavioural_distributions'); show('06_fraud_rate_by_hour'); show('07_correlation_matrix')""",
"""tbl('eda_medians_by_class')""",
"md:## 5. Experiment 1 result — how much does leakage inflate performance?",
"""display(tbl('exp1_leakage_tiers')[['model','setting','n_features','pr_auc','roc_auc','precision','recall','f1']]); show('10_exp1_leakage')""",
],

"02_features_split_imbalance.ipynb": [
"md:# Member 2 — Feature engineering, data split & class imbalance\nModules: `src/features.py`, `src/splits.py`",
SETUP,
"""from src.features import prepare_dataset, feature_sets
from src.splits import get_split, split_summary
df = prepare_dataset(); split = get_split(df); fs = feature_sets(df)
{k: len(v) for k, v in fs.items()}""",
"md:## 1. Feature tiers\n| Tier | Content |\n|---|---|\n| T0 | every usable column |\n| T1 | T0 minus 3 label proxies |\n| T2a | T2 + the leaky geographic pair (Exp 1 only) |\n| T2 | 12 strict behavioural / contextual features |\n| T2H | T2 + 8 leakage-safe user-history features |",
"""for k, v in fs.items(): print(k, '->', v[:8], '...' if len(v) > 8 else '')""",
"md:## 2. User-history features — computed from *earlier* transactions only\nEach statistic uses `shift(1)` before the expanding window, so the current transaction never sees itself or the future. Labels are never used.",
"""u = df.user_id.value_counts().index[0]
df[df.user_id == u][['user_id','order_key','amount','hist_prior_count','hist_amount_ratio','hist_amount_z','hist_device_new','is_fraud']].head(10)""",
"md:## 3. Shared split — grouped by user\n59 % of users have both fraud and legitimate transactions, so rows of the same user must stay in one partition.",
"""ss = split_summary(df, split); print('user overlap between partitions:', ss.attrs['user_overlap']); ss""",
"md:## 4. Experiment 3 — class imbalance (none vs class weights vs SMOTE)\nSMOTE sits *inside* the pipeline, so it is fitted on training data only; validation and test keep the natural 17 % fraud rate. The strategy for each model is chosen on **validation** PR-AUC.",
"""e3 = tbl('exp3_imbalance'); e3[['model','setting','val_pr_auc','pr_auc','precision','recall','f1','fp','fn']]""",
"""show('12_exp3_imbalance_prauc'); show('12_exp3_imbalance_recall')""",
"md:## 5. Experiment 4 — do user-history features help?",
"""display(tbl('exp4_feature_engineering')[['model','setting','val_pr_auc','pr_auc','recall','precision','f1']]); show('13_exp4_features')""",
"md:## 6. Experiment 5 — grouped vs random split",
"""tbl('exp5_split_strategy')[['model','setting','pr_auc','roc_auc','recall','precision','f1']]""",
],

"03_classical_ml.ipynb": [
"md:# Member 3 — Logistic Regression, Random Forest, XGBoost\nModule: `src/models_classical.py`. Hyper-parameters tuned with `RandomizedSearchCV` + `StratifiedGroupKFold(5)` on the training users, scored on PR-AUC.",
SETUP,
"""import joblib
from src.features import prepare_dataset
from src.splits import get_split
from src.evaluation import evaluate
meta = json.loads((C.MODELS_DIR / 'model_meta.json').read_text())
df = prepare_dataset(); split = get_split(df); cols = meta['feature_columns']; y = df.is_fraud.values
print('feature set:', meta['feature_set']); json.dumps(meta['best_params'], indent=1)""",
"md:## 1. Re-evaluate the saved tuned models on the test set",
"""rows = []
for f, name in [('logreg','Logistic Regression'), ('random_forest','Random Forest'), ('xgboost','XGBoost')]:
    m = joblib.load(C.MODELS_DIR / f'{f}.joblib')
    pv = m.predict_proba(df.iloc[split['val']][cols])[:, 1]
    pt = m.predict_proba(df.iloc[split['test']][cols])[:, 1]
    r = evaluate(y[split['val']], pv, y[split['test']], pt); r['model'] = name; rows.append(r)
pd.DataFrame(rows).set_index('model')[['pr_auc','roc_auc','precision','recall','f1','fp','fn','threshold']].round(4)""",
"md:## 2. Logistic Regression coefficients (standardised features)",
"""lr = joblib.load(C.MODELS_DIR / 'logreg.joblib')
names = lr.named_steps['prep'].get_feature_names_out()
pd.Series(lr.named_steps['model'].coef_[0], index=names).sort_values(key=np.abs, ascending=False).round(3).to_frame('coef')""",
"md:## 3. Experiment 2 — full comparison incl. deep models, bootstrap CI, hard subset",
"""tbl('exp2_model_comparison')""",
"""show('11_pr_curves'); show('16_confusion_matrix_champion')""",
"md:## 4. Is the champion significantly better? (McNemar on test decisions)",
"""tbl('exp2_mcnemar_tests')""",
"md:## 5. Threshold & illustrative cost analysis",
"""show('17_threshold_and_cost')""",
],

"04_deep_learning_mlp_lstm.ipynb": [
"md:# Member 4 — Deep learning: MLP and LSTM\nModule: `src/models_deep.py` (PyTorch).\n\n* **MLP**: 3 hidden layers (128-64-32), BatchNorm, ReLU, Dropout 0.3, AdamW, early stopping on validation PR-AUC.\n* **LSTM**: 2-layer LSTM (64 units) over the user's last 5 transactions; the head also sees the current transaction.\n* Each configuration is trained with 3 seeds; mean ± sd is reported.",
SETUP,
"""from src.features import prepare_dataset
from src.splits import get_split
from src.models_deep import TorchClassifier, build_sequences
from sklearn.metrics import average_precision_score
import joblib
meta = json.loads((C.MODELS_DIR / 'model_meta.json').read_text())
df = prepare_dataset(); split = get_split(df); y = df.is_fraud.values""",
"md:## 1. What an LSTM input looks like\nOne row = one transaction + up to 4 earlier transactions of the same user (left-aligned, zero-padded).",
"""sc = joblib.load(C.MODELS_DIR / 'lstm_scaler.joblib')
F = sc.transform(df[C.BEHAVIOURAL_COLS].values).astype('float32')
S, L = build_sequences(df, F)
print('tensor shape (transactions, steps, features):', S.shape)
print('sequence-length distribution:', dict(zip(*np.unique(L, return_counts=True))))
i = int(np.where(L == 5)[0][0]); pd.DataFrame(S[i], columns=C.BEHAVIOURAL_COLS).round(2)""",
"md:## 2. Short demonstration training of an MLP (verbose)",
"""X = df[meta['feature_columns']].values.astype('float32')
demo = TorchClassifier('mlp', strategy='class_weight', epochs=8, verbose=True)
demo.fit(X[split['train']], y[split['train']], X[split['val']], y[split['val']])""",
"md:## 3. Evaluate the saved models on the test set",
"""mlp = TorchClassifier.load(C.MODELS_DIR / 'mlp.pt'); lstm = TorchClassifier.load(C.MODELS_DIR / 'lstm.pt')
te = split['test']
print('MLP  test PR-AUC:', round(average_precision_score(y[te], mlp.predict_proba(X[te])), 4))
print('LSTM test PR-AUC:', round(average_precision_score(y[te], lstm.predict_proba(S[te], L[te])), 4))""",
"md:## 4. Experiment 6 — does transaction order carry signal?",
"""tbl('exp6_sequence_summary')""",
"""show('14_exp6_sequence'); show('15_dl_training_curves')""",
"""tbl('exp6_sequence_runs')[['model','setting','seed','epochs','val_pr_auc','pr_auc','recall','precision']]""",
"md:**Interpretation guide.** If *ordered ≈ shuffled*, order carries no extra signal. If *sequence length 1 ≥ length 5*, earlier transactions add noise rather than information — consistent with a generator that labels transactions independently of the user's history.",
],

"05_explainability_and_dashboard.ipynb": [
"md:# Member 5 — Explainable AI, dashboard & integration\nModule: `src/explain.py`, app: `app/streamlit_app.py`.",
SETUP,
"""summary = json.loads((C.ROOT / 'reports' / 'summary.json').read_text()); summary['explainability']""",
"md:## 1. XGBoost — SHAP (TreeExplainer)",
"""show('20_shap_global_xgboost'); show('21_shap_beeswarm_xgboost'); show('22_shap_local_example')""",
"md:## 2. MLP — Integrated Gradients\nIG integrates the gradient along a straight path from a baseline (the training mean) to the input; attributions sum to the change in logit.",
"""from src.features import prepare_dataset
from src.splits import get_split
from src.models_deep import TorchClassifier
from src.explain import integrated_gradients
meta = json.loads((C.MODELS_DIR / 'model_meta.json').read_text())
df = prepare_dataset(); split = get_split(df); cols = meta['feature_columns']
mlp = TorchClassifier.load(C.MODELS_DIR / 'mlp.pt')
X = df.iloc[split['test']][cols].values.astype('float32')[:300]
ig = integrated_gradients(mlp, X)
import torch
with torch.no_grad():
    lx = mlp.model(torch.from_numpy(mlp._scale(X))); l0 = mlp.model(torch.zeros(1, X.shape[1]))
print('completeness check (sum of attributions vs logit difference), mean abs error:',
      float(np.abs(ig.sum(1) - (lx - l0).numpy()).mean()).__round__(4))
show('23_ig_global_mlp')""",
"md:## 3. LSTM — permutation importance",
"""show('24_lstm_permutation_importance'); tbl('exp7_lstm_permutation_importance')""",
"md:## 4. Do the explanations agree?",
"""tbl('exp7_explainability_comparison')""",
"md:## 5. Explanations as a leakage detector (Exp 7b)\nIf two very different models explain the same data differently, ask why. With the leaky geographic pair the MLP ranked `geographic_disparity` in its top 3 while XGBoost did not; once the pair was removed, agreement rose sharply.",
"""tbl('exp7b_agreement_vs_leakage')""",
"md:## 6. Dashboard\nFrom the project root run:\n```bash\nstreamlit run app/streamlit_app.py\n```\nPages: Overview · Leakage audit · Model comparison · Deep learning & sequences · Transaction monitor (risk scores + per-transaction SHAP) · Threshold & cost (editable cost assumptions) · Score a transaction (what-if) · Explainability.",
],
}

if __name__ == "__main__":
    for fname, cells in NOTEBOOKS.items():
        nbf.write(nb(cells), HERE / fname)
        print("wrote", fname)
