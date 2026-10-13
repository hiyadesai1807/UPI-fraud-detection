"""Member 1 — data-quality and leakage audit.

Four checks:
  1. single-feature ROC-AUC (how well one column alone predicts fraud)
  2. category levels that are 100 % fraud or 100 % legitimate
  3. 'legit-constant' columns: every legitimate row has the same value, so any
     other value means fraud (a scenario-injection artifact of the generator)
  4. out-of-legitimate-range values for continuous features
  5. pairwise equality between numeric columns (interaction leakage)
Plus a transparent rule baseline that uses only these artifacts.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C


def _encode_for_auc(s: pd.Series, y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(s.median() if s.notna().any() else 0)
    # categorical: replace with in-sample fraud rate (audit only, not a model)
    rate = y.groupby(s.fillna("NA")).mean()
    return s.fillna("NA").map(rate)


def single_feature_auc(df: pd.DataFrame, max_card: int = 200) -> pd.DataFrame:
    y = df[C.TARGET]
    rows = []
    for c in df.columns:
        if c in (C.TARGET, "order_key", "mmss_seconds") or c in C.ID_TEXT_COLS:
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) and s.nunique() > max_card:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        auc = roc_auc_score(y, _encode_for_auc(s, y))
        rows.append({"feature": c, "auc": round(max(auc, 1 - auc), 4)})
    out = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    out["flag"] = np.select(
        [out.auc >= 0.99, out.auc >= 0.90], ["LABEL PROXY", "SUSPICIOUS"], default="")
    return out


def pure_levels(df: pd.DataFrame, min_count: int = 50, max_card: int = 30) -> pd.DataFrame:
    y = df[C.TARGET]
    rows = []
    for c in df.columns:
        if c == C.TARGET or df[c].nunique() > max_card:
            continue
        t = y.groupby(df[c].astype(str)).agg(["mean", "size"])
        for lvl, r in t.iterrows():
            if r["size"] >= min_count and r["mean"] in (0.0, 1.0):
                rows.append({"feature": c, "level": lvl, "fraud_rate": r["mean"], "rows": int(r["size"])})
    return pd.DataFrame(rows)


def legit_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    y = df[C.TARGET]
    rows = []
    for c in df.columns:
        if c == C.TARGET:
            continue
        legit = df.loc[y == 0, c]
        if legit.nunique(dropna=False) == 1:
            legit_val = legit.iloc[0]
            fraud = df.loc[y == 1, c]
            differs = ~(fraud.astype(str) == str(legit_val))
            rows.append({"feature": c, "legit_value": legit_val,
                         "fraud_rows_with_other_value_pct": round(differs.mean() * 100, 2)})
    return pd.DataFrame(rows).sort_values("fraud_rows_with_other_value_pct", ascending=False)


def out_of_range(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Share of fraud rows lying outside the [min, max] range of legitimate rows."""
    y = df[C.TARGET]
    cols = cols or C.BEHAVIOURAL_COLS
    rows = []
    for c in cols:
        lo, hi = df.loc[y == 0, c].min(), df.loc[y == 0, c].max()
        f = df.loc[y == 1, c]
        rows.append({"feature": c, "legit_min": lo, "legit_max": hi,
                     "fraud_outside_pct": round(((f < lo) | (f > hi)).mean() * 100, 2)})
    return pd.DataFrame(rows).sort_values("fraud_outside_pct", ascending=False)


def pairwise_equality(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Interaction leakage: pairs of numeric columns that are *exactly equal*
    far more often in one class than the other. Univariate AUC cannot see this."""
    y = df[C.TARGET].values
    cols = cols or (C.BEHAVIOURAL_COLS + C.GEO_PAIR_COLS)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            eq = np.isclose(df[a].values, df[b].values)
            le, fe = eq[y == 0].mean(), eq[y == 1].mean()
            if abs(le - fe) > 0.20:          # large class asymmetry only
                rows.append({"feature_a": a, "feature_b": b, "equal_in_legit_pct": round(le * 100, 2),
                             "equal_in_fraud_pct": round(fe * 100, 2),
                             "auc_of_abs_difference": round(roc_auc_score(y, np.abs(df[a] - df[b])), 4)})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "equal_in_legit_pct",
                                       "equal_in_fraud_pct", "auc_of_abs_difference"])


def rule_baseline(df: pd.DataFrame, train_mask: np.ndarray, cols=None) -> np.ndarray:
    """Transparent rule: flag a transaction if ANY behavioural feature lies outside
    the range seen for legitimate *training* transactions. Learns only min/max."""
    cols = cols or C.BEHAVIOURAL_COLS
    tr = df.loc[train_mask & (df[C.TARGET] == 0).values, cols]
    lo, hi = tr.min(), tr.max()
    X = df[cols]
    return ((X.lt(lo, axis=1)) | (X.gt(hi, axis=1))).any(axis=1).astype(int).values


def run_full_audit(df: pd.DataFrame) -> dict:
    from .data_loader import empty_or_constant_columns, basic_profile
    res = {
        "profile": basic_profile(df),
        "empty_or_constant": pd.DataFrame({"feature": empty_or_constant_columns(df)}),
        "single_feature_auc": single_feature_auc(df),
        "pure_levels": pure_levels(df),
        "legit_constant": legit_constant_columns(df),
        "out_of_range": out_of_range(df),
        "pairwise_equality": pairwise_equality(df),
    }
    for k, v in res.items():
        v.to_csv(C.TABLE_DIR / f"audit_{k}.csv", index=False)
    return res
