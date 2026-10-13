"""Member 2 — feature tiers, leakage-safe user-history features, preprocessing."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .data_loader import empty_or_constant_columns


# ---------------------------------------------------------------------------
# User-history features (prior transactions only, no labels used)
# ---------------------------------------------------------------------------
def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expects df sorted by (user_id, order_key). Every statistic is computed
    from the user's *earlier* transactions only (shift(1) before expanding), so
    no information from the current or future transactions leaks in."""
    df = df.copy()
    g = df.groupby(C.GROUP, sort=False)

    def prior_mean(col):
        return g[col].transform(lambda s: s.shift(1).expanding().mean())

    def prior_std(col):
        return g[col].transform(lambda s: s.shift(1).expanding().std())

    df["hist_prior_count"] = g.cumcount()
    pm_amt, ps_amt = prior_mean("amount"), prior_std("amount")
    df["hist_amount_ratio"] = (df["amount"] / pm_amt).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["hist_amount_z"] = ((df["amount"] - pm_amt) / ps_amt).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10, 10)
    df["hist_keyboard_dev"] = (df["keyboard_input_speed"] - prior_mean("keyboard_input_speed")).fillna(0.0)
    df["hist_pin_dev"] = (df["pin_entry_speed"] - prior_mean("pin_entry_speed")).fillna(0.0)
    df["hist_timing_dev"] = (df["input_timing_consistency"] - prior_mean("input_timing_consistency")).fillna(0.0)
    df["hist_secs_since_prev"] = g["order_key"].diff().fillna(86400.0)
    # is this device new for the user?
    df["hist_device_new"] = (~df.duplicated(subset=[C.GROUP, "device_id"])).astype(int)
    return df


# ---------------------------------------------------------------------------
# Feature tiers
# ---------------------------------------------------------------------------
def feature_sets(df: pd.DataFrame) -> dict:
    drop = set(C.ID_TEXT_COLS) | set(empty_or_constant_columns(df)) | {
        C.TARGET, "mmss_seconds", "order_key"} | set(C.HISTORY_COLS)
    t0 = [c for c in df.columns if c not in drop]
    t1 = [c for c in t0 if c not in C.LABEL_PROXY_COLS]
    t2 = list(C.BEHAVIOURAL_COLS)
    t2h = t2 + list(C.HISTORY_COLS)
    return {
        "T0_all_features": t0,
        "T1_proxies_removed": t1,
        "T2a_behavioural_with_geo_pair": t2 + list(C.GEO_PAIR_COLS),
        "T2_behavioural": t2,
        "T2H_behavioural_history": t2h,
    }


def make_preprocessor(df: pd.DataFrame, cols: list, scale: bool = True) -> ColumnTransformer:
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num)]
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
        ]), cat))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def prepare_dataset() -> pd.DataFrame:
    """Load -> clean -> history features. Cached to data/processed."""
    from .data_loader import load_clean
    out = C.PROCESSED_DIR / "transactions_with_history.parquet"
    if out.exists():
        return pd.read_parquet(out)
    df = add_history_features(load_clean())
    try:
        df.to_parquet(out, index=False)
    except Exception:
        pass
    return df
