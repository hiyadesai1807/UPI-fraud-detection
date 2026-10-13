"""Member 1 — data loading and cleaning."""
import numpy as np
import pandas as pd

from . import config as C


def _mmss_to_seconds(s: pd.Series) -> pd.Series:
    """The raw 'timestamp' only holds 'MM:SS.f' (the date and hour were lost,
    most likely when the CSV was saved from a spreadsheet). Convert to seconds."""
    parts = s.astype(str).str.split(":", n=1, expand=True)
    mins = pd.to_numeric(parts[0], errors="coerce")
    secs = pd.to_numeric(parts[1], errors="coerce")
    return mins * 60 + secs


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def empty_or_constant_columns(df: pd.DataFrame, missing_threshold: float = 0.95):
    """Columns that are >=95 % missing or have a single value."""
    out = []
    for c in df.columns:
        if c == C.TARGET:
            continue
        if df[c].isna().mean() >= missing_threshold or df[c].nunique(dropna=True) <= 1:
            out.append(c)
    return out


def load_clean() -> pd.DataFrame:
    """Load the CSV and add parsed time fields and an intra-day ordering key.

    order_key = hour * 3600 + MM:SS seconds. Because the date is missing this is
    an *intra-day pseudo-chronology*, not a true timeline (documented limitation).
    """
    df = load_raw()
    df = df.drop_duplicates(subset="transaction_id").reset_index(drop=True)
    df["mmss_seconds"] = _mmss_to_seconds(df["timestamp"])
    df["order_key"] = df["transaction_time_of_day"] * 3600 + df["mmss_seconds"].fillna(0)
    df = df.sort_values([C.GROUP, "order_key", "transaction_id"]).reset_index(drop=True)
    return df


def basic_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "missing_pct": round(df[c].isna().mean() * 100, 2),
            "n_unique": df[c].nunique(dropna=True),
            "example": df[c].dropna().iloc[0] if df[c].notna().any() else None,
        })
    return pd.DataFrame(rows)
