"""Member 2 — shared train / validation / test split.

Split is by USER: 59 % of users have both fraud and legitimate transactions,
so a random row-level split would let models memorise users. All members
must use `get_split()` so that comparisons are fair.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

from . import config as C

SPLIT_FILE = C.PROCESSED_DIR / "split_indices.json"


def make_group_split(df: pd.DataFrame, seed: int = C.SEED) -> dict:
    idx = np.arange(len(df))
    groups = df[C.GROUP].values
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=seed)
    trval, test = next(gss.split(idx, groups=groups))
    rel_val = C.VAL_SIZE / (1 - C.TEST_SIZE)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tr_rel, val_rel = next(gss2.split(trval, groups=groups[trval]))
    return {"train": trval[tr_rel].tolist(), "val": trval[val_rel].tolist(), "test": test.tolist()}


def make_random_split(df: pd.DataFrame, seed: int = C.SEED) -> dict:
    """Row-level stratified split — used ONLY in the split-strategy experiment."""
    idx = np.arange(len(df))
    y = df[C.TARGET].values
    trval, test = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y, random_state=seed)
    tr, val = train_test_split(trval, test_size=C.VAL_SIZE / (1 - C.TEST_SIZE),
                               stratify=y[trval], random_state=seed)
    return {"train": tr.tolist(), "val": val.tolist(), "test": test.tolist()}


def get_split(df: pd.DataFrame) -> dict:
    if SPLIT_FILE.exists():
        s = json.loads(SPLIT_FILE.read_text())
        if sum(len(v) for v in s.values()) == len(df):
            return {k: np.array(v) for k, v in s.items()}
    s = make_group_split(df)
    SPLIT_FILE.write_text(json.dumps(s))
    return {k: np.array(v) for k, v in s.items()}


def group_cv(n_splits: int = 5, seed: int = C.SEED):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def split_summary(df: pd.DataFrame, split: dict) -> pd.DataFrame:
    rows = []
    for k, ix in split.items():
        part = df.iloc[ix]
        rows.append({"partition": k, "rows": len(part), "users": part[C.GROUP].nunique(),
                     "fraud_rows": int(part[C.TARGET].sum()),
                     "fraud_rate_pct": round(part[C.TARGET].mean() * 100, 2)})
    users = [set(df.iloc[ix][C.GROUP]) for ix in split.values()]
    overlap = len(users[0] & users[1]) + len(users[0] & users[2]) + len(users[1] & users[2])
    out = pd.DataFrame(rows)
    out.attrs["user_overlap"] = overlap
    return out
