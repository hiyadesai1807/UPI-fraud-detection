"""Shared evaluation protocol (all members).

- threshold is chosen on the VALIDATION set (max F1), never on test
- test metrics: ROC-AUC, PR-AUC, precision, recall, F1, FPR, FP, FN
- user-cluster bootstrap 95 % CI for PR-AUC (resamples users, not rows)
- McNemar test between two models' test decisions
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from . import config as C


def best_f1_threshold(y_true, proba) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i])


def threshold_for_recall(y_true, proba, target_recall=0.90) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    ok = np.where(r[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(t[0])
    i = ok[np.argmax(p[:-1][ok])]
    return float(t[i])


def metrics_at(y_true, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "fpr": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": float(thr),
    }


def evaluate(y_val, p_val, y_test, p_test) -> dict:
    thr = best_f1_threshold(y_val, p_val)
    m = metrics_at(y_test, p_test, thr)
    thr90 = threshold_for_recall(y_val, p_val, 0.90)
    m90 = metrics_at(y_test, p_test, thr90)
    m["precision_at_val_recall90"] = m90["precision"]
    m["recall_at_val_recall90"] = m90["recall"]
    return m


def cluster_bootstrap_ci(y, proba, groups, n_boot=500, seed=C.SEED, metric=average_precision_score):
    rng = np.random.default_rng(seed)
    y, proba, groups = np.asarray(y), np.asarray(proba), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array).to_dict()
    vals = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx_by_g[g] for g in gs])
        if y[ix].min() == y[ix].max():
            continue
        vals.append(metric(y[ix], proba[ix]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y, pred_a, pred_b) -> dict:
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    stat = (abs(b - c) - 1) ** 2 / max(b + c, 1)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "chi2": stat, "p_value": float(chi2.sf(stat, 1))}


def expected_cost(y, proba, thr, amounts, review_cost=100.0, false_alarm_cost=1000.0):
    """Illustrative cost model (assumptions, NOT real UPI economics):
      * a missed fraud costs its transaction amount (INR)
      * every alert costs a manual-review fee
      * every false alarm additionally costs customer friction / churn risk."""
    pred = proba >= thr
    fn_cost = float(np.sum(amounts[(y == 1) & ~pred]))
    review = float(np.sum(pred) * review_cost + np.sum(pred & (y == 0)) * false_alarm_cost)
    return fn_cost + review, fn_cost, review
