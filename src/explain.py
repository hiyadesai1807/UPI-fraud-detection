"""Member 5 — explainability.

* XGBoost / RF : SHAP TreeExplainer (global + local)
* MLP          : Integrated Gradients (implemented directly in PyTorch)
* LSTM         : permutation importance on the test sequences (feature-wise)
* Agreement    : Spearman rank correlation between SHAP and IG global rankings
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def shap_tree(pipeline, X: pd.DataFrame, max_rows=2000, seed=0):
    import shap
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    # keep the original row order when explaining the whole set (row i of the
    # SHAP matrix must correspond to row i of X)
    Xs = X if len(X) <= max_rows else X.sample(max_rows, random_state=seed)
    Xt = prep.transform(Xs)
    names = list(prep.get_feature_names_out())
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    base = explainer.expected_value
    base = base[1] if np.ndim(base) else base
    return sv, pd.DataFrame(Xt, columns=names, index=Xs.index), float(base)


def global_importance(values: np.ndarray, names) -> pd.Series:
    return pd.Series(np.abs(values).mean(axis=0), index=names).sort_values(ascending=False)


def integrated_gradients(torch_clf, X: np.ndarray, steps=50, baseline=None) -> np.ndarray:
    """IG attributions for the MLP in *scaled* feature space. Baseline = zero
    vector = the training mean after standardisation."""
    model = torch_clf.model.eval()
    Xs = torch.from_numpy(torch_clf._scale(np.asarray(X, dtype=np.float32)))
    base = torch.zeros_like(Xs) if baseline is None else torch.as_tensor(baseline).expand_as(Xs)
    total = torch.zeros_like(Xs)
    for a in np.linspace(0, 1, steps + 1)[1:]:
        xi = (base + a * (Xs - base)).requires_grad_(True)
        out = model(xi).sum()
        grad, = torch.autograd.grad(out, xi)
        total += grad
    return ((Xs - base) * total / steps).detach().numpy()


def lstm_permutation_importance(torch_clf, X_seq, lengths, y, names, n_repeats=3, seed=0):
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, torch_clf.predict_proba(X_seq, lengths))
    rows = []
    for j, nm in enumerate(names):
        drops = []
        for _ in range(n_repeats):
            Xp = X_seq.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, :, j] = Xp[perm][:, :, j]
            drops.append(base - average_precision_score(y, torch_clf.predict_proba(Xp, lengths)))
        rows.append({"feature": nm, "pr_auc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False).reset_index(drop=True)


def rank_agreement(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index)
    return float(spearmanr(a[common], b[common]).statistic)
