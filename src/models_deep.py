"""Member 4 — deep learning: MLP (per transaction) and LSTM (per user sequence).

Both models:
  * PyTorch, CPU friendly
  * class imbalance via BCE pos_weight ('class_weight') or SMOTE (MLP only)
  * AdamW + ReduceLROnPlateau, early stopping on validation PR-AUC
"""
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence

from . import config as C

torch.set_num_threads(2)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in, hidden=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMNet(nn.Module):
    """Reads a user's last k transactions (oldest -> current) and scores the current one."""

    def __init__(self, d_in, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + d_in, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        last = x[torch.arange(x.size(0)), lengths - 1]       # current transaction
        return self.head(torch.cat([h[-1], last], dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------
def build_sequences(df: pd.DataFrame, feats: np.ndarray, seq_len=C.SEQ_LEN,
                    shuffle_history=False, seed=C.SEED):
    """df must be sorted by (user_id, order_key) and aligned with feats rows.
    Returns X (n, seq_len, d) left-aligned + lengths (n,). Row i's sequence ends
    with transaction i and contains only that user's EARLIER transactions."""
    rng = np.random.default_rng(seed)
    n, d = feats.shape
    X = np.zeros((n, seq_len, d), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    users = df[C.GROUP].values
    start = 0
    for i in range(n):
        if i > 0 and users[i] != users[i - 1]:
            start = i
        lo = max(start, i - seq_len + 1)
        window = list(range(lo, i + 1))
        if shuffle_history and len(window) > 2:
            hist = window[:-1]
            rng.shuffle(hist)
            window = hist + [window[-1]]
        X[i, :len(window)] = feats[window]
        lengths[i] = len(window)
    return X, lengths


# ---------------------------------------------------------------------------
# Generic trainer
# ---------------------------------------------------------------------------
class TorchClassifier:
    def __init__(self, kind="mlp", strategy="class_weight", seed=C.SEED, epochs=80,
                 batch_size=256, lr=1e-3, weight_decay=1e-4, patience=10, verbose=False, **arch):
        self.kind, self.strategy, self.seed = kind, strategy, seed
        self.epochs, self.bs, self.lr, self.wd, self.patience = epochs, batch_size, lr, weight_decay, patience
        self.arch, self.verbose = arch, verbose
        self.scaler = StandardScaler()
        self.history = []

    # ---- scaling helpers --------------------------------------------------
    def _scale_fit(self, X):
        if self.kind == "mlp":
            return self.scaler.fit_transform(X).astype(np.float32)
        n, L, d = X.shape
        return X  # sequences are scaled before building (see fit_lstm_features)

    def _scale(self, X):
        return self.scaler.transform(X).astype(np.float32) if self.kind == "mlp" else X

    def _forward(self, xb, lb=None):
        return self.model(xb) if self.kind == "mlp" else self.model(xb, lb)

    def predict_proba(self, X, lengths=None):
        self.model.eval()
        X = self._scale(X)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.from_numpy(X[i:i + 2048])
                lb = None if lengths is None else torch.from_numpy(lengths[i:i + 2048])
                out.append(torch.sigmoid(self._forward(xb, lb)).numpy())
        return np.concatenate(out)

    def fit(self, X, y, X_val, y_val, len_tr=None, len_val=None):
        set_seed(self.seed)
        X = self._scale_fit(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32)
        if self.strategy == "smote":
            if self.kind != "mlp":
                raise ValueError("SMOTE is not defined for sequences")
            X, y = SMOTE(random_state=self.seed).fit_resample(X, y)
            X, y = X.astype(np.float32), y.astype(np.float32)
        pos_weight = (1 - y.mean()) / y.mean() if self.strategy == "class_weight" else 1.0
        d_in = X.shape[-1]
        self.model = MLP(d_in, **self.arch) if self.kind == "mlp" else LSTMNet(d_in, **self.arch)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        Lt = None if len_tr is None else torch.from_numpy(len_tr)
        best, best_state, bad = -1.0, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xt), generator=g)
            tot = 0.0
            for i in range(0, len(perm), self.bs):
                b = perm[i:i + self.bs]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                logits = self._forward(Xt[b], None if Lt is None else Lt[b])
                loss = loss_fn(logits, yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                tot += loss.item() * len(b)
            pv = self.predict_proba(X_val, len_val)
            ap = average_precision_score(y_val, pv)
            sched.step(ap)
            self.history.append({"epoch": ep + 1, "train_loss": tot / len(Xt), "val_pr_auc": ap})
            if self.verbose:
                print(f"epoch {ep+1:3d} loss {tot/len(Xt):.4f} val PR-AUC {ap:.4f}")
            if ap > best + 1e-4:
                best, best_state, bad = ap, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.best_val_pr_auc = best
        return self

    def save(self, path):
        torch.save({"kind": self.kind, "arch": self.arch, "state": self.model.state_dict(),
                    "scaler_mean": getattr(self.scaler, "mean_", None),
                    "scaler_scale": getattr(self.scaler, "scale_", None),
                    "d_in": next(self.model.parameters()).shape[-1] if self.kind == "mlp" else self.model.lstm.input_size,
                    "history": self.history}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, weights_only=False)
        obj = cls(kind=ck["kind"], **ck["arch"])
        obj.model = (MLP if ck["kind"] == "mlp" else LSTMNet)(ck["d_in"], **ck["arch"])
        obj.model.load_state_dict(ck["state"])
        obj.model.eval()
        if ck["scaler_mean"] is not None:
            obj.scaler.mean_, obj.scaler.scale_ = ck["scaler_mean"], ck["scaler_scale"]
            obj.scaler.var_ = obj.scaler.scale_ ** 2
            obj.scaler.n_features_in_ = len(obj.scaler.mean_)
        obj.history = ck["history"]
        return obj
