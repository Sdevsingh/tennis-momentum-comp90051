"""Complex model: Relational Recurrent Neural Network (Santoro et al., 2018).

Paper: Santoro, Faulkner, Raposo, Rae, Chrzanowski, Weber, Wierstra, Vinyals,
Pascanu, Lillicrap. "Relational Recurrent Neural Networks." NeurIPS 2018.
https://arxiv.org/abs/1806.01822

WHY IT FITS THE PROJECT
A standard LSTM (which the subject covers) squeezes the past into one hidden
vector. The Relational Memory Core (RMC) keeps a MATRIX of memory slots and, at
each step, lets them interact through multi-head self-attention before a gated
update. It reads the RAW memory of earlier points (src/sequences.py: who won,
who served, rally length) instead of the engineered memory features, so it can
discover order-dependent "momentum" structure the hand-built features never
encode. That is exactly the research question. Pre-match (MATCH) features and
the current server (POINT_STRUCTURE) go in as static context, so it is not
blind to player strength or to who serves.

WHY IT WAS NOT TAUGHT IN CLASS (for the literature review)
The syllabus covers standard RNN/LSTM; the RMC's defining component -- self
attention AMONG memory slots -- is a post-2017 development outside it.

INTERFACE
Matches the other models: fit(X, y) / predict_proba(X). X is the flat sequence
matrix from src/sequences.build_sequence_matrix: the first RNN_HISTORY *
RNN_STEP_FEATURES columns are the flattened (time, features) sequence, the rest
are static context. The wrapper reshapes internally, so the from-scratch nested
CV (nested_cv.py) tunes and evaluates it exactly like the others. The static
columns arrive raw (NaN allowed): fit() standardises them with the training
fold's mean/sd (constant columns zeroed, NaN -> the mean) and predict_proba
reuses those statistics, so nothing leaks from the test fold.

HYPERPARAMETER (2.f): mem_slots, the number of memory slots, grid {2, 4, 8}
(middle = 4). More slots = more relational capacity.

DEVICE: runs on a GPU when this PyTorch build can see one -- NVIDIA (cuda),
Intel Arc (xpu) or Apple (mps) -- else the CPU (config.RNN_DEVICE, or the
TENNIS_RNN_DEVICE environment variable). The training data is moved to the
device once per fit; predictions come back as numpy, so nested CV is unchanged.

COMPUTE: fit() trains on the whole training fold by default, with a small
net, few epochs and early stopping on an internal split. RNN_MAX_TRAIN (or
max_train) can cap it to a random subset when time matters. We never report
loss-vs-epochs (2.g forbids it); epochs are only a training detail.
"""
from __future__ import annotations

import os

import numpy as np

from ..config import (RNN_HISTORY, RNN_STEP_FEATURES, RNN_MAX_TRAIN,
                      RNN_EPOCHS, RNN_BATCH, RNN_LR, RNN_DEVICE, SEED)

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                                    # torch optional at import
    _TORCH = False


def pick_device(preference: str | None = None) -> "torch.device":
    """cuda -> xpu -> mps -> cpu, the first this PyTorch build can use.

    `preference` (or TENNIS_RNN_DEVICE, or config.RNN_DEVICE) forces one;
    "auto" means pick. Asking for a device that is not available falls back
    to the CPU with a warning rather than crashing a long run.
    """
    pref = (preference or os.environ.get("TENNIS_RNN_DEVICE") or RNN_DEVICE).lower()
    available = {
        "cuda": torch.cuda.is_available(),
        "xpu": hasattr(torch, "xpu") and torch.xpu.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu": True,
    }
    if pref == "auto":
        return torch.device(next(d for d in ("cuda", "xpu", "mps", "cpu") if available[d]))
    if not available.get(pref, False):
        print(f"  [relational_rnn] device '{pref}' not available in this PyTorch build; using cpu")
        return torch.device("cpu")
    return torch.device(pref)


if _TORCH:
    class _RMC(nn.Module):
        """A compact Relational Memory Core + a linear read-out to one logit."""

        def __init__(self, d_step, n_static, mem_slots=4, head_size=16, n_heads=2):
            super().__init__()
            self.mem_slots = mem_slots
            self.mem_size = head_size * n_heads       # size of each slot
            self.n_heads = n_heads

            self.in_proj = nn.Linear(d_step, self.mem_size)
            self.attn = nn.MultiheadAttention(self.mem_size, n_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(self.mem_size)
            self.norm2 = nn.LayerNorm(self.mem_size)
            self.mlp = nn.Sequential(nn.Linear(self.mem_size, self.mem_size),
                                     nn.ReLU(), nn.Linear(self.mem_size, self.mem_size))
            # LSTM-style input/forget gates (Santoro et al. gated update)
            self.gate = nn.Linear(2 * self.mem_size, 2 * self.mem_size)
            self.mem0 = nn.Parameter(torch.zeros(mem_slots, self.mem_size))
            self.head = nn.Linear(mem_slots * self.mem_size + n_static, 1)

        def forward(self, seq, static):
            # seq: (B, T, d_step); static: (B, n_static)
            b, t, _ = seq.shape
            x = self.in_proj(seq)                      # (B, T, mem_size)
            M = self.mem0.unsqueeze(0).expand(b, -1, -1).contiguous()
            for ti in range(t):
                xt = x[:, ti:ti + 1, :]                 # (B, 1, mem_size)
                rows = torch.cat([M, xt], dim=1)        # attend over slots + input
                a, _ = self.attn(M, rows, rows)         # queries = slots
                M_hat = self.norm1(M + a)
                M_hat = self.norm2(M_hat + self.mlp(M_hat))
                # gated update, input xt broadcast across slots
                g = self.gate(torch.cat([M, xt.expand(-1, self.mem_slots, -1)], dim=-1))
                i_gate, f_gate = g.chunk(2, dim=-1)
                M = torch.sigmoid(f_gate) * M + torch.sigmoid(i_gate) * torch.tanh(M_hat)
            flat = torch.cat([M.reshape(b, -1), static], dim=1)
            return self.head(flat).squeeze(1)          # logit


class RelationalRNN:
    def __init__(self, mem_slots: int = 4, head_size: int = 16, n_heads: int = 2,
                 history: int = RNN_HISTORY, d_step: int = RNN_STEP_FEATURES,
                 n_epochs: int = RNN_EPOCHS, lr: float = RNN_LR,
                 batch: int = RNN_BATCH, max_train: int | None = RNN_MAX_TRAIN,
                 seed: int = SEED, device: str | None = None):
        if not _TORCH:
            raise ImportError("PyTorch is required for the Relational RNN "
                              "(pip install torch).")
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.n_heads = n_heads
        self.history = history
        self.d_step = d_step
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch = batch
        self.max_train = max_train
        self.seed = seed
        self.device = device
        self._dev = None
        self._net = None
        self._n_static = None
        self._mu = None
        self._sd = None

    def _split(self, X):
        """Flat X -> (sequence (N,T,d_step), static (N,n_static))."""
        cut = self.history * self.d_step
        seq = X[:, :cut].reshape(-1, self.history, self.d_step)
        static = X[:, cut:]
        return seq.astype(np.float32), static.astype(np.float32)

    def _fit_scaler(self, static):
        """Training-fold mean/sd of each static column. A column that is
        constant on this fold (spread tiny next to its size) gets sd = inf,
        so it becomes 0 rather than float noise blown up."""
        with np.errstate(all="ignore"):
            mu = np.nanmean(static, axis=0)
            sd = np.nanstd(static, axis=0)
            spread = np.nanmax(static, axis=0) - np.nanmin(static, axis=0)
        mu = np.nan_to_num(mu)
        const = ~(spread > 1e-9 * np.maximum(1.0, np.abs(mu)))
        self._mu = mu.astype(np.float32)
        self._sd = np.where(const, np.inf, sd).astype(np.float32)

    def _scale(self, static):
        z = (static - self._mu) / self._sd
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        # optional subsample (max_train=None trains on everything)
        if self.max_train is not None and len(X) > self.max_train:
            idx = rng.choice(len(X), self.max_train, replace=False)
            X, y = X[idx], y[idx]
        seq, static = self._split(X)
        self._n_static = static.shape[1]
        self._fit_scaler(static)
        static = self._scale(static)

        # internal train/val split for early stopping
        n = len(seq)
        perm = rng.permutation(n)
        n_val = max(1, int(0.1 * n))
        val, tr = perm[:n_val], perm[n_val:]

        dev = self._dev = pick_device(self.device)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self._net = _RMC(self.d_step, self._n_static, self.mem_slots,
                         self.head_size, self.n_heads).to(dev)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        # the whole (subsampled) training set goes to the device once
        seq_t = torch.from_numpy(seq).to(dev)
        static_t = torch.from_numpy(static).to(dev)
        y_t = torch.from_numpy(y).to(dev)
        val_t = torch.from_numpy(val).to(dev)

        def batches(ix):
            ix = torch.from_numpy(ix).to(dev)
            for s in range(0, len(ix), self.batch):
                b = ix[s:s + self.batch]
                yield seq_t[b], static_t[b], y_t[b]

        best_val, best_state, patience = float("inf"), None, 0
        for _ in range(self.n_epochs):
            self._net.train()
            for sb, stb, yb in batches(rng.permutation(tr)):
                opt.zero_grad()
                loss = loss_fn(self._net(sb, stb), yb)
                loss.backward()
                opt.step()
            # validation loss for early stopping (NOT reported as a result)
            self._net.eval()
            with torch.no_grad():
                vl = loss_fn(self._net(seq_t[val_t], static_t[val_t]),
                             y_t[val_t]).item()
            if vl < best_val - 1e-4:
                best_val, best_state, patience = vl, {
                    k: v.clone() for k, v in self._net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 2:
                    break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=np.float32)
        seq, static = self._split(X)
        static = self._scale(static)
        self._net.eval()
        out = np.empty(len(seq), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(seq), self.batch):
                sb = torch.from_numpy(seq[s:s + self.batch]).to(self._dev)
                stb = torch.from_numpy(static[s:s + self.batch]).to(self._dev)
                out[s:s + self.batch] = torch.sigmoid(self._net(sb, stb)).cpu().numpy()
        return out


# Centred so the middle memory size should win most often (2.f requirement).
PARAM_GRID = {"mem_slots": [2, 4, 8]}


def make_model(**params) -> RelationalRNN:
    return RelationalRNN(**params)
