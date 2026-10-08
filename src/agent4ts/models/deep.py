"""PatchTST-style and iTransformer-style forecasters (torch, tim env).

These are faithful re-implementations of the core architectural idea, trained
here on the agent's search split, NOT copies of the authors' release code. The
`paper_url` class attribute records the canonical source for traceability. If a
future run needs the exact authors' implementation, `heavy.py` tracks that.

Both take the causal lookback grid as input (length T) and the forecast query
grid as output (length H). Input and output are separate axes, so no future leaks.
"""
from __future__ import annotations

import numpy as np
from torch import nn

from ..logging_utils import get_logger

log = get_logger("models.deep")


class _GridModel:
    """Shared fit scaffolding for the torch grid models."""

    def _prepare(self, train, bundle, grid, in_grid):
        from .runners import (resample_series, lookback_grid, fit_stats,
                              fit_target_stats, _build_target_tensor,
                              normalize_targets)
        self.grid = grid
        self.n_vars = len(bundle.variable_names)
        view = getattr(self, "_view", None)
        if view is None:
            from ..schemas import DataView
            view = DataView()
        if in_grid is None:
            in_grid = lookback_grid(bundle, view, n_steps=max(16, grid.size))
        self.in_grid = in_grid
        X = np.stack([resample_series(lw, self.n_vars, in_grid) for lw in train])
        Y, valid = _build_target_tensor(train, bundle, grid)
        self.x_mean, self.x_std = fit_stats(X)
        self.y_mean, self.y_std = fit_target_stats(Y, valid)
        Xn = (np.where(np.isnan(X), self.x_mean, X) - self.x_mean) / self.x_std
        Yn = normalize_targets(Y, valid, self.y_mean, self.y_std)
        return Xn, Yn, valid

    def _predict_grid(self, lw):
        import torch
        from .runners import resample_series
        from .base import _nearest_grid_index
        series = resample_series(lw, self.n_vars, self.in_grid)
        Xn = (np.where(np.isnan(series), self.x_mean[0], series) - self.x_mean[0]) / self.x_std[0]
        with torch.no_grad():
            pred_n = self._forward(torch.tensor(Xn[None], dtype=torch.float32))[0].numpy()
        pred = pred_n * self.y_std[0] + self.y_mean[0]
        gi = _nearest_grid_index(self.grid, lw.query_times)
        return np.array([pred[int(lw.query_var_ids[q]), gi[q]]
                         if int(lw.query_var_ids[q]) < self.n_vars else np.nan
                         for q in range(len(lw.query_times))])


class _PatchNet(nn.Module):
    def __init__(self, patch, stride, d_model, n_heads, n_layers, n_patch, h):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.proj = nn.Linear(patch, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model * n_patch, h)

    def forward(self, x):                       # x: (B, T)
        patches = x.unfold(-1, self.patch, self.stride)
        h = self.enc(self.proj(patches))
        return self.head(h.flatten(1))


class _InvNet(nn.Module):
    def __init__(self, T, d_model, n_heads, n_layers, h):
        super().__init__()
        self.proj = nn.Linear(T, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, h)

    def forward(self, x):        # x: (B, V, T)
        return self.head(self.enc(self.proj(x)))


class PatchTST(_GridModel):
    """Patch-based transformer over the lookback, channel-independent."""
    model_id = "PatchTST"
    paper_url = "https://github.com/yuqinie98/PatchTST"

    def __init__(self, patch: int = 4, stride: int = 2, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2, epochs: int = 120,
                 lr: float = 1e-3, seed: int = 0):
        self.patch, self.stride = patch, stride
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def fit(self, train, bundle, grid, var_ids):
        import torch
        torch.manual_seed(self.seed)
        Xn, Yn, valid = self._prepare(train, bundle, grid, getattr(self, "in_grid", None))
        T, H = Xn.shape[-1], Yn.shape[-1]
        n_patch = max(1, (T - self.patch) // self.stride + 1)
        net = _PatchNet(self.patch, self.stride, self.d_model, self.n_heads,
                        self.n_layers, n_patch, H)
        self.net = net
        self._forward = lambda x: net(x.reshape(-1, T)).reshape(x.shape[0], self.n_vars, H)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        lossf = torch.nn.MSELoss(reduction="none")
        xt = torch.tensor(Xn.reshape(-1, T), dtype=torch.float32)   # (E*V, T)
        yt = torch.tensor(Yn.reshape(-1, H), dtype=torch.float32)
        vm = torch.tensor(valid.reshape(-1, H).astype(np.float32))
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = (lossf(net(xt), yt) * vm).sum() / vm.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
        self.trained = True

    def predict(self, lw):
        return self._predict_grid(lw)


class ITransformer(_GridModel):
    """Inverted attention: variables are tokens, the lookback is the feature axis."""
    model_id = "iTransformer"
    paper_url = "https://github.com/thuml/iTransformer"

    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 2,
                 epochs: int = 120, lr: float = 1e-3, seed: int = 0):
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def fit(self, train, bundle, grid, var_ids):
        import torch
        torch.manual_seed(self.seed)
        Xn, Yn, valid = self._prepare(train, bundle, grid, getattr(self, "in_grid", None))
        T, H = Xn.shape[-1], Yn.shape[-1]
        net = _InvNet(T, self.d_model, self.n_heads, self.n_layers, H)
        self.net = net
        self._forward = lambda x: net(x)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        lossf = torch.nn.MSELoss(reduction="none")
        xt = torch.tensor(Xn, dtype=torch.float32)     # (E, V, T)
        yt = torch.tensor(Yn, dtype=torch.float32)
        vm = torch.tensor(valid.astype(np.float32))
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = (lossf(net(xt), yt) * vm).sum() / vm.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
        self.trained = True

    def predict(self, lw):
        return self._predict_grid(lw)
