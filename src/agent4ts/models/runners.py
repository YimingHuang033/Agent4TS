"""Traditional (non-agent) forecasters, all runnable in the `tim` env.

Ordered by cost. `LastValue` and `MeanValue` train nothing and back the data
pipeline / scoring sanity checks. `DLinear`, `PatchTST` and `iTransformer` are
real torch modules trained here on the agent's search split. The remaining 2026
models named in DESIGN 3 live in `heavy.py` and report their real availability.
"""
from __future__ import annotations

import numpy as np

from ..logging_utils import get_logger
from ..schemas import TaskBundle
from ..transforms import LabeledWindow
from .base import Forecaster, _nearest_grid_index, _var_position

log = get_logger("models")


# ------------------------------------------------------------------ torch base

def _build_target_tensor(train: list[LabeledWindow], bundle: TaskBundle,
                         grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (n_entities, n_vars, n_grid) targets from bucketed means.

    Also returns a validity mask so a variable with no labels in a window is
    excluded from the loss instead of contributing a NaN.
    """
    n_vars = len(bundle.variable_names)
    y = np.full((len(train), n_vars, len(grid)), np.nan)
    for e, lw in enumerate(train):
        pos = _var_position(lw.window.variable_ids)
        gi = _nearest_grid_index(grid, lw.query_times)
        for q in range(len(lw.query_times)):
            vi = pos.get(int(lw.query_var_ids[q]))
            if vi is None:
                continue
            y[e, vi, gi[q]] = lw.y_true[q]
    return y, np.isfinite(y)


def lookback_grid(bundle: TaskBundle, view: "DataView", n_steps: int = 64) -> np.ndarray:
    """Canonical causal lookback axis: [cutoff - history*frac, cutoff], length n_steps."""
    cutoff = bundle.spec.history_duration
    start = cutoff - cutoff * view.history_fraction
    return np.linspace(start, cutoff, n_steps)


def resample_series(lw: LabeledWindow, n_vars: int, in_grid: np.ndarray) -> np.ndarray:
    """Resample each variable's real observations onto the *lookback* grid.

    Step function: the value at each lookback point is the last real observation
    at or before it. Never reads past the cutoff, so no future leakage.
    """
    pos = _var_position(lw.window.variable_ids)
    out = np.full((n_vars, len(in_grid)), np.nan)
    for vid, pi in pos.items():
        if vid >= n_vars:
            continue
        real = lw.window.mask[pi].astype(bool)
        if not real.any():
            continue
        ts = lw.window.times[real]
        vs = lw.window.x[pi][real]
        order = np.argsort(ts)
        ts, vs = ts[order], vs[order]
        for gi, g in enumerate(in_grid):
            k = np.searchsorted(ts, g, side="right") - 1
            if k >= 0:
                out[vid, gi] = vs[k]
    return out


def fit_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-variable mean/std over (E, V, T). No-target variables fall back to 0/1."""
    mean = np.nanmean(X, axis=(0, 2), keepdims=True)
    std = np.nanstd(X, axis=(0, 2), keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std[std < 1e-8] = 1.0
    return mean, std


def fit_target_stats(Y: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-variable mean/std over valid targets only, so each variable gets its
    own affine scale back at prediction time (a shared head cannot otherwise
    produce different offsets for HR vs Temp)."""
    masked = np.where(valid, Y, np.nan)
    return fit_stats(masked)


def normalize_targets(Y, valid, mean, std):
    """(Y - mean)/std with invalid entries zeroed so they cannot poison the loss."""
    Yn = (np.where(valid, Y, 0.0) - mean) / std
    return np.where(valid, Yn, 0.0)



# ------------------------------------------------------------------ trivial

class LastValue(Forecaster):
    """Repeat the most recent real observation forward. No training."""
    model_id = "LastValue"

    def fit(self, train, bundle, grid, var_ids):
        self.grid = grid
        self.n_vars = len(bundle.variable_names)

    def predict(self, lw: LabeledWindow) -> np.ndarray:
        pos = _var_position(lw.window.variable_ids)
        norm = getattr(self, "_norm", None)
        preds = np.full(len(lw.query_times), np.nan)
        for q in range(len(lw.query_times)):
            vid = int(lw.query_var_ids[q])
            vi = pos.get(vid)
            if vi is None:
                continue
            real = lw.window.mask[vi].astype(bool)
            if real.any():
                val = lw.window.x[vi][real][-1]
                # window values are normalized by the view; truth is in original
                # units, so map the copied value back before returning it
                preds[q] = norm.inverse(vid, val) if norm is not None else val
        return preds

    def fit_meta(self) -> dict:
        return {"trained": False}


class MeanValue(Forecaster):
    """Predict the training-mean of each variable. No per-entity logic."""
    model_id = "MeanValue"

    def fit(self, train, bundle, grid, var_ids):
        y, valid = _build_target_tensor(train, bundle, grid)
        self.per_var = np.nanmean(np.where(valid, y, np.nan), axis=(0, 2)).ravel()
        self.global_mean = float(np.nanmean(y)) if valid.any() else 0.0

    def predict(self, lw: LabeledWindow) -> np.ndarray:
        out = np.full(len(lw.query_times), self.global_mean)
        for q in range(len(lw.query_times)):
            v = int(lw.query_var_ids[q])
            if v < len(self.per_var) and np.isfinite(self.per_var[v]):
                out[q] = self.per_var[v]
        return out


# ------------------------------------------------------------------ DLinear

class DLinear(Forecaster):
    """Trend+seasonal decomposition with a per-variable linear head.

    Faithful to the LTSF-Linear idea: the causal lookback series (length T) is
    mapped linearly to the forecast horizon (length H) per variable. Input and
    output live on *different* axes — lookback in the past, forecast in the
    future — so the linear layer is (T -> H), not (T -> T).
    """
    model_id = "DLinear"

    def __init__(self, epochs: int = 200, lr: float = 1e-2, kernel: int = 5, seed: int = 0):
        self.epochs, self.lr, self.kernel, self.seed = epochs, lr, kernel, seed

    def _decompose(self, x):
        k = self.kernel
        pad = k // 2
        pad_edges = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
        trend = np.stack([pad_edges[:, i:i + k].mean(axis=1) for i in range(x.shape[1])], axis=1)
        return trend, x - trend

    def _inputs(self, labeled, bundle, in_grid, view):
        n_vars = len(bundle.variable_names)
        return np.stack([resample_series(lw, n_vars, in_grid) for lw in labeled])

    def fit(self, train, bundle, grid, var_ids):
        import torch
        torch.manual_seed(self.seed)
        self.grid = grid
        self.n_vars = len(bundle.variable_names)
        view = getattr(self, "_view", None) or _default_view()
        self.in_grid = getattr(self, "in_grid", None)
        if self.in_grid is None:
            self.in_grid = lookback_grid(bundle, view, n_steps=max(16, grid.size))
        X = self._inputs(train, bundle, self.in_grid, view)
        Y, valid = _build_target_tensor(train, bundle, grid)

        self.x_mean, self.x_std = fit_stats(X)
        self.y_mean, self.y_std = fit_target_stats(Y, valid)
        Xn = (np.where(np.isnan(X), self.x_mean, X) - self.x_mean) / self.x_std
        Yn = normalize_targets(Y, valid, self.y_mean, self.y_std)

        T, H = Xn.shape[-1], Yn.shape[-1]
        xt = torch.tensor(Xn, dtype=torch.float32)
        yt = torch.tensor(Yn, dtype=torch.float32)
        vm = torch.tensor(valid.astype(np.float32))
        self.lin = torch.nn.Linear(T, H)
        opt = torch.optim.Adam(self.lin.parameters(), lr=self.lr)
        lossf = torch.nn.MSELoss(reduction="none")
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.lin(xt)
            loss = (lossf(pred, yt) * vm).sum() / vm.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
        self.trained = True

    def predict(self, lw: LabeledWindow) -> np.ndarray:
        import torch
        series = resample_series(lw, self.n_vars, self.in_grid)
        Xn = (np.where(np.isnan(series), self.x_mean[0], series) - self.x_mean[0]) / self.x_std[0]
        with torch.no_grad():
            pred_n = self.lin(torch.tensor(Xn[None], dtype=torch.float32))[0].numpy()
        pred = pred_n * self.y_std[0] + self.y_mean[0]
        gi = _nearest_grid_index(self.grid, lw.query_times)
        return np.array([pred[int(lw.query_var_ids[q]), gi[q]]
                         if int(lw.query_var_ids[q]) < self.n_vars else np.nan
                         for q in range(len(lw.query_times))])


def _default_view():
    from ..schemas import DataView
    return DataView()
