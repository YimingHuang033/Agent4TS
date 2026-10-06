"""Forecaster interface shared by every model.

A forecaster consumes padded windows (with mask/age) and predicts a value for
each query `(relative_time, variable_id)` in *original* units. The evaluator
normalizes both prediction and truth with train-set statistics to compute the
primary normalized MAE, so models stay physically interpretable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..schemas import TaskBundle
from ..transforms import LabeledWindow


class Forecaster:
    model_id: str = "base"

    def fit(self, train: list[LabeledWindow], bundle: TaskBundle,
            grid: np.ndarray, var_ids: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, lw: LabeledWindow) -> np.ndarray:
        """Return predictions aligned to lw.query_times / lw.query_var_ids."""
        raise NotImplementedError


# ------------------------------------------------------------------ helpers

def _nearest_grid_index(grid: np.ndarray, qtimes: np.ndarray) -> np.ndarray:
    if grid.size == 0:
        return np.zeros(len(qtimes), dtype=np.int64)
    idx = np.searchsorted(grid, qtimes)
    idx = np.clip(idx, 0, grid.size - 1)
    left = np.clip(idx - 1, 0, grid.size - 1)
    take_left = np.abs(grid[left] - qtimes) <= np.abs(grid[idx] - qtimes)
    return np.where(take_left, left, idx)


def _var_position(var_ids: np.ndarray) -> dict[int, int]:
    return {int(v): i for i, v in enumerate(var_ids)}
