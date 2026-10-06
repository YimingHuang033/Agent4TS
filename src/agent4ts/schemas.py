"""Core data contracts for Agent4TS.

Event-based irregular time series interface. Everything downstream (data
backends, transforms, models, evaluator, agent loop) speaks these types.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------- time / events

@dataclass
class EventTable:
    """Columnar irregular observation store.

    All arrays are length N (one row per real observation). No padding here;
    padding + boolean mask is produced by the grid/window builder, never stored
    as if it were a real measurement.
    """
    entity_id: np.ndarray          # (N,) int
    event_time: np.ndarray         # (N,) float, relative hours from entity anchor
    available_time: np.ndarray     # (N,) float, when the value became known
    variable_id: np.ndarray        # (N,) int
    value: np.ndarray              # (N,) float

    def __post_init__(self) -> None:
        n = len(self.value)
        for name in ("entity_id", "event_time", "available_time", "variable_id"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"EventTable.{name} length {len(arr)} != {n}")

    def __len__(self) -> int:
        return len(self.value)

    def subset(self, mask: np.ndarray) -> "EventTable":
        return EventTable(
            entity_id=self.entity_id[mask],
            event_time=self.event_time[mask],
            available_time=self.available_time[mask],
            variable_id=self.variable_id[mask],
            value=self.value[mask],
        )

    def entities(self) -> np.ndarray:
        return np.unique(self.entity_id)

    def to_records(self) -> list[dict[str, Any]]:
        return [
            {
                "entity_id": int(self.entity_id[i]),
                "event_time": float(self.event_time[i]),
                "available_time": float(self.available_time[i]),
                "variable_id": int(self.variable_id[i]),
                "value": float(self.value[i]),
            }
            for i in range(len(self))
        ]

    @classmethod
    def concat(cls, tables: list["EventTable"]) -> "EventTable":
        if not tables:
            raise ValueError("cannot concat empty table list")
        return cls(
            entity_id=np.concatenate([t.entity_id for t in tables]),
            event_time=np.concatenate([t.event_time for t in tables]),
            available_time=np.concatenate([t.available_time for t in tables]),
            variable_id=np.concatenate([t.variable_id for t in tables]),
            value=np.concatenate([t.value for t in tables]),
        )


class Track(str, Enum):
    NATIVE_IRR = "native_irr"
    ROBUST_IMPROVE = "robust_improve"


class QueryPolicy(str, Enum):
    BUCKETED = "bucketed"      # fixed query grid, bucketed real observation mean
    EVENT = "event"            # per-query (time, variable), BITS style
    POINT = "point"            # fixed query times, latest available value at cutoff


# ---------------------------------------------------------------- task specs

@dataclass
class TaskSpec:
    task_id: str
    dataset_version: str
    track: Track
    backend: str                              # bits | p12 | physiome | timeimm | mimic
    split_ids: dict[str, list[int]]           # entity ids per split
    target_variables: list[str]
    covariate_variables: list[str]
    cutoff_rule: str                          # e.g. "anchor + history_duration"
    history_duration: float                   # hours
    horizon: float                            # hours
    query_policy: QueryPolicy = QueryPolicy.BUCKETED
    query_grid: list[float] = field(default_factory=list)  # relative to cutoff
    metric_version: str = "v1"
    degradation: dict[str, Any] | None = None  # B class controlled degradation
    f0_model: str = "DLinear"
    notes: str = ""

    def validate(self) -> None:
        if not self.split_ids:
            raise ValueError(f"{self.task_id}: empty split_ids")
        if self.history_duration <= 0 or self.horizon <= 0:
            raise ValueError(f"{self.task_id}: non-positive history/horizon")
        overlap = set(self.target_variables) & set(self.covariate_variables)
        if self.track == Track.NATIVE_IRR and overlap:
            raise ValueError(f"{self.task_id}: target also listed as covariate: {overlap}")


@dataclass
class DatasetCard:
    dataset_version: str
    backend: str
    variables: dict[str, dict[str, Any]]   # name -> {unit, kind, missing_meaning, ...}
    time_semantics: dict[str, Any]
    provenance: list[dict[str, Any]] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    def unit(self, var: str) -> str:
        return self.variables.get(var, {}).get("unit", "unknown")


@dataclass
class TaskBundle:
    """Agent-visible view + evaluator-private targets.

    `history` is the full event table. Splits index into `history`/`targets` by
    entity id. The agent receives the bundle but the harness only ever hands
    `search` labels to the search path; `test` labels stay behind `Predictor`.
    """
    spec: TaskSpec
    card: DatasetCard
    history: EventTable                       # observations before cutoff per entity
    targets: EventTable                       # ground-truth future observations (private)
    variable_names: list[str]                 # index -> name
    entity_anchors: dict[int, float]          # entity_id -> cutoff time
    train_stats: dict[str, dict[str, float]]  # fitted on train entities only

    def var_name(self, vid: int) -> str:
        return self.variable_names[vid]

    def var_index(self, name: str) -> int:
        return self.variable_names.index(name)

    def entities_for(self, split: str) -> np.ndarray:
        return np.asarray(self.spec.split_ids[split], dtype=np.int64)

    def labels_for(self, split: str) -> EventTable:
        ents = set(self.spec.split_ids[split].tolist())
        mask = np.isin(self.targets.entity_id, list(ents))
        return self.targets.subset(mask)


# ---------------------------------------------------------------- pipeline

@dataclass
class DataView:
    history_fraction: float = 1.0             # fraction of history_duration kept
    variable_subset: str = "all"              # all | targets_only
    grid_width: float = 1.0                   # hours; events | fixed grid
    grid_mode: str = "events"                 # events | grid
    fill: str = "mean"                        # mean | ffill
    forward_limit: float = 0.0                # hours of ffill validity
    age_feature: bool = False
    normalize: str = "zscore"                 # zscore | minmax | none

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineSpec:
    pipeline_id: str
    data_view: DataView
    model_id: str
    model_config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    parent_id: str | None = None

    def fingerprint(self, upstream_hash: str, dataset_version: str, splits_hash: str) -> str:
        import hashlib
        import json as _json
        payload = _json.dumps(
            {
                "data_view": self.data_view.to_dict(),
                "model_id": self.model_id,
                "model_config": self.model_config,
                "seed": self.seed,
                "upstream": upstream_hash,
                "dataset": dataset_version,
                "splits": splits_hash,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "data_view": self.data_view.to_dict(),
            "model_id": self.model_id,
            "model_config": self.model_config,
            "seed": self.seed,
            "parent_id": self.parent_id,
        }


@dataclass
class Feedback:
    status: str                                # ok | rejected | error | timeout
    checks: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, Any] = field(default_factory=dict)
    delta_vs_parent: float | None = None
    delta_vs_f0: float | None = None
    support_counts: dict[str, int] = field(default_factory=dict)
    plot_paths: list[str] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    remaining_budget: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Action:
    op: str
    target_field: str | None = None      # renamed from `field` to avoid shadowing
    value: Any = None
    parent_id: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    expected_effect: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass
class Observation:
    """What the agent is allowed to see in one round."""
    task_id: str
    round_index: int
    dataset_card: dict[str, Any]
    statistics: dict[str, Any]
    plot_paths: list[str]
    archive_summary: list[dict[str, Any]]
    incumbent: dict[str, Any] | None
    last_feedback: dict[str, Any] | None
    budget: dict[str, Any]
    allowed_ops: list[str]
    action_space: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
