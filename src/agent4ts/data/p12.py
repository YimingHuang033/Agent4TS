"""P12 / PhysioNet Challenge 2012 backend.

Raw Set A records are irregular clinical observations: (relative time, variable,
value) with no fixed sampling grid and a per-record anchor at ICU admission.
This loader keeps event_time and available_time separate and never fabricates a
regular grid; the window builder adds a boolean mask instead of padding values.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..logging_utils import get_logger
from ..schemas import DatasetCard, EventTable, QueryPolicy, TaskBundle, TaskSpec, Track

log = get_logger("data.p12")

# PhysioNet 2012 static descriptors that appear in the Value column as text.
TEXT_VALUES = {"-1": np.nan, "MISSING": np.nan}

# Clinical variables, with audited units. Values recorded as text ("-1") are
# dropped to NaN so they never enter a mean as a real measurement.
VARIABLE_UNITS: dict[str, str] = {
    "HR": "bpm", "Temp": "degC", "SysABP": "mmHg", "DiasABP": "mmHg",
    "MAP": "mmHg", "NISysABP": "mmHg", "NIDiasABP": "mmHg", "NIMAP": "mmHg",
    "RespRate": "bpm", "SaO2": "%", "SpO2": "%", "FiO2": "%",
    "Urine": "mL", "GCS": "score", "pH": "unitless", "HCO3": "mmol/L",
    "BUN": "mg/dL", "Creatinine": "mg/dL", "Na": "mmol/L", "K": "mmol/L",
    "Cl": "mmol/L", "Glucose": "mg/dL", "Mg": "mmol/L", "Ca": "mmol/L",
    "Cholesterol": "mg/dL", "AST": "IU/L", "ALT": "IU/L", "Albumin": "g/dL",
    "Bilirubin": "mg/dL", "ALK": "IU/L", "Lactate": "mmol/L", "TroponinI": "ug/L",
    "TroponinT": "ug/L", "WBC": "cells/nL", "Platelets": "cells/nL", "Hgb": "g/dL",
    "HCT": "fraction", "PT": "sec", "INR": "unitless", "PTT": "sec",
    "Weight": "kg", "DiasABP.1": "mmHg",
}

_T_RE = re.compile(r"^(\d+):(\d+)$")


def _parse_time(token: str) -> float | None:
    """'HH:MM' -> hours since admission. Rejects non-clock tokens."""
    m = _T_RE.match(token.strip())
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    return hh + mm / 60.0


def read_record(path: Path, variable_index: dict[str, int]) -> tuple[list[tuple], dict[int, float]]:
    """Parse one record file -> (rows, static_outcomes)."""
    rows: list[tuple] = []
    with path.open("r", encoding="latin-1") as fh:
        header = fh.readline()
        if "Time" not in header:
            log.warning("unexpected header in %s: %r", path.name, header.strip())
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            t = _parse_time(parts[0])
            if t is None:
                continue
            name = parts[1].strip()
            raw = parts[2].strip()
            if name not in variable_index:
                continue
            if raw in TEXT_VALUES or raw == "":
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            rows.append((t, name, val))
    return rows, {}


def load_p12(
    root: Path,
    task_id: str = "p12_native_irr",
    track: Track = Track.NATIVE_IRR,
    history_hours: float = 24.0,
    horizon_hours: float = 6.0,
    target_variables: tuple[str, ...] = ("HR", "SysABP", "Temp", "Urine"),
    max_entities: int | None = None,
) -> TaskBundle:
    """Build a TaskBundle from PhysioNet Set A.

    Protocol (our own, NOT the official Challenge protocol): from the admission
    anchor, keep the first `history_hours` as history, and predict the bucketed
    mean of each target variable over the following `horizon_hours`.
    """
    rec_dir = root / "set-a"
    if not rec_dir.exists():
        raise FileNotFoundError(
            f"P12 raw data missing at {rec_dir}; run scripts/gen_eval/00_download_resources.sh p12"
        )
    files = sorted(rec_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no records under {rec_dir}")
    if max_entities:
        files = files[:max_entities]

    variables = sorted(VARIABLE_UNITS)
    variable_index = {name: i for i, name in enumerate(variables)}
    target_idx = [variable_index[v] for v in target_variables]

    ev_ent, ev_t, ev_av, ev_v, ev_val = [], [], [], [], []
    tg_ent, tg_t, tg_av, tg_v, tg_val = [], [], [], [], []
    anchors: dict[int, float] = {}

    bucket = 1.0  # query bucket width in hours
    n_buckets = int(round(horizon_hours / bucket))

    for eid, path in enumerate(files):
        rows, _ = read_record(path, variable_index)
        if not rows:
            continue
        cutoff = history_hours
        anchors[eid] = cutoff
        for t, name, val in rows:
            vid = variable_index[name]
            available = t  # charted at event time in Set A
            if t <= cutoff:
                ev_ent.append(eid); ev_t.append(t); ev_av.append(available)
                ev_v.append(vid); ev_val.append(val)
            elif t < cutoff + horizon_hours and vid in target_idx:
                b = int((t - cutoff) // bucket)
                if b < n_buckets:
                    qtime = cutoff + (b + 0.5) * bucket
                    tg_ent.append(eid); tg_t.append(qtime); tg_av.append(available)
                    tg_v.append(vid); tg_val.append(val)

    history = EventTable(
        entity_id=np.asarray(ev_ent, dtype=np.int64),
        event_time=np.asarray(ev_t, dtype=np.float64),
        available_time=np.asarray(ev_av, dtype=np.float64),
        variable_id=np.asarray(ev_v, dtype=np.int64),
        value=np.asarray(ev_val, dtype=np.float64),
    )
    # Build the fixed query grid: coarse buckets in hours after cutoff.
    query_grid = [round((b + 0.5) * bucket, 3) for b in range(n_buckets)]
    targets = EventTable(
        entity_id=np.asarray(tg_ent, dtype=np.int64),
        event_time=np.asarray(tg_t, dtype=np.float64),
        available_time=np.asarray(tg_av, dtype=np.float64),
        variable_id=np.asarray(tg_v, dtype=np.int64),
        value=np.asarray(tg_val, dtype=np.float64),
    )

    entity_ids = sorted(anchors)
    split_ids = split_entities(entity_ids)

    card = p12_card()
    spec = TaskSpec(
        task_id=task_id,
        dataset_version="p12_set_a_1.0.0",
        track=track,
        backend="p12",
        split_ids=split_ids,
        target_variables=list(target_variables),
        covariate_variables=[v for v in variables if v not in target_variables],
        cutoff_rule=f"anchor + {history_hours}h",
        history_duration=history_hours,
        horizon=horizon_hours,
        query_policy=QueryPolicy.BUCKETED,
        query_grid=query_grid,
        degradation=None,
        f0_model="DLinear",
        notes="Our protocol: 24h history -> bucketed-mean future of target variables.",
    )
    spec.validate()
    bundle = TaskBundle(
        spec=spec,
        card=card,
        history=history,
        targets=targets,
        variable_names=variables,
        entity_anchors=anchors,
        train_stats={},
    )
    log.info(
        "P12 loaded: %d entities, %d history events, %d target buckets, targets=%s",
        len(entity_ids), len(history), len(targets), task_id,
    )
    return bundle


def split_entities(entity_ids: list[int], ratios: dict[str, float] | None = None) -> dict[str, list[int]]:
    """Entity-level split; no entity ever appears in two splits."""
    ratios = ratios or {"train": 0.6, "search": 0.2, "confirm": 0.1, "test": 0.1}
    ids = np.asarray(entity_ids, dtype=np.int64)
    rng = np.random.default_rng(1234)
    perm = rng.permutation(len(ids))
    n = len(ids)
    n_train = int(n * ratios["train"])
    n_search = int(n * ratios["search"])
    n_confirm = int(n * ratios["confirm"])
    return {
        "train": ids[perm[:n_train]].tolist(),
        "search": ids[perm[n_train:n_train + n_search]].tolist(),
        "confirm": ids[perm[n_train + n_search:n_train + n_search + n_confirm]].tolist(),
        "test": ids[perm[n_train + n_search + n_confirm:]].tolist(),
    }


def p12_card() -> DatasetCard:
    return DatasetCard(
        dataset_version="p12_set_a_1.0.0",
        backend="p12",
        variables={
            name: {
                "unit": unit,
                "kind": "instantaneous",
                "missing_meaning": "not charted (no value recorded at that time)",
            }
            for name, unit in VARIABLE_UNITS.items()
        },
        time_semantics={
            "origin": "ICU admission (t=0 at first record)",
            "unit": "hours",
            "available_time_assumption": "Set A charts at event time; available_time == event_time",
            "irregularity": "native (asynchronous per-variable charting)",
        },
        provenance=[
            {"name": "PhysioNet Challenge 2012", "version": "1.0.0",
             "url": "https://physionet.org/content/challenge-2012/1.0.0/"},
        ],
        unknown=["Exact charting latency for lab values is not published"],
    )
