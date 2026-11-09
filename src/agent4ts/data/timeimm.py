"""Time-IMM / EPA-Air backend (air-quality forecasting).

Uses the processed CSV layout from the Time-IMM repository (datetime + pollutant
columns). Kept separate from BITS's EPA-Air entry with its own dataset_version so
the two splits are never mixed, per DESIGN 2.2.

If the processed CSV is absent, raises `blocked_resource` naming the script that
fetches it, rather than fabricating a series.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..schemas import DatasetCard, EventTable, QueryPolicy, TaskBundle, TaskSpec, Track
from ..data.bits import BlockedResource

log = get_logger("data.timeimm")

CITIES = ("Los_Angeles", "Dallas", "Denver")
POLLUTANTS = ("O3", "PM2.5", "PM10", "NO2", "CO", "SO2", "TEMP", "PRES", "WSPD")
_UNITS = {"O3": "ppb", "PM2.5": "ug/m3", "PM10": "ug/m3", "NO2": "ppb",
          "CO": "ppm", "SO2": "ppb", "TEMP": "degC", "PRES": "mbar", "WSPD": "m/s"}


def load_epa_air(
    root: Path,
    city: str = "Los_Angeles",
    track: Track = Track.NATIVE_IRR,
    history_days: float = 7.0,
    horizon_days: float = 1.0,
    target_variables: tuple[str, ...] = ("O3", "PM2.5"),
    max_entities: int | None = 400,
    seed: int = 0,
) -> TaskBundle:
    if city not in CITIES:
        raise KeyError(f"unknown city {city}; have {CITIES}")
    csv = _find_csv(root, city)
    if csv is None:
        raise BlockedResource(
            f"timeimm EPA-Air {city}",
            f"processed CSV not found under {root}; run scripts/gen_eval/00_download_resources.sh timeimm",
        )
    df = _read_epa(csv)
    return _build_from_frame(df, city, track, history_days, horizon_days,
                             target_variables, max_entities, seed)


def _find_csv(root: Path, city: str) -> Path | None:
    for pat in (f"{city}.csv", f"{city.lower()}.csv", f"*{city}*.csv"):
        hits = list(root.rglob(pat)) if root.exists() else []
        if hits:
            return sorted(hits)[0]
    return None


def _read_epa(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df.rename(columns={"date_time": "datetime", "temp": "TEMP", "pm2_5": "PM2.5", "ozone": "O3"})
    time_col = next((c for c in df.columns if c.lower() in ("datetime", "date", "time")), df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if df[time_col].duplicated().any():
        raise ValueError("Duplicate source timestamps require explicit aggregation")
    df = df.set_index(time_col).reindex(pd.date_range(df[time_col].min(), df[time_col].max(), freq="h"))
    return df.rename_axis("timestamp").reset_index()


def _build_from_frame(df, city, track, history_days, horizon_days,
                      target_variables, max_entities, seed) -> TaskBundle:
    """Split the long series into overlapping entities, each with a cutoff."""
    step_hours = 1.0  # source cadence expected hourly
    hist_steps = int(history_days * 24 / step_hours)
    horiz_steps = int(horizon_days * 24 / step_hours)
    window = hist_steps + horiz_steps

    present = [v for v in POLLUTANTS if v in df.columns]
    variables = present
    vidx = {v: i for i, v in enumerate(variables)}
    target_idx = [vidx[v] for v in target_variables if v in vidx]
    if not target_idx:
        raise BlockedResource(f"timeimm EPA-Air {city}",
                              f"none of {target_variables} present in {list(df.columns)}")

    vals = df[variables].to_numpy(dtype=np.float64)
    times = df["timestamp"].to_numpy()

    n = len(df)
    starts = list(range(0, max(0, n - window + 1), window))
    if max_entities:
        starts = starts[:max_entities]

    ev_ent, ev_t, ev_av, ev_v, ev_val = [], [], [], [], []
    tg_ent, tg_t, tg_av, tg_v, tg_val = [], [], [], [], []
    anchors: dict[int, float] = {}

    for eid, s0 in enumerate(starts):
        cutoff_step = s0 + hist_steps
        if cutoff_step >= n:
            continue
        anchors[eid] = history_days * 24.0
        for k in range(s0, cutoff_step):
            for vi in range(len(variables)):
                if np.isnan(vals[k, vi]):
                    continue  # missing pollutant hour stays missing
                rel = (k - s0 + 1) * step_hours
                ev_ent.append(eid); ev_t.append(rel); ev_av.append(rel)
                ev_v.append(vi); ev_val.append(float(vals[k, vi]))
        for k in range(cutoff_step, min(cutoff_step + horiz_steps, n)):
            rel = (k - s0 + 1) * step_hours
            for vi in target_idx:
                if np.isnan(vals[k, vi]):
                    continue
                tg_ent.append(eid); tg_t.append(rel); tg_av.append(rel)
                tg_v.append(vi); tg_val.append(float(vals[k, vi]))

    if not anchors:
        raise BlockedResource(f"timeimm EPA-Air {city}", "series too short for the window")

    history = EventTable(
        entity_id=np.asarray(ev_ent, dtype=np.int64), event_time=np.asarray(ev_t, dtype=np.float64),
        available_time=np.asarray(ev_av, dtype=np.float64), variable_id=np.asarray(ev_v, dtype=np.int64),
        value=np.asarray(ev_val, dtype=np.float64))
    targets = EventTable(
        entity_id=np.asarray(tg_ent, dtype=np.int64), event_time=np.asarray(tg_t, dtype=np.float64),
        available_time=np.asarray(tg_av, dtype=np.float64), variable_id=np.asarray(tg_v, dtype=np.int64),
        value=np.asarray(tg_val, dtype=np.float64))

    ids = np.asarray(sorted(anchors), dtype=np.int64)
    a, b, c = [int(len(ids) * r) for r in (.6, .8, .9)]
    split_ids = {k: v.tolist() for k, v in zip(("train", "search", "confirm", "test"), np.split(ids, [a,b,c]))}
    if any(len(v) == 0 for v in split_ids.values()):
        raise BlockedResource("EPA-Air", "Too few disjoint windows for four chronological splits")
    spec = TaskSpec(
        task_id=f"epa_air_{city}_native_irr",
        dataset_version=f"timeimm_epa_air_{city}_chronological_disjoint_v2",
        track=track, backend="timeimm", split_ids=split_ids,
        target_variables=[v for v in target_variables if v in vidx],
        covariate_variables=[v for v in variables if v not in target_variables],
        cutoff_rule=f"anchor + {history_days}d", history_duration=history_days * 24.0,
        horizon=horizon_days * 24.0, query_policy=QueryPolicy.POINT,
        query_grid=[float(x) for x in range(1, horiz_steps + 1)],
        f0_model="DLinear",
        notes=f"Time-IMM EPA-Air {city}; separate dataset_version from BITS EPA-Air.",
    )
    spec.validate()
    card = DatasetCard(
        dataset_version=spec.dataset_version, backend="timeimm",
        variables={v: {"unit": "unknown", "kind": "instantaneous",
                       "missing_meaning": "no sensor reading for that hour (NaN in source)"}
                   for v in variables},
        time_semantics={"origin": "series start", "unit": "hours after window start",
                        "source_cadence": "hourly", "irregularity": "gaps from sensor downtime"},
        provenance=[{"name": "Time-IMM EPA-Air", "city": city,
                     "url": "https://github.com/blacksnail789521/Time-IMM"}],
        unknown=["Source numeric units await authoritative metadata verification; no unit conversion applied"],
    )
    log.info("EPA-Air %s: %d entities, %d history, %d targets",
             city, len(anchors), len(history), len(targets))
    return TaskBundle(spec=spec, card=card, history=history, targets=targets,
                      variable_names=variables, entity_anchors=anchors, train_stats={})
