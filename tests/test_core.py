"""Core invariants: information boundaries, masks, splits, scoring, closed loop."""
import json
from pathlib import Path

import numpy as np
import pytest

from agent4ts import config as cfgmod, evaluator, tests_checks
from agent4ts.data import load_task
from agent4ts.degradation import apply_degradation
from agent4ts.schemas import Action, DataView
from agent4ts.transforms import Normalizer, build_labeled_windows


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_config("smoke/quick")


@pytest.fixture(scope="module")
def bundle(cfg):
    return load_task(cfg["tasks"]["physiome_dupont"], cfg)


def test_all_invariants(bundle, cfg):
    report = tests_checks.run_checks(bundle, cfg)
    assert report["status"] == "ok", report["failed"]


def test_no_entity_crosses_splits(bundle):
    seen = {}
    for name, ids in bundle.spec.split_ids.items():
        for e in ids:
            assert e not in seen, f"{e} in both {seen.get(e)} and {name}"
            seen[e] = name


def test_mask_marks_only_real_observations(bundle):
    view = DataView()
    windows = build_labeled_windows(bundle, view, "train")
    for lw in windows[:5]:
        n_real = int(lw.window.mask.sum())
        ids = bundle.history.entity_id == lw.window.entity_id
        sub = bundle.history.subset(ids)
        cells = {(round(float(t), 6), int(v)) for t, v in zip(sub.event_time, sub.variable_id)}
        assert n_real == len(cells)


def test_degradation_preserves_targets(bundle):
    deg = apply_degradation(bundle, {"kind": "drop_random", "ratio": 0.5, "tag": "t"}, seed=0)
    assert len(deg.targets) == len(bundle.targets)
    assert np.allclose(deg.targets.value, bundle.targets.value)
    assert len(deg.history) < len(bundle.history)
    assert deg.spec.task_id != bundle.spec.task_id


def test_normalizer_train_only(bundle):
    norm = Normalizer.fit(bundle.history, bundle.entities_for("train"))
    norm_all = Normalizer.fit(bundle.history, np.concatenate(
        [bundle.entities_for(s) for s in ("train", "search", "confirm", "test")]))
    # stats must come from train only -> generally differ from all-split stats
    diffs = [abs(norm.mean[k] - norm_all.mean.get(k, norm.mean[k])) for k in norm.mean]
    assert any(d > 0 for d in diffs)


def test_empty_group_returns_null(bundle, cfg):
    view = DataView()
    norm = Normalizer.fit(bundle.history, bundle.entities_for("train"))
    ev = build_labeled_windows(bundle, view, "search", norm)
    preds = [np.full(len(lw.query_times), np.nan) for lw in ev]  # all invalid
    r = evaluator.score_windows(ev, preds, bundle, norm, cfg["evaluation"])
    assert r["metrics"]["normalized_mae"] is None
    assert r["metrics"]["n_valid_targets"] == 0


def test_improvement_zero_denominator_is_null():
    assert evaluator.improvement(0, 1.0) is None
    assert evaluator.improvement(None, 1.0) is None
    assert evaluator.improvement(2.0, 1.0) == 0.5


def test_action_wire_format_roundtrip():
    a = Action(op="SET_TRANSFORM", target_field="fill", value="ffill")
    d = a.to_dict()
    assert d["op"] == "SET_TRANSFORM" and d["target_field"] == "fill"


def test_faults_disabled():
    from agent4ts import faults
    with pytest.raises(NotImplementedError):
        faults.inject_fault()
    assert faults.detect_irr_fault()["status"] == "not_implemented"


def test_bits_blocked(cfg):
    from agent4ts.data.bits import BlockedResource, load_bits_task
    with pytest.raises(BlockedResource) as ei:
        load_bits_task({}, cfg)
    assert ei.value.as_status()["status"] == "blocked_resource"


def test_upstream_models_require_real_source():
    from agent4ts.models.registry import make
    for mid in ('APN','ASTGI','TFMixer','KAFNet','HyperIMTS','TiWeaver','tPatchGNN'):
        with pytest.raises(NotImplementedError): make(mid,{},{})


def test_tsrouter_is_honest_not_faked():
    """TSRouter must report unavailable and refuse to run, not imitate upstream."""
    from agent4ts.baselines import tsrouter
    st = tsrouter.status({})
    assert st["status"] == "unavailable"
    with pytest.raises(RuntimeError):
        tsrouter.run()
    assert tsrouter.UPSTREAM_URL.endswith("tianyi-lab/TSRouter")


def test_mock_ladder_first_action_changes_state(bundle, cfg):
    """The offline mock's first action must be a real change vs the f0 view, so
    the smoke run actually exercises the accept path (no opaque no-op step)."""
    from agent4ts.controller import Controller
    from agent4ts.schemas import Observation, DataView
    from agent4ts.search import apply_action

    ctrl = Controller(cfg, Path("."))
    f0_view = DataView()
    obs = Observation(
        task_id=bundle.spec.task_id, round_index=0, dataset_card={},
        statistics={}, plot_paths=[], archive_summary=[],
        incumbent={"pipeline_id": "p0_f0", "model_id": bundle.spec.f0_model,
                   "data_view": f0_view.to_dict()},
        last_feedback=None, budget={"trials": 4}, allowed_ops=["SET_TRANSFORM"],
        action_space={})
    action = ctrl._parse(ctrl._mock(ctrl.build_messages(obs)))
    assert action.op in ("SET_TRANSFORM", "SELECT_MODEL", "SELECT_DATA")
    # applying it must actually move the incumbent (field value or model)
    child = apply_action(action, _spec(bundle, f0_view), n=1)
    assert (child.data_view.to_dict() != f0_view.to_dict()
            or child.model_id != bundle.spec.f0_model)


def _spec(bundle, view):
    from agent4ts.schemas import PipelineSpec
    return PipelineSpec("p0_f0", view, bundle.spec.f0_model, {}, 0)


def test_dlinear_kernel_option_removed():
    """`kernel` was dead code; the registry must not advertise a no-op option."""
    from agent4ts.models.registry import CATALOG, make
    assert "kernel" not in CATALOG["DLinear"]["options"]
    with pytest.raises(ValueError):
        make("DLinear", {"kernel": 7})


def test_lastvalue_predicts_original_units(bundle):
    """Window x is normalized by the view; LastValue must return raw units."""
    from agent4ts.models.runners import LastValue
    norm = Normalizer.fit(bundle.history, bundle.entities_for("train"))
    ev = build_labeled_windows(bundle, DataView(), "search", norm)
    m = LastValue()
    m._norm = norm
    m.fit([], bundle, np.array([0.0]), None)
    lw = ev[0]
    preds = m.predict(lw)
    sub = bundle.history.subset(bundle.history.entity_id == lw.window.entity_id)
    checked = 0
    for q in range(len(lw.query_times)):
        vid = int(lw.query_var_ids[q])
        mvar = sub.variable_id == vid
        if not mvar.any() or not np.isfinite(preds[q]):
            continue
        order = np.argsort(sub.event_time[mvar])
        last_raw = sub.value[mvar][order][-1]
        assert abs(preds[q] - last_raw) < 1e-6
        checked += 1
    assert checked > 0


def test_closed_loop_smoke(tmp_path, bundle, cfg):
    """One search run: proposals land, feedback flows, trace is replayable."""
    from agent4ts.search import run_search
    out = run_search(bundle, cfg, tmp_path, seed=0)
    assert out["rounds"] >= 1
    assert out["f0_metric"] is not None
    trace = (tmp_path / "trace.jsonl").read_text().splitlines()
    events = [json.loads(l) for l in trace]
    kinds = {e["event"] for e in events}
    assert {"f0", "propose"} <= kinds


def test_macro_scoring_equal_entity_variable_weight(bundle, cfg):
    import copy
    norm = Normalizer.fit(bundle.history, bundle.entities_for('train'))
    ws = copy.deepcopy(build_labeled_windows(bundle, DataView(), 'search', norm)[:2])
    preds = []
    for index, lw in enumerate(ws):
        lw.y_scored[:] = False
        count = 1 if index == 0 else 3
        lw.y_scored[:count] = True
        lw.query_var_ids[:] = lw.query_var_ids[0]
        lw.y_true[:] = 0
        preds.append(np.full(len(lw.y_true), (1 if index == 0 else 3) * norm.std[int(lw.query_var_ids[0])]))
    score = evaluator.score_windows(ws, preds, bundle, norm, cfg['evaluation'])
    assert score['metrics']['normalized_mae'] == pytest.approx(2)
    assert score['metrics']['n_valid_targets'] == 4


def test_private_splits_and_cache_are_isolated(tmp_path, bundle, cfg):
    from agent4ts.harness import Harness
    spec = _spec(bundle, DataView())
    public = Harness(bundle, cfg, tmp_path / 'public')
    public.run_trial(spec)
    with pytest.raises(PermissionError):
        public.run_trial(spec, 'test')
    private = Harness(bundle, cfg, tmp_path / 'private', allow_test=True)
    a = private.run_trial(spec, 'confirm')
    b = private.run_trial(spec, 'test')
    assert a.status == b.status == 'ok'
    assert a is not b
    assert len(private.cache) == 2
    assert private.used["trials"] == 1
    assert [x['split'] for x in map(json.loads, private.trace_path.read_text().splitlines())] == ['confirm', 'test']


def test_frozen_export_predicts_without_fit(tmp_path, bundle, cfg, monkeypatch):
    import pickle
    from agent4ts.harness import Harness
    from agent4ts.models.runners import DLinear
    h = Harness(bundle, cfg, tmp_path / 'run')
    spec = _spec(bundle, DataView())
    assert h.run_trial(spec).status == 'ok'
    payload = h.export(spec, tmp_path / 'export')
    model = pickle.loads((tmp_path / 'export' / payload['model_artifact']).read_bytes())
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must not train')
    monkeypatch.setattr(type(model), 'fit', forbidden)
    predictions = model.predict(h.windows('search', spec.data_view)[0])
    assert np.isfinite(predictions).all()


def test_training_timeout_preserves_failure(tmp_path, bundle, cfg, monkeypatch):
    import copy, time
    from agent4ts.harness import Harness
    import agent4ts.harness as module
    class SlowModel:
        def fit(self, *args): time.sleep(.1)
    monkeypatch.setattr(module, 'make_model', lambda *args: SlowModel())
    small = copy.deepcopy(cfg);small['budget']['trial_timeout_seconds'] = .01
    h=Harness(bundle,small,tmp_path)
    result=h.run_trial(_spec(bundle,DataView()))
    assert result.status == 'error'
    assert 'TimeoutError' in result.error
    assert not result.metrics
    assert h.used['trials'] == 1


def test_robustness_thresholds_ignore_test_history(bundle):
    import copy
    _,before = evaluator.robustness_diagnostics(bundle)
    changed=copy.deepcopy(bundle)
    mask=np.isin(changed.history.entity_id,changed.entities_for('test'))
    changed.history.event_time[mask]=0
    _,after=evaluator.robustness_diagnostics(changed)
    assert before == after


def test_controller_resolves_secret_environment(cfg, monkeypatch, tmp_path):
    from agent4ts.controller import Controller
    monkeypatch.setenv('ATS_LLM_API_KEY','test-only-key')
    monkeypatch.setenv('ATS_LLM_BASE_URL','https://example.invalid/v1')
    controller=Controller(cfg,tmp_path)
    assert controller.llm['api_key']=='test-only-key'
    assert controller.llm['base_url']=='https://example.invalid/v1'
    assert cfg['llm']['api_key']!='test-only-key'


def test_transform_space_is_finite_and_rejects_invalid_values(bundle):
    from agent4ts.transforms import TRANSFORM_SPACE, compatible
    assert all(isinstance(values,list) for values in TRANSFORM_SPACE.values())
    assert not compatible(bundle,DataView(grid_width='invalid'))[0]


def test_age_covariates_change_upstream_input(bundle,cfg,tmp_path):
    from agent4ts.harness import Harness
    from agent4ts.models.apn import APN
    h=Harness(bundle,cfg,tmp_path)
    lw=h.windows('search',DataView())[0]
    m=APN(cfg['resources']['apn_source'],epochs=1)
    m._view=DataView(age_feature=True);m.n_original=len(bundle.variable_names)
    m.params={'enc_in':2*m.n_original,'history':bundle.spec.history_duration}
    inputs,_=m._inputs(lw)
    assert inputs['x'].shape[-1] == 2*m.n_original
    assert inputs['x_mask'][...,m.n_original:].all()
    assert inputs['x'][...,m.n_original:].max() > 0
