"""Controller: one multimodal LLM that emits a structured action per round.

Constraints enforced here (DESIGN 5.2):
  * output is a single JSON action, one field changed at a time,
  * the multimodal prompt attaches the *real* PNG bytes, not just paths,
  * a malformed reply gets at most one retry, counted against the LLM budget,
  * `mock` transport produces a deterministic action for offline smoke tests so
    the loop runs without a live model.
"""
from __future__ import annotations

import base64
import json
from typing import Any

from .logging_utils import get_logger
from .observe import load_png_bytes
from .schemas import Action, Observation
from .transforms import TRANSFORM_SPACE

log = get_logger("controller")

SYSTEM_PROMPT = """You are an agent that improves an irregular time-series forecasting pipeline.
You see numeric statistics, a semantic data card, and (when enabled) real PNG plots of the history.
Reply with ONE JSON object and nothing else. No prose, no markdown, no explanation.

Allowed ops: %s
Allowed data_view fields (use as "field"): %s
Allowed models (use as "value" when op is SELECT_MODEL): %s
Allowed transform values (SET_TRANSFORM/SELECT_DATA must use ONLY these): %s

Schema:
{"op": "<one of the allowed ops>", "field": "<a data_view field, or null>",
 "value": <new value>, "parent_id": null, "evidence_refs": [],
 "expected_effect": "<one short sentence>"}

Rules:
- Change exactly ONE field per reply.
- For SELECT_MODEL put the model id in "value" and set "field": null.
- For SET_TRANSFORM put a data_view field name in "field" and the new value in
  "value"; the value MUST come from the allowed transform values above.
- Never invent field names like "statistics.n_events" or "model"; only use the
  allowed ops, fields and models listed above.
- If a previous reply was rejected, read last_feedback.checks and propose a
  DIFFERENT, allowed action; never repeat a rejected one.
- Never re-propose the incumbent itself (same model AND same values); that is
  rejected as a no-op. Change exactly one thing to a NEW allowed value, or STOP.
- Do not repeat a proposal already tried: check archive_summary (each entry
  lists its data_view/model_config and resulting normalized_mae). If a field
  value is already in the archive, pick a DIFFERENT field or value, or STOP.
- Lower normalized_mae is better. Tried-and-not-better values are dead ends.
- If nothing is worth changing, reply {"op": "STOP"}.

Example valid reply:
{"op": "SET_TRANSFORM", "field": "history_fraction", "value": 0.5, "parent_id": null,
 "evidence_refs": ["plots show a long gap"], "expected_effect": "focus on recent regime"}

When the plots or statistics show long gaps or stale observations, prefer a short
history_fraction, ffill with a forward_limit, or a model that accepts dt."""


class Controller:
    def __init__(self, cfg: dict[str, Any], run_dir):
        self.llm = cfg["llm"]
        self.cfg = cfg
        self.run_dir = run_dir

    # ------------------------------------------------------------- prompt
    def build_messages(self, obs: Observation) -> list[dict[str, Any]]:
        from .models.registry import runnable_ids
        from .schemas import DataView
        system = SYSTEM_PROMPT % (
            json.dumps(obs.allowed_ops),
            json.dumps(sorted(DataView().__dict__.keys())),
            json.dumps(runnable_ids()),
            json.dumps(TRANSFORM_SPACE),
        )
        text = json.dumps({
            "task_id": obs.task_id,
            "round": obs.round_index,
            "dataset_card": obs.dataset_card,
            "statistics": obs.statistics,
            "archive_summary": obs.archive_summary,
            "incumbent": obs.incumbent,
            "last_feedback": obs.last_feedback,
            "budget": obs.budget,
            "action_space": obs.action_space,
        }, indent=2, default=str)

        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        # a text-only model (e.g. Qwen2.5-0.5B used while the GPUs are busy) runs
        # as the documented no_vision mode rather than being sent images it can't read
        if self.llm.get("supports_vision", True):
            for fig in obs.plot_paths:
                if isinstance(fig, dict) and "path" in fig:
                    try:
                        b64 = base64.b64encode(load_png_bytes(fig)).decode()
                    except Exception as exc:
                        log.warning("could not read figure %s: %s", fig.get("path"), exc)
                        continue
                    content.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"}})
        return [{"role": "system", "content": system},
                {"role": "user", "content": content}]

    # ------------------------------------------------------------- call
    def propose(self, obs: Observation) -> Action:
        messages = self.build_messages(obs)
        raw = self._call(messages)
        action = self._parse(raw)
        if action is None:
            # honest diagnosis: show how the reply ended so truncation vs
            # garbage is visible in the trace of the run
            log.warning("malformed action (tail=%r), retrying once", (raw or "")[-120:])
            retry = messages + [{"role": "user", "content": "Reply with ONLY the JSON action object."}]
            raw = self._call(retry)
            action = self._parse(raw)
        if action is None:
            return Action(op="STOP", expected_effect="unparseable reply after retry")
        return action

    def _call(self, messages: list[dict[str, Any]]) -> str:
        transport = self.llm.get("transport", "openai")
        if transport == "mock":
            return self._mock(messages)
        from openai import OpenAI
        client = OpenAI(base_url=self.llm["base_url"], api_key=self.llm.get("api_key", "EMPTY"),
                        timeout=self.llm.get("timeout", 180))
        extra: dict[str, Any] = {}
        if "enable_thinking" in self.llm:
            # Qwen3-style chat templates accept this toggle via vLLM; keeps the
            # action JSON out of a reasoning block and inside the token budget
            extra["chat_template_kwargs"] = {"enable_thinking": bool(self.llm["enable_thinking"])}
        try:
            resp = client.chat.completions.create(
                model=self.llm["model"], messages=messages,
                temperature=self.llm.get("temperature", 0.2),
                max_tokens=self.llm.get("max_tokens", 2048),
                extra_body=extra)
        except Exception as exc:
            # endpoint down / context overflow / timeout: a failed call must not
            # kill the run; the empty reply goes down the malformed-action path
            # (one retry, then an honest STOP) and is visible in the log
            log.warning("LLM call failed: %s: %s", type(exc).__name__, exc)
            return ""
        return resp.choices[0].message.content or ""

    def _parse(self, raw: str) -> Action | None:
        if not raw:
            return None
        text = raw.strip()
        # reasoning models emit a <think>…</think> block; the block routinely
        # contains braces (e.g. an echoed schema) that would break the JSON
        # scan below, so the final answer is whatever follows the last closing
        # tag. vLLM's reasoning parser consumes the opening tag, so the closing
        # tag alone is the normal marker here; an unclosed <think> means
        # generation was cut off mid-reasoning: nothing to parse.
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1].strip()
        elif "<think>" in text:
            return None
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
        if "op" not in obj:
            return None
        return Action(
            op=obj.get("op"),
            target_field=obj.get("field"),
            value=obj.get("value"),
            parent_id=obj.get("parent_id"),
            evidence_refs=obj.get("evidence_refs", []),
            expected_effect=obj.get("expected_effect", ""),
        )

    # ------------------------------------------------------------- mock
    def _mock(self, messages: list[dict[str, Any]]) -> str:
        """Deterministic scripted policy for offline smoke runs.

        Reads the round from the text payload and proposes a fixed ladder so the
        loop exercises the accept path, a no-improvement fallback and STOP end to
        end, without a live model.

        The ladder is data-driven, not arbitrary: on the smoke tasks the default
        view is already event-direct on the full history (the f0 view), and a
        measured sweep shows the one change that reliably *improves* f0 for both
        physiome and P12 is switching the window builder to a fixed grid
        (`grid_mode=grid`). Steps that do not improve (shortening history, a
        heavier model) follow so the rollback path is exercised too.
        """
        text = messages[1]["content"]
        if isinstance(text, list):
            text = text[0]["text"]
        payload = json.loads(text)
        rnd = payload["round"]
        ladder = [
            {"op": "SET_TRANSFORM", "field": "grid_mode", "value": "grid",
             "expected_effect": "bucket the irregular history onto a fixed grid"},
            {"op": "SET_TRANSFORM", "field": "grid_width", "value": 2.0,
             "expected_effect": "widen buckets to smooth sparse variables"},
            {"op": "SET_TRANSFORM", "field": "fill", "value": "ffill",
             "expected_effect": "hold last value forward instead of mean fill"},
            {"op": "SET_TRANSFORM", "field": "history_fraction", "value": 0.5,
             "expected_effect": "shorten lookback (expect a step back)"},
            {"op": "SELECT_MODEL", "field": None, "value": "PatchTST",
             "expected_effect": "try a patch transformer (expect a step back)"},
            {"op": "STOP", "expected_effect": "budget-conscious stop"},
        ]
        return json.dumps(ladder[min(rnd, len(ladder) - 1)])
