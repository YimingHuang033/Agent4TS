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

log = get_logger("controller")

SYSTEM_PROMPT = """You are an agent that improves an irregular time-series forecasting pipeline.
You see numeric statistics, a semantic data card, and (when enabled) real PNG plots of the history.
Reply with ONE JSON object and nothing else. No prose, no markdown, no explanation.

Allowed ops: %s
Allowed data_view fields (use as "field"): %s
Allowed models (use as "value" when op is SELECT_MODEL): %s

Schema:
{"op": "<one of the allowed ops>", "field": "<a data_view field, or null>",
 "value": <new value>, "parent_id": null, "evidence_refs": [],
 "expected_effect": "<one short sentence>"}

Rules:
- Change exactly ONE field per reply.
- For SELECT_MODEL put the model id in "value" and set "field": null.
- For SET_TRANSFORM put a data_view field name in "field" and the new value in "value".
- Never invent field names like "statistics.n_events" or "model"; only use the
  allowed ops, fields and models listed above.
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
            log.warning("malformed action, retrying once")
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
        resp = client.chat.completions.create(
            model=self.llm["model"], messages=messages,
            temperature=self.llm.get("temperature", 0.2),
            max_tokens=self.llm.get("max_tokens", 2048))
        return resp.choices[0].message.content or ""

    def _parse(self, raw: str) -> Action | None:
        if not raw:
            return None
        text = raw.strip()
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
        loop exercises accept, no-improvement and fallback paths end to end.
        """
        text = messages[1]["content"]
        if isinstance(text, list):
            text = text[0]["text"]
        payload = json.loads(text)
        rnd = payload["round"]
        ladder = [
            {"op": "SELECT_MODEL", "field": None, "value": "DLinear",
             "expected_effect": "start from a linear baseline"},
            {"op": "SET_TRANSFORM", "field": "history_fraction", "value": 0.5,
             "expected_effect": "shorten lookback to focus on recent regime"},
            {"op": "SET_TRANSFORM", "field": "fill", "value": "ffill",
             "expected_effect": "hold last value forward instead of mean fill"},
            {"op": "SET_TRANSFORM", "field": "forward_limit", "value": 6.0,
             "expected_effect": "cap forward hold to avoid stale carry"},
            {"op": "SELECT_MODEL", "field": None, "value": "PatchTST",
             "expected_effect": "try a patch transformer"},
            {"op": "STOP", "expected_effect": "budget-conscious stop"},
        ]
        return json.dumps(ladder[min(rnd, len(ladder) - 1)])
