"""MIMIC-IV backend stub.

Requires PhysioNet credentialed access. Without it, every entry point returns
`blocked_access`; the semantic-card template is kept so the adapter can be filled
in once access is granted. No MIMIC data is shipped or auto-downloaded.
"""
from __future__ import annotations

from pathlib import Path

from ..logging_utils import get_logger
from ..schemas import DatasetCard

log = get_logger("data.mimic")

ACCESS_URL = "https://physionet.org/content/mimiciv/3.0/"


class BlockedAccess(RuntimeError):
    def __init__(self, what: str = "MIMIC-IV v3.0", reason: str = "credentialed access not granted"):
        self.what, self.reason = what, reason
        super().__init__(f"blocked_access: {what} ({reason})")

    def as_status(self) -> dict:
        return {"status": "blocked_access", "what": self.what, "reason": self.reason,
                "how_to_obtain": ACCESS_URL}


def card_template() -> DatasetCard:
    """Filled only when access exists; unknown fields stay 'unknown'."""
    return DatasetCard(
        dataset_version="mimic_iv_3.0 (not loaded)",
        backend="mimic",
        variables={},
        time_semantics={"note": "charttime (event) vs storetime (available) must be audited per table"},
        provenance=[{"name": "MIMIC-IV", "version": "3.0", "url": ACCESS_URL}],
        unknown=["all variables", "event/record time semantics per table"],
    )


def load_mimic(*_args, **_kwargs):
    raise BlockedAccess()


def access_status(root: Path | None = None) -> dict:
    present = bool(root and Path(root).exists() and any(Path(root).iterdir()))
    if present:
        return {"status": "present_unaudited", "root": str(root),
                "note": "files exist but schema/time semantics not yet audited"}
    return BlockedAccess().as_status()
