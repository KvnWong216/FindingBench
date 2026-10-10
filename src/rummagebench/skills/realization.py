"""Realization commit discipline for the symbolic interaction skills.

Pipeline per execution (remediation R01):

    verify -> snapshot -> standardized realization -> postcondition
    consistency -> commit

The benchmark-owned state stays the semantic truth, but a skill may commit a
semantic transition ONLY when its standardized realization actually took
effect in the backend. A realization that reports failure (or leaves the
backend half-committed) is an INFRASTRUCTURE fault: the pre-call snapshot is
restored and ``FeasibilityBackendError`` is raised — the semantic state is
never updated and the caller can never report EXECUTED for that step. The
visual session's existing ENGINE_ERROR path turns this into an invalid run.

Snapshot/restore reuse the backend's own counterfactual machinery
(``capture_observe_state`` / ``restore_observe_state``): simulation state,
commanded pose, holding relations and judgement caches. Backends without the
capability snapshot to ``None`` and simply fail loudly without rollback.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from rummagebench.core.errors import FeasibilityBackendError

logger = logging.getLogger(__name__)


def snapshot(backend) -> Any:
    capture = getattr(backend, "capture_observe_state", None)
    return capture() if callable(capture) else None


def rollback(backend, snap: Any) -> None:
    restore = getattr(backend, "restore_observe_state", None)
    if snap is not None and callable(restore):
        restore(snap)


def commit_realization(
    backend,
    realize: Callable[[], bool],
    what: str,
    verify: Callable[[], bool] | None = None,
) -> None:
    """Run ONE standardized realization under commit discipline.

    ``realize`` is the single transition attempt the validation stage
    selected; any backend-internal retries (other arm, alternative placement
    point) stay inside the backend — the skill never re-plans after failure.
    On a False return, a failing postcondition check or an exception, the
    backend is rolled back to the pre-call snapshot and the realization is
    reported as an infrastructure fault.
    """
    snap = snapshot(backend)
    try:
        realized = bool(realize())
    except Exception:
        rollback(backend, snap)
        raise
    if not realized or (verify is not None and not verify()):
        rollback(backend, snap)
        raise FeasibilityBackendError(
            f"{what}: realization did not establish its postcondition and was "
            f"rolled back; the step cannot be reported as EXECUTED"
        )
