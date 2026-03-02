from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from .controller import RuntimeController

_STATE_FILE = Path.home() / ".runtimectl" / "current_queue_dir"
_CONTROLLER: RuntimeController | None = None


def init_control(queue_dir: str, *, ddp: bool = False, dist_module: Any = None) -> RuntimeController:
    global _CONTROLLER
    _CONTROLLER = RuntimeController(queue_dir=queue_dir, ddp=ddp, dist_module=dist_module)

    # Persist queue location so CLI can target the same queue without extra args.
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(queue_dir, encoding="utf-8")
    return _CONTROLLER


def get_controller() -> RuntimeController:
    if _CONTROLLER is None:
        raise RuntimeError("runtimectl not initialized. Call init_control(queue_dir=...) first.")
    return _CONTROLLER


def register_control(
    path: str,
    apply_fn: Callable[[Any, Any], None],
    validate_fn: Optional[Callable[[Any], Any]] = None,
) -> None:
    get_controller().register_control(path=path, apply_fn=apply_fn, validate_fn=validate_fn)


def poll_and_apply(ctx: Any = None, every_s: float = 2.0):
    return get_controller().poll_and_apply(ctx=ctx, every_s=every_s)


def get_current_queue_dir() -> str:
    if not _STATE_FILE.exists():
        raise RuntimeError(
            "No queue configured for CLI. Run init_control(queue_dir=...) first in your training setup."
        )
    q = _STATE_FILE.read_text(encoding="utf-8").strip()
    if not q:
        raise RuntimeError("Invalid queue configuration: empty queue_dir.")
    return q
