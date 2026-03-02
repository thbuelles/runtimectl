from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class _Control:
    apply_fn: Callable[[Any, Any], None]
    validate_fn: Optional[Callable[[Any], Any]] = None


class RuntimeController:
    def __init__(self, queue_dir: str, *, ddp: bool = False, dist_module: Any = None):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.commands_path = self.queue_dir / "commands.jsonl"
        self.acks_path = self.queue_dir / "acks.jsonl"

        self._controls: Dict[str, _Control] = {}
        self._last_poll_ts = 0.0
        self._read_offset = 0

        self.ddp = ddp
        self.dist = dist_module

        self.commands_path.touch(exist_ok=True)
        self.acks_path.touch(exist_ok=True)

    def register(
        self,
        path: str,
        apply_fn: Callable[[Any, Any], None],
        validate_fn: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if not path or not isinstance(path, str):
            raise ValueError("path must be a non-empty string")
        if path in self._controls:
            raise ValueError(f"control already registered: {path}")
        self._controls[path] = _Control(apply_fn=apply_fn, validate_fn=validate_fn)

    def poll_and_apply(self, ctx: Any = None, every_s: float = 2.0) -> List[dict]:
        now = time.time()
        if every_s > 0 and (now - self._last_poll_ts) < every_s:
            return []
        self._last_poll_ts = now

        if self._is_ddp_active():
            instructions = self._ddp_collect_and_broadcast()
        else:
            instructions = self._read_new_commands_local()

        applied = []
        for cmd in instructions:
            result = self._apply_command(cmd, ctx)
            applied.append(result)
            self._append_ack(result)
        return applied

    def _is_ddp_active(self) -> bool:
        if not self.ddp or self.dist is None:
            return False
        try:
            return bool(self.dist.is_available() and self.dist.is_initialized())
        except Exception:
            return False

    def _ddp_collect_and_broadcast(self) -> List[dict]:
        rank = self.dist.get_rank()
        payload = [self._read_new_commands_local()] if rank == 0 else [None]
        self.dist.barrier()
        self.dist.broadcast_object_list(payload, src=0)
        self.dist.barrier()
        return payload[0] or []

    def _read_new_commands_local(self) -> List[dict]:
        lines = self.commands_path.read_text(encoding="utf-8").splitlines()
        if self._read_offset >= len(lines):
            return []

        new_lines = lines[self._read_offset :]
        self._read_offset = len(lines)
        out = []
        for ln in new_lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                out.append(
                    {
                        "id": str(uuid.uuid4()),
                        "ts": time.time(),
                        "op": "invalid",
                        "path": "",
                        "value": None,
                        "_raw": ln,
                    }
                )
        return out

    def _apply_command(self, cmd: dict, ctx: Any) -> dict:
        cmd_id = cmd.get("id") or str(uuid.uuid4())
        op = cmd.get("op", "set")
        path = cmd.get("path")
        value = cmd.get("value")

        base = {"id": cmd_id, "ts": time.time(), "path": path, "op": op}

        if op != "set":
            return {**base, "status": "rejected", "error": f"unsupported op: {op}"}

        control = self._controls.get(path)
        if control is None:
            return {**base, "status": "rejected", "error": f"unknown path: {path}"}

        try:
            if control.validate_fn is not None:
                value = control.validate_fn(value)
            control.apply_fn(value, ctx)
            return {**base, "status": "applied", "value": value}
        except Exception as e:
            return {**base, "status": "failed", "error": str(e)}

    def _append_ack(self, ack: dict) -> None:
        with self.acks_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ack, ensure_ascii=False) + "\n")

    @staticmethod
    def enqueue(queue_dir: str, path: str, value: Any, op: str = "set") -> dict:
        qdir = Path(queue_dir)
        qdir.mkdir(parents=True, exist_ok=True)
        cmd = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "op": op,
            "path": path,
            "value": value,
        }
        commands_path = qdir / "commands.jsonl"
        with commands_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cmd, ensure_ascii=False) + "\n")
        return cmd
