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
    def __init__(self, queue_dir: Optional[str] = None, *, ddp: bool = False):
        self.queue_dir: Optional[Path] = None
        self.commands_path: Optional[Path] = None
        self.acks_path: Optional[Path] = None
        self.configured = False

        self._controls: Dict[str, _Control] = {}
        self._last_poll_ts = 0.0
        self._read_offset = 0

        self.ddp = ddp

        if queue_dir is not None:
            self.configure(queue_dir)

    def configure(self, queue_dir: str) -> None:
        if self.configured:
            if self.queue_dir == Path(queue_dir):
                return
            raise ValueError(
                f"RuntimeController already configured with {self.queue_dir}"
            )
        q = Path(queue_dir)
        q.mkdir(parents=True, exist_ok=True)
        self.queue_dir = q
        self.commands_path = q / "commands.jsonl"
        self.acks_path = q / "acks.jsonl"
        self.commands_path.touch(exist_ok=True)
        self.acks_path.touch(exist_ok=True)
        self.configured = True


    def register(
        self,
        path: str,
        apply_fn: Callable[[Any, Any], None],
        validate_fn: Optional[Callable[[Any], Any]] = None,
        *,
        overwrite: bool = False,
    ) -> None:
        if not path or not isinstance(path, str):
            raise ValueError("path must be a non-empty string")
        if path in self._controls and not overwrite:
            return
        self._controls[path] = _Control(apply_fn=apply_fn, validate_fn=validate_fn)

    def poll_and_apply(self, ctx: Any = None, every_s: float = 2.0) -> List[dict]:
        self._ensure_queue_ready()

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

    def _ensure_queue_ready(self) -> None:
        if not self.configured or self.commands_path is None or self.acks_path is None:
            raise RuntimeError(
                "queue_dir is not set. Call configure(...) before poll_and_apply()."
            )

    def _is_ddp_active(self) -> bool:
        if not self.ddp:
            return False
        try:
            import torch.distributed as dist  # type: ignore

            return bool(dist.is_available() and dist.is_initialized())
        except Exception:
            return False

    def _ddp_collect_and_broadcast(self) -> List[dict]:
        import torch.distributed as dist  # type: ignore

        rank = dist.get_rank()
        payload = [self._read_new_commands_local()] if rank == 0 else [None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0] or []

    def _read_new_commands_local(self) -> List[dict]:
        self._ensure_queue_ready()
        assert self.commands_path is not None

        with self.commands_path.open("r", encoding="utf-8") as f:
            f.seek(self._read_offset)
            data = f.read()
            self._read_offset = f.tell()

        if not data:
            return []

        new_lines = data.splitlines()
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
        self._ensure_queue_ready()
        assert self.acks_path is not None
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
