from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .api import get_current_queue_dir
from .controller import RuntimeController


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw


def cmd_enqueue(path: str, value_raw: str) -> int:
    queue_dir = get_current_queue_dir()
    value = _parse_value(value_raw)
    cmd = RuntimeController.enqueue(queue_dir, path, value, op="set")
    print(json.dumps(cmd, ensure_ascii=False))
    return 0


def cmd_status(last: int) -> int:
    q = Path(get_current_queue_dir())
    commands = q / "commands.jsonl"
    acks = q / "acks.jsonl"

    def tail(path: Path, n: int):
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[-n:]

    out = {
        "queue_dir": str(q),
        "commands": len(commands.read_text(encoding="utf-8").splitlines()) if commands.exists() else 0,
        "acks": len(acks.read_text(encoding="utf-8").splitlines()) if acks.exists() else 0,
        "last_acks": [json.loads(x) for x in tail(acks, last)],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="runtimectl", description="Runtime queue control CLI")
    p.add_argument("path", nargs="?", help="Registered control path, or 'status'")
    p.add_argument("value", nargs="?", help="Value (JSON or raw string)")
    p.add_argument("--last", type=int, default=10, help="For status: how many ack lines to show")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.path == "status" and args.value is None:
        return cmd_status(args.last)

    if args.path is None or args.value is None:
        parser.print_help()
        return 2

    return cmd_enqueue(args.path, args.value)


if __name__ == "__main__":
    raise SystemExit(main())
