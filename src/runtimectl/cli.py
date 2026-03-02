from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .controller import RuntimeController


def _parse_value(raw: str) -> Any:
    # Try JSON first so users can pass numbers/bools/objects cleanly.
    try:
        return json.loads(raw)
    except Exception:
        return raw


def cmd_set(args: argparse.Namespace) -> int:
    value = _parse_value(args.value)
    cmd = RuntimeController.enqueue(args.queue_dir, args.path, value, op="set")
    print(json.dumps(cmd, ensure_ascii=False))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    q = Path(args.queue_dir)
    commands = (q / "commands.jsonl")
    acks = (q / "acks.jsonl")

    def tail(path: Path, n: int):
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[-n:]

    out = {
        "queue_dir": str(q),
        "commands": len(commands.read_text(encoding="utf-8").splitlines()) if commands.exists() else 0,
        "acks": len(acks.read_text(encoding="utf-8").splitlines()) if acks.exists() else 0,
        "last_acks": [json.loads(x) for x in tail(acks, args.last)],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="runtimectl", description="Runtime queue control CLI")
    p.add_argument("-q", "--queue-dir", required=True, help="Queue directory")

    sub = p.add_subparsers(dest="command", required=True)

    p_set = sub.add_parser("set", help="Enqueue a set command")
    p_set.add_argument("path", help="Registered control path")
    p_set.add_argument("value", help="Value (JSON or raw string)")
    p_set.set_defaults(func=cmd_set)

    p_status = sub.add_parser("status", help="Show queue/ack summary")
    p_status.add_argument("--last", type=int, default=10, help="How many ack lines to show")
    p_status.set_defaults(func=cmd_status)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
