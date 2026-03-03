from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .controller import RuntimeController


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw


def cmd_enqueue(
    queue_dir: str,
    path: str,
    args_raw: list[str],
    kwargs_raw: str | None,
    parser: argparse.ArgumentParser,
) -> int:
    args = [_parse_value(v) for v in args_raw]
    kwargs = _parse_value(kwargs_raw) if kwargs_raw is not None else {}
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        parser.error("--kwargs must be a JSON object (e.g. --kwargs '{\"label\":\"manual\"}')")
    cmd = RuntimeController.enqueue(queue_dir, path, args=args, kwargs=kwargs, op="set")
    print(json.dumps(cmd, ensure_ascii=False))
    return 0


def cmd_status(queue_dir: str, last: int) -> int:
    q = Path(queue_dir)
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
    p.add_argument("-q", "--queue-dir", required=True, help="Queue directory")
    p.add_argument("path", nargs="?", help="Registered control path, or 'status'")
    p.add_argument("args", nargs="*", help="Positional args (JSON or raw strings)")
    p.add_argument("--kwargs", default=None, help="Keyword args as JSON object")
    p.add_argument("--last", type=int, default=10, help="For status: how many ack lines to show")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.path == "status" and not args.args and args.kwargs is None:
        return cmd_status(args.queue_dir, args.last)

    if args.path is None:
        parser.print_help()
        return 2

    return cmd_enqueue(args.queue_dir, args.path, args.args, args.kwargs, parser)


if __name__ == "__main__":
    raise SystemExit(main())
