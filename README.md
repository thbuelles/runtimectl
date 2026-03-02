# runtimectl

Minimal runtime control plane for training loops.

## Goals

- Keep training code changes minimal.
- Register controls with path + apply/validate functions.
- Poll control queue periodically.
- DDP-safe pattern: rank 0 polls queue, broadcasts instructions, all ranks apply.

## Install (dev)

```bash
pip install -e .
```

## Quick example

```python
from runtimectl import RuntimeController

rt = RuntimeController(queue_dir="/tmp/runtimectl")

rt.register_control(
    "optimizer.lr.multiplier",
    apply_fn=lambda v, ctx: [pg.__setitem__("lr", pg["lr"] * v) for pg in ctx["optimizer"].param_groups],
    validate_fn=lambda v: float(v) if float(v) > 0 else (_ for _ in ()).throw(ValueError("must be > 0")),
)

# inside training loop
rt.poll_and_apply(ctx={"optimizer": optimizer}, every_s=2.0)
```

## DDP behavior

- If `ddp=True` and `dist` is initialized:
  - rank 0 reads and drains queue
  - instructions are broadcast to all ranks
  - all ranks apply the same instructions in order

## Queue format

`runtimectl` writes one JSON command per line into `queue/commands.jsonl`.

Example command:

```json
{"id":"...","ts":1700000000.0,"op":"set","path":"optimizer.lr.multiplier","value":0.5}
```

Acks are appended to `queue/acks.jsonl`.
