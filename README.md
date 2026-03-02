# runtimectl

Minimal runtime control plane for training loops.

## Goals

- Keep training code changes minimal.
- Initialize once with queue location.
- Register controls with path + apply/validate functions.
- Poll control queue periodically.
- DDP-safe pattern: rank 0 polls queue, broadcasts instructions, all ranks apply.

## Python API

```python
from runtimectl import init_control, register_control, poll_and_apply

# initialize once (single source of truth for queue_dir)
init_control(queue_dir="/tmp/runtimectl", ddp=False)

register_control(
    "optimizer.lr.multiplier",
    apply_fn=lambda v, ctx: [pg.__setitem__("lr", pg["lr"] * float(v)) for pg in ctx["optimizer"].param_groups],
    validate_fn=lambda v: float(v),
)

# inside training loop
poll_and_apply(ctx={"optimizer": optimizer}, every_s=2.0)
```

If `register_control(...)` or `poll_and_apply(...)` is called before `init_control(...)`, a clear error is raised.

## DDP behavior

- If initialized with `ddp=True` and `dist_module=torch.distributed`:
  - rank 0 reads and drains queue
  - barrier
  - broadcast instructions to all ranks
  - barrier
  - all ranks apply same instructions in order

## CLI

`runtimectl` uses the same queue dir configured by `init_control(...)` (stored in `~/.runtimectl/current_queue_dir`).

Enqueue command:

```bash
runtimectl optimizer.lr.multiplier 0.5
runtimectl model.dropout.p 0.2
```

Status:

```bash
runtimectl status
runtimectl status --last 20
```

## Queue format

`runtimectl` writes one JSON command per line into `commands.jsonl`.

Example command:

```json
{"id":"...","ts":1700000000.0,"op":"set","path":"optimizer.lr.multiplier","value":0.5}
```

Acks are appended to `acks.jsonl`.
