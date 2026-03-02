# runtimectl

Minimal runtime control plane for training loops.

## Design

`runtimectl` is explicit by default:
- you create a `RuntimeController` instance in your code
- register controls on that instance
- call `poll_and_apply` on that instance

No global singleton convenience wrappers.

## Recommended integration pattern

Create a dedicated owner module in your training project, e.g. `runtime_control.py`.

```python
# runtime_control.py
import torch.distributed as dist
from runtimectl import RuntimeController

rt = RuntimeController(
    queue_dir="/tmp/runtimectl",
    ddp=True,
    dist_module=dist,
)
```

Then import `rt` where needed.

```python
# controls.py
from runtime_control import rt

def validate_mult(v):
    v = float(v)
    if v <= 0:
        raise ValueError("multiplier must be > 0")
    return v

def apply_lr_multiplier(mult, ctx):
    for pg in ctx["optimizer"].param_groups:
        pg["lr"] *= mult

rt.register_control("optimizer.lr.multiplier", apply_lr_multiplier, validate_mult)
```

```python
# train_loop.py
from runtime_control import rt

# inside training loop
rt.poll_and_apply(ctx={"optimizer": optimizer, "model": model}, every_s=2.0)
```

## DDP behavior

If initialized with `ddp=True` and `dist_module=torch.distributed`:
- rank 0 reads and drains queue
- barrier
- broadcast instructions to all ranks
- barrier
- all ranks apply same instructions in order

## CLI

Enqueue command:

```bash
runtimectl -q /tmp/runtimectl optimizer.lr.multiplier 0.5
runtimectl -q /tmp/runtimectl model.dropout.p 0.2
```

Status:

```bash
runtimectl -q /tmp/runtimectl status
runtimectl -q /tmp/runtimectl status --last 20
```

## DDP example

See `examples/ddp_train.py` for a full minimal torch.distributed example.

Run:

```bash
torchrun --nproc_per_node=2 examples/ddp_train.py
```

Then from another shell:

```bash
runtimectl -q /tmp/runtimectl-ddp optimizer.lr.multiplier 0.5
runtimectl -q /tmp/runtimectl-ddp model.dropout.p 0.2
runtimectl -q /tmp/runtimectl-ddp status
```

## Queue format

`runtimectl` writes one JSON command per line into `commands.jsonl`.

Example command:

```json
{"id":"...","ts":1700000000.0,"op":"set","path":"optimizer.lr.multiplier","value":0.5}
```

Acks are appended to `acks.jsonl`.
