"""Minimal DDP example for runtimectl.

Run (example):
  torchrun --nproc_per_node=2 examples/ddp_train.py

Then in another shell:
  runtimectl -q /tmp/runtimectl-ddp optimizer.lr.multiplier 0.5
  runtimectl -q /tmp/runtimectl-ddp model.dropout.p 0.2
  runtimectl -q /tmp/runtimectl-ddp status
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from runtimectl import RuntimeController


def setup_ddp() -> tuple[int, int, torch.device]:
    dist.init_process_group(backend="gloo")  # use "nccl" on CUDA
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return rank, world_size, device


def main() -> None:
    rank, world_size, device = setup_ddp()

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(32, 4),
    ).to(device)
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    rt = RuntimeController(queue_dir="/tmp/runtimectl-ddp", ddp=True, dist_module=dist)

    def validate_mult(v):
        v = float(v)
        if v <= 0:
            raise ValueError("multiplier must be > 0")
        return v

    def apply_lr_multiplier(mult, ctx):
        for pg in ctx["optimizer"].param_groups:
            pg["lr"] *= mult

    def validate_dropout(v):
        v = float(v)
        if not (0.0 <= v < 1.0):
            raise ValueError("dropout p must be in [0,1)")
        return v

    def apply_dropout(p, ctx):
        for m in ctx["model"].module.modules():  # model is DDP-wrapped
            if isinstance(m, torch.nn.Dropout):
                m.p = p

    rt.register_control("optimizer.lr.multiplier", apply_lr_multiplier, validate_mult)
    rt.register_control("model.dropout.p", apply_dropout, validate_dropout)

    if rank == 0:
        print(f"[rank0] world_size={world_size}")
        print("[rank0] queue: /tmp/runtimectl-ddp")
        print("[rank0] try: runtimectl -q /tmp/runtimectl-ddp optimizer.lr.multiplier 0.5")
        print("[rank0] try: runtimectl -q /tmp/runtimectl-ddp model.dropout.p 0.2")

    for step in range(200):
        x = torch.randn(32, 16, device=device)
        y = torch.randn(32, 4, device=device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        # End-of-step polling: rank0 reads, all ranks apply same commands.
        results = rt.poll_and_apply(ctx={"optimizer": optimizer, "model": model}, every_s=2.0)

        if rank == 0 and results:
            print(f"[rank0][step={step}] applied: {results}")

        time.sleep(0.05)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
