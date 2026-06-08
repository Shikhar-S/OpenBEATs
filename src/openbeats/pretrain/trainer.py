"""A slim DeepSpeed training loop for OpenBEATs encoder pretraining (stage B).

DeepSpeed owns the optimizer, LR schedule, bf16, grad clipping, accumulation,
checkpointing, and (via its Monitor) logging — all from the JSON config. This
module is just the loop around it: move batch to device, call the model's
``(loss, stats, weight)`` contract, apply the proven ``loss/weight*world_size``
rescale (to offset DeepSpeed's gradient averaging), step, periodically validate
and checkpoint. An ``iterator_stop`` all-reduce keeps ranks with uneven shard
counts in lockstep.

For machines without DeepSpeed (and CPU smoke tests) a ``PlainEngine`` fallback
presents the same interface and mirrors DeepSpeed's on-disk checkpoint layout
(``<dir>/global_step{N}/mp_rank_00_model_states.pt`` with a ``module`` key + a
``latest`` file), so the loop, resume, and the export step are format-identical.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("openbeats.pretrain")

MODEL_STATES = "mp_rank_00_model_states.pt"


# ----------------------------------------------------------------- device helper
def to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


# ----------------------------------------------------- plain (no-DeepSpeed) engine
class _NoopMonitor:
    def write_events(self, events):  # events: list[(name, value, step)]
        pass


class PlainEngine:
    """Minimal stand-in for a DeepSpeed engine (single process, no sharding).

    Mirrors the subset of the DeepSpeed engine API the loop uses, and the on-disk
    checkpoint layout, so code paths are identical whether or not DeepSpeed is
    installed. Intended for CPU smoke tests and DeepSpeed-less machines.
    """

    def __init__(self, model, lr=1e-4, weight_decay=0.01, grad_clip=1.0, device="cpu"):
        self.module = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.grad_clip = grad_clip
        self.global_steps = 0
        self.monitor = _NoopMonitor()

    def __call__(self, **batch):
        return self.module(**batch)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def backward(self, loss):
        loss.backward()

    def step(self):
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_steps += 1

    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def save_checkpoint(self, save_dir, tag=None, client_state=None, **kwargs):
        tag = tag or f"global_step{self.global_steps}"
        ckpt_dir = os.path.join(save_dir, tag)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                "module": self.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_steps": self.global_steps,
                "client_state": client_state or {},
            },
            os.path.join(ckpt_dir, MODEL_STATES),
        )
        with open(os.path.join(save_dir, "latest"), "w") as f:
            f.write(tag)
        return ckpt_dir

    def load_checkpoint(self, load_dir, tag=None, **kwargs):
        latest = os.path.join(load_dir, "latest")
        if tag is None:
            if not os.path.isfile(latest):
                return None, None
            tag = open(latest).read().strip()
        path = os.path.join(load_dir, tag, MODEL_STATES)
        obj = torch.load(path, map_location=self.device, weights_only=False)
        self.module.load_state_dict(obj["module"])
        if "optimizer" in obj:
            self.optimizer.load_state_dict(obj["optimizer"])
        self.global_steps = obj.get("global_steps", 0)
        return path, obj.get("client_state", {})


# --------------------------------------------------------------------- the loop
def _maybe_dist():
    """Return (rank, world_size, dist_module_or_None)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size(), dist
    except Exception:  # noqa: BLE001
        pass
    return 0, 1, None


def _step_loss(loss, weight, world_size):
    """The proven rescale: undo DeepSpeed's grad averaging, weight by batch.

    `loss`/`weight` are shape-[1] (force_gatherable); sum collapses to a 0-dim
    scalar with the same value so engine.backward gets a scalar."""
    return (loss / weight * world_size).sum()


def run(
    engine,
    train_loader,
    train_sampler,
    *,
    valid_loader=None,
    max_steps=100000,
    max_epochs=1000,
    save_dir="exp",
    save_interval=10000,
    valid_interval=10000,
    log_interval=50,
    device="cuda",
    resume=True,
):
    """Drive training to ``max_steps``/``max_epochs`` with periodic valid + ckpt."""
    rank, world_size, dist = _maybe_dist()
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    if resume:
        _, client = engine.load_checkpoint(save_dir)
        if client:
            start_epoch = client.get("epoch", 0)
            logger.info("resumed at step %d, epoch %d", engine.global_steps, start_epoch)

    for epoch in range(start_epoch, max_epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        engine.train()
        _train_one_epoch(
            engine, train_loader, epoch, world_size, dist, device,
            max_steps, log_interval, save_dir, save_interval, valid_loader,
            valid_interval, rank,
        )
        # epoch-boundary checkpoint (collective)
        engine.save_checkpoint(save_dir, client_state={"epoch": epoch + 1})
        if engine.global_steps >= max_steps:
            logger.info("reached max_steps=%d; stopping.", max_steps)
            break


def _train_one_epoch(
    engine, loader, epoch, world_size, dist, device, max_steps, log_interval,
    save_dir, save_interval, valid_loader, valid_interval, rank,
):
    iterator_stop = torch.zeros(1, device=device) if dist is not None else None

    it = iter(loader)
    while True:
        if dist is not None:
            from torch.distributed import ReduceOp

            dist.all_reduce(iterator_stop, ReduceOp.SUM)
            if iterator_stop.item() > 0:
                break
        try:
            batch = next(it)
        except StopIteration:
            if dist is not None:
                from torch.distributed import ReduceOp

                iterator_stop.fill_(1)
                dist.all_reduce(iterator_stop, ReduceOp.SUM)
            break

        batch = to_device(batch, device)
        loss, stats, weight = engine(**batch)
        loss = _step_loss(loss, weight, world_size)
        engine.backward(loss)
        engine.step()

        step = engine.global_steps
        if step % log_interval == 0 and rank == 0:
            lr = engine.get_lr()[0] if hasattr(engine, "get_lr") else float("nan")
            logger.info(
                "epoch %d step %d | loss %.4f | acc_mask %.3f | lr %.2e",
                epoch, step, float(stats["loss"]), float(stats.get("acc_mask", 0)), lr,
            )
        if valid_loader is not None and valid_interval and step % valid_interval == 0:
            validate(engine, valid_loader, world_size, dist, device, step, rank)
            engine.train()
        if save_interval and step % save_interval == 0:
            engine.save_checkpoint(save_dir, client_state={"epoch": epoch})
        if step >= max_steps:
            break


@torch.no_grad()
def validate(engine, loader, world_size, dist, device, step, rank):
    engine.eval()
    total_loss = torch.zeros(1, device=device)
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        loss, stats, weight = engine(**batch)
        total_loss += stats["loss"].reshape(1).to(device)
        n += 1
    if dist is not None:
        from torch.distributed import ReduceOp

        cnt = torch.tensor([n], device=device, dtype=total_loss.dtype)
        dist.all_reduce(total_loss, ReduceOp.SUM)
        dist.all_reduce(cnt, ReduceOp.SUM)
        n = int(cnt.item())
    mean_loss = (total_loss / max(n, 1)).item()
    if rank == 0:
        logger.info("[valid] step %d | loss %.4f", step, mean_loss)
        engine.monitor.write_events([("valid/loss", mean_loss, step)])
    return mean_loss
