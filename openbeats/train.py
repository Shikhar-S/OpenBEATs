"""Common training runner for OpenBEATs (encoder pretraining today; the acoustic
tokenizer later).

Two halves: this module owns everything **objective-agnostic** -- DeepSpeed/Plain
engine setup, the epoch loop (equal-count batches, ``loss/weight*world_size``
rescale, ``iterator_stop``, bf16 ``torch_autocast``), checkpoint/resume, logging --
and drives a pluggable **engine** (``openbeats.pretraining.engine`` /, later,
``openbeats.tokenization.engine``) that supplies ``build_model`` /
``build_dataloaders`` and the model's ``forward(**batch) -> (loss, stats, weight)``
contract.

Console script ``openbeats-train-encoder`` (``train_encoder_main``). Launch with
torchrun (single- or multi-node):

    torchrun --standalone --nnodes=1 --nproc_per_node=8 -m openbeats.train \\
        --config conf/pretrain_large.yaml \\
        --deepspeed_config conf/ds_openbeats_large.json \\
        --train_data data/tokens_train --valid_data data/tokens_valid \\
        --output_dir exp/openbeats_large

Auto-resumes from ``--output_dir`` (DeepSpeed ``latest``). DeepSpeed owns the
optimizer, LR schedule, bf16, grad clipping, accumulation, checkpointing, and
logging -- all from the JSON config; this module is just the loop around it. A
``PlainEngine`` fallback presents the same interface and mirrors DeepSpeed's
on-disk checkpoint layout (``<dir>/global_step{N}/mp_rank_00_model_states.pt`` with
a ``module`` key + a ``latest`` file) for CPU smoke tests / DeepSpeed-less machines.
Heavy imports are deferred so ``--help`` stays fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch

logger = logging.getLogger("openbeats.train")

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
            engine.monitor.write_events([
                ("train/loss", float(stats["loss"]), step),
                ("train/acc_mask", float(stats.get("acc_mask", 0)), step),
                ("train/lr", lr, step),
            ])
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


# ----------------------------------------------------------------- the dispatch
def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _train(engine_module, argv=None):
    """Build model + data via ``engine_module`` and run the common loop."""
    p = argparse.ArgumentParser(prog="openbeats-train")
    p.add_argument("--config", required=True, help="run config YAML")
    p.add_argument("--deepspeed_config", default=None, help="DeepSpeed JSON config")
    p.add_argument("--train_data", required=True, help="train token dataset dir")
    p.add_argument("--valid_data", default=None, help="valid token dataset dir")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--no-deepspeed",
        action="store_true",
        help="use the plain torch fallback engine (CPU / no DeepSpeed)",
    )
    p.add_argument("--device", default=None, help="override device (default: auto)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    use_deepspeed = not args.no_deepspeed
    if use_deepspeed:
        try:
            import deepspeed  # noqa: F401
        except ImportError:
            logger.warning("deepspeed not installed; falling back to plain engine.")
            use_deepspeed = False

    if args.device:
        device = args.device
    elif use_deepspeed or os.environ.get("LOCAL_RANK") is not None:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- distributed init -----
    if use_deepspeed and world_size > 1:
        import deepspeed

        deepspeed.init_distributed(dist_backend="nccl")

    # persist the run config next to checkpoints so openbeats-convert finds it
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    # ----- model + data (objective-specific, from the engine module) -----
    # data paths reach build_dataloaders via the config (runtime injection)
    data_conf = config.setdefault("data_conf", {})
    data_conf["train_data"] = args.train_data
    if args.valid_data:
        data_conf["valid_data"] = args.valid_data

    model = engine_module.build_model(config)
    train_loader, train_sampler, valid_loader = engine_module.build_dataloaders(
        config, rank, world_size
    )

    # ----- engine -----
    if use_deepspeed:
        import deepspeed

        with open(args.deepspeed_config) as f:
            ds_config = json.load(f)
        # per-run TensorBoard: descriptive logdir/name single-sourced from output_dir
        # (a logdir/name in the JSON wins; this just fills the default).
        run_name = os.path.basename(os.path.normpath(args.output_dir))
        tb = ds_config.setdefault("tensorboard", {})
        tb.setdefault("enabled", True)
        tb.setdefault("output_path", os.path.join(args.output_dir, "tb"))
        tb.setdefault("job_name", run_name)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
    else:
        train_conf = config.get("train_conf", {})
        engine = PlainEngine(
            model,
            lr=train_conf.get("lr", 1e-4),
            weight_decay=train_conf.get("weight_decay", 0.01),
            grad_clip=train_conf.get("grad_clip", 1.0),
            device=device,
        )

    train_conf = config.get("train_conf", {})
    run(
        engine,
        train_loader,
        train_sampler,
        valid_loader=valid_loader,
        max_steps=train_conf.get("max_steps", 400_000),
        max_epochs=train_conf.get("max_epochs", 1000),
        save_dir=args.output_dir,
        save_interval=train_conf.get("save_interval", 10_000),
        valid_interval=train_conf.get("valid_interval", 10_000),
        log_interval=train_conf.get("log_interval", 50),
        device=device,
        resume=True,
    )


def train_encoder_main(argv=None):
    """``openbeats-train-encoder``: pretrain the BEATs encoder (masked acoustic modeling)."""
    from .pretraining import engine

    _train(engine, argv)


if __name__ == "__main__":
    train_encoder_main()
