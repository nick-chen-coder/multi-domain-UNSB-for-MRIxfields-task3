from __future__ import annotations

import math
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_utils import get_mem_usage, get_train_logger
from mrixfields_unsb.config import load_config
from mrixfields_unsb.data import PairedSampler
from mrixfields_unsb.models import generator_from_checkpoint_state
from mrixfields_unsb.training.batches import load_paired_batch
from mrixfields_unsb.training.bridge import deterministic_rollout
from mrixfields_unsb.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    permanent_checkpoint_path,
    resolve_resume,
    restore_rng_state,
    save_checkpoint,
)
from mrixfields_unsb.training.cli import training_parser
from mrixfields_unsb.training.runtime import (
    StopController,
    autocast_context,
    build_index,
    initialize_run,
    resolved_payload,
)
from mrixfields_unsb.utils.ema import EMA
from mrixfields_unsb.utils.logging import JsonlLogger, configure_logging
from mrixfields_unsb.utils.seed import seed_everything


def paired_lr_lambda(
    step: int,
    warmup_steps: int,
    constant_until_step: int,
    total_steps: int,
    start_ratio: float,
    min_ratio: float,
) -> float:
    if step < warmup_steps:
        progress = (step + 1) / max(warmup_steps, 1)
        return start_ratio + (1.0 - start_ratio) * progress
    if step < constant_until_step:
        return 1.0
    progress = min(max((step - constant_until_step) / max(total_steps - constant_until_step, 1), 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = training_parser("Train fixed-length paired calibration without validation").parse_args()
    cfg = load_config(args.config, args.override)
    initialize_run(cfg)
    logger = configure_logging(Path(cfg.paths.log_dir) / "full.log")
    metrics = JsonlLogger(cfg.audit.metrics_jsonl)
    audit_logger = get_train_logger()
    seed_everything(
        int(cfg.training.seed),
        deterministic=bool(cfg.reproducibility.get("deterministic_preflight", True)),
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    index = build_index(cfg.paths.train_npz_roots)
    expected_subjects = set(str(value) for value in cfg.data.train_subjects)
    found_subjects = {item.subject for item in index.records}
    if any(item.cohort != "P" for item in index.records):
        raise RuntimeError("paired training is restricted to prospective training slices")
    if expected_subjects != {"0006", "0007", "0009"} or found_subjects != expected_subjects:
        raise RuntimeError(
            f"paired training must use exactly subjects 0006/0007/0009; "
            f"configured={sorted(expected_subjects)} found={sorted(found_subjects)}"
        )
    index.validate_coverage()
    sampler = PairedSampler(index, int(cfg.training.seed) + 401, tuple(cfg.data.train_subjects))
    stage_b = load_checkpoint(cfg.paths.stage_b_final, device)
    generator = generator_from_checkpoint_state(stage_b).to(device)
    generator.load_state_dict(stage_b["ema"]["module"])
    ema = EMA(generator, float(cfg.ema.decay))
    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=float(cfg.optimizer.lr),
        betas=tuple(cfg.optimizer.betas),
        eps=float(cfg.optimizer.eps),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    scheduler_type = str(cfg.scheduler.type)
    if scheduler_type != "warmup_constant_cosine":
        raise ValueError(f"unknown paired scheduler: {scheduler_type}")
    peak_lr = float(cfg.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: paired_lr_lambda(
            step,
            int(cfg.scheduler.warmup_steps),
            int(cfg.scheduler.constant_until_step),
            int(cfg.training.optimizer_steps),
            float(cfg.scheduler.start_lr) / peak_lr,
            float(cfg.scheduler.min_lr) / peak_lr,
        ),
    )
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("paired training requires lpips") from error
    perceptual = lpips.LPIPS(net=str(cfg.loss.lpips_backbone)).to(device).eval().requires_grad_(False)
    start = 0
    clipped_steps = 0
    stop = StopController()
    resume = resolve_resume(args.resume, cfg.paths.checkpoint_dir)
    if resume:
        state = load_checkpoint(resume, "cpu")
        generator.load_state_dict(state["models"]["G"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizers"]["G"])
        scheduler.load_state_dict(state["schedulers"]["G"])
        sampler.load_state_dict(state["sampler"])
        restore_rng_state(state["rng"])
        start = int(state["step"])
        clipped_steps = int(state.get("diagnostics", {}).get("clipped_steps", 0))
        del state
        logger.info("resumed exact state from %s at optimizer step %d", resume, start)
    total = 1 if args.dry_run else int(cfg.training.optimizer_steps)
    accumulation = int(cfg.training.accumulation_steps)

    def checkpoint_payload(step: int) -> dict:
        return {
            "format_version": 1,
            "stage": str(cfg.project.stage),
            "step": step,
            "models": {"G": generator.state_dict()},
            "ema": ema.state_dict(),
            "optimizers": {"G": optimizer.state_dict()},
            "schedulers": {"G": scheduler.state_dict()},
            "sampler": sampler.state_dict(),
            "rng": capture_rng_state(),
            "resolved_config": resolved_payload(cfg),
            "diagnostics": {"clipped_steps": clipped_steps},
        }

    for global_step in range(start, total):
        optimizer.zero_grad(set_to_none=True)
        sums = {"l1": 0.0, "lpips": 0.0, "total": 0.0}
        for _ in range(accumulation):
            descriptors = sampler.sample(int(cfg.training.micro_batch))
            batch = load_paired_batch(descriptors, int(cfg.data.workers), device)
            values = {key: batch[key] for key in ("modality", "source_field", "target_field", "position")}
            with autocast_context(device, str(cfg.training.precision)):
                generated = deterministic_rollout(generator, batch["source"], values)
                loss_l1 = F.l1_loss(generated, batch["target"])
                loss_lpips = perceptual(
                    generated.repeat(1, 3, 1, 1), batch["target"].repeat(1, 3, 1, 1)
                ).mean()
                loss = float(cfg.loss.lambda_l1) * loss_l1 + float(cfg.loss.lambda_lpips) * loss_lpips
            (loss / accumulation).backward()
            sums["l1"] += float(loss_l1.detach()) / accumulation
            sums["lpips"] += float(loss_lpips.detach()) / accumulation
            sums["total"] += float(loss.detach()) / accumulation
        if start == 0 and global_step == 0:
            audit_logger.info(
                "[PretrainedBaseline] Stage:B-EMA, Dataset:TRAINING_BATCH_NOT_VALIDATION, "
                "Iteration:0, BatchSize:%d, Loss:%s, LossWeight:%s",
                int(cfg.training.effective_batch),
                {"total": sums["total"], "L1": sums["l1"], "LPIPS": sums["lpips"]},
                {"L1": float(cfg.loss.lambda_l1), "LPIPS": float(cfg.loss.lambda_lpips)},
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), float(cfg.training.gradient_clip))
        optimizer.step()
        scheduler.step()
        ema.update(generator)
        completed_steps = global_step + 1
        clipped = bool(float(grad_norm) > float(cfg.training.gradient_clip))
        clipped_steps += int(clipped)
        record = {
            "step": completed_steps,
            "loss": sums["total"],
            "l1": sums["l1"],
            "lpips": sums["lpips"],
            "grad_norm": float(grad_norm),
            "clipped": clipped,
            "clip_fraction": clipped_steps / completed_steps,
            "lr": scheduler.get_last_lr()[0],
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        }
        metrics.write(record)
        logical_epoch = (completed_steps - 1) // int(cfg.training.logical_epoch_steps) + 1
        audit_logger.info(
            "Epoch:%d, Iteration:%d, LR:%s, Scheduler:%s, BatchSize:%d, "
            "Loss:%s, LossWeight:%s, MemUsage:%s",
            logical_epoch,
            completed_steps,
            record["lr"],
            scheduler_type,
            int(cfg.training.effective_batch),
            {"total": record["loss"], "L1": record["l1"], "LPIPS": record["lpips"]},
            {"L1": float(cfg.loss.lambda_l1), "LPIPS": float(cfg.loss.lambda_lpips)},
            get_mem_usage(),
        )
        if global_step == start or completed_steps % 100 == 0:
            logger.info(
                "step=%d loss=%.5f l1=%.5f lpips=%.5f",
                completed_steps, sums["total"], sums["l1"], sums["lpips"],
            )
        should_save = completed_steps % int(cfg.checkpoint.interval) == 0 or completed_steps == total or stop.requested
        if should_save and not args.dry_run:
            payload = checkpoint_payload(completed_steps)
            latest = Path(cfg.paths.checkpoint_dir) / "latest.pt"
            save_checkpoint(latest, payload)
            audit_logger.info(
                "[Checkpoint] Saved to:%s | Epoch:%d, Iteration:%d, IsBest:False, "
                "ValLoss:N/A, ValMetrics:{}, ValidationStatus:%s",
                latest, logical_epoch, completed_steps, str(cfg.audit.validation_status),
            )
            permanent = permanent_checkpoint_path(
                cfg.paths.checkpoint_dir,
                completed_steps,
                total,
                int(cfg.checkpoint.milestone_interval),
            )
            if permanent is not None:
                shutil.copy2(latest, permanent)
                audit_logger.info(
                    "[Checkpoint] Saved to:%s | Epoch:%d, Iteration:%d, IsBest:False, "
                    "ValLoss:N/A, ValMetrics:{}, ValidationStatus:%s",
                    permanent, logical_epoch, completed_steps, str(cfg.audit.validation_status),
                )
        if completed_steps % int(cfg.training.logical_epoch_steps) == 0:
            audit_logger.info(
                "[Validation] Epoch:%d, Iteration:%d, Status:%s, ValTotalLoss:N/A, "
                "ValLosses:{}, ValMetrics:{}",
                logical_epoch, completed_steps, str(cfg.audit.validation_status),
            )
        if stop.requested:
            logger.info("received signal %s; checkpointed at optimizer step %d", stop.signal, completed_steps)
            break
    if args.dry_run:
        logger.info("dry-run passed")


if __name__ == "__main__":
    main()
