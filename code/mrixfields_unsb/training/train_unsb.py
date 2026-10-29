from __future__ import annotations

import math
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_utils import get_mem_usage, get_train_logger
from mrixfields_unsb.config import load_config
from mrixfields_unsb.data import BridgeSampler
from mrixfields_unsb.losses import (
    PatchNCELoss,
    PatchProjection,
    anatomical_structure_loss,
    mine_objective,
)
from mrixfields_unsb.models import Discriminator, Generator, MineEstimator
from mrixfields_unsb.training.batches import load_bridge_batch
from mrixfields_unsb.training.bridge import transition
from mrixfields_unsb.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    permanent_checkpoint_path,
    resolve_resume,
    restore_rng_state,
    save_checkpoint,
)
from mrixfields_unsb.training.cli import training_parser
from mrixfields_unsb.training.mine import mine_scores
from mrixfields_unsb.training.runtime import (
    autocast_context,
    build_index,
    constant_cosine_lambda,
    frozen,
    initialize_run,
    resolved_payload,
    StopController,
)
from mrixfields_unsb.utils.ema import EMA
from mrixfields_unsb.utils.logging import JsonlLogger, configure_logging
from mrixfields_unsb.utils.seed import seed_everything


def condition(batch: dict) -> dict:
    return {
        "modality": batch["modality"],
        "source_field": batch["source_field"],
        "target_field": batch["target_field"],
        "position": batch["position"],
    }


def rollout_to_step(generator: Generator, source: torch.Tensor, step: int, values: dict):
    state = source
    with torch.no_grad():
        for history_step in range(step):
            endpoint = generator(state, step=history_step, **values)
            state = transition(state, endpoint, history_step).detach()
    return state


def endpoint_at_step(generator: Generator, source: torch.Tensor, step: int, values: dict):
    state = rollout_to_step(generator, source, step, values)
    return state, generator(state, step=step, **values)


def mine_pairs(state_a, endpoint_a, state_b, endpoint_b):
    pair_a = torch.cat((state_a, endpoint_a), dim=1)
    pair_b = torch.cat((state_b, endpoint_b), dim=1)
    return torch.cat((pair_a, pair_a), dim=1), torch.cat((pair_a, pair_b), dim=1)


def available_steps(global_step: int) -> list[int]:
    return list(range(min(global_step // 2000 + 1, 5)))


def adam(module, cfg):
    return torch.optim.Adam(
        module.parameters(),
        lr=float(cfg.lr),
        betas=tuple(cfg.betas),
        eps=float(cfg.eps),
        weight_decay=float(cfg.weight_decay),
    )


def main() -> None:
    args = training_parser("Train Stage B UNSB warmup").parse_args()
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
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    index = build_index(cfg.paths.train_npz_roots)
    if any(item.cohort != "R" for item in index.records):
        raise RuntimeError("UNSB training is restricted to retrospective training slices")
    index.validate_coverage()
    sampler = BridgeSampler(index, int(cfg.training.seed) + 201)
    generator = Generator(
        bottleneck_blocks=int(cfg.model.generator.get("bottleneck_blocks", 2))
    ).to(device)
    discriminator = Discriminator().to(device)
    mine = MineEstimator().to(device)
    projection = PatchProjection().to(device)
    nce_loss = PatchNCELoss(projection, float(cfg.patchnce.temperature), int(cfg.patchnce.max_patches))
    ema = EMA(generator, float(cfg.ema.decay))
    modules = {"G": generator, "D": discriminator, "T": mine, "P": projection}
    optimizers = {name: adam(module, cfg.optimizer[name]) for name, module in modules.items()}
    schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step, start=float(cfg.optimizer[name].lr), minimum=float(cfg.optimizer[name].get("min_lr", cfg.scheduler.min_lr)): constant_cosine_lambda(
                step,
                int(cfg.scheduler.constant_steps),
                int(cfg.scheduler.cosine_steps),
                minimum / start,
            ),
        )
        for name, optimizer in optimizers.items()
    }
    start = 0
    clip_counts = {name: 0 for name in modules}
    unstable_streak = 0
    stop = StopController()
    resume = resolve_resume(args.resume, cfg.paths.checkpoint_dir)
    if resume:
        state = load_checkpoint(resume, "cpu")
        for name, module in modules.items():
            module.load_state_dict(state["models"][name])
            optimizers[name].load_state_dict(state["optimizers"][name])
            schedulers[name].load_state_dict(state["schedulers"][name])
        ema.load_state_dict(state["ema"])
        sampler.load_state_dict(state["sampler"])
        restore_rng_state(state["rng"])
        start = int(state["step"])
        diagnostics = state.get("diagnostics", {})
        clip_counts.update(diagnostics.get("clip_counts", {}))
        unstable_streak = int(diagnostics.get("unstable_streak", 0))
        del state
        logger.info("resumed exact state from %s at step %d", resume, start)
    total = 1 if args.dry_run else int(cfg.training.generator_steps)
    for global_step in range(start, total):
        descriptors = sampler.sample(int(cfg.training.batch_size))
        batch = load_bridge_batch(descriptors, int(cfg.data.workers), device)
        values = condition(batch)
        step = sampler.rng.choice(available_steps(global_step))

        optimizers["D"].zero_grad(set_to_none=True)
        real = batch["target"].detach().requires_grad_(True)
        with torch.no_grad(), autocast_context(device, str(cfg.training.precision)):
            _, fake_d = endpoint_at_step(generator, batch["source_a"], step, values)
        with autocast_context(device, str(cfg.training.precision)):
            real_logit = discriminator(real, step, batch["target_domain"], batch["position"])
            fake_logit = discriminator(fake_d.detach(), step, batch["target_domain"], batch["position"])
            adversarial_d = F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()
        real_gradient = torch.autograd.grad(real_logit.sum(), real, create_graph=True)[0]
        r1 = real_gradient.float().square().flatten(1).sum(1).mean()
        loss_d = adversarial_d + float(cfg.loss.lambda_r1) * 0.5 * r1
        loss_d.backward()
        grad_d = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float(cfg.training.gradient_clip.D))
        optimizers["D"].step()

        optimizers["T"].zero_grad(set_to_none=True)
        with torch.no_grad(), autocast_context(device, str(cfg.training.precision)):
            state_a_t, endpoint_a_t = endpoint_at_step(generator, batch["source_a"], step, values)
            state_b_t, endpoint_b_t = endpoint_at_step(generator, batch["source_b"], step, values)
        positive_pair, negative_pair = mine_pairs(state_a_t, endpoint_a_t, state_b_t, endpoint_b_t)
        with autocast_context(device, str(cfg.training.precision)):
            positive_score, negative_score = mine_scores(
                mine,
                positive_pair,
                negative_pair,
                step,
                batch["target_domain"],
                batch["position"],
                bool(cfg.mine.get("force_fp32", False)),
            )
            loss_t, mine_stats_t = mine_objective(
                positive_score,
                negative_score,
                bool(cfg.mine.get("subtract_log_spatial", False)),
                float(cfg.mine.get("positive_quadratic_weight", 0.0)),
            )
        loss_t.backward()
        grad_t = torch.nn.utils.clip_grad_norm_(mine.parameters(), float(cfg.training.gradient_clip.T))
        optimizers["T"].step()

        optimizers["G"].zero_grad(set_to_none=True)
        optimizers["P"].zero_grad(set_to_none=True)
        with autocast_context(device, str(cfg.training.precision)):
            state_a, endpoint_a = endpoint_at_step(generator, batch["source_a"], step, values)
            state_b, endpoint_b = endpoint_at_step(generator, batch["source_b"], step, values)
            with frozen(discriminator):
                fake_logit_g = discriminator(endpoint_a, step, batch["target_domain"], batch["position"])
            loss_adv = F.softplus(-fake_logit_g).mean()
            positive_pair, negative_pair = mine_pairs(state_a, endpoint_a, state_b, endpoint_b)
            with frozen(mine):
                positive_score, negative_score = mine_scores(
                    mine,
                    positive_pair,
                    negative_pair,
                    step,
                    batch["target_domain"],
                    batch["position"],
                    bool(cfg.mine.get("force_fp32", False)),
                )
                _, mine_stats_g = mine_objective(
                    positive_score,
                    negative_score,
                    bool(cfg.mine.get("subtract_log_spatial", False)),
                )
                entropy = mine_stats_g["entropy"]
            tau = float(cfg.bridge.tau)
            loss_sb = tau * F.mse_loss(state_a, endpoint_a) - tau * (5 - step) / 5.0 * entropy
            with torch.no_grad():
                _, source_features = generator.encode(batch["source_a"])
            _, generated_features = generator.encode(endpoint_a)
            loss_nce = nce_loss(source_features, generated_features)
            loss_asp, asp_parts = anatomical_structure_loss(
                batch["source_a"],
                endpoint_a,
                float(cfg.asp.mask_threshold),
                float(cfg.asp.mask_scale),
                float(cfg.asp.foreground_threshold),
                float(cfg.asp.background_threshold),
                float(cfg.asp.tolerance_pixels),
                float(cfg.asp.gate_gamma),
                float(cfg.asp.presence_epsilon),
                output_scale=float(cfg.asp.output_scale),
                mode="legacy",
            )
            loss_g = (
                float(cfg.loss.lambda_adv) * loss_adv
                + float(cfg.loss.lambda_sb) * loss_sb
                + float(cfg.loss.lambda_nce) * loss_nce
                + float(cfg.loss.lambda_asp) * loss_asp
            )
        loss_g.backward()
        grad_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), float(cfg.training.gradient_clip.G))
        grad_p = torch.nn.utils.clip_grad_norm_(projection.parameters(), float(cfg.training.gradient_clip.P))
        optimizers["G"].step()
        optimizers["P"].step()
        ema.update(generator)
        for scheduler in schedulers.values():
            scheduler.step()

        clipped = {
            "D": bool(float(grad_d) > float(cfg.training.gradient_clip.D)),
            "T": bool(float(grad_t) > float(cfg.training.gradient_clip.T)),
            "G": bool(float(grad_g) > float(cfg.training.gradient_clip.G)),
            "P": bool(float(grad_p) > float(cfg.training.gradient_clip.P)),
        }
        for name, value in clipped.items():
            clip_counts[name] += int(value)
        completed_steps = global_step + 1
        values_log = {
            "step": completed_steps,
            "bridge_step": step,
            "loss_d": float(loss_d.detach()),
            "r1": float(r1.detach()),
            "loss_t": float(loss_t.detach()),
            "loss_g": float(loss_g.detach()),
            "loss_adv": float(loss_adv.detach()),
            "loss_sb": float(loss_sb.detach()),
            "entropy": float(entropy.detach()),
            "mine_positive_mean": float(mine_stats_t["positive_mean"].detach()),
            "mine_negative_logsumexp": float(mine_stats_t["negative_logsumexp"].detach()),
            "mine_normalization_log": float(mine_stats_t["normalization_log"].detach()),
            "mine_temp": float(mine_stats_t["temp"].detach()),
            "mine_positive_mean_squared": float(mine_stats_t["positive_mean_squared"].detach()),
            "mine_temp_squared": float(mine_stats_t["temp_squared"].detach()),
            "mine_entropy_t": float(mine_stats_t["entropy"].detach()),
            "loss_nce": float(loss_nce.detach()),
            "loss_asp": float(loss_asp.detach()),
            "loss_trimap": float(asp_parts["trimap"].detach()),
            "loss_boundary": float(asp_parts["boundary"].detach()),
            "loss_asp_inner": float(asp_parts["inner"].detach()),
            "loss_asp_outer": float(asp_parts["outer"].detach()),
            "grad_norm_D": float(grad_d),
            "grad_norm_T": float(grad_t),
            "grad_norm_G": float(grad_g),
            "grad_norm_P": float(grad_p),
            **{f"clipped_{name}": value for name, value in clipped.items()},
            **{f"clip_fraction_{name}": clip_counts[name] / completed_steps for name in modules},
            **{f"lr_{name}": optimizers[name].param_groups[0]["lr"] for name in modules},
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        }
        metrics.write(values_log)
        logical_epoch = (completed_steps - 1) // int(cfg.training.logical_epoch_steps) + 1
        audit_logger.info(
            "Epoch:%d, Iteration:%d, LR:%s, Scheduler:%s, BatchSize:%d, "
            "Loss:%s, LossWeight:%s, MemUsage:%s",
            logical_epoch,
            completed_steps,
            {name: values_log[f"lr_{name}"] for name in modules},
            str(cfg.scheduler.type),
            int(cfg.training.batch_size),
            {
                "total_G": values_log["loss_g"], "D": values_log["loss_d"],
                "T": values_log["loss_t"], "adv": values_log["loss_adv"],
                "SB": values_log["loss_sb"], "NCE": values_log["loss_nce"],
                "ASP": values_log["loss_asp"], "R1": values_log["r1"],
            },
            {
                "adv": float(cfg.loss.lambda_adv), "SB": float(cfg.loss.lambda_sb),
                "NCE": float(cfg.loss.lambda_nce), "ASP": float(cfg.loss.lambda_asp),
                "R1": float(cfg.loss.lambda_r1),
            },
            get_mem_usage(),
        )
        if global_step == start or completed_steps % 100 == 0:
            logger.info(
                "step=%d k=%d D=%.4f T=%.4f G=%.4f adv=%.4f SB=%.4f H=%.4f gradT=%.2f",
                completed_steps,
                step,
                values_log["loss_d"],
                values_log["loss_t"],
                values_log["loss_g"],
                values_log["loss_adv"],
                values_log["loss_sb"],
                values_log["entropy"],
                values_log["grad_norm_T"],
            )
        stability = cfg.get("stability", {})
        finite_keys = (
            "loss_d", "loss_t", "loss_g", "entropy", "mine_entropy_t",
            "mine_positive_mean", "mine_negative_logsumexp", "mine_temp",
            "mine_positive_mean_squared", "mine_temp_squared",
            "loss_asp_inner", "loss_asp_outer",
            "grad_norm_D", "grad_norm_T", "grad_norm_G", "grad_norm_P",
        )
        finite = all(math.isfinite(values_log[key]) for key in finite_keys)
        exceeds_limit = (
            abs(values_log["entropy"]) > float(stability.get("max_abs_entropy", math.inf))
            or abs(values_log["mine_entropy_t"]) > float(stability.get("max_abs_entropy", math.inf))
            or abs(values_log["loss_t"]) > float(stability.get("max_abs_loss_t", math.inf))
            or abs(values_log["grad_norm_T"]) > float(stability.get("max_grad_norm_t", math.inf))
        )
        unstable_streak = unstable_streak + 1 if (not finite or exceeds_limit) else 0
        threshold_failure = bool(stability.get("thresholds_enabled", False)) and (
            unstable_streak >= int(stability.get("consecutive_steps", 5))
        )
        stability_failure = not finite or threshold_failure
        if stability_failure and not args.dry_run:
            failure_payload = {
                "format_version": 1,
                "stage": "B-unstable",
                "step": completed_steps,
                "models": {name: module.state_dict() for name, module in modules.items()},
                "ema": ema.state_dict(),
                "optimizers": {name: value.state_dict() for name, value in optimizers.items()},
                "schedulers": {name: value.state_dict() for name, value in schedulers.items()},
                "sampler": sampler.state_dict(),
                "rng": capture_rng_state(),
                "resolved_config": resolved_payload(cfg),
                "diagnostics": {
                    "clip_counts": clip_counts,
                    "unstable_streak": unstable_streak,
                    "last": values_log,
                },
            }
            failure_path = Path(cfg.paths.checkpoint_dir) / f"unstable_step_{completed_steps:07d}.pt"
            save_checkpoint(failure_path, failure_payload)
            audit_logger.info(
                "[Checkpoint] Saved to:%s | Epoch:%d, Iteration:%d, IsBest:False, "
                "ValLoss:N/A, ValMetrics:{}, ValidationStatus:%s",
                failure_path, logical_epoch, completed_steps, str(cfg.audit.validation_status),
            )
            logger.error(
                "numerical safety stop at step %d; diagnostic checkpoint: %s",
                completed_steps,
                failure_path,
            )
            raise RuntimeError(f"non-finite or unsafe numerical state at step {completed_steps}")
        should_save = (global_step + 1) % int(cfg.checkpoint.interval) == 0 or global_step + 1 == total or stop.requested
        if should_save and not args.dry_run:
            payload = {
                "format_version": 1,
                "stage": "B",
                "step": global_step + 1,
                "models": {name: module.state_dict() for name, module in modules.items()},
                "ema": ema.state_dict(),
                "optimizers": {name: value.state_dict() for name, value in optimizers.items()},
                "schedulers": {name: value.state_dict() for name, value in schedulers.items()},
                "sampler": sampler.state_dict(),
                "rng": capture_rng_state(),
                "resolved_config": resolved_payload(cfg),
                "diagnostics": {"clip_counts": clip_counts, "unstable_streak": unstable_streak},
            }
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
            logger.info("received signal %s; checkpointed at step %d", stop.signal, global_step + 1)
            break
    if args.dry_run:
        logger.info("dry-run passed")


if __name__ == "__main__":
    main()
