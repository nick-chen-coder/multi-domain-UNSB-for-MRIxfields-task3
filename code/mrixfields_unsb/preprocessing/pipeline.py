from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from audit_utils import audit_file_loading
from mrixfields_unsb.config import config_hash, load_config
from mrixfields_unsb.constants import FIELDS, MODALITIES, PROSPECTIVE_TRAIN_SUBJECTS


NIFTI_RE = re.compile(
    r"^(?P<cohort>[PR])_(?P<modality>T1W|T2W|T2FLAIR)_"
    r"(?P<field>0\.1T|1\.5T|3T|5T|7T)_(?P<subject>\d{4})\.nii\.gz$"
)


@dataclass(frozen=True, slots=True)
class VolumeRecord:
    source_path: Path
    cohort: str
    modality: str
    field: str
    subject: str

    @property
    def volume_id(self) -> str:
        return self.source_path.name.removesuffix(".nii.gz")

    def manifest_record(self, split: str) -> dict[str, Any]:
        value = asdict(self)
        value["source_path"] = str(self.source_path)
        value["split"] = split
        return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, compressed: bool, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        writer = np.savez_compressed if compressed else np.savez
        writer(handle, **arrays)
    os.replace(temporary, path)


def _versions() -> dict[str, str]:
    result = {}
    for package in ("numpy", "nibabel", "omegaconf", "psutil", "torch"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def discover_volumes(raw_root: Path, allowed_prospective: set[str]) -> list[VolumeRecord]:
    records: list[VolumeRecord] = []
    roots = {
        "R": raw_root / "Training_retrospective",
        "P": raw_root / "Training_prospective",
    }
    for cohort, cohort_root in roots.items():
        if not cohort_root.is_dir():
            raise FileNotFoundError(cohort_root)
        for modality in MODALITIES:
            for field in FIELDS:
                directory = cohort_root / modality / field
                if not directory.is_dir():
                    raise FileNotFoundError(directory)
                for path in sorted(directory.glob("*.nii.gz")):
                    match = NIFTI_RE.fullmatch(path.name)
                    if match is None:
                        raise ValueError(f"unexpected training filename: {path}")
                    values = match.groupdict()
                    if values["cohort"] != cohort or values["modality"] != modality or values["field"] != field:
                        raise ValueError(f"file path and filename disagree: {path}")
                    if cohort == "P" and values["subject"] not in allowed_prospective:
                        raise RuntimeError(f"unauthorized prospective subject in training root: {path}")
                    records.append(
                        VolumeRecord(path.resolve(), cohort, modality, field, values["subject"])
                    )
    if not records:
        raise RuntimeError(f"no official training NIfTI files found below {raw_root}")
    return records


def reserved_names(records: list[VolumeRecord], quotas, seed: int) -> list[str]:
    selected: list[str] = []
    rng = random.Random(seed)
    for modality in MODALITIES:
        for field in FIELDS:
            candidates = sorted(
                record.source_path.name
                for record in records
                if record.cohort == "R" and record.modality == modality and record.field == field
            )
            quota = int(quotas[modality][field])
            if len(candidates) < quota:
                raise RuntimeError(f"{modality}/{field}: need {quota} retrospective holdout volumes, found {len(candidates)}")
            selected.extend(sorted(rng.sample(candidates, quota)))
    if len(selected) != 97 or len(set(selected)) != 97:
        raise RuntimeError("the deterministic retrospective holdout must contain 97 unique volumes")
    return sorted(selected)


def split_for(record: VolumeRecord, reserved_names: set[str]) -> str:
    if record.cohort == "P":
        return "train"
    return "val" if record.source_path.name in reserved_names else "train"


def foreground(volume: np.ndarray, minimum: float, scale: float) -> tuple[np.ndarray, float]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("volume has no finite voxels")
    threshold = max(minimum, scale * float(np.percentile(finite, 99.9)))
    return np.isfinite(volume) & (volume > threshold), threshold


def classify(ratio: float, edge_minimum: float, core_minimum: float) -> str:
    if ratio < edge_minimum:
        return "background"
    if ratio < core_minimum:
        return "edge"
    return "core"


def output_path(root: Path, record: VolumeRecord, split: str, z: int, category: str) -> Path:
    return (
        root
        / f"all_slices_{split}"
        / ("prospective" if record.cohort == "P" else "retrospective")
        / record.modality
        / record.field
        / record.volume_id
        / f"{record.volume_id}_z{z:04d}_{category}.npz"
    )


def _parse_slice_indices(text: str | None, depth: int) -> list[int]:
    if text is None:
        return list(range(depth))
    values = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not values or values[0] < 0 or values[-1] >= depth:
        raise ValueError(f"invalid slice indices for depth {depth}: {values}")
    return values


def process_volume(record: VolumeRecord, split: str, output_root: Path, cfg, args) -> dict[str, Any]:
    started = time.monotonic()
    audit_file_loading(record.source_path)
    image = nib.load(str(record.source_path))
    expected_shape = tuple(int(value) for value in cfg.preprocessing.source_shape)
    if tuple(image.shape) != expected_shape:
        raise ValueError(f"expected {expected_shape}, got {image.shape}: {record.source_path}")
    volume = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError(f"non-finite source volume: {record.source_path}")
    source_min, source_max = float(volume.min()), float(volume.max())
    if source_min < -1e-5 or source_max > 1.00001:
        raise ValueError(f"expected [0,1], got [{source_min},{source_max}]: {record.source_path}")
    volume = np.clip(volume, 0.0, 1.0)
    fg_cfg = cfg.preprocessing.foreground
    mask, threshold = foreground(
        volume, float(fg_cfg.minimum_threshold), float(fg_cfg.percentile_scale)
    )
    z_indices = _parse_slice_indices(args.slice_indices, volume.shape[2])
    pad = cfg.preprocessing.padding
    padding = ((int(pad.top), int(pad.bottom)), (int(pad.left), int(pad.right)))
    expected_padded = tuple(int(value) for value in cfg.preprocessing.padded_shape)
    counts: Counter[str] = Counter()
    bytes_written = 0
    for z in z_indices:
        source_slice = np.ascontiguousarray(volume[:, :, z])
        ratio = float(mask[:, :, z].mean())
        category = classify(ratio, float(fg_cfg.edge_minimum), float(fg_cfg.core_minimum))
        padded = np.pad(source_slice, padding, mode="constant", constant_values=float(pad.value))
        if padded.shape != expected_padded:
            raise RuntimeError(f"unexpected padded shape {padded.shape}")
        destination = output_path(output_root, record, split, z, category)
        if destination.exists() and not args.overwrite:
            if not args.resume:
                raise FileExistsError(destination)
        else:
            _atomic_npz(
                destination,
                bool(cfg.preprocessing.compression),
                image_u16=np.rint(padded * 65535.0).astype(np.uint16),
                z_index=np.asarray(z, dtype=np.int16),
                foreground_ratio=np.asarray(ratio, dtype=np.float32),
            )
        bytes_written += destination.stat().st_size
        counts[category] += 1
    return {
        "source_path": str(record.source_path),
        "volume_id": record.volume_id,
        "cohort": record.cohort,
        "modality": record.modality,
        "field": record.field,
        "subject": record.subject,
        "split": split,
        "source_min": source_min,
        "source_max": source_max,
        "foreground_threshold": threshold,
        "slice_count": len(z_indices),
        "slice_class_counts": dict(counts),
        "bytes": bytes_written,
        "elapsed_s": time.monotonic() - started,
    }


def write_split_files(task_root: Path, output_root: Path, records: list[VolumeRecord], reserved: list[str]) -> None:
    reserved_set = set(reserved)
    payload = {
        "format_version": 1,
        "prospective_training_subjects": list(PROSPECTIVE_TRAIN_SUBJECTS),
        "training": [record.manifest_record(split_for(record, reserved_set)) for record in records if split_for(record, reserved_set) == "train"],
        "reserved_retrospective_not_used_for_training_or_validation": [
            record.manifest_record("val") for record in records if split_for(record, reserved_set) == "val"
        ],
        "validation_subjects": [],
        "validation_status": "NOT_PERFORMED_NO_AUTHORIZED_PAIRED_GROUND_TRUTH",
    }
    manifests = task_root / "manifests"
    _atomic_json(manifests / "data_split.json", payload)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "data_split.json", payload)
    csv_path = manifests / "retrospective_reserved_volumes.csv"
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_filename"])
        writer.writerows([[name] for name in reserved])
    os.replace(temporary, csv_path)
    lines = [
        "# Task 3 data split",
        "",
        "- Training retrospective volumes: all records listed in `manifests/data_split.json` with split `train`.",
        "- Training prospective subjects: `0006`, `0007`, `0009` (all available slices and fields).",
        "- Validation subjects: none.",
        "- The 97 deterministic retrospective reserved volumes reproduce the historical cache layout but are not read during training or validation.",
        "- No validation, early stopping, hyperparameter tuning, or checkpoint selection is performed.",
        "- Withdrawn paired validation data, `Validating_prospective`, and `Testing_prospective` are never read by preprocessing or training.",
    ]
    split_md = task_root / "data-split.md"
    temporary_md = split_md.with_name(split_md.name + ".tmp")
    temporary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary_md, split_md)


def write_preprocessing_config(output_root: Path, cfg, reserved: list[str], record_count: int) -> None:
    value = {
        "format_version": 3,
        "config_sha256": config_hash(cfg),
        "source_shape": [364, 436, 364],
        "padded_slice_shape": [368, 448],
        "padding": {"top": 2, "bottom": 2, "left": 6, "right": 6, "value": 0.0},
        "foreground": {
            "threshold": "max(0.01, 0.02 * volume_finite_p99.9)",
            "background": "ratio < 0.02",
            "edge": "0.02 <= ratio < 0.25",
            "core": "ratio >= 0.25",
            "ratio_denominator": "original 364x436 slice",
        },
        "npz": {
            "compression": "numpy.savez_compressed" if cfg.preprocessing.compression else "numpy.savez",
            "filename_pattern": "<P|R>_<modality>_<field>_<subject>_z<4-digit>_<core|edge|background>.npz",
            "arrays": {
                "image_u16": {"shape": [368, 448], "dtype": "uint16"},
                "z_index": {"shape": [], "dtype": "int16"},
                "foreground_ratio": {"shape": [], "dtype": "float32"},
            },
            "excluded_arrays": ["canny1_packed", "canny2_packed", "latent_mode_fp16"],
        },
        "prospective_training_subjects": list(PROSPECTIVE_TRAIN_SUBJECTS),
        "retrospective_reserved_volume_count": len(reserved),
        "source_volume_count": record_count,
        "software_versions": _versions(),
    }
    _atomic_json(output_root / "preprocessing_config.json", value)


def check_full_run_disk_space(output_root: Path, volume_count: int, padded_shape: tuple[int, int]) -> None:
    """Fail before a new uncompressed full run if the cache cannot fit."""
    parent = output_root
    while not parent.exists():
        parent = parent.parent
    image_bytes = int(np.prod(padded_shape)) * np.dtype(np.uint16).itemsize
    # np.savez adds a small ZIP container and three NPY headers per slice.
    estimated = volume_count * 364 * (image_bytes + 1024)
    required = int(estimated * 1.05)
    free = shutil.disk_usage(parent).free
    if free < required:
        raise RuntimeError(
            "insufficient free space for a new uncompressed full slice cache: "
            f"estimated={estimated / 2**30:.1f} GiB, required_with_5pct_margin={required / 2**30:.1f} GiB, "
            f"free={free / 2**30:.1f} GiB, output={output_root}"
        )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Create audited plain MRI slice NPZ cache")
    value.add_argument("--config", required=True)
    value.add_argument("--override", action="append", default=[], help="OmegaConf key=value override")
    value.add_argument("--cohort", choices=("all", "retrospective", "prospective"), default="all")
    value.add_argument("--limit-volumes", type=int)
    value.add_argument("--slice-indices")
    value.add_argument("--workers", type=int)
    mode = value.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    cfg = load_config(args.config, args.override)
    raw_root = Path(cfg.paths.raw_data_root).resolve()
    output_root = Path(cfg.paths.slices_root).resolve()
    task_root = Path(cfg.paths.task_root).resolve()
    allowed = set(str(value) for value in cfg.preprocessing.prospective_subjects)
    expected_allowed = set(PROSPECTIVE_TRAIN_SUBJECTS)
    if allowed != expected_allowed:
        raise RuntimeError(f"prospective allowlist must be exactly {sorted(expected_allowed)}, got {sorted(allowed)}")
    if any(name in raw_root.parts for name in ("Validating_prospective", "Testing_prospective")):
        raise RuntimeError(f"refusing forbidden raw data root: {raw_root}")
    records = discover_volumes(raw_root, allowed)
    reserved = reserved_names(records, cfg.preprocessing.retrospective_reserved_quotas, int(cfg.preprocessing.seed))
    if not args.resume and not args.overwrite and any(output_root.rglob("*.npz")):
        raise RuntimeError(
            f"slice cache already contains NPZ files: {output_root}; use --resume to continue"
        )
    write_split_files(task_root, output_root, records, reserved)
    write_preprocessing_config(output_root, cfg, reserved, len(records))
    selected = records
    if args.cohort != "all":
        cohort = "R" if args.cohort == "retrospective" else "P"
        selected = [record for record in selected if record.cohort == cohort]
    if args.limit_volumes is not None:
        if args.limit_volumes < 1:
            raise ValueError("--limit-volumes must be positive")
        selected = selected[: args.limit_volumes]
    if not selected:
        raise RuntimeError("no volumes selected")
    is_full_new_run = (
        args.cohort == "all"
        and args.limit_volumes is None
        and args.slice_indices is None
        and not args.resume
        and not args.overwrite
        and not any(output_root.rglob("*.npz"))
    )
    if is_full_new_run and not bool(cfg.preprocessing.compression):
        check_full_run_disk_space(
            output_root,
            len(selected),
            tuple(int(value) for value in cfg.preprocessing.padded_shape),
        )
    workers = int(args.workers or cfg.preprocessing.workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    reserved_set = set(reserved)
    print(f"preprocessing volumes={len(selected)} workers={workers} raw={raw_root} output={output_root}", flush=True)
    jobs = [(record, split_for(record, reserved_set), output_root, cfg, args) for record in selected]
    summaries: list[dict[str, Any]] = []
    if workers == 1:
        iterator = (process_volume(*job) for job in jobs)
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        iterator = pool.map(lambda job: process_volume(*job), jobs)
    try:
        for number, summary in enumerate(iterator, start=1):
            summaries.append(summary)
            print(
                f"[{number}/{len(jobs)}] {summary['volume_id']} split={summary['split']} "
                f"slices={summary['slice_count']} classes={summary['slice_class_counts']} "
                f"size_mib={summary['bytes'] / 2**20:.1f} elapsed_s={summary['elapsed_s']:.1f}",
                flush=True,
            )
    finally:
        if workers != 1:
            pool.shutdown(wait=True)
    partial_run = bool(args.limit_volumes is not None or args.slice_indices is not None or args.cohort != "all")
    total_slices = sum(int(item["slice_count"]) for item in summaries)
    manifest = {
        "format_version": 1,
        "config": str(Path(args.config).resolve()),
        "config_sha256": config_hash(cfg),
        "partial_run": partial_run,
        "volume_count": len(summaries),
        "slice_count": total_slices,
        "summaries": summaries,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    _atomic_json(task_root / "manifests" / f"preprocessing_{stamp}.json", manifest)
    if not partial_run:
        expected_slices = len(records) * int(cfg.preprocessing.source_shape[2])
        if len(summaries) != len(records) or total_slices != expected_slices:
            raise RuntimeError(
                f"full preprocessing count mismatch: volumes={len(summaries)}/{len(records)} "
                f"slices={total_slices}/{expected_slices}"
            )
        _atomic_json(task_root / "manifests" / "preprocessing_complete.json", manifest)
    print(f"preprocessing complete volumes={len(summaries)} slices={total_slices}", flush=True)
