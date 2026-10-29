from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from mrixfields_unsb.config import config_hash, load_config
from mrixfields_unsb.constants import FIELDS, INFERENCE_SUBJECTS, MODALITIES, VOLUME_SHAPE
from mrixfields_unsb.inference.volume import infer_volume, load_generator


INPUT_RE = re.compile(r"P_(T1W|T2W|T2FLAIR)_(0\.1T|1\.5T|3T|5T|7T)_(\d{4})\.nii\.gz")


def _valid_existing(path: Path) -> bool:
    if not path.is_file():
        return False
    image = nib.load(str(path))
    if image.shape != VOLUME_SHAPE or np.dtype(image.get_data_dtype()) != np.dtype(np.float32):
        return False
    data = np.asarray(image.dataobj, dtype=np.float32)
    return bool(np.isfinite(data).all() and data.min() >= -1e-6 and data.max() <= 1.0 + 1e-6)


def discover_jobs(input_root: Path, output_root: Path) -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for modality in MODALITIES:
        for source_field in FIELDS:
            directory = input_root / modality / source_field
            expected = set(INFERENCE_SUBJECTS[source_field])
            found: dict[str, Path] = {}
            for path in sorted(directory.glob("*.nii.gz")):
                match = INPUT_RE.fullmatch(path.name)
                if match and match.group(1) == modality and match.group(2) == source_field:
                    found[match.group(3)] = path
            if set(found) != expected:
                raise RuntimeError(
                    f"input IDs mismatch for {modality}/{source_field}: "
                    f"expected={sorted(expected)} found={sorted(found)}"
                )
            for target_field in FIELDS:
                if target_field == source_field:
                    continue
                pair = f"{source_field}_to_{target_field}"
                for subject in sorted(found):
                    relative = Path("task3") / modality / pair / "pred" / f"P_{modality}_{target_field}_{subject}.nii.gz"
                    jobs.append(
                        {
                            "input": str(found[subject].resolve()),
                            "output": str((output_root / "full" / relative).resolve()),
                            "relative_output": str(relative),
                            "modality": modality,
                            "source_field": source_field,
                            "target_field": target_field,
                            "subject": subject,
                        }
                    )
    if len(jobs) != 180:
        raise RuntimeError(f"expected 180 inference-input conversions, got {len(jobs)}")
    return jobs


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run native full-volume Task 3 inference")
    value.add_argument("--config", required=True)
    value.add_argument("--override", action="append", default=[], help="OmegaConf key=value override")
    value.add_argument("--device", default="cuda")
    mode = value.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    value.add_argument("--limit", type=int, help="test only: infer the first N jobs")
    return value


def main() -> None:
    args = parser().parse_args()
    cfg = load_config(args.config, args.override)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = Path(cfg.paths.checkpoint).resolve()
    input_root = Path(cfg.paths.input_root).resolve()
    output_root = Path(cfg.paths.output_root).resolve()
    jobs = discover_jobs(input_root, output_root)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        jobs = jobs[: args.limit]
    model = load_generator(checkpoint, device)
    records = []
    started = time.monotonic()
    for number, job in enumerate(jobs, start=1):
        output = Path(job["output"])
        if output.exists() and not args.overwrite:
            if not args.resume:
                raise FileExistsError(f"existing output requires --resume or --overwrite: {output}")
            if _valid_existing(output):
                result = {**job, "resumed_existing": True}
            else:
                print(f"regenerating invalid partial output: {output}", flush=True)
                result = infer_volume(
                    model,
                    job["input"],
                    output,
                    job["modality"],
                    job["source_field"],
                    job["target_field"],
                    int(cfg.inference.batch_size),
                    device,
                    str(cfg.inference.precision),
                )
                result.update({"relative_output": job["relative_output"], "subject": job["subject"]})
        else:
            result = infer_volume(
                model,
                job["input"],
                output,
                job["modality"],
                job["source_field"],
                job["target_field"],
                int(cfg.inference.batch_size),
                device,
                str(cfg.inference.precision),
            )
            result.update({"relative_output": job["relative_output"], "subject": job["subject"]})
        records.append(result)
        elapsed = time.monotonic() - started
        eta = elapsed / number * (len(jobs) - number)
        print(
            f"inference progress={number}/{len(jobs)} elapsed_min={elapsed / 60:.1f} "
            f"eta_min={eta / 60:.1f} {job['modality']} {job['source_field']}->{job['target_field']} "
            f"subject={job['subject']}",
            flush=True,
        )
    manifest = {
        "format_version": 1,
        "partial": args.limit is not None,
        "config": str(Path(args.config).resolve()),
        "config_sha256": config_hash(cfg),
        "checkpoint": str(checkpoint),
        "records": records,
    }
    manifest_path = Path(cfg.paths.inference_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"inference complete outputs={len(records)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
