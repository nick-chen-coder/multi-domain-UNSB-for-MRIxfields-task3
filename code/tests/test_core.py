from pathlib import Path

import numpy as np

from mrixfields_unsb.config import load_config
from mrixfields_unsb.constants import FIELDS, MODALITIES
from mrixfields_unsb.data.index import SliceDescriptor, SliceIndex
from mrixfields_unsb.data.io import EXPECTED_KEYS, load_npz_slice
from mrixfields_unsb.data.samplers import BridgeSampler, PairedSampler
from mrixfields_unsb.inference.batch import discover_jobs
from mrixfields_unsb.preprocessing.pipeline import discover_volumes, reserved_names
from mrixfields_unsb.training.checkpoint import permanent_checkpoint_path


CODE_ROOT = Path(__file__).resolve().parents[1]


def descriptor(cohort, modality, field, subject, z, category):
    name = f"{cohort}_{modality}_{field}_{subject}_z{z:04d}_{category}.npz"
    return SliceDescriptor(name, cohort, modality, field, subject, z, category)


def test_configs_resolve():
    for name in ("preprocess.yaml", "train_unsb.yaml", "train_paired.yaml", "inference.yaml"):
        cfg = load_config(CODE_ROOT / "configs" / name)
        assert Path(cfg.paths.task_root).is_absolute()
    assert load_config(CODE_ROOT / "configs" / "preprocess.yaml").preprocessing.compression is False


def test_plain_npz_schema_and_values(tmp_path):
    path = tmp_path / "R_T1W_0.1T_0001_z0165_core.npz"
    image = np.arange(368 * 448, dtype=np.uint16).reshape(368, 448)
    np.savez(
        path,
        image_u16=image,
        z_index=np.asarray(165, dtype=np.int16),
        foreground_ratio=np.asarray(0.5, dtype=np.float32),
    )
    loaded = load_npz_slice(path)
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == EXPECTED_KEYS
    assert loaded["image"].shape == (1, 368, 448)
    assert loaded["z"] == 165


def test_samplers_cover_plain_slice_index():
    records = []
    for modality in MODALITIES:
        for field in FIELDS:
            for category, z in (("background", 0), ("edge", 80), ("core", 165)):
                records.extend(
                    descriptor("R", modality, field, subject, z, category)
                    for subject in ("1000", "1001")
                )
                records.extend(
                    descriptor("P", modality, field, subject, z, category)
                    for subject in ("0006", "0007", "0009")
                )
    index = SliceIndex(records)
    index.validate_coverage()
    assert len(BridgeSampler(index, 7).sample(8)) == 8
    assert len(PairedSampler(index, 9).sample(8)) == 8


def test_official_raw_inventory_and_split():
    cfg = load_config(CODE_ROOT / "configs" / "preprocess.yaml")
    records = discover_volumes(Path(cfg.paths.raw_data_root), {"0006", "0007", "0009"})
    assert len(records) == 1984
    assert sum(record.cohort == "P" for record in records) == 45
    reserved = reserved_names(records, cfg.preprocessing.retrospective_reserved_quotas, 20260809)
    assert len(reserved) == 97


def test_inference_inventory_builds_180_jobs(tmp_path):
    cfg = load_config(CODE_ROOT / "configs" / "inference.yaml")
    jobs = discover_jobs(Path(cfg.paths.input_root), tmp_path)
    assert len(jobs) == 180
    assert len({job["output"] for job in jobs}) == 180


def test_checkpoint_retention_has_no_final_step_duplicate(tmp_path):
    assert permanent_checkpoint_path(tmp_path, 2000, 30000, 10000) is None
    assert permanent_checkpoint_path(tmp_path, 10000, 30000, 10000).name == "step_0010000.pt"
    assert permanent_checkpoint_path(tmp_path, 30000, 30000, 10000).name == "final.pt"
    assert permanent_checkpoint_path(tmp_path, 5000, 20000, 5000).name == "step_0005000.pt"
    assert permanent_checkpoint_path(tmp_path, 20000, 20000, 5000).name == "final.pt"
