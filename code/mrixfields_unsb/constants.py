from __future__ import annotations

from dataclasses import dataclass

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
CLASSES = ("core", "edge", "background")
CLASS_WEIGHTS_A = (0.60, 0.30, 0.10)
CLASS_WEIGHTS_BRIDGE = (0.65, 0.30, 0.05)
PROSPECTIVE_TRAIN_SUBJECTS = ("0006", "0007", "0009")
INFERENCE_SUBJECTS = {
    "0.1T": ("0001", "0002", "0003"),
    "1.5T": ("0004", "0005", "0008"),
    "3T": ("0010", "0011", "0012"),
    "5T": ("0013", "0014", "0015"),
    "7T": ("0016", "0017", "0018"),
}
SLICE_SHAPE = (368, 448)
MODEL_SHAPE = (384, 448)
VOLUME_SHAPE = (364, 436, 364)
PAD_TB = 8
BRIDGE_TIMES = (0.0, 0.50, 0.74, 0.86, 0.94, 1.0)
TAU_SB = 0.01


def domain_id(modality: str, field: str) -> int:
    return 5 * MODALITIES.index(modality) + FIELDS.index(field)


def decode_domain(value: int) -> tuple[str, str]:
    if not 0 <= value < 15:
        raise ValueError(f"domain id must be in [0, 14], got {value}")
    return MODALITIES[value // 5], FIELDS[value % 5]


@dataclass(frozen=True)
class Domain:
    modality: str
    field: str

    @property
    def id(self) -> int:
        return domain_id(self.modality, self.field)
