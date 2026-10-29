from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from mrixfields_unsb.constants import (
    CLASSES,
    CLASS_WEIGHTS_A,
    CLASS_WEIGHTS_BRIDGE,
    FIELDS,
    MODALITIES,
    PROSPECTIVE_TRAIN_SUBJECTS,
)

from .index import SliceDescriptor, SliceIndex


def _weighted_choice(rng: random.Random, values: Sequence[str], weights: Sequence[float]) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def _available_class(rng, index, cohort, modality, field, weights, subject=None):
    available, available_weights = [], []
    for cls_name, weight in zip(CLASSES, weights):
        exists = (
            bool(index.group(cohort, modality, field, subject, cls_name))
            if subject is not None
            else bool(index.volumes(cohort, modality, field, cls_name))
        )
        if exists:
            available.append(cls_name); available_weights.append(weight)
    if not available:
        raise RuntimeError(f"domain has no available slices: {cohort}/{modality}/{field}/{subject or '*'}")
    return _weighted_choice(rng, available, available_weights)


class StatefulSampler:
    def __init__(self, index: SliceIndex, seed: int):
        self.index = index
        self.rng = random.Random(seed)
        self.draws = 0

    def state_dict(self) -> dict:
        return {"rng_state": self.rng.getstate(), "draws": self.draws}

    def load_state_dict(self, state: dict) -> None:
        self.rng.setstate(state["rng_state"])
        self.draws = int(state["draws"])

    def _choose(self, values: Sequence):
        if not values:
            raise RuntimeError("sampling candidate set is empty")
        return values[self.rng.randrange(len(values))]


class ScoreSampler(StatefulSampler):
    def sample(self, batch_size: int) -> list[SliceDescriptor]:
        result = []
        for _ in range(batch_size):
            modality = self._choose(MODALITIES)
            field = self._choose(FIELDS)
            cls_name = _available_class(self.rng, self.index, "R", modality, field, CLASS_WEIGHTS_A)
            volume = self._choose(self.index.volumes("R", modality, field, cls_name))
            result.append(self._choose(volume))
            self.draws += 1
        return result


@dataclass(frozen=True)
class BridgeDescriptor:
    source_a: SliceDescriptor
    source_b: SliceDescriptor
    target: SliceDescriptor
    target_field: str


class BridgeSampler(StatefulSampler):
    """Unpaired source/target draws with an independent MINE-B source volume."""

    def _different_volume(
        self, source: SliceDescriptor, candidates: Sequence[SliceDescriptor]
    ) -> SliceDescriptor:
        valid = [item for item in candidates if item.volume_key != source.volume_key]
        if not valid:
            raise RuntimeError(
                "MINE-B requires a different source volume, but no candidate exists for "
                f"{source.modality}/{source.field}/z{source.z:04d}"
            )
        return self._choose(valid)

    def sample(self, batch_size: int) -> list[BridgeDescriptor]:
        result = []
        for _ in range(batch_size):
            modality = self._choose(MODALITIES)
            source_field = self._choose(FIELDS)
            target_field = self._choose([field for field in FIELDS if field != source_field])
            cls_name = _available_class(self.rng, self.index, "R", modality, source_field, CLASS_WEIGHTS_BRIDGE)
            source_volume = self._choose(self.index.volumes("R", modality, source_field, cls_name))
            source_a = self._choose(source_volume)
            same_z = self.index.group("R", modality, source_field, str(source_a.z))
            source_b = self._different_volume(source_a, same_z)
            target_candidates = self.index.group("R", modality, target_field, str(source_a.z))
            target = self._choose(target_candidates)
            result.append(BridgeDescriptor(source_a, source_b, target, target_field))
            self.draws += 1
        return result


@dataclass(frozen=True)
class PairedDescriptor:
    source: SliceDescriptor
    target: SliceDescriptor
    target_field: str


class PairedSampler(StatefulSampler):
    def __init__(self, index: SliceIndex, seed: int, subjects: Sequence[str] = PROSPECTIVE_TRAIN_SUBJECTS):
        super().__init__(index, seed)
        self.subjects = tuple(subjects)

    def sample(self, batch_size: int) -> list[PairedDescriptor]:
        result = []
        for _ in range(batch_size):
            modality = self._choose(MODALITIES)
            source_field = self._choose(FIELDS)
            target_field = self._choose([field for field in FIELDS if field != source_field])
            subject = self._choose(self.subjects)
            cls_name = _available_class(self.rng, self.index, "P", modality, source_field, CLASS_WEIGHTS_BRIDGE, subject)
            source = self._choose(
                self.index.group("P", modality, source_field, subject, cls_name)
            )
            target = self.index.exact("P", modality, target_field, subject, source.z)
            if target is None:
                raise RuntimeError(
                    f"missing paired target {modality}/{target_field}/{subject}/z{source.z:04d}"
                )
            result.append(PairedDescriptor(source, target, target_field))
            self.draws += 1
        return result
