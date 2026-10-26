from __future__ import annotations

import torch

from mrixfields_unsb.constants import FIELDS, MODALITIES, domain_id
from mrixfields_unsb.data.index import SliceDescriptor
from mrixfields_unsb.data.io import load_npz_batch
from mrixfields_unsb.data.samplers import BridgeDescriptor, PairedDescriptor


def descriptor_condition(items: list[SliceDescriptor], device: torch.device) -> dict:
    return {
        "modality": torch.tensor([MODALITIES.index(x.modality) for x in items], device=device),
        "field": torch.tensor([FIELDS.index(x.field) for x in items], device=device),
        "position": torch.tensor([2.0 * x.z / 363.0 - 1.0 for x in items], device=device),
        "domain": torch.tensor([domain_id(x.modality, x.field) for x in items], device=device),
    }


def load_bridge_batch(
    items: list[BridgeDescriptor], workers: int, device: torch.device,
) -> dict:
    source_a_items = [x.source_a for x in items]
    source_b_items = [x.source_b for x in items]
    target_items = [x.target for x in items]
    source_a_loaded = load_npz_batch(source_a_items, workers, pin_memory=True)
    source_a = source_a_loaded["image"].to(device, non_blocking=True)
    source_b = load_npz_batch(source_b_items, workers, pin_memory=True)["image"].to(device, non_blocking=True)
    target = load_npz_batch(target_items, workers, pin_memory=True)["image"].to(device, non_blocking=True)
    source_condition = descriptor_condition(source_a_items, device)
    target_condition = descriptor_condition(target_items, device)
    return {
        "source_a": source_a,
        "source_b": source_b,
        "target": target,
        "modality": source_condition["modality"],
        "source_field": source_condition["field"],
        "target_field": target_condition["field"],
        "target_domain": target_condition["domain"],
        "position": source_condition["position"],
        "z": torch.tensor([item.z for item in source_a_items], device=device),
        "source_is_core": torch.tensor(
            [item.slice_class == "core" for item in source_a_items], device=device
        ),
        "source_has_anatomy": torch.tensor(
            [item.slice_class in {"core", "edge"} for item in source_a_items], device=device
        ),
    }


def load_paired_batch(
    items: list[PairedDescriptor], workers: int, device: torch.device,
) -> dict:
    source_items = [x.source for x in items]
    target_items = [x.target for x in items]
    source = load_npz_batch(source_items, workers, pin_memory=True)["image"].to(device, non_blocking=True)
    target_loaded = load_npz_batch(target_items, workers, pin_memory=True)
    target = target_loaded["image"].to(device, non_blocking=True)
    source_condition = descriptor_condition(source_items, device)
    target_condition = descriptor_condition(target_items, device)
    return {
        "source": source,
        "target": target,
        "modality": source_condition["modality"],
        "source_field": source_condition["field"],
        "target_field": target_condition["field"],
        "position": source_condition["position"],
    }
