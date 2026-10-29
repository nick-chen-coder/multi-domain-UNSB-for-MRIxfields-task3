from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from mrixfields_unsb.constants import CLASSES, FIELDS, MODALITIES

FILENAME_RE = re.compile(
    r"^(?P<cohort>[PR])_(?P<modality>T1W|T2W|T2FLAIR)_"
    r"(?P<field>0\.1T|1\.5T|3T|5T|7T)_(?P<subject>\d{4})_"
    r"z(?P<z>\d{4})_(?P<slice_class>core|edge|background)\.npz$"
)


@dataclass(frozen=True, slots=True)
class SliceDescriptor:
    path: str
    cohort: str
    modality: str
    field: str
    subject: str
    z: int
    slice_class: str

    @property
    def volume_key(self) -> tuple[str, str, str, str]:
        return self.cohort, self.modality, self.field, self.subject

    def to_dict(self) -> dict:
        return asdict(self)


class SliceIndex:
    """Serializable metadata index; NPZ payloads are never loaded while indexing."""

    def __init__(self, records: list[SliceDescriptor]):
        if not records:
            raise ValueError("slice index is empty")
        self.records = records
        self._lookup: dict[tuple[str, str, str, str, int], SliceDescriptor] = {}
        self._groups: dict[tuple[str, ...], list[SliceDescriptor]] = {}
        self._volume_groups: dict[
            tuple[str, str, str, str], dict[tuple[str, str, str, str], list[SliceDescriptor]]
        ] = {}
        for item in records:
            self._lookup[(item.cohort, item.modality, item.field, item.subject, item.z)] = item
            keys = (
                (item.modality, item.field, item.slice_class),
                (item.cohort, item.modality, item.field, item.slice_class),
                (item.cohort, item.modality, item.field, item.subject, item.slice_class),
                (item.cohort, item.modality, item.field, str(item.z)),
            )
            for key in keys:
                self._groups.setdefault(key, []).append(item)
            volume_group_key = (item.cohort, item.modality, item.field, item.slice_class)
            self._volume_groups.setdefault(volume_group_key, {}).setdefault(item.volume_key, []).append(item)

    @classmethod
    def scan(cls, roots: list[str | Path]) -> "SliceIndex":
        records: list[SliceDescriptor] = []
        for root in map(Path, roots):
            if not root.is_dir():
                raise FileNotFoundError(root)
            for path in root.rglob("*.npz"):
                match = FILENAME_RE.match(path.name)
                if match is None:
                    continue
                values = match.groupdict()
                records.append(
                    SliceDescriptor(
                        path=str(path.resolve()),
                        cohort=values["cohort"],
                        modality=values["modality"],
                        field=values["field"],
                        subject=values["subject"],
                        z=int(values["z"]),
                        slice_class=values["slice_class"],
                    )
                )
        records.sort(key=lambda item: item.path)
        return cls(records)

    @classmethod
    def from_json(cls, path: str | Path) -> "SliceIndex":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls([SliceDescriptor(**item) for item in payload["records"]])

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"version": 1, "records": [x.to_dict() for x in self.records]}, handle)

    def group(self, *key: str) -> list[SliceDescriptor]:
        return self._groups.get(tuple(key), [])

    def exact(
        self, cohort: str, modality: str, field: str, subject: str, z: int
    ) -> SliceDescriptor | None:
        return self._lookup.get((cohort, modality, field, subject, z))

    def volumes(self, cohort: str, modality: str, field: str, slice_class: str):
        return tuple(self._volume_groups.get((cohort, modality, field, slice_class), {}).values())

    def validate_coverage(self) -> None:
        missing = []
        for modality in MODALITIES:
            for field in FIELDS:
                for cls_name in CLASSES:
                    if not self.group(modality, field, cls_name):
                        missing.append((modality, field, cls_name))
        if missing:
            raise RuntimeError(f"index has missing domain/class groups: {missing}")
