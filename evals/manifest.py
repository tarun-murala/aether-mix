"""Typed loaders and validators for the eval manifests (sources, services, gates)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evals import MANIFEST_DIR

SLOTS = ("keys", "acoustic", "electric", "bass", "drums", "vocal_song", "choir", "sermon", "room")
PROFILES = ("full", "quick", "smoke")
FETCH_METHODS = ("cambridge_mt", "archive_org", "librivox_page", "openslr_librispeech", "repo_asset", "local", "synth")
# Parts a source can contribute and the 16-ch tap columns (0-based) they land on.
PART_COLUMNS: dict[str, tuple[int, ...]] = {
    "speech": (0,),
    "lead": (1,),
    "bv1": (2,),
    "bv2": (3,),
    "choir": (2, 3),
    "keys": (4, 5),
    "guitar": (6, 7),
    "drums": (8, 9),
    "bass": (10, 11),
    "room": (12, 13),
    "howl": (),
}
# Engine channel_id that owns each tap column group (fader that scales it in the closed-loop emulation).
# Meta-parts expand to several concrete parts of the same source (one coherent multitrack).
META_PARTS: dict[str, tuple[str, ...]] = {
    "band": ("keys", "guitar", "drums", "bass"),
    "vocals": ("lead", "bv1", "bv2"),
    "song": ("lead", "bv1", "bv2", "keys", "guitar", "drums", "bass"),
}
COLUMN_CHANNEL: dict[int, str] = {
    0: "ch1_pastor", 1: "ch2_lead_vox", 2: "ch3_bv1", 3: "ch4_bv2", 4: "ch5_keys", 5: "ch5_keys",
    6: "ch7_guitars", 7: "ch7_guitars", 8: "ch9_drums", 9: "ch9_drums", 10: "ch9_drums", 11: "ch9_drums",
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class License(_Model):
    id: str
    commercial_use: Literal["allowed", "forbidden", "unknown"]
    redistribute: bool = False
    attribution: str | None = None
    terms_url: str | None = None


class Fetch(_Model):
    method: Literal["cambridge_mt", "archive_org", "librivox_page", "openslr_librispeech", "repo_asset", "local", "synth"]
    anchor: str | None = None  # Cambridge-MT artist anchor/name; the index lists no instruments
    archive_id: str | None = None
    asset: str | None = None  # repo-relative path for repo_asset
    tier: Literal["excerpt", "full"] | None = None
    hint: str | None = None  # human instructions for local/manual sources
    target_minutes: float | None = None
    gender: Literal["male", "female"] | None = None


class Source(_Model):
    id: str
    slots: list[str]
    title: str
    artist: str | None = None
    provider: str
    genre: str | None = None
    tracks: int | None = None
    size_mb: int | None = None
    page_url: str | None = None
    license: str  # key in licenses
    fetch: Fetch
    picks: dict[str, list[str]] = Field(default_factory=dict)  # part -> filename globs (override auto-classify)
    notes: str | None = None

    @model_validator(mode="after")
    def _slots_known(self) -> Source:
        bad = [s for s in self.slots if s not in SLOTS]
        if bad:
            raise ValueError(f"{self.id}: unknown slots {bad}")
        return self


class SourcesFile(_Model):
    schema_version: int
    licenses: dict[str, License]
    sources: list[Source]

    def by_id(self) -> dict[str, Source]:
        return {s.id: s for s in self.sources}

    @model_validator(mode="after")
    def _check(self) -> SourcesFile:
        ids = [s.id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source ids")
        for s in self.sources:
            if s.license not in self.licenses:
                raise ValueError(f"{s.id}: unknown license {s.license}")
        return self


class Layer(_Model):
    id: str
    src: str | None = None  # service slot name; None for synthetic howl
    part: str
    at_frac: float = 0.0
    at_s: float | None = None
    dur_frac: float | None = None
    dur_s: float | None = None
    gain_db: float = 0.0
    src_offset_s: float = 0.0
    continue_: bool = Field(default=False, alias="continue")
    fade_out_s: float | None = None
    # howl-only parameters
    col: int | None = None
    freq_hz: float = 2500.0
    start_amp: float = 0.04
    growth_db_per_s: float = 3.0
    expect: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _part_known(self) -> Layer:
        if self.part not in PART_COLUMNS and self.part not in META_PARTS:
            raise ValueError(f"layer {self.id}: unknown part {self.part}")
        return self


class Action(_Model):
    kind: Literal["human_move"]
    channel: str
    at_frac: float
    delta_db: float
    hold_s: float = 15.0


class Segment(_Model):
    id: str
    title: str
    phase: str | None  # ground-truth macro phase or null (unscored)
    dur: dict[str, float]  # profile -> seconds; a segment absent from a profile is skipped
    step_idx: int | None = None
    layers: list[Layer] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    expect: dict[str, Any] = Field(default_factory=dict)


class Service(_Model):
    id: str
    title: str
    seed: int
    slots: dict[str, str]  # slot -> source id


class ServicesFile(_Model):
    schema_version: int
    sample_rate: int = 48000
    bleed_db: float = -22.0
    trim_range_db: float = 6.0
    settle_s: float = 4.5
    segments: list[Segment]
    services: list[Service]

    def segments_for(self, profile: str) -> list[Segment]:
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        return [s for s in self.segments if profile in s.dur]


class Gate(_Model):
    metric: str
    description: str
    min: float | None = None
    max: float | None = None
    equals: float | None = None
    source: str
    mode: Literal["headless", "mainstage", "manual"] = "headless"
    informational: bool = False


class GatesFile(_Model):
    schema_version: int
    gates: dict[str, Gate]


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_sources(path: Path | None = None) -> SourcesFile:
    return SourcesFile.model_validate(_load(path or MANIFEST_DIR / "sources.yaml"))


def load_gates(path: Path | None = None) -> GatesFile:
    return GatesFile.model_validate(_load(path or MANIFEST_DIR / "gates.yaml"))


def load_services(path: Path | None = None, sources: SourcesFile | None = None) -> ServicesFile:
    svc = ServicesFile.model_validate(_load(path or MANIFEST_DIR / "services.yaml"))
    if sources is not None:
        validate_cross_refs(svc, sources)
    return svc


def validate_cross_refs(svc: ServicesFile, sources: SourcesFile) -> None:
    """Every service fills every slot with a source that supports it; slots are Latin-square distinct."""
    by_id = sources.by_id()
    for s in svc.services:
        missing = [slot for slot in SLOTS if slot not in s.slots]
        if missing:
            raise ValueError(f"{s.id}: missing slots {missing}")
        for slot, sid in s.slots.items():
            if sid not in by_id:
                raise ValueError(f"{s.id}: unknown source {sid}")
            if slot not in by_id[sid].slots:
                raise ValueError(f"{s.id}: source {sid} does not declare slot {slot}")
    for slot in SLOTS:
        picked = [s.slots[slot] for s in svc.services]
        if len(set(picked)) != len(picked):
            raise ValueError(f"slot {slot}: sources must differ across services, got {picked}")
    used_slots = {lay.src for seg in svc.segments for lay in seg.layers if lay.src}
    if not used_slots <= set(SLOTS):
        raise ValueError(f"segments reference unknown slots {used_slots - set(SLOTS)}")
