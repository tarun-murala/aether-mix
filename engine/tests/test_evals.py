"""Tests for the multitrack MainStage eval suite (evals/, scripts/run_eval.py, docs/EVAL_SPEC.md)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from engine.core.protocol import MixingRules
from evals import BLOCK_FRAMES, REPO_ROOT, SAMPLE_RATE
from evals import fetch as ev_fetch
from evals.manifest import PROFILES, SLOTS, load_gates, load_services, load_sources, validate_cross_refs
from evals.report import aggregate, write_reports
from evals.runner import CHANNELS, DT, PHASES, TickLog, run_timeline
from evals.scoring import GateResult, judge, macro_f1, score_run
from evals.stems import PartSource, classify_filename, load_parts, synth_block
from evals.timeline import audio_onset_s, build_timeline

RULES_PATH = REPO_ROOT / "engine" / "schemas" / "mixing_rules.json"
SOURCES = load_sources()
SERVICES = load_services(sources=SOURCES)
GATES = load_gates()


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


def test_manifests_are_consistent_latin_square() -> None:
    assert len(SERVICES.services) == 5
    validate_cross_refs(SERVICES, SOURCES)
    for slot in SLOTS:
        assert len({s.slots[slot] for s in SERVICES.services}) == 5, f"{slot}: 5 distinct sources expected"


def test_every_source_has_license_and_audio_is_never_committed() -> None:
    for s in SOURCES.sources:
        lic = SOURCES.licenses[s.license]
        assert lic.id
        if s.fetch.method == "local":
            assert s.fetch.hint, f"{s.id}: manual sources need instructions"
    cmt = SOURCES.licenses["cambridge_mt"]
    assert cmt.commercial_use == "unknown" and cmt.redistribute is False
    cc0_hints = [s for s in SOURCES.sources if s.license == "cc0"]
    assert cc0_hints and all("CC0" in (s.fetch.hint or "") or "Creative Commons 0" in (s.fetch.hint or "") for s in cc0_hints)
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("recordings/", "*.wav", "*.flac"):
        assert pattern in ignore


def test_profiles_and_gates_cover_the_spec() -> None:
    totals = {p: sum(s.dur[p] for s in SERVICES.segments_for(p)) for p in PROFILES}
    assert 1500 <= totals["full"] <= 1600 and 300 <= totals["quick"] <= 420 and totals["smoke"] < 60
    headless = {n for n, g in GATES.gates.items() if g.mode == "headless"}
    assert {"phase_macro_f1", "speech_open_latency_s", "instrument_open_latency_s", "false_feedback_cuts",
            "human_override_drift_db", "loop_p99_ms"} <= headless
    assert GATES.gates["phase_macro_f1"].min == 0.85
    assert GATES.gates["speech_open_latency_s"].max == 1.0
    assert GATES.gates["instrument_open_latency_s"].max == 0.8
    assert GATES.gates["loop_p99_ms"].max == 25.0


def test_closedloop_plan_matches_plan_except_device() -> None:
    base = json.loads((REPO_ROOT / "engine" / "schemas" / "service_plan.json").read_text())
    loop = json.loads((REPO_ROOT / "engine" / "schemas" / "service_plan.closedloop.json").read_text())
    assert loop.pop("audio_device_hint") == "BlackHole 64ch"
    base.pop("audio_device_hint")
    assert loop == base


# ---------------------------------------------------------------------------
# Stems and timeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,part", [
    ("01_Kick_In.wav", "drums"), ("Snare_Top.wav", "drums"), ("OH_L.wav", "drums"), ("BassDI.wav", "bass"),
    ("Bass Drum.wav", "drums"), ("LeadVox_01.wav", "lead"), ("BV2.wav", "bv"), ("Backing Vocals.wav", "bv"),
    ("Piano_L.wav", "keys"), ("Rhodes.wav", "keys"), ("Ac_Gtr.wav", "guitar"), ("ElecGtr_Dist.wav", "guitar"),
    ("mystery_track.wav", "other"),
])
def test_filename_classification(name: str, part: str) -> None:
    assert classify_filename(name) == part


def test_synthetic_blocks_are_deterministic_and_stateless() -> None:
    a = synth_block("drums", 96000, BLOCK_FRAMES, seed=5)
    b = synth_block("drums", 96000, BLOCK_FRAMES, seed=5)
    assert a.shape == (BLOCK_FRAMES, 2) and np.array_equal(a, b)
    assert not np.array_equal(a, synth_block("drums", 96000, BLOCK_FRAMES, seed=6))


def test_timeline_is_deterministic_and_marks_synthetic_parts(tmp_path: Path) -> None:
    tl1 = build_timeline(SERVICES, SOURCES, "svc1", "smoke", stems_root=tmp_path)
    tl2 = build_timeline(SERVICES, SOURCES, "svc1", "smoke", stems_root=tmp_path)
    assert tl1.total_s == 38.0 and tl1.n_ticks == 1900
    assert np.array_equal(tl1.block(700), tl2.block(700))
    assert tl1.block(0).shape == (BLOCK_FRAMES, 16) and tl1.block(0).dtype == np.float32
    assert tl1.synthetic_parts, "with no stems on disk every part must be flagged synthetic"
    assert tl1.trims_db != build_timeline(SERVICES, SOURCES, "svc2", "smoke", stems_root=tmp_path).trims_db
    assert all(abs(v) <= 6.0 for v in tl1.trims_db.values())
    kinds = {e.kind for e in tl1.events}
    assert {"speech_onset", "instrument_open", "ducking"} <= kinds
    tl1.write_markers_csv(tmp_path / "m.csv")
    assert (tmp_path / "m.csv").read_text().splitlines()[0] == "t0_s,t1_s,kind,id,phase,detail"


def test_layer_boundary_aligns_to_the_exact_sample(tmp_path: Path) -> None:
    """A layer that starts mid-block must begin at its own first sample (no negative source index, no shift)."""
    tl = build_timeline(SERVICES, SOURCES, "svc1", "smoke", stems_root=tmp_path)
    lay = next(lay for lay in tl.layers if lay.id == "applause_1.room")
    k = int(lay.t0 * SAMPLE_RATE // BLOCK_FRAMES)
    blk = tl.block(k)[:, 12]
    assert blk.shape == (BLOCK_FRAMES,)
    # an 8 s layer starting on a block boundary: energy appears from sample 0 of its first block
    assert np.sqrt(np.mean(blk[:480] ** 2)) > 10 ** (-40 / 20)
    mid = Path(tmp_path)
    shifted = lay.__class__(**{**lay.__dict__, "t0": lay.t0 + 0.5 / SAMPLE_RATE * 480})  # start 240 samples into the block
    tl.layers = [shifted]
    first = tl.block(k)[:, 12]
    assert np.sqrt(np.mean(first[:200] ** 2)) < 10 ** (-45 / 20) and np.sqrt(np.mean(first[300:] ** 2)) > 10 ** (-40 / 20)
    assert mid.exists()


def test_real_stems_replace_synthetic_and_are_not_flagged(tmp_path: Path) -> None:
    folder = tmp_path / "jesper_buhl_trio"
    folder.mkdir()
    t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    sf.write(str(folder / "Piano_L.wav"), (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SAMPLE_RATE)
    tl = build_timeline(SERVICES, SOURCES, "svc1", "quick", stems_root=tmp_path)
    assert not any(p.startswith("keys:jesper_buhl_trio") for p in tl.synthetic_parts)
    assert any(p.startswith("drums:ripe") for p in tl.synthetic_parts)
    lay = next(lay for lay in tl.layers if lay.id == "warm_keys.keys")
    assert lay.source is not None and not lay.source.synthetic
    # 3 s of real audio loops instead of running out
    assert lay.source.read(SAMPLE_RATE * 10, BLOCK_FRAMES).shape == (BLOCK_FRAMES, 2)


def test_sermon_files_play_back_to_back_not_overlaid(tmp_path: Path) -> None:
    folder = tmp_path / "spurgeon_1858"
    folder.mkdir()
    t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    for i, f0 in enumerate((180.0, 220.0, 260.0)):
        sf.write(str(folder / f"sermon_{i:02d}.wav"), (0.2 * np.sin(2 * np.pi * f0 * t)).astype(np.float32), SAMPLE_RATE)
    parts = load_parts(SOURCES.by_id()["spurgeon_1858"], tmp_path, seed=1)
    assert len(parts["speech"].data) == 3 * len(t), "speech files must be concatenated (one talker at a time), not summed"


def test_audio_onset_skips_leading_silence() -> None:
    t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    sig = np.zeros((SAMPLE_RATE * 6, 2), dtype=np.float32)
    sig[SAMPLE_RATE * 2 : SAMPLE_RATE * 5] = (0.3 * np.sin(2 * np.pi * 200 * t))[:, None]
    ps = PartSource("x", "speech", False, data=sig)
    assert audio_onset_s(ps, 0.0, 6.0) == pytest.approx(2.0, abs=0.03)
    assert audio_onset_s(ps, 5.5, 0.4) is None, "silence only: no onset inside the window"
    tl = build_timeline(SERVICES, SOURCES, "svc1", "smoke", stems_root=Path("/nonexistent"))
    ev = next(e for e in tl.events if e.layer == "interrupt_speech")
    assert ev.t0 <= ev.params["t_onset"] <= ev.t1, "scored onset must come from the audio inside the layer window"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_judge_and_macro_f1() -> None:
    g = GATES.gates
    assert judge(g["speech_open_latency_s"], 0.9) is True and judge(g["speech_open_latency_s"], 1.2) is False
    assert judge(g["phase_macro_f1"], 0.9) is True and judge(g["phase_macro_f1"], 0.8) is False
    assert judge(g["false_feedback_cuts"], 0.0) is True and judge(g["false_feedback_cuts"], 1.0) is False
    assert judge(g["lufs_error_lu"], 99.0) is None, "informational gates never fail"
    assert judge(g["loop_p99_ms"], None) is None
    f1, per = macro_f1(["A"] * 10 + ["B"] * 10, ["A"] * 10 + ["B"] * 5 + ["A"] * 5)
    assert 0.7 < f1 < 0.8 and per["B"]["recall"] == 0.5


def _blank_log(n: int) -> TickLog:
    z = np.zeros(n)
    return TickLog(
        service="svcX", profile="smoke", t=np.arange(n) * DT, phase=np.zeros(n, dtype=np.int8),
        vol_db={c: np.full(n, -20.0) for c in CHANNELS}, speech_open=np.zeros(n, dtype=bool),
        instrument_open={c: np.zeros(n, dtype=bool) for c in CHANNELS},
        feedback_ceiling={c: np.full(n, np.nan) for c in CHANNELS}, lufs_short=np.full(n, -70.0),
        pastor_e_mix=z.copy(), band_e_mix=z.copy(), pastor_e_raw=z.copy(), band_e_raw=z.copy(), tick_ms=np.full(n, 2.0),
    )


def test_scoring_detects_slow_speech_open_and_false_feedback(tmp_path: Path) -> None:
    tl = build_timeline(SERVICES, SOURCES, "svc1", "smoke", stems_root=tmp_path)
    rules = MixingRules.model_validate_json(RULES_PATH.read_text())
    log = _blank_log(tl.n_ticks)
    onset = next(e for e in tl.events if e.layer == "interrupt_speech").params["t_onset"]
    k0 = round(onset / DT)
    log.speech_open[k0 + 60 :] = True  # opens 1.2 s after onset
    log.vol_db["ch1_pastor"][k0 + 60 :] = 0.0
    log.feedback_ceiling["ch2_lead_vox"][50:80] = -4.0  # a cut with no howl in this segment
    by = {r.gate: r for r in score_run(log, tl, GATES, rules, SERVICES.settle_s)}
    assert by["speech_open_latency_s"].passed is False and by["speech_open_latency_s"].value == pytest.approx(1.2, abs=0.05)
    assert by["false_feedback_cuts"].passed is False and by["false_feedback_cuts"].value == 1.0
    assert by["feedback_cut_latency_s"].passed is None, "no howl in the smoke profile, so not measured"


def test_report_aggregation_fails_if_any_service_fails(tmp_path: Path) -> None:
    gate = "speech_open_latency_s"
    results = [GateResult(gate, "svc1", 0.7, True, ""), GateResult(gate, "svc2", 1.4, False, "slow")]
    agg = aggregate(results, GATES)
    row = next(r for r in agg["rows"] if r["gate"] == gate)
    assert row["status"] == "FAIL" and row["worst"] == 1.4
    manual = [r for r in agg["rows"] if r["status"] in ("MANUAL", "MAINSTAGE")]
    assert {r["gate"] for r in manual} >= {"mainstage_overloads_30min", "listening_score"}
    write_reports(tmp_path, results, GATES, {"profile": "smoke", "synthetic": {"svc1": ["keys:x:keys"]}})
    md = (tmp_path / "report.md").read_text()
    assert "PROVISIONAL" in md and "`speech_open_latency_s`" in md


# ---------------------------------------------------------------------------
# Fetching (resolvers on fixtures; no network)
# ---------------------------------------------------------------------------

CMT_HTML = """
<div><h3>Jesper Buhl Trio</h3> Acoustic Jazz
 <a href="/ms/mtk/dl/JesperBuhl-Full.zip">Full Multitrack</a>
 <a href="/ms/mtk/dl/JesperBuhl-Excerpt.zip">Edited Excerpt</a></div>
<div><h3>Jesse Joy</h3><a href="https://x.example/JesseJoy.zip">Full</a></div>
"""


def test_resolve_cambridge_zip_prefers_requested_tier() -> None:
    ex = ev_fetch.resolve_cambridge_zip(CMT_HTML, "Jesper Buhl Trio", "excerpt")
    full = ev_fetch.resolve_cambridge_zip(CMT_HTML, "Jesper Buhl Trio", "full")
    assert ex[0].endswith("JesperBuhl-Excerpt.zip") and full[0].endswith("JesperBuhl-Full.zip")
    assert ev_fetch.resolve_cambridge_zip(CMT_HTML, "Nobody Here", None) == []
    assert ev_fetch.resolve_cambridge_zip(CMT_HTML, "Jesse Joy", None) == ["https://x.example/JesseJoy.zip"]


def test_archive_and_librispeech_resolvers() -> None:
    assert ev_fetch.find_archive_ids('<a href="https://archive.org/details/spurgeon_may_1858_librivox">') == ["spurgeon_may_1858_librivox"]
    meta = {"files": [{"name": f"c{i:02d}_64kb.mp3", "length": "300"} for i in range(1, 9)] + [{"name": "cover.jpg"}]}
    assert ev_fetch.pick_archive_audio(meta, 12.0) == ["c01_64kb.mp3", "c02_64kb.mp3", "c03_64kb.mp3"]
    spk = "; header\n1 | F | dev-clean | 8.0 | A\n2 | M | dev-clean | 9.0 | B\n3 | F | dev-clean | 9.5 | C\n4 | F | train | 9 | D\n"
    assert ev_fetch.select_librispeech_speakers(spk, "dev-clean", "female", 2) == ["3", "1"]


def test_zip_extraction_blocks_path_traversal(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ok/vox.wav", b"RIFF")
        zf.writestr("../../evil.wav", b"RIFF")
        zf.writestr("notes.txt", b"x")
    zpath = tmp_path / "a.zip"
    zpath.write_bytes(buf.getvalue())
    dest = tmp_path / "dest"
    out = ev_fetch.safe_extract_zip(zpath, dest)
    assert [p.name for p in out] == ["vox.wav"] and not (tmp_path.parent / "evil.wav").exists()


def test_fetch_requires_terms_and_writes_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = SOURCES.by_id()["jesper_buhl_trio"]
    msg = ev_fetch.fetch_source(src, SOURCES, tmp_path, accept_terms=False, max_gb=1, dry_run=False)
    assert msg.startswith("SKIPPED") and not (tmp_path / src.id).exists()
    monkeypatch.setattr(ev_fetch, "_http_get", lambda url, timeout=60.0: CMT_HTML.encode())
    assert "would download" in ev_fetch.fetch_source(src, SOURCES, tmp_path, True, 1, dry_run=True)

    def fake_download(url: str, dest: Path, max_bytes: int) -> tuple[int, str]:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("Piano.wav", b"RIFF")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(buf.getvalue())
        return len(buf.getvalue()), "abc"

    monkeypatch.setattr(ev_fetch, "_download", fake_download)
    assert "downloaded" in ev_fetch.fetch_source(src, SOURCES, tmp_path, True, 1, dry_run=False)
    ledger = json.loads((tmp_path / ev_fetch.LEDGER).read_text())
    assert ledger[0]["id"] == "jesper_buhl_trio" and ledger[0]["license"] == "cambridge-mt-library"
    assert (tmp_path / src.id / "Piano.wav").exists()
    assert "already present" in ev_fetch.fetch_source(src, SOURCES, tmp_path, True, 1, dry_run=False)


def test_manual_sources_explain_where_to_put_files(tmp_path: Path) -> None:
    msg = ev_fetch.fetch_source(SOURCES.by_id()["church_own_sermon"], SOURCES, tmp_path, False, 1, False)
    assert msg.startswith("manual") and "recordings/stems/eval/church_own_sermon" in msg


# ---------------------------------------------------------------------------
# End to end (headless closed loop through the real engine) and the CLI
# ---------------------------------------------------------------------------


async def test_smoke_run_scores_every_headless_gate_and_reports(tmp_path: Path) -> None:
    tl = build_timeline(SERVICES, SOURCES, "svc1", "smoke", stems_root=tmp_path / "none")
    log = await run_timeline(tl)
    assert log.vad_backend == "dsp", "synthetic speech must use the DSP VAD (Silero rejects the synthetic voice)"
    assert len(log.t) == tl.n_ticks and set(PHASES) >= {PHASES[i] for i in set(log.phase.tolist())}
    assert log.vol_db["ch1_pastor"].min() > -61.0 and log.lufs_short.max() > -60.0, "closed-loop master must reach the LUFS meter"
    rules = MixingRules.model_validate_json(RULES_PATH.read_text())
    results = score_run(log, tl, GATES, rules, SERVICES.settle_s)
    names = {r.gate for r in results}
    assert names == {n for n, g in GATES.gates.items() if g.mode == "headless"}
    by = {r.gate: r for r in results}
    assert by["phase_macro_f1"].value is not None and by["loop_p99_ms"].passed is True
    assert by["speech_open_latency_s"].value is not None, "interrupt onset from a closed pastor fader must be measured"
    agg = write_reports(tmp_path / "out", results, GATES, {"profile": "smoke", "synthetic": {"svc1": tl.synthetic_parts}})
    assert agg["scored"] >= 8 and (tmp_path / "out" / "report.json").exists()
    log.save_npz(tmp_path / "ticks.npz")
    assert np.load(tmp_path / "ticks.npz")["t"].shape == log.t.shape


def test_cli_refuses_synthetic_by_default_and_lists(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts import run_eval

    assert run_eval.main(["--stems-root", str(tmp_path), "list"]) == 0
    assert "svc1" in capsys.readouterr().out
    assert run_eval.main(["--stems-root", str(tmp_path), "run", "svc1", "--profile", "smoke", "--out", str(tmp_path / "o")]) == 3
    assert "synthetic" in capsys.readouterr().out
    assert not (tmp_path / "o" / "report.md").exists()


def test_docs_cover_every_source_and_gate() -> None:
    spec = (REPO_ROOT / "docs" / "EVAL_SPEC.md").read_text(encoding="utf-8")
    runbook = (REPO_ROOT / "docs" / "MAINSTAGE_EVAL_RUNBOOK.md").read_text(encoding="utf-8")
    used = {sid for s in SERVICES.services for sid in s.slots.values()}
    for sid in used:
        assert SOURCES.by_id()[sid].title.split(" (")[0] in spec, f"EVAL_SPEC.md is missing source {sid}; regenerate the track table"
    for name in GATES.gates:
        assert f"`{name}`" in spec, f"EVAL_SPEC.md is missing gate {name}"
    assert "BlackHole 64ch" in runbook and "closedloop-plan" in runbook
