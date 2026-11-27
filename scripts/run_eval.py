#!/usr/bin/env python3
"""AetherMix multitrack MainStage eval CLI (see docs/EVAL_SPEC.md).

  list            services x slots -> sources, and which are on disk
  status          per-source availability and license state
  fetch           download sources (explicit license acceptance; ledgered)
  audition        show how each multitrack's files were classified into parts (listen before you trust a pick)
  run             closed-loop headless run through the real engine, scored against manifest/gates.yaml
  build           write one service as a 16-channel WAV + markers CSV for a DAW / BlackHole player
  play            stream one service to BlackHole 16ch in real time (Mode B, MainStage in the loop)
  live            Mode B scored run: play through MainStage while recording the running engine, then score
  live-score      re-score a finished live run from its saved recordings
  prefader-check  sweep a MainStage fader via the bridge and verify the BlackHole 64ch pre-fader tap does not move
  routing-check   verify BlackHole 16ch/64ch and the AetherMix-Live aggregate exist
  closedloop-plan write engine/schemas/service_plan.closedloop.json (taps from BlackHole 64ch)
  report          print a finished run's report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "engine"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evals import BLOCK_FRAMES, DEFAULT_RUNS_ROOT, DEFAULT_STEMS_ROOT, SAMPLE_RATE
from evals.manifest import PROFILES, SLOTS, load_gates, load_services, load_sources

PLAN = REPO_ROOT / "engine" / "schemas" / "service_plan.json"
RULES = REPO_ROOT / "engine" / "schemas" / "mixing_rules.json"


def _svc_ids(arg: list[str], services) -> list[str]:  # type: ignore[no-untyped-def]
    ids = [s.id for s in services.services]
    if not arg or arg == ["all"]:
        return ids
    bad = [a for a in arg if a not in ids]
    if bad:
        raise SystemExit(f"unknown service(s) {bad}; choose from {ids}")
    return arg


def cmd_list(args: argparse.Namespace) -> int:
    from evals.fetch import status

    sources, root = load_sources(), Path(args.stems_root)
    services = load_services(sources=sources)
    have = {s.id: s.present for s in status(sources, root)}
    print(f"{'slot':<11}" + "".join(f"{s.id:<34}" for s in services.services))
    for slot in SLOTS:
        print(f"{slot:<11}" + "".join(f"{s.slots[slot] + ('' if have[s.slots[slot]] else ' [missing]'):<34}" for s in services.services))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from evals.fetch import status

    sources = load_sources()
    for r in status(sources, Path(args.stems_root)):
        print(f"{'OK     ' if r.present else 'MISSING'} {r.id:<34} {r.method:<20} {r.license:<22} commercial={r.commercial_use:<8} files={r.files}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from evals.fetch import FetchError

    sources = load_sources()
    services = load_services(sources=sources)
    wanted = set(args.source or [])
    for sid in args.service or []:
        wanted |= set(next(s for s in services.services if s.id == sid).slots.values())
    todo = [s for s in sources.sources if not wanted or s.id in wanted]
    rc = 0
    for s in todo:
        try:
            print(f"{s.id:<34} {fetch_msg(s, sources, args)}")
        except FetchError as exc:
            rc = 1
            print(f"{s.id:<34} FAILED: {exc}")
        except OSError as exc:
            rc = 1
            print(f"{s.id:<34} NETWORK ERROR: {exc}")
    return rc


def fetch_msg(source, sources, args) -> str:  # type: ignore[no-untyped-def]
    from evals.fetch import fetch_source

    return fetch_source(source, sources, Path(args.stems_root), args.accept_terms, args.max_gb, args.dry_run)


def cmd_audition(args: argparse.Namespace) -> int:
    from evals.stems import PART_TARGET_DB, classify_folder, load_source

    sources = load_sources()
    for s in sources.sources:
        if args.source and s.id not in args.source:
            continue
        grouped = classify_folder(s, Path(args.stems_root))
        if not any(grouped.values()):
            continue
        parts = load_source(s, Path(args.stems_root), 0)
        print(f"\n{s.id}  ({s.title})")
        for part, files in sorted(grouped.items()):
            ps = parts.get(part if part != "bv" else "bv1")
            dur = f"{ps.duration_s:.0f}s" if ps and ps.duration_s else "-"
            print(f"  {part:<8} {dur:>6}  {', '.join(f.name for f in files[:6])}{' ...' if len(files) > 6 else ''}")
        missing = [p for p in PART_TARGET_DB if p in ("lead", "keys", "drums", "bass", "speech", "room", "choir") and s.slots[0] in ("vocal_song",) and p not in parts]
        if missing:
            print(f"  note: no {', '.join(missing)} found; add `picks:` globs to sources.yaml or the part will be synthetic")
    return 0


def _build(args: argparse.Namespace, sid: str):  # type: ignore[no-untyped-def]
    from evals.timeline import build_timeline

    sources = load_sources()
    services = load_services(sources=sources)
    return build_timeline(services, sources, sid, args.profile, Path(args.stems_root), bleed_db=args.bleed_db)


def cmd_run(args: argparse.Namespace) -> int:
    from engine.core.protocol import MixingRules
    from evals.report import write_reports
    from evals.runner import run_timeline
    from evals.scoring import score_run

    sources = load_sources()
    services = load_services(sources=sources)
    gates = load_gates()
    rules = MixingRules.model_validate_json(RULES.read_text(encoding="utf-8"))
    ids = _svc_ids(args.service, services)
    timelines = {sid: _build(args, sid) for sid in ids}
    synthetic = {sid: tl.synthetic_parts for sid, tl in timelines.items()}
    if any(synthetic.values()) and not args.allow_synthetic:
        print("Refusing to score: these slots would use synthetic stand-ins instead of real recordings.\n"
              "Fetch them (`make eval-fetch`) or pass --allow-synthetic for a clearly labelled PROVISIONAL run.\n")
        for sid, parts in synthetic.items():
            for p in parts:
                print(f"  {sid}: {p}")
        return 3
    out_dir = Path(args.out) if args.out else DEFAULT_RUNS_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    vad_backends: dict[str, str] = {}
    for sid, tl in timelines.items():
        print(f"== {sid}: {tl.total_s:.0f}s ({args.profile}), synthetic parts: {len(tl.synthetic_parts)}", flush=True)
        t0 = time.time()
        log = asyncio.run(run_timeline(tl, PLAN, RULES, use_service_plan=args.use_service_plan,
                                       keep_audio=args.write_audio, progress=True))
        vad_backends[sid] = log.vad_backend
        (out_dir / sid).mkdir(exist_ok=True)
        log.save_npz(out_dir / sid / "ticks.npz")
        tl.write_markers_csv(out_dir / sid / "markers.csv")
        (out_dir / sid / "trims_db.json").write_text(json.dumps(tl.trims_db, indent=2), encoding="utf-8")
        if args.write_audio and log.masters is not None:
            import soundfile as sf

            sf.write(str(out_dir / sid / "aethermix_master.flac"), log.masters[0], SAMPLE_RATE, subtype="PCM_16")
            sf.write(str(out_dir / sid / "static_mix.flac"), log.masters[1], SAMPLE_RATE, subtype="PCM_16")
            _write_listening_sheet(out_dir / sid / "listening_sheet.csv", tl)
        results += score_run(log, tl, gates, rules, services.settle_s)
        print(f"   done in {time.time() - t0:.0f}s", flush=True)
    meta = {"profile": args.profile, "services": ids, "synthetic": synthetic, "bleed_db": args.bleed_db,
            "use_service_plan": args.use_service_plan, "vad_backend": vad_backends, "mode": "headless closed-loop emulation (not MainStage)"}
    agg = write_reports(out_dir, results, gates, meta)
    print((out_dir / "report.md").read_text(encoding="utf-8"))
    print(f"Report: {out_dir / 'report.md'}")
    return 0 if agg["all_passed"] or args.no_fail else 1


def _write_listening_sheet(path: Path, tl) -> None:  # type: ignore[no-untyped-def]
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["segment", "t0_s", "t1_s", "title", "score_aethermix_1to5", "score_static_1to5", "notes"])
        for s in tl.segments:
            w.writerow([s.id, f"{s.t0:.1f}", f"{s.t1:.1f}", s.title, "", "", ""])


def cmd_build(args: argparse.Namespace) -> int:
    import numpy as np
    import soundfile as sf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sid in _svc_ids(args.service, load_services(sources=load_sources())):
        tl = _build(args, sid)
        if tl.synthetic_parts and not args.allow_synthetic:
            print(f"{sid}: refusing, synthetic parts present {tl.synthetic_parts}; pass --allow-synthetic")
            return 3
        path = out / f"{sid}_{args.profile}_16ch.wav"
        big = tl.total_s * SAMPLE_RATE * 16 * 2 > 3.8e9
        with sf.SoundFile(str(path), "w", SAMPLE_RATE, 16, subtype="PCM_16", format="RF64" if big else "WAV") as fh:
            for k in range(tl.n_ticks):
                fh.write(np.clip(tl.block(k), -1.0, 1.0))
        tl.write_markers_csv(out / f"{sid}_{args.profile}_markers.csv")
        (out / f"{sid}_{args.profile}_trims_db.json").write_text(json.dumps(tl.trims_db, indent=2), encoding="utf-8")
        print(f"{sid}: {path} ({tl.total_s:.0f}s, 16ch) + markers; tap trims {tl.trims_db}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    import sounddevice as sd  # type: ignore[import-untyped]

    tl = _build(args, args.service)
    if tl.synthetic_parts and not args.allow_synthetic:
        print(f"refusing: synthetic parts {tl.synthetic_parts}; pass --allow-synthetic")
        return 3
    dev = next((i for i, d in enumerate(sd.query_devices()) if args.device.lower() in d["name"].lower() and d["max_output_channels"] >= 16), None)
    if dev is None:
        print(f"no 16-channel output device matching '{args.device}' (run: routing-check)")
        return 1
    print(f"Streaming {tl.service.id} ({tl.total_s:.0f}s) to {sd.query_devices(dev)['name']}. Segment starts:")
    for s in tl.segments:
        print(f"  {s.t0:7.1f}s  {s.id:<18} {s.phase or 'unscored'}")
    with sd.OutputStream(device=dev, samplerate=SAMPLE_RATE, channels=16, dtype="float32", blocksize=BLOCK_FRAMES) as stream:
        for k in range(tl.n_ticks):
            stream.write(tl.block(k))
    return 0


CLOSEDLOOP_PLAN = REPO_ROOT / "engine" / "schemas" / "service_plan.closedloop.json"


def _score_live(out_dir: Path, ids: list[str], timelines: dict, meta: dict) -> int:  # type: ignore[type-arg]
    from engine.core.protocol import MixingRules, ServicePlan
    from evals.live import LiveRecording, estimate_offset_s, ticklog_from_recording
    from evals.report import write_reports
    from evals.scoring import score_run

    services, gates = load_services(sources=load_sources()), load_gates()
    rules = MixingRules.model_validate_json(RULES.read_text(encoding="utf-8"))
    plan = ServicePlan.model_validate_json(CLOSEDLOOP_PLAN.read_text(encoding="utf-8"))
    results, align = [], {}
    for sid in ids:
        tl = timelines[sid]
        rec = LiveRecording.load(out_dir / sid / "live_recording.json.gz")
        off, r = estimate_offset_s(rec, tl)
        align[sid] = {"offset_s": round(off, 3), "corr": round(r, 3), "frames": len(rec.frames), **rec.meta}
        log = ticklog_from_recording(rec, tl, plan, off)
        log.save_npz(out_dir / sid / "ticks.npz")
        tl.write_markers_csv(out_dir / sid / "markers.csv")
        results += score_run(log, tl, gates, rules, services.settle_s)
    meta = {**meta, "alignment": align}
    agg = write_reports(out_dir, results, gates, meta)
    print((out_dir / "report.md").read_text(encoding="utf-8"))
    print(f"Report: {out_dir / 'report.md'}")
    return 0 if agg["all_passed"] else 1


def cmd_live(args: argparse.Namespace) -> int:
    import sounddevice as sd  # type: ignore[import-untyped]
    from evals.live import capture

    services = load_services(sources=load_sources())
    ids = _svc_ids(args.service, services)
    timelines = {sid: _build(args, sid) for sid in ids}
    synthetic = {sid: tl.synthetic_parts for sid, tl in timelines.items()}
    if any(synthetic.values()) and not args.allow_synthetic:
        print("refusing: synthetic parts present; pass --allow-synthetic for a PROVISIONAL live run")
        return 3
    dev = next((i for i, d in enumerate(sd.query_devices()) if args.device.lower() in d["name"].lower() and d["max_output_channels"] >= 16), None)
    if dev is None:
        print(f"no 16-channel output device matching '{args.device}' (run: routing-check)")
        return 1
    token = Path(args.token_path).expanduser().read_text(encoding="utf-8").strip()
    out_dir = Path(args.out) if args.out else DEFAULT_RUNS_ROOT / ("live-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        tl = timelines[sid]
        print(f"== {sid}: live {tl.total_s:.0f}s ({args.profile}) -> {sd.query_devices(dev)['name']}", flush=True)
        from engine.core.protocol import ServicePlan

        plan = ServicePlan.model_validate_json(CLOSEDLOOP_PLAN.read_text(encoding="utf-8"))
        sock = None if args.no_human_moves else Path(args.bridge_socket).expanduser()
        rec = asyncio.run(capture(tl, dev, token, url=args.url, arm=not args.no_arm, bridge_socket=sock, plan=plan))
        (out_dir / sid).mkdir(exist_ok=True)
        rec.save(out_dir / sid / "live_recording.json.gz")
        (out_dir / sid / "trims_db.json").write_text(json.dumps(tl.trims_db, indent=2), encoding="utf-8")
        print(f"   recorded {len(rec.frames)} frames, underruns {rec.meta.get('underruns')}", flush=True)
    meta = {"profile": args.profile, "services": ids, "synthetic": synthetic, "bleed_db": args.bleed_db,
            "mode": "LIVE Mode B: MainStage in the loop (engine taps from BlackHole 64ch)",
            "human_moves": "none" if args.no_human_moves else "sent as UI fader moves through the bridge"}
    (out_dir / "live_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rc = _score_live(out_dir, ids, timelines, meta)
    return 0 if args.no_fail else rc


def cmd_prefader_check(args: argparse.Namespace) -> int:
    import sounddevice as sd  # type: ignore[import-untyped]
    from evals.live import prefader_check

    devs = sd.query_devices()
    out_dev = next((i for i, d in enumerate(devs) if args.out_device.lower() in d["name"].lower() and d["max_output_channels"] >= 16), None)
    in_dev = next((i for i, d in enumerate(devs) if args.in_device.lower() in d["name"].lower() and d["max_input_channels"] >= 16), None)
    if out_dev is None or in_dev is None:
        print(f"devices not found: out={args.out_device} in={args.in_device} (run: routing-check)")
        return 1
    token = Path(args.token_path).expanduser().read_text(encoding="utf-8").strip()
    res = asyncio.run(prefader_check(out_dev, in_dev, token, Path(args.bridge_socket).expanduser(), args.channel,
                                     in_col=args.in_col, tap=args.tap - 1, master=(args.master - 1, args.master)))
    curves = {k: res.pop(k) for k in ("pre_curve_db", "master_curve_db")}
    out = Path(args.out) if args.out else DEFAULT_RUNS_ROOT / ("prefader-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, **curves}, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    rng_db = res["prefader_tap_db_range"]
    ok = rng_db is not None and rng_db <= 0.1 and (res["master_tap_db_range"] or 0) >= 20.0
    print(f"mainstage_prefader_invariance_db = {rng_db} dB (gate <= 0.1; master tap must follow: {res['master_tap_db_range']} dB) -> {'PASS' if ok else 'FAIL'}")
    print(f"saved {out}")
    return 0 if ok else 1


def cmd_live_score(args: argparse.Namespace) -> int:
    out_dir = Path(args.run)
    meta = json.loads((out_dir / "live_meta.json").read_text(encoding="utf-8"))
    args.profile, args.bleed_db = meta["profile"], meta.get("bleed_db")
    timelines = {sid: _build(args, sid) for sid in meta["services"]}
    return _score_live(out_dir, meta["services"], timelines, meta)


def cmd_routing_check(args: argparse.Namespace) -> int:
    import sounddevice as sd  # type: ignore[import-untyped]

    devs = [(d["name"], d["max_input_channels"], d["max_output_channels"]) for d in sd.query_devices()]
    checks = {
        "BlackHole 16ch (player -> MainStage)": any("BlackHole 16ch" in n and o >= 16 for n, _, o in devs),
        "BlackHole 64ch (MainStage taps -> engine)": any("BlackHole 64ch" in n and i >= 16 for n, i, _ in devs),
        f"Aggregate '{args.aggregate}'": any(args.aggregate in n for n, _, _ in devs),
    }
    for name, ok in checks.items():
        print(f"{'OK     ' if ok else 'MISSING'} {name}")
    for n, i, o in devs:
        print(f"   device: {n} in={i} out={o}")
    return 0 if all(checks.values()) else 1


def cmd_closedloop_plan(args: argparse.Namespace) -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    plan["audio_device_hint"] = "BlackHole 64ch"
    out = REPO_ROOT / "engine" / "schemas" / "service_plan.closedloop.json"
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.run) / "report.md"
    print(path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems-root", default=str(DEFAULT_STEMS_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn, **kw):  # type: ignore[no-untyped-def]
        p = sub.add_parser(name, help=kw.get("help"))
        p.set_defaults(fn=fn)
        return p

    add("list", cmd_list, help="services x slots")
    add("status", cmd_status, help="source availability")
    p = add("fetch", cmd_fetch, help="download sources")
    p.add_argument("--source", action="append")
    p.add_argument("--service", action="append", help="fetch everything a service needs")
    p.add_argument("--accept-terms", action="store_true", help="I have read the license terms of the sources being fetched")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-gb", type=float, default=2.0, help="per-download size cap")
    p = add("audition", cmd_audition, help="show part classification")
    p.add_argument("--source", action="append")

    def common(p) -> None:  # type: ignore[no-untyped-def]
        p.add_argument("--profile", choices=PROFILES, default="quick")
        p.add_argument("--bleed-db", type=float, default=None, help="override stage bleed (plan suggests -15 for stress)")
        p.add_argument("--allow-synthetic", action="store_true", help="allow synthetic stand-ins (report is PROVISIONAL)")

    p = add("run", cmd_run, help="headless closed-loop scored run")
    p.add_argument("service", nargs="*", help="svc1..svc5 or all")
    common(p)
    p.add_argument("--out")
    p.add_argument("--write-audio", action="store_true", help="write AetherMix master + static mix FLACs and a listening sheet")
    p.add_argument("--use-service-plan", action="store_true", help="fuse the Order of Service into phase detection")
    p.add_argument("--no-fail", action="store_true", help="exit 0 even when gates fail")
    p = add("build", cmd_build, help="write 16ch WAV + markers")
    p.add_argument("service", nargs="*")
    common(p)
    p.add_argument("--out", default=str(DEFAULT_RUNS_ROOT / "build"))
    p = add("play", cmd_play, help="stream to BlackHole 16ch")
    p.add_argument("service")
    common(p)
    p.add_argument("--device", default="BlackHole 16ch")
    p = add("live", cmd_live, help="Mode B: play through MainStage, record the running engine, score")
    p.add_argument("service", nargs="*", help="svc1..svc5 or all")
    common(p)
    p.add_argument("--device", default="BlackHole 16ch")
    p.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    p.add_argument("--token-path", default="~/.aethermix/session.token")
    p.add_argument("--no-arm", action="store_true", help="do not arm channels / switch the engine to ACTIVE")
    p.add_argument("--bridge-socket", default="~/.aethermix/bridge.sock")
    p.add_argument("--no-human-moves", action="store_true", help="do not send the scripted human fader moves")
    p.add_argument("--out")
    p.add_argument("--no-fail", action="store_true")
    p = add("prefader-check", cmd_prefader_check, help="gate mainstage_prefader_invariance_db: sweep a fader, watch the taps")
    p.add_argument("--channel", default="ch1_pastor")
    p.add_argument("--in-col", type=int, default=0, help="BlackHole 16ch column (0-based) feeding that strip")
    p.add_argument("--tap", type=int, default=1, help="BlackHole 64ch pre-fader tap channel (1-based)")
    p.add_argument("--master", type=int, default=15, help="first BlackHole 64ch master-tap channel (1-based)")
    p.add_argument("--out-device", default="BlackHole 16ch")
    p.add_argument("--in-device", default="BlackHole 64ch")
    p.add_argument("--token-path", default="~/.aethermix/session.token")
    p.add_argument("--bridge-socket", default="~/.aethermix/bridge.sock")
    p.add_argument("--out")
    p = add("live-score", cmd_live_score, help="re-score a live run directory")
    p.add_argument("run")
    p = add("routing-check", cmd_routing_check, help="verify loopback devices")
    p.add_argument("--aggregate", default="AetherMix-Live")
    add("closedloop-plan", cmd_closedloop_plan, help="write service_plan.closedloop.json")
    p = add("report", cmd_report, help="print a run's report")
    p.add_argument("run")
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
