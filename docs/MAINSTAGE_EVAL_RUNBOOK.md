# MainStage Eval Runbook (Mode B and C rig)

Companion to [EVAL_SPEC.md](EVAL_SPEC.md). Everything here happens on the Mac; none of it was run while writing the suite,
so treat each numbered step as a procedure to verify, and report what differs.

## Why two loopback buses

`scripts/play_mainstage_pack.py` writes stems to BlackHole 16ch, and both MainStage and the engine read that same bus. The
engine then hears the raw stems: MainStage fader moves never reach it and master taps 15-16 carry nothing. For a real
closed loop the engine must read MainStage's outputs from a second bus.

```text
DAW / run_eval.py play -> BlackHole 16ch -> MainStage strips -> speakers
                                              |  Send 1 (Pre fader) -> aux -> BlackHole 64ch 1-12   -> engine taps
                                              |  Room strip (no CC)            -> BlackHole 64ch 13-14
                                              |  Send 2 (Post fader) -> Master Tap aux -> BlackHole 64ch 15-16
engine -> bridge daemon -> "AetherMix To MainStage" (CC 20-26, 30-31) -> MainStage faders
```

## Setup

1. `brew install blackhole-64ch` (BlackHole 16ch is already required). Rogue Amoeba Loopback is a paid alternative.
2. Audio MIDI Setup -> Aggregate Device `AetherMix-Live`, 48 kHz. Sub-devices in this order: interface or speakers (clock source), BlackHole 16ch, BlackHole 64ch. Drift correction on for both BlackHole devices. Note the resulting channel numbers; aggregate numbering follows that order.
3. `python scripts/run_eval.py routing-check` should report OK for 16ch, 64ch and the aggregate.
4. MainStage -> Settings -> Audio: input and output `AetherMix-Live`, buffer 128. Create the 8 Concert-level audio strips from `docs/MAINSTAGE_LIVE_TEST.md`.
5. On every strip set Send 1 to **Pre** fader at 0 dB into its own aux, and route that aux to the matching BlackHole 64ch channel (1-12). Create a room strip on In 13-14 outputting straight to BlackHole 64ch 13-14 with no CC mapped. Set Send 2 on every strip to **Post** fader at 0 dB into one Master Tap aux on BlackHole 64ch 15-16, including the reverb return. Keep the main output strip flat.
6. `python scripts/run_eval.py closedloop-plan` writes `engine/schemas/service_plan.closedloop.json` (audio hint `BlackHole 64ch`). Start the engine with `--service-plan engine/schemas/service_plan.closedloop.json`.
7. MIDI: remove any direct controller assignments in MainStage, then use Setup Wizard -> MIDI Learn Pulse for CC 20-26 and 30-31 and run Auto-Verify.
8. **Pre-fader invariance check (gate `mainstage_prefader_invariance_db`)**: sweep one MainStage fader from -60 to +6 dB while recording BlackHole 64ch. Channels 1-12 must not move more than 0.1 dB; channels 15-16 must follow the fader. If a Send cannot be set Pre fader or an aux cannot output to a single BlackHole channel, fall back to Mode A+ for perception and read fader effects from the bridge log only.

## Running a service in Mode B

1. Calibrate with the Setup Wizard using service 1's sources (room silence, isolated spoken, isolated band, bleed floor, baselines).
2. `python scripts/run_eval.py play svc1 --profile quick` prints each segment's start time and streams the 16 channels to BlackHole 16ch. For a DAW instead, `run_eval.py build svc1` writes a 16-channel WAV and `*_markers.csv` (ground truth: segment phases, onsets, trap windows).
3. Switch the engine to ACTIVE with all channels armed. Record the master tap (BlackHole 64ch 15-16) and keep the engine's `telemetry.db` and the bridge log with the markers CSV.
4. Type five critiques per run in the Brain Stream and note whether each produced valid JSON and respected the +/-3 dB auto-apply gate (gate `llm_critique_validity`).
5. Repeat for svc2-svc5. Trims differ per service (`*_trims_db.json`), so a quiet pastor and a hot lead vocal both occur.

## Mode C and fault injection

- **Load**: loop one service for 30 minutes with the heaviest concert at the live buffer size. Watch MainStage's CPU meter and `log stream --predicate 'process == "MainStage"'` for CoreAudio overloads (gate `mainstage_overloads_30min`: 0).
- **Resilience** (gate `resilience_checks`): `kill -9` the engine, then the bridge (`launchd` should restart it in under 2 s); unplug and replug the controller. Manual faders must stay live and no fader may go to zero. `scripts/verify_audio_midi.py` and `scripts/phase0_spike.py` automate parts of this.
- **Physical inputs**: add the interface's mic/line inputs to the aggregate, then play keys or guitar and have someone speak into the pulpit mic to get real bleed, real feedback and real latency.

## Not implemented

Scoring a Mode B capture automatically. The headless scorer (`evals/scoring.py`) works from per-tick engine state; a live scorer needs the bridge CC log with timestamps and `telemetry.db` events parsed against the markers CSV.
