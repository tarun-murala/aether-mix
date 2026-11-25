# Project Specification v5.1: **AetherMix** (Autonomous Agentic Live Sound Engineer for Apple MainStage)

> **Status:** Locked v5.1 — Ready for Phase 0 Hardware Spike  
> **Target Hardware:** Apple Silicon macOS 15+ (optimized for M4 Max, 128 GB Unified Memory)  
> **Host Integration:** Apple MainStage (Concert-Level Audio Channel Strips)

---

## 1. Executive Summary, Scope & Operating Assumptions

Build **AetherMix**, a native macOS application, isolated MIDI bridge daemon, and local multi-model AI engine that runs alongside **Apple MainStage** to autonomously mix live multi-channel audio (pulpit/spoken mics, vocals, keys/pads, guitars, drums/bass/tracks) in real time across dynamic church service phases:

* `SPOKEN_WORD`
* `PRAYER_UNDERSCORE`
* `WORSHIP_LOW`
* `WORSHIP_HIGH`
* `TRANSITION` *(hold & gentle-settle state; not a fixed baseline target)*

### Operating Assumptions

1. **MainStage as FOH Host:** MainStage hosts the live audio channel strips for these inputs at the **Concert Level** (not per-Patch) and routes PA output + pre-fader BlackHole taps via a 48 kHz macOS Aggregate Device.
2. **Control Surface Discipline:** All live fader and reverb-send control flows through the **hardware MIDI controller** or **AetherMix's SwiftUI faders**, both routed directly through the standalone `midi_bridge_daemon`. Unless Phase 0 proves MainStage reliably emits GUI feedback CCs, **direct mouse drags inside MainStage's GUI are invisible to AetherMix**—operators must mix via the controller or AetherMix UI while channels are armed.

### Core Non-Negotiable Constraints

1. **100% Local Execution / Zero Cloud Latency:** All inference (VAD, CLAP scene classification, supervised offset learner, supervisory LLM) and CoreAudio/CoreMIDI I/O run locally on Apple Silicon.
2. **Always-On Manual Control (Process Isolation):** The MIDI bridge (`midi_bridge_daemon`) runs as an **independent, zero-ML `launchd`-supervised process** separate from the AI engine. Both the hardware controller AND SwiftUI's virtual faders/Emergency Bypass connect directly to `midi_bridge_daemon`, remaining live (`< 2 ms`) even if the AI engine crashes, stalls, hits a watchdog trip, or is in `SHADOW_OBSERVE`.
3. **Deterministic Safety Firewall:** No ML model or LLM ever emits raw MIDI directly. Every agent command passes through a deterministic firewall enforcing calibrated dB/CC bounds, speech-open and ensemble-gated instrument-open reflexes, drift-free per-control human ephemeral offsets, 5-path asymmetric slew-rate limits, bleed-rejected anti-feedback protection, and strict no-auto-mute protection on spoken mics.
4. **Fail-Safe Live Reliability:**
   * Engine always boots into **`SHADOW OBSERVE`** (agent writes frozen; human passthrough active) with all channels disarmed (`Restore Previous Arming` available in 1 click).
   * Never sends zero/mute CCs on shutdown, crash, or watchdog trip—MainStage holds its last fader state while `midi_bridge_daemon` persists authoritative `last_sent_cc` to disk and keeps manual control live.
5. **Zero MainStage Audio Contention:** Audio ML threads are hard-pinned (`1` thread, Background-QoS subprocess), and live deployment requires passing a 30-minute zero-overload stress gate alongside the heaviest MainStage concert.

---

## 2. Architectural Decision Records (ADRs)

| Area | Decision | Rationale |
| :--- | :--- | :--- |
| **Standalone MIDI Bridge Process** | **`midi_bridge_daemon`** (zero-ML process supervised by `launchd` `KeepAlive`, Unix Domain Socket accepting **both** `engine` and `ui` clients, atomic disk persistence of `last_sent_cc`) | Guarantees an engine crash, GC pause, or watchdog trip **never** kills hardware controller passthrough, SwiftUI fallback faders, or Emergency Bypass, and preserves authoritative `last_sent_cc` across engine or bridge restarts. |
| **MainStage MIDI Port Persistence** | `mainstage_out_mode: "virtual_port" \| "iac_bus"` validated via Phase 0 `kill -9` restart test | If recreating a CoreMIDI virtual port on bridge restart breaks MainStage's MIDI Learn binding, an OS-owned IAC bus preserves endpoint identity permanently across restarts and reboots. |
| **Per-Channel Arming & Sync** | Master mode (`SHADOW` / `ACTIVE` / `BYPASS`) + per-channel `armed` + `sync_status`, hydrated from `midi_bridge_daemon.last_sent_cc` | Prevents fader jumps on `SHADOW → ACTIVE` or post-crash reconnect, enables 1-click `Restore Previous Arming`, and supports staged live rollout. |
| **Audio Tap Topology** | **Pre-fader Ch 1–12** (perception + internal gain-normalized sum for CLAP) + **Direct Input Ch 13–14** (room mic) + **Post-fader Ch 15–16 Master** (LUFS metering only) | Keeps VAD, RMS, feedback detection, and CLAP strictly invariant to fader moves, preventing phase lock-in loops. |
| **Sequential 3-Step Bleed Calibration** | Soundcheck: (A1) room silence → (A2) isolated spoken nominals then isolated band nominals → (B) band-only bleed floor | Prevents band bleed from contaminating spoken nominals, measures true `mic_noise_floor_norm_db`, and makes `speech_to_band_snr_db = 0 dB` mean *"expected bleed only"* regardless of preamp trims. |
| **Guarded Fast Reflexes** | Tier-1 **`speech_open` (`48 dB/s`)** + ensemble/silent-onset-gated **`instrument_open` (`24 dB/s`, `8 s` auto-release)** + `-16 dB` standby baselines + segment pre-staging | Prevents both buried pastors (`< 1.0 s`) and late band entrances (`< 800 ms`) without falsely raising keys during prayer or a lone tuning guitar during the sermon. |
| **Drift-Free Per-Control Human Offsets & Learning** | **Per-`(phase, channel, control)`** Ridge Offset Regression (1 sample per operator event); while human override is active, freeze learned offset AND suspend conditional rules on that control; clear smoothly via `4 dB/s` musical slew | Eliminates learner double-counting, conditional-rule snap-downs (e.g. `-24 dB` on BVs), and phase-change jumps; applies identically to volume faders and reverb sends. |
| **Reverb Sends in dB** | Logarithmic **dB + measured `reverb_cc_taper`** | MainStage bus sends are logarithmic dB controls, not linear percentages. |
| **CPU & Memory Isolation** | **Silero VAD (`onnxruntime`, 1 thread)** + **CLAP (`torch` CPU, 1 thread in Background-QoS subprocess)**; Ollama model benchmarked in Phase 0 | Prevents PyTorch thread pools from preempting MainStage's real-time `AudioWorkgroup` threads; validated by a 30-minute zero-overload stress gate. |
| **LLM Rule Safety** | Strict Ollama JSON Schema; auto-apply only single-channel non-mute within `±3.0 dB` vol / `±4.0 dB` reverb, preview required otherwise; continuous cap `[-12, +6 dB]` separate from conditional `[-36, 0 dB]` | Prevents misheard critiques or stacked rules from slamming vocals; keeps LLM advisory-only on automatic phase/safety triggers. |

---

## 3. Technology Stack & Repository Layout

### Stack
* **Standalone MIDI Bridge (`bridge/`):** Python 3.12 stdlib + `python-rtmidi` ONLY (zero NumPy/PyTorch/FastAPI imports, `< 20 MB` RSS, `< 1 ms` latency), supervised by macOS `launchd` (`KeepAlive: true`, `ThrottleInterval: 1`, `ProcessType: Interactive`), exposing local Unix Domain Socket (`~/.aethermix/bridge.sock`, `0600`, authenticated with `~/.aethermix/session.token`) to both `engine` and `macos-app`, persisting `last_sent_cc` atomically (`100 ms` debounce) to `~/.aethermix/bridge_state.json`.
* **AI & Real-Time Audio Engine (`engine/`):** Python 3.12 via `uv`, FastAPI + WebSocket (`ws://127.0.0.1:8765/ws`, authenticated via `~/.aethermix/session.token`, `0600`), Pydantic v2, `sounddevice` (PortAudio), `numpy`/`scipy`, `onnxruntime` (1-thread CPU Silero VAD), `transformers` + `torch` (1-thread CPU CLAP in Background-QoS subprocess), `httpx` (async Ollama client), `sqlite3` (WAL mode).
* **Native macOS UI (`macos-app/`):** Swift 6 + SwiftUI (macOS 15+), `XcodeGen` (`project.yml`) + `Package.swift`. Connects to **both** the Engine WebSocket (telemetry, Brain Stream, rules) AND `~/.aethermix/bridge.sock` (direct virtual faders + Emergency Bypass independent of engine health). Quick-chips and keyboard typing are the primary critique UI; Apple `Speech` dictation is optional (rehearsal use).
* **Audio/MIDI Infrastructure:** BlackHole 16ch + 48 kHz macOS Aggregate Device (physical PA interface as clock master, drift correction enabled on BlackHole), CoreMIDI Virtual Port or OS-owned IAC Driver bus (decided in Phase 0) + Hardware MIDI Controller.

### Directory Layout

```text
aethermix/
├── SPEC.md                           # This engineering specification (v5)
├── Makefile                          # setup, models, test, spike, install-bridge, build-app, dev, simulate
├── scripts/
│   ├── bootstrap_mac.sh              # Homebrew deps, launchd bridge plist, Ollama pull, HF/Silero cache
│   ├── com.aethermix.midibridge.plist # launchd KeepAlive manifest for standalone MIDI bridge
│   ├── phase0_spike.py               # MainStage feedback, kill -9 virtual-port persistence, vol/reverb tapers, pre-fader & 30m stress spike
│   ├── verify_audio_midi.py          # Aggregate/BlackHole RMS + standalone bridge passthrough smoke test
│   ├── record_shadow_session.py      # Records 16ch pre-fader FLAC/WAV + phase annotations in SHADOW mode
│   └── simulate_service.py           # Synthetic multi-stem church service runner (headless + BlackHole)
├── bridge/
│   └── midi_bridge_daemon.py         # Zero-ML standalone MIDI router, fader_map, pickup takeover, HW+UI bypass, persisted last_sent_cc
├── engine/
│   ├── pyproject.toml
│   ├── main.py                       # Orchestrator: Bridge Client, Audio Tap, VAD, CLAP Subprocess, 50Hz Loop, WS Server
│   ├── core/
│   │   ├── protocol.py               # Versioned Pydantic v5 schemas (WS envelopes, Bridge NDJSON IPC, LLM schema, rules)
│   │   ├── bridge_client.py          # Async Unix-socket client to midi_bridge_daemon (state hydrate, 20Hz heartbeat, agent writes)
│   │   ├── audio_tap.py              # Lock-free ring buffer, 50Hz RMS/peak/FFT, bleed-normalized SNR, pre-fader CLAP sum, 10Hz LUFS, TCC guard
│   │   ├── vad_worker.py             # 1-thread ONNX Silero VAD (16kHz) + vocal harmonicity/RMS detector
│   │   ├── clap_worker.py            # Isolated Background-QoS subprocess running 1-thread CPU CLAP at 0.5Hz
│   │   ├── scene_classifier.py       # Pre-fader CLAP + quantitative P_telemetry + Order-of-Service fusion + hysteresis
│   │   ├── offset_learner.py         # Per-(phase, channel, control) Ridge offset regression, decay, versioned snapshots, reset
│   │   ├── llm_supervisor.py         # Async Ollama client (strict JSON schema format, 5s timeout, advisory vs critique)
│   │   ├── safety_firewall.py        # 5-path slew, speech-open, ensemble instrument-open, per-control human offset, feedback guard, watchdog
│   │   ├── rules_manager.py          # Validated rule patching, ±3dB preview gate, continuous vs conditional caps, atomic backups
│   │   └── telemetry_store.py        # SQLite WAL: 1Hz summaries, discrete events, last_known_state, policy_snapshots, review queue
│   ├── schemas/
│   │   ├── mixing_rules.json         # Versioned baselines (vol & reverb dB), LUFS targets, firewall params, learned rules
│   │   └── service_plan.json         # Order of Service, 16ch map, controller fader_map/buttons, noise/bleed calibration, Ollama config
│   ├── storage/
│   │   ├── telemetry.db              # SQLite WAL database (gitignored)
│   │   └── rule_backups/             # Timestamped atomic backups of mixing_rules.json
│   └── tests/
│       ├── mock_io.py                # Headless deterministic Audio + Bridge IPC simulator
│       ├── test_midi_bridge.py       # Engine death passthrough, 500ms dead-engine detect, UI direct socket, persisted state, HW bypass, pickup
│       ├── test_safety_firewall.py   # Speech-open <1s, instrument-open gating & 8s release, conditional suspend under human offset, feedback, watchdog
│       ├── test_scene_fusion.py      # Pre-fader sum invariance, 3-step bleed SNR, TRANSITION hold-settle, applause/room mic
│       ├── test_offset_learner.py    # 1-sample updates, no double-counting, per-control vol/reverb, freeze queue, decay, reset
│       ├── test_rules_and_llm.py     # ±3dB preview gate, continuous vs conditional caps, schema bounds, default expires_on
│       └── test_end_to_end.py        # Full synthetic service lifecycle + WS auth/RPC + direct bridge fallback verification
└── macos-app/
    ├── project.yml                   # XcodeGen spec (Info.plist TCC keys, entitlements, macOS 15 target)
    ├── Package.swift                 # SwiftPM manifest for CLI build & model tests
    └── Sources/AetherMix/
        ├── App/AetherMixApp.swift    # Lifecycle, LaunchAgent check, managed engine subprocess, update guard
        ├── Models/
        │   ├── ProtocolModels.swift  # Codable mirror of engine/core/protocol.py (v5)
        │   └── MeterStore.swift      # Isolated 30-60Hz meter buffer keyed by channel_ids[]
        ├── Services/
        │   ├── EngineWebSocketClient.swift  # Token auth, req_id RPC acks, heartbeat monitor, last-state cache
        │   ├── BridgeSocketClient.swift     # Direct Unix-socket connection to midi_bridge_daemon (virtual faders + bypass)
        │   └── EngineProcessManager.swift   # Installs/verifies bridge LaunchAgent + spawns/monitors local uv engine
        └── Views/
            ├── MainDashboardView.swift      # Status bar, 1-click Restore Arming, crash banner w/ direct bridge faders
            ├── MeterCanvasView.swift        # Dedicated 30-60Hz SwiftUI Canvas meters + clip/VAD/speech-open LEDs
            ├── MixerTelemetryView.swift     # Per-channel Arm toggle, dual-ring vol/reverb knobs, pickup indicator, human offset & feedback badges
            ├── AgentBrainStreamView.swift   # Filterable feed + Preview/Pending Rule cards ([Apply Session]/[Save Permanent]/[Dismiss])
            ├── RulesManagerView.swift       # Drawer to inspect, toggle, delete permanent rules & reset phase learning
            ├── PostServiceReviewView.swift  # Review & commit/discard queued samples collected under Freeze Learning
            ├── FeedbackControlBar.swift     # 👍/👎 rollback dock, Freeze Learning toggle, chips & text critique
            └── SetupWizardView.swift        # TCC/BlackHole check, Bridge/Controller map, Mute polarity, 4-step Soundcheck Calibration
```

---

## 4. Model Manifest & Automated Provisioning (`scripts/bootstrap_mac.sh`)

1. **System Dependencies (Homebrew):** `blackhole-16ch`, `ollama`, `uv`, `portaudio`, `xcodegen`.
2. **Standalone Bridge Service:** Installs `scripts/com.aethermix.midibridge.plist` to `~/Library/LaunchAgents/` (`KeepAlive: true`, `ThrottleInterval: 1`, `ProcessType: Interactive`).
3. **Supervisory Reasoning LLM (Ollama — configurable in `service_plan.json`):**
   * Benchmark current local structured-JSON models in Phase 0 (`qwen3:14b` with `think: false` / `qwen3:8b` / `qwen2.5:14b`) and select the fastest model passing JSON-schema accuracy + 30-minute zero-overload stress gate.
   * Optional offline/post-service analysis model (`--with-32b` flag): `qwen3:32b` or `qwen2.5:32b`.
4. **Semantic Scene Classifier (`~/.aethermix/models/clap/`):**
   * `laion/clap-htsat-unfused` via Hugging Face `transformers` (1-thread CPU in Background-QoS subprocess).
   * Prompts:
     * `"pastor or speaker talking alone without music"` → `SPOKEN_WORD`
     * `"soft ambient keyboard pad underneath spoken prayer"` → `PRAYER_UNDERSCORE`
     * `"acoustic guitar or piano gentle worship verse"` → `WORSHIP_LOW`
     * `"full church band with drums electric guitar and loud singing"` → `WORSHIP_HIGH`
     * `"silence applause or stage transition between songs"` → `TRANSITION`
5. **Voice Activity Detection (`~/.aethermix/models/silero_vad.onnx`):**
   * `silero-vad` ONNX executed via `onnxruntime` (`intra_op_num_threads=1`, `inter_op_num_threads=1`) at 16 kHz / 512-sample chunks (`32 ms`).

---

## 5. Detailed Subsystem Architecture

### 5.1 Audio & MIDI Routing Specification

#### BlackHole 16ch Tap Map (`service_plan.json`)
MainStage drives a **48 kHz macOS Aggregate Device** combining the physical PA interface (clock master) and **BlackHole 16ch** (drift correction enabled). All AetherMix-controlled strips reside at MainStage's **Concert Level**.

| BlackHole Ch | Tap Type | Logical Strip (`channel_id`) | Role | Controlled MIDI Params (Ch 1 default) |
| :--- | :--- | :--- | :--- | :--- |
| `Ch 1` (mono) | Pre-Fader | `ch1_pastor` (Pastor / Pulpit) | `spoken_primary` | `CC 20` Vol (dB), `CC 30` Reverb Send (dB) *(auto hard-mute forbidden)* |
| `Ch 2` (mono) | Pre-Fader | `ch2_lead_vox` (Worship Leader) | `vocal_lead` | `CC 21` Vol (dB), `CC 31` Reverb Send (dB), `CC 41` Mute |
| `Ch 3` (mono) | Pre-Fader | `ch3_bv1` (Backup Vocal 1) | `vocal_bg` | `CC 22` Vol (dB), `CC 42` Mute |
| `Ch 4` (mono) | Pre-Fader | `ch4_bv2` (Backup Vocal 2) | `vocal_bg` | `CC 23` Vol (dB), `CC 43` Mute |
| `Ch 5–6` (stereo) | Pre-Fader | `ch5_keys` (Keys / Pad) | `instrument_pad` | `CC 24` Vol (dB) |
| `Ch 7–8` (stereo) | Pre-Fader | `ch7_guitars` (Guitars) | `instrument_guitar` | `CC 25` Vol (dB) |
| `Ch 9–12` (bus) | Pre-Fader | `ch9_drums` (Drums / Bass / Tracks) | `instrument_band` | `CC 26` Vol (dB) |
| `Ch 13–14` (stereo) | **Direct Input (No Strip)** | `ch13_room` (Room Ambience, optional) | `room_ambience` | Telemetry only (applause / transition context) |
| `Ch 15–16` (stereo) | **Post-Fader** | `master_out` (Stereo Master Mix) | `master_meter` | **LUFS & clip telemetry only** *(never fed to CLAP)* |

#### Standalone MIDI Bridge (`bridge/midi_bridge_daemon.py`)
```text
[Hardware MIDI Controller] ──► [midi_bridge_daemon (launchd KeepAlive)] ──► (Virtual Port OR IAC Bus) ──► [MainStage]
                                    ▲                      ▲
                      (role: engine)│   ~/.aethermix/      │(role: ui — direct virtual faders + bypass)
                                    ▼     bridge.sock      ▼
                            [AetherMix Engine]      [SwiftUI Mission Control]
```
* **Always-On Manual Control:** Both the hardware controller (`midi.fader_map`) AND SwiftUI's direct `BridgeSocketClient` (`ui_manual_cc`) pass through to MainStage in **every** mode (`SHADOW`, `ACTIVE`, `EMERGENCY_BYPASS`, watchdog trip, or engine crash). `SHADOW`, `BYPASS`, and watchdog freeze **agent writes only**.
* **Bridge-Latched Emergency Bypass & Hardware Buttons:** Hardware `emergency_bypass` (`CC 104`) OR SwiftUI `set_bypass(true)` latches directly inside `midi_bridge_daemon`, dropping all incoming `agent_cc_write` frames in `< 1 ms` regardless of engine health. Hardware `👍` (`CC 102`) and `👎` (`CC 103`) forward to the engine immediately (works even when MainStage has macOS focus).
* **Dead-Engine Watchdog inside Bridge:** Expects `engine_heartbeat` (or agent frame) every `<= 500 ms` while engine socket is connected. On `> 500 ms` silence or socket EOF, sets `engine_connected = false`, holds `last_sent_cc` (never zeros!), broadcasts `bridge_state` to SwiftUI, and continues serving hardware + SwiftUI manual control seamlessly.
* **Bridge-Level In-Flight Race Guard (`human_seq` Fence + `250 ms` Window):** Every human hardware or UI CC move increments a per-control monotonic `human_seq` inside `midi_bridge_daemon` and starts a `250 ms` local drop window. Any `agent_cc_write` arriving with `ack_human_seq < current_human_seq` (or within the `250 ms` guard) is dropped immediately inside the bridge, guaranteeing an in-flight `50 Hz` agent glide can never overwrite an operator's touch.
* **Authoritative Persisted `last_sent_cc` & Restart Resilience:**
  * Persists `last_sent_cc[(channel_id, control)]` atomically (`tmp` + `os.replace` to `~/.aethermix/bridge_state.json`) on every change (debounced `100 ms` + immediate on `SIGTERM`), reloading on startup so bridge restarts or reboots retain pickup reference values.
  * `mainstage_out_mode` supports `"virtual_port"` or OS-owned `"iac_bus"` (selected after Phase 0 `kill -9` bridge-restart test verifies MainStage binding persistence and `< 2 s` recovery via `ThrottleInterval: 1`).
* **Soft-Takeover (With Pre-Pickup Touch Lock) & USB Hot-Plug:** Supports `pickup_catchup` (default for non-motorized faders), `immediate`, and `relative_encoder`. In `pickup_catchup`, **any physical fader movement immediately emits a `human_touch_event` (engaging the bridge race guard and engine touch lock so the operator never chases a moving agent target)**, while MIDI passthrough to MainStage waits until the fader crosses within `±2 CC` of `last_sent_cc`. Polls USB-MIDI ports at `1 Hz`; on unplug, alerts `CONTROLLER_DISCONNECTED`, allows SwiftUI direct bridge faders as immediate fallback, and re-attaches in `pickup_catchup` on replug.
* **Canonical Full-Range `dB ↔ CC` Tapers (0..127, measured in Phase 0):**
  * **Volume Taper:** `[(0, null), (1, -60.0), (16, -36.0), (32, -24.0), (64, -12.0), (96, 0.0), (114, +6.0), (127, +6.0)]`
  * **Reverb Send Taper (dB):** `[(0, null), (1, -60.0), (32, -24.0), (64, -12.0), (96, 0.0), (114, +6.0), (127, +6.0)]`
  * `CC 0` = `-inf` (off); `CC 1..114` strictly monotonic (validated at load); `CC 115..127` maps to `+6.0 dB` (inverse returns `114`).
* **Mute Safety & Deduplication:** Automatic mute CCs are hard-forbidden on `spoken_primary`; `SetupWizardView` mandates Mute Polarity Verification. Emits MIDI CC only when quantized integer value changes (or during `MIDI Learn Pulse`).

---

### 5.2 Calibration, Initial State Sync & Handover

#### Sequential 4-Step Soundcheck Calibration (`SetupWizardView`)
1. **Step A1 — `capture_room_silence` (`5 s`, "Room Quiet — Nobody Speaks or Plays"):** Measures ambient voice-band noise floor `mic_noise_floor_vband_db[m]` for each mic `m`.
2. **Step A2 — `capture_input_nominals` (Sequential, Isolated):**
   * *Part 1 (`10 s`, Band Silent):* Pastor / vocalists speak/sing normally → records clean `nominal_vband_rms_db[ch]` & `nominal_rms_db[ch]` with zero band bleed, and computes `mic_noise_floor_norm_db[m] = mic_noise_floor_vband_db[m] - nominal_vband_rms_db[m]`.
   * *Part 2 (`15 s`, Spoken Mics Silent):* Keys, guitars, and drums/band play normally → records clean instrument nominals.
3. **Step B — `capture_bleed_floor` (`10 s`, "Band Plays, Nobody Speaks"):** Measures per-mic acoustic bleed transfer ratio in gain-normalized units:
   ```text
   norm_vband_db(ch) = rms_300_3400(ch) - nominal_vband_rms_db[ch]
   bleed_offset_db[m] = clamp(median(norm_vband_db(m) - max_{b in band}(norm_vband_db(b))), -35.0, -3.0)
   ```
   *(Default prior if uncalibrated: `-12.0 dB`, noise floor `-36.0 dB` norm, with Setup warning badge.)*
4. **Step C — `capture_baseline(phase, quick_derive_others)`:** Snapshots **`midi_bridge_daemon`'s authoritative `last_sent_cc`** (volume + reverb dB—never un-synced physical non-motorized fader positions!) into `phase_baselines_db[phase]` and `phase_reverb_baselines_db[phase]`, updating `calibrated_at`.

#### Startup, Sync & Fast Post-Crash Re-Arming
1. Engine always boots into `SHADOW_OBSERVE` with all channels disarmed (`armed = false`) and immediately hydrates `current_db` for every `(channel_id, control)` from `midi_bridge_daemon` (`get_bridge_state`):
   * Controls with an observed live `last_sent_cc` in the running bridge (`restored_from_disk_at == null`) are immediately `sync_status = "SYNCED"`.
   * Cold-boot controls or values reloaded from `bridge_state.json` after a bridge crash/reboot (`restored_from_disk_at != null`) are flagged `UNVERIFIED` until touched on the controller/UI, captured via baseline, or confirmed via `Confirm & Sync` (`2.0 dB/s` handover glide). If the bridge crashes and restarts while the engine is `ACTIVE`, the engine automatically drops to `SHADOW_OBSERVE`, marks restored controls `UNVERIFIED`, and flashes a reconnect banner.
2. **1-Click `Restore Previous Arming`:** Persists the prior `armed` bitmask in SQLite `last_known_state`; after an engine restart, restores all previously armed channels in 1 click (still requiring explicit switch from `SHADOW_OBSERVE` → `ACTIVE`).

---

### 5.3 Three-Tier Control Loop

#### Tier 1: Real-Time Reflex & Safety Loop (`50 Hz` / `20 ms` tick)

1. **Decoupled Audio Ingestion (`audio_tap.py`, `vad_worker.py`):**
   * PortAudio callback pushes 48 kHz float32 blocks into a lock-free ring buffer (zero allocations, zero ML).
   * `audio_tap.py` computes at `50 Hz`:
     * Per-channel wideband RMS, voice-band (`300–3400 Hz`) RMS, `norm_vband_db`, true-peak dBFS, clip flags (`>= -0.5 dBFS`), and 1024-point Hann FFT spectra.
     * **Gain- & Bleed-Normalized Speech SNR:**
       ```text
       expected_bleed_norm_db(m) = max(max_{b in band}(norm_vband_db(b)) + bleed_offset_db[m], mic_noise_floor_norm_db[m])
       speech_to_band_snr_db(m)  = norm_vband_db(m) - expected_bleed_norm_db(m)
       ```
     * **Sung-Vocal Activity (`Ch 2–4`):** Gain-normalized `200–4000 Hz` RMS + autocorrelation harmonicity salience (`vocal_snr_db >= +3.0 dB`), NOT Silero VAD.
     * **Pre-Fader CLAP Sum:** Gain-normalized unity sum of **pre-fader `Ch 1–12`** into a 3-second rolling buffer for the CLAP subprocess.
     * **Master LUFS (`10 Hz`):** ITU-R BS.1770 momentary (`400 ms`) & short-term (`3 s`) LUFS strictly on **post-fader `Ch 15–16`**.
     * **TCC Zero-Buffer Guard:** Bitwise all-zero input for `> 2.0 s` raises `HARDWARE_SILENCE_WARNING`.
   * `vad_worker.py`: 1-thread CPU ONNX Silero VAD on 16 kHz downsampled `Ch 1` (and spoken `Ch 2`, `32 ms` hop / ~31 Hz).

2. **Drift-Free Supervised Offset Learner (`offset_learner.py`):**
   * Maintains per-`(phase p, channel i, control c)` Ridge regression `(A_{p,i,c}, b_{p,i,c})` predicting **target dB offset `y ∈ [-4.0, +4.0]`** (`control ∈ {"vol", "reverb"}`) from context `phi_i(S_t)`:
     ```text
     phi_i(S_t) = [norm_vband_db_i / 24.0, vocal_to_band_ratio_norm, speech_or_vocal_active_i, master_lufs_error_norm, 1.0]
     live_learned_db[p][i, c] = clip(w[p][i, c]^T * phi_i(S_t), -4.0, +4.0)
     ```
     *(In `TRANSITION`, evaluates using the prior macro phase's weights.)*
   * **Anti-Double-Count Freeze Invariant:** While `human_override_active[i, c] == true`, uses `frozen_learned_db[i, c]` (captured at touch time) instead of `live_learned_db[p][i, c]`. Updated weights take effect only after the human offset clears.
   * **Exact 1-Sample Updates (all 3 types queue to `pending_review_samples` if `freeze_learning == true`):**
     * **Operator Fader / Reverb Move (`alpha = 2.0`):** `y = clip(human_db - (phase_base_db + continuous_rule_sum_db), -4.0, +4.0)` with 2-second pre-move mean `phi_bar`.
     * **Thumbs Up (`👍`, `alpha = 1.0`):** 1 sample per armed `(channel, control)` (`y = current_effective_offset_db`, trailing 5 s mean `phi_bar`).
     * **Thumbs Down (`👎`, `alpha = 1.5`):** 1 sample per rolled-back `(channel, control)` (`y = pre_change_offset_db`, pre-change `phi_bar`).
   * **Decay, Snapshots & Reset:** Inter-service forgetting factor (`gamma = 0.98`), SQLite `policy_snapshots`, and `reset_learning(phase?, channel_id?)`.

3. **Target Assembly & Deterministic Safety Firewall (`safety_firewall.py`):**
   * Per-control `(i, c)` target assembly (`c ∈ {"vol", "reverb"}`):
     ```text
     continuous_rule_sum_db = clip(sum(matching vol_db_offset or reverb_db_offset rules), -12.0, +6.0)
     conditional_delta_db   = 0.0 if human_override_active[i, c] else clip(sum(triggered conditional_attenuate rules), -36.0, 0.0)
     effective_learned_db   = frozen_learned_db[i, c] if human_override_active[i, c] else live_learned_db[p][i, c]
     base_target_db[i, c]   = phase_base_db[i, c] + continuous_rule_sum_db + effective_learned_db + human_session_offset_db[i, c] + conditional_delta_db
     ```
     *(In `TRANSITION`, `phase_base_db[i, "vol"]` holds the prior macro phase's baseline with a `-4.0 dB` gentle settle on instrument roles; reverb holds prior macro baseline.)*
   * Every `20 ms` tick enforces these invariants in strict precedence order:
     1. **Watchdog & Mode Gate:** If `> 5` consecutive ticks exceed `50 ms` or audio stalls `> 500 ms`, drops engine to `SHADOW_OBSERVE` (freezes **agent writes only**; `midi_bridge_daemon` preserves hardware + SwiftUI manual control) and emits `WATCHDOG_DROP_TO_SHADOW`.
     2. **Per-Control Touch Lock, Persistent Human Offset & Smooth Clearing:**
        * Non-echo human move on `(i, c)` (from hardware controller or SwiftUI direct bridge fader) sets `touch_lock_until[i, c] = now + 15.0 s`, captures `human_anchor_db[i, c] = human_db`, freezes `frozen_learned_db[i, c] = live_learned_db[p][i, c]`, **suspends conditional attenuate/mute rules on `(i, c)`**, and sets `human_session_offset_db[i, c] = human_db - (phase_base_db[i, c] + continuous_rule_sum_db + frozen_learned_db[i, c])`.
        * **While `touch_lock_until[i, c]` is active:** **NO reflexes (`speech_open`, `instrument_open`, ducking, phase glides) may act on `(i, c)`**, EXCEPT escalating **Anti-Feedback Cuts** on volume (announced prominently in UI).
        * **After `touch_lock_until[i, c]` expires:** `human_session_offset_db[i, c]` holds the operator's exact value until the next macro phase transition, segment advance, or manual *Release Offset*. When it clears (unfreezing `live_learned_db` and re-enabling conditional rules), the transition glides **smoothly through the `4.0 dB/s` musical slew limiter** (never jumping instantaneously).
     3. **Tier-1 `Speech-Open` Reflex (Fast Pastor Onset, `< 1.0 s`):**
        * For `spoken_primary` volume (when not touch-locked), triggers whenever `vad_prob >= 0.70` AND `speech_to_band_snr_db >= +6.0 dB` sustained for:
          * **`150 ms`** in `SPOKEN_WORD`, `PRAYER_UNDERSCORE`, or `TRANSITION`,
          * **`400 ms`** in `WORSHIP_LOW` or `WORSHIP_HIGH` (`400 ms + ~420 ms = ~820 ms < 1.0 s`).
        * Target = `phase_baselines_db["SPOKEN_WORD"][i] + human_session_offset_db[i, "vol"]`, slewed via **`slew_speech_open_up = 48.0 dB/s`**.
        * Holds until phase confirms `SPOKEN_WORD`/`PRAYER_UNDERSCORE` OR sustained speech silence `>= 3.0 s`, then releases at `4.0 dB/s`.
     4. **Tier-1 Ensemble-Gated `Instrument-Open` Reflex (Fast Song Start, `< 800 ms`, Zero Prayer/Sermon False Opens):**
        * **Never** triggers while `active_phase == PRAYER_UNDERSCORE`.
        * Tracks per-instrument silent→active onset: channel was below silent threshold (`norm_vband_db <= -30.0 dB`) for **`>= 3.0 s`** and rises to `norm_vband_db >= -12.0 dB` for **`>= 200 ms`**.
        * **Song-Start Ensemble Gate:** Fires only when **either** `instrument_band` (`ch9_drums`) has a silent→active onset, **or** `>= 2` distinct instrument roles (`instrument_pad`, `instrument_guitar`, `instrument_band`) have silent→active onsets within a `1.0 s` window.
        * **Speech-Over-Intro Handling:** Opens active non-touch-locked instrument channels toward `WORSHIP_LOW` (or next Order-of-Service worship segment) at **`slew_instrument_open_up = 24.0 dB/s`** (`-16 dB → -2 dB` in `583 ms`). If `speech_open_active` is simultaneously `true` (e.g., pastor saying *"let's stand and sing"* over the intro), the ensemble gate still fires, but targets are capped at **`worship_target + duck_depth_db`** (`-6 dB` under the pastor) until speech ends.
        * **Auto-Release Guard:** If macro phase does NOT transition to `WORSHIP_LOW` or `WORSHIP_HIGH` within **`8.0 s`** (or instrument returns silent for `>= 3.0 s`), releases smoothly back to active phase target at musical slew (`4.0 dB/s`).
        * Additionally, operator advancing `service_step_idx` to a worship segment pre-stages instruments toward `WORSHIP_LOW` at `4.0 dB/s`.
     5. **Bleed-Safe, Role-Configurable Auto-Ducking Prior:**
        * Ducked roles configured per phase via `firewall.duck_targets_by_phase` (`["instrument_pad"]` in spoken/prayer/transition; `["instrument_pad", "instrument_guitar", "instrument_band"]` during worship Pastor Interrupt).
        * Triggers when `Ch 1` (or `Ch 2` in spoken/prayer) meets `vad_prob >= 0.65` & `speech_to_band_snr_db >= +3.0 dB` (`150 ms`) in spoken/prayer/transition, OR when `Ch 1` `speech_open` is active (`>= 400 ms`) during worship.
        * Runs inside the unified 5-path slew limiter (`48 dB/s` attack → `125 ms`; `8 dB/s` release → `750 ms`).
     6. **Bleed-Rejected, Escalating Anti-Feedback Detector (Overrides Touch Lock!):**
        * Monitors `feedback_guard == true` mics (`Ch 1–4`) for high-Q FFT peaks `> +14 dB` above local median floor (`±10` bins).
        * Rejects peaks matching `Ch 5–12` instrument taps (`±1 bin`, within `6 dB` prominence); requires `>= 250 ms` persistence with `>= +2.0 dB` growth (or sustained `> +18 dB` lock-on).
        * Cuts `-4.0 dB` (`48 dB/s`), escalates `-4.0 dB` every `400 ms` up to `-16.0 dB`, holds sticky `feedback_ceiling_db[i]` until `5.0 s` clear, then lifts at `+1.0 dB/s`.
     7. **Unified 5-Path Float Slew-Rate Limiter & Human-Respecting Clamps:**
        * `slew_protect_down = 48.0 dB/s` | `slew_speech_open_up = 48.0 dB/s` | `slew_instrument_open_up = 24.0 dB/s` | `slew_duck_release_up = 8.0 dB/s` | `slew_musical = 4.0 dB/s` (`2.0 dB/s` unverified sync handover).
        * Agent-originated moves clamp to `[hard_min_db, hard_max_db]`; human-set levels above `hard_max_db` hold `max(hard_max_db, human_anchor_db[i, c])` until human offset clears (anti-feedback ceiling always wins).

---

#### Tier 2: Semantic Sensor Fusion & LLM Supervisor (`0.5 Hz` / `2 s` + event triggers)

* **Isolated CLAP Subprocess (`clap_worker.py`):** `multiprocessing.spawn`, `torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`, macOS Background QoS (`PRIO_DARWIN_BG` / `os.nice(10)`). Classifies 3-second rolling **pre-fader sum (`Ch 1–12`)** every `2.0 s`.
* **Quantitative Telemetry Evidence (`P_telemetry`):** Calibrated `norm_vband_db` (`active >= -18 dB`, `silent <= -30 dB` rel. nominal), bleed-normalized `speech_to_band_snr_db`, sung-vocal harmonicity (`Ch 2–4`), and `ch13_room` broadband spectral flatness (`> 0.55` for applause/transitions).
* **Fusion & Hysteresis:** `0.40 * P_clap + 0.40 * P_telemetry + 0.20 * P_service_plan` (`0.50 / 0.50` without Order of Service); requires 2 consecutive `2 s` windows (`4.0 s`) at `>= 0.70` confidence. `TRANSITION` holds prior macro phase with `-4 dB` instrument settle.
* **Async Ollama Supervisor (`llm_supervisor.py`):** Warm-pinned (`keep_alive: "60m"`), strict JSON Schema `format`, `5.0 s` timeout. **Advisory-only** on `phase_transition` and `safety_event` (`proposed_patches = []` enforced in code); emits typed patches only on `operator_critique`.

---

#### Tier 3: Human-in-the-Loop Adaptation & Rule Lifecycle (`rules_manager.py`)

* **`👍` / `👎` & Operator Fader/Reverb Moves:** All three feed 1 sample per event into `offset_learner.py` (or queue into SQLite `pending_review_samples` when `freeze_learning == true`). `👎` rolls back non-touch-locked armed controls to `T - 5 s` (second press within `5 s` steps back to `T - 15 s` / pre-transition).
* **Natural-Language Critique & Rule Promotion:**
  * Single-channel non-mute `vol_db_offset` within `[-3.0, +3.0] dB` or `reverb_db_offset` within `[-4.0, +4.0] dB` auto-applies as ephemeral session rule and shows **`[Save Permanent]`** (`promote`) / **`[Dismiss]`** (`dismiss`).
  * Any larger, multi-channel, or conditional attenuate/mute patch requires **`PREVIEW REQUIRED`** approval (**`[Apply Session]`** → `apply_session` | **`[Save Permanent]`** → `promote` | **`[Dismiss]`** → `dismiss`).
  * `expires_on` defaults to `"segment_change"` for `ephemeral_session` rules. Conflicts resolve by most-recent rule wins; permanent rules write atomically with backups in `engine/storage/rule_backups/`.

---

## 6. Reference Schemas & IPC Protocols (v5)

### 6.1 `engine/schemas/service_plan.json`

```json
{
  "schema_version": 5,
  "sample_rate": 48000,
  "audio_device_hint": "BlackHole 16ch",
  "ollama_model": "qwen3:14b",
  "ollama_think": false,
  "midi": {
    "hardware_in_port": "X-Touch Mini",
    "takeover_mode": "pickup_catchup",
    "mainstage_out_mode": "virtual_port",
    "mainstage_out_port": "AetherMix To MainStage",
    "mainstage_feedback_in_port": null,
    "fader_map": [
      {"hw_channel": 1, "hw_cc": 1,  "channel_id": "ch1_pastor",  "control": "vol"},
      {"hw_channel": 1, "hw_cc": 2,  "channel_id": "ch2_lead_vox","control": "vol"},
      {"hw_channel": 1, "hw_cc": 3,  "channel_id": "ch3_bv1",     "control": "vol"},
      {"hw_channel": 1, "hw_cc": 4,  "channel_id": "ch4_bv2",     "control": "vol"},
      {"hw_channel": 1, "hw_cc": 5,  "channel_id": "ch5_keys",    "control": "vol"},
      {"hw_channel": 1, "hw_cc": 6,  "channel_id": "ch7_guitars", "control": "vol"},
      {"hw_channel": 1, "hw_cc": 7,  "channel_id": "ch9_drums",   "control": "vol"},
      {"hw_channel": 1, "hw_cc": 11, "channel_id": "ch1_pastor",  "control": "reverb"},
      {"hw_channel": 1, "hw_cc": 12, "channel_id": "ch2_lead_vox","control": "reverb"}
    ],
    "action_bindings": {
      "thumbs_up": {"hw_channel": 1, "hw_cc": 102},
      "thumbs_down": {"hw_channel": 1, "hw_cc": 103},
      "emergency_bypass": {"hw_channel": 1, "hw_cc": 104}
    }
  },
  "room_tap_channels": [13, 14],
  "master_tap_channels": [15, 16],
  "channels": [
    {
      "id": "ch1_pastor",
      "label": "Pastor Mic",
      "tap_channels": [1],
      "role": "spoken_primary",
      "vad_enabled": true,
      "feedback_guard": true,
      "nominal_rms_db": -18.0,
      "nominal_vband_rms_db": -20.0,
      "mic_noise_floor_norm_db": -36.0,
      "bleed_offset_db": -16.0,
      "midi_channel": 1,
      "vol_cc": 20,
      "reverb_cc": 30,
      "mute_cc": null,
      "hard_min_db": -60.0,
      "hard_max_db": 3.0,
      "cc_taper": [[0, null], [1, -78.2], [4, -54.1], [8, -42.1], [16, -30.1], [24, -23.0], [32, -18.0], [48, -11.0], [64, -6.0], [80, -2.1], [96, 1.1], [112, 3.7], [127, 6.0]],
      "reverb_cc_taper": [[0, null], [1, -60.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0], [127, 6.0]]
    },
    {
      "id": "ch2_lead_vox",
      "label": "Lead Vocal",
      "tap_channels": [2],
      "role": "vocal_lead",
      "vad_enabled": true,
      "feedback_guard": true,
      "nominal_rms_db": -18.0,
      "nominal_vband_rms_db": -20.0,
      "mic_noise_floor_norm_db": -36.0,
      "bleed_offset_db": -14.0,
      "midi_channel": 1,
      "vol_cc": 21,
      "reverb_cc": 31,
      "mute_cc": 41,
      "mute_active_high": true,
      "hard_min_db": -60.0,
      "hard_max_db": 3.0,
      "cc_taper": [[0, null], [1, -78.2], [4, -54.1], [8, -42.1], [16, -30.1], [24, -23.0], [32, -18.0], [48, -11.0], [64, -6.0], [80, -2.1], [96, 1.1], [112, 3.7], [127, 6.0]],
      "reverb_cc_taper": [[0, null], [1, -60.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0], [127, 6.0]]
    },
    {
      "id": "ch3_bv1",
      "label": "Backup Vocal 1",
      "tap_channels": [3],
      "role": "vocal_bg",
      "vad_enabled": false,
      "feedback_guard": true,
      "nominal_rms_db": -20.0,
      "nominal_vband_rms_db": -22.0,
      "mic_noise_floor_norm_db": -36.0,
      "bleed_offset_db": -14.0,
      "midi_channel": 1,
      "vol_cc": 22,
      "mute_cc": 42,
      "mute_active_high": true,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -78.2], [4, -54.1], [8, -42.1], [16, -30.1], [24, -23.0], [32, -18.0], [48, -11.0], [64, -6.0], [80, -2.1], [96, 1.1], [112, 3.7], [127, 6.0]]
    },
    {
      "id": "ch4_bv2",
      "label": "Backup Vocal 2",
      "tap_channels": [4],
      "role": "vocal_bg",
      "vad_enabled": false,
      "feedback_guard": true,
      "nominal_rms_db": -20.0,
      "nominal_vband_rms_db": -22.0,
      "mic_noise_floor_norm_db": -36.0,
      "bleed_offset_db": -14.0,
      "midi_channel": 1,
      "vol_cc": 23,
      "mute_cc": 43,
      "mute_active_high": true,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -78.2], [4, -54.1], [8, -42.1], [16, -30.1], [24, -23.0], [32, -18.0], [48, -11.0], [64, -6.0], [80, -2.1], [96, 1.1], [112, 3.7], [127, 6.0]]
    },
    {
      "id": "ch5_keys",
      "label": "Keys / Pad",
      "tap_channels": [5, 6],
      "role": "instrument_pad",
      "vad_enabled": false,
      "feedback_guard": false,
      "nominal_rms_db": -16.0,
      "nominal_vband_rms_db": -18.0,
      "midi_channel": 1,
      "vol_cc": 24,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -78.2], [4, -54.1], [8, -42.1], [16, -30.1], [24, -23.0], [32, -18.0], [48, -11.0], [64, -6.0], [80, -2.1], [96, 1.1], [112, 3.7], [127, 6.0]]
    },
    {
      "id": "ch7_guitars",
      "label": "Guitars",
      "tap_channels": [7, 8],
      "role": "instrument_guitar",
      "vad_enabled": false,
      "feedback_guard": false,
      "nominal_rms_db": -16.0,
      "nominal_vband_rms_db": -18.0,
      "midi_channel": 1,
      "vol_cc": 25,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -78.2], [4, -54.1], [8, -42.1], [16, -30.1], [24, -23.0], [32, -18.0], [48, -11.0], [64, -6.0], [80, -2.1], [96, 1.1], [112, 3.7], [127, 6.0]]
    },
    {
      "id": "ch9_drums",
      "label": "Drums / Bass / Tracks",
      "tap_channels": [9, 10, 11, 12],
      "role": "instrument_band",
      "vad_enabled": false,
      "feedback_guard": false,
      "nominal_rms_db": -14.0,
      "nominal_vband_rms_db": -18.0,
      "midi_channel": 1,
      "vol_cc": 26,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -78.2], [4, -54.1], [8, -42.1], [16, -30.1], [24, -23.0], [32, -18.0], [48, -11.0], [64, -6.0], [80, -2.1], [96, 1.1], [112, 3.7], [127, 6.0]]
    }
  ],
  "order_of_service": [
    {"id": "seg_1", "title": "Opening Worship (Songs 1–2)", "expected_phase": "WORSHIP_HIGH"},
    {"id": "seg_2", "title": "Song 3 (Acoustic Refrain)", "expected_phase": "WORSHIP_LOW"},
    {"id": "seg_3", "title": "Pastoral Prayer", "expected_phase": "PRAYER_UNDERSCORE"},
    {"id": "seg_4", "title": "Sermon", "expected_phase": "SPOKEN_WORD"},
    {"id": "seg_5", "title": "Closing Song & Benediction", "expected_phase": "WORSHIP_HIGH"}
  ]
}
```

### 6.2 `engine/schemas/mixing_rules.json`

```json
{
  "schema_version": 5,
  "calibrated_at": null,
  "firewall": {
    "touch_lock_seconds": 15.0,
    "echo_suppress_ms": 200,
    "transition_instrument_trim_db": -4.0,
    "speech_open_vad_min": 0.70,
    "speech_open_snr_db": 6.0,
    "speech_open_hold_spoken_ms": 150,
    "speech_open_hold_worship_ms": 400,
    "speech_open_release_silence_s": 3.0,
    "instrument_open_silent_norm_db": -30.0,
    "instrument_open_silent_min_s": 3.0,
    "instrument_open_active_norm_db": -12.0,
    "instrument_open_hold_ms": 200,
    "instrument_open_ensemble_window_s": 1.0,
    "instrument_open_max_hold_s": 8.0,
    "duck_depth_db": -6.0,
    "duck_snr_spoken_db": 3.0,
    "duck_targets_by_phase": {
      "SPOKEN_WORD": ["instrument_pad"],
      "PRAYER_UNDERSCORE": ["instrument_pad"],
      "TRANSITION": ["instrument_pad"],
      "WORSHIP_LOW": ["instrument_pad", "instrument_guitar", "instrument_band"],
      "WORSHIP_HIGH": ["instrument_pad", "instrument_guitar", "instrument_band"]
    },
    "slew_protect_down_db_per_sec": 48.0,
    "slew_speech_open_up_db_per_sec": 48.0,
    "slew_instrument_open_up_db_per_sec": 24.0,
    "slew_duck_release_up_db_per_sec": 8.0,
    "slew_musical_db_per_sec": 4.0,
    "slew_handover_db_per_sec": 2.0,
    "feedback_peak_above_floor_db": 14.0,
    "feedback_growth_db": 2.0,
    "feedback_hold_ms": 250,
    "feedback_cut_step_db": -4.0,
    "feedback_max_cut_db": -16.0,
    "feedback_release_delay_s": 5.0,
    "feedback_release_rate_db_per_sec": 1.0,
    "watchdog_max_tick_ms": 50.0,
    "watchdog_max_consecutive_late": 5
  },
  "phase_target_lufs": {
    "SPOKEN_WORD": -21.0,
    "PRAYER_UNDERSCORE": -24.0,
    "WORSHIP_LOW": -17.0,
    "WORSHIP_HIGH": -13.0
  },
  "phase_baselines_db": {
    "SPOKEN_WORD":       {"ch1_pastor": 0.0,   "ch2_lead_vox": -20.0, "ch3_bv1": -24.0, "ch4_bv2": -24.0, "ch5_keys": -16.0, "ch7_guitars": -16.0, "ch9_drums": -16.0},
    "PRAYER_UNDERSCORE": {"ch1_pastor": 0.0,   "ch2_lead_vox": -18.0, "ch3_bv1": -24.0, "ch4_bv2": -24.0, "ch5_keys": -10.0, "ch7_guitars": -16.0, "ch9_drums": -16.0},
    "WORSHIP_LOW":       {"ch1_pastor": -18.0, "ch2_lead_vox": 0.0,   "ch3_bv1": -8.0,  "ch4_bv2": -8.0,  "ch5_keys": -4.0,  "ch7_guitars": -6.0,  "ch9_drums": -12.0},
    "WORSHIP_HIGH":      {"ch1_pastor": -20.0, "ch2_lead_vox": 0.0,   "ch3_bv1": -5.0,  "ch4_bv2": -5.0,  "ch5_keys": -3.0,  "ch7_guitars": -4.0,  "ch9_drums": -2.0}
  },
  "phase_reverb_baselines_db": {
    "SPOKEN_WORD":       {"ch1_pastor": -36.0, "ch2_lead_vox": -24.0},
    "PRAYER_UNDERSCORE": {"ch1_pastor": -20.0, "ch2_lead_vox": -14.0},
    "WORSHIP_LOW":       {"ch1_pastor": -24.0, "ch2_lead_vox": -10.0},
    "WORSHIP_HIGH":      {"ch1_pastor": -30.0, "ch2_lead_vox": -8.0}
  },
  "learned_rules": [
    {
      "id": "rule_001",
      "description": "Keep keys softer under prayer",
      "phase": "PRAYER_UNDERSCORE",
      "channel_id": "ch5_keys",
      "parameter": "vol_db_offset",
      "value": -3.0,
      "condition": null,
      "enabled": true,
      "source": "operator_confirmed"
    },
    {
      "id": "rule_002",
      "description": "Attenuate backup vocals whenever pastor speaks",
      "phase": "ANY",
      "channel_id": "ch3_bv1",
      "parameter": "conditional_attenuate",
      "value": -24.0,
      "condition": {"when_channel": "ch1_pastor", "predicate": "speech_open"},
      "enabled": true,
      "source": "operator_confirmed"
    }
  ]
}
```

### 6.3 Ollama Supervisory LLM Output JSON Schema

Validated by Pydantic with per-parameter hard bounds (`vol_db_offset: [-12, +6]`, `reverb_db_offset: [-18, +12]`, `duck_depth_db: [-12, 0]`, `conditional_attenuate: [-36, 0]`, `conditional_mute: [0, 1]`; `expires_on` defaults to `"segment_change"`):

```json
{
  "type": "object",
  "required": ["thought_summary", "reasoning", "confidence", "proposed_patches"],
  "properties": {
    "thought_summary": {"type": "string", "maxLength": 140},
    "reasoning": {"type": "string", "maxLength": 500},
    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "proposed_patches": {
      "type": "array",
      "maxItems": 4,
      "items": {
        "type": "object",
        "required": ["description", "scope", "phase", "channel_id", "parameter", "value"],
        "properties": {
          "description": {"type": "string", "maxLength": 120},
          "scope": {"type": "string", "enum": ["ephemeral_session", "candidate_permanent"]},
          "expires_on": {"type": "string", "enum": ["phase_change", "segment_change", "end_of_service"], "default": "segment_change"},
          "phase": {"type": "string", "enum": ["ANY", "SPOKEN_WORD", "PRAYER_UNDERSCORE", "WORSHIP_LOW", "WORSHIP_HIGH"]},
          "channel_id": {"type": "string"},
          "parameter": {"type": "string", "enum": ["vol_db_offset", "reverb_db_offset", "duck_depth_db", "conditional_attenuate", "conditional_mute"]},
          "value": {"type": "number", "minimum": -36.0, "maximum": 12.0},
          "condition": {
            "type": ["object", "null"],
            "properties": {
              "when_channel": {"type": "string"},
              "predicate": {"type": "string", "enum": ["speech_open", "active"]}
            }
          }
        }
      }
    }
  }
}
```

### 6.4 Engine ↔ SwiftUI WebSocket Protocol (`ws://127.0.0.1:8765/ws`, v5)

* **Auth & Handshake:** Bearer token from `~/.aethermix/session.token` (`0600`) + `hello { protocol_version: "5.0" }`.
* **Envelope:** `{ "type": "<str>", "seq": <int>, "req_id": "<uuid|null>", "ts_ms": <int>, "payload": { ... } }`
* **Server → Client Messages:**
  * `ack` / `error`: `{ "req_id": "...", "ok": true|false, "code": "...", "message": "..." }`
  * `state_snapshot` (on connect + every state delta):
    ```json
    {
      "channel_ids": ["ch1_pastor", "ch2_lead_vox", "ch3_bv1", "ch4_bv2", "ch5_keys", "ch7_guitars", "ch9_drums"],
      "autonomy_mode": "SHADOW_OBSERVE",
      "bypass_latched": false,
      "freeze_learning": false,
      "active_phase": "PRAYER_UNDERSCORE",
      "phase_confidence": 0.94,
      "hysteresis_progress": 1.0,
      "service_step_idx": 2,
      "channels": {
        "ch1_pastor": {
          "armed": false, "sync_status": "SYNCED",
          "vol_current_db": 0.0, "vol_target_db": 0.0, "vol_actual_cc": 96, "vol_target_cc": 96,
          "vol_touch_lock_s": 0.0, "vol_human_offset_db": 0.0, "vol_pickup_dir": 0,
          "reverb_current_db": -20.0, "reverb_target_db": -20.0, "reverb_actual_cc": 43,
          "reverb_touch_lock_s": 0.0, "reverb_human_offset_db": 0.0,
          "feedback_ceiling_db": null, "speech_open": true, "instrument_open": false
        }
      },
      "active_rules": [], "pending_rules": [], "queued_review_count": 0, "hardware_warnings": []
    }
    ```
  * `meter_frame` (`30–50 Hz`, aligned strictly to `state_snapshot.channel_ids[]`, routed exclusively to `MeterStore`):
    `{ "rms_db": [...], "peak_db": [...], "clip": [...], "vad": [...], "master_lufs": -24.1, "vol_target_cc": [...], "vol_actual_cc": [...], "reverb_target_cc": [...], "reverb_actual_cc": [...] }`
  * `brain_event`: `{ "id": "...", "category": "PERCEPTION|ACTION|SAFETY|LLM|RLHF", "summary": "...", "detail": "...", "confidence": 0.92, "ts_ms": 0 }`
  * `heartbeat` (`1 Hz`): `{ "uptime_s": 1240, "loop_p95_ms": 1.8, "loop_p99_ms": 6.2, "audio_ok": true, "bridge_ok": true, "clap_ok": true, "ollama_ok": true }`
* **Client → Server RPCs (all return `ack` / `error`):**
  * `set_mode`, `set_channel_armed`, `restore_previous_arming`, `sync_channel`, `release_human_offset` (`channel_id`, `control`), `set_freeze_learning`, `override_phase`, `advance_service_step`, `feedback_thumb` (`"up"` | `"down"`), `submit_critique`, `resolve_pending_rule` (`rule_id`, `"apply_session"` | `"promote"` | `"dismiss"` — UI `[Apply Session]` / `[Save Permanent]` / `[Dismiss]`), `toggle_rule`, `delete_rule`, `reset_learning`, `commit_queued_reviews`, `capture_room_silence`, `capture_input_nominals` (`group: "spoken" | "band"`), `capture_bleed_floor`, `capture_baseline`, `midi_learn_pulse`, `verify_mute_polarity`.

### 6.5 Standalone Bridge Unix Socket Protocol (`~/.aethermix/bridge.sock`, `0600`)

Newline-delimited JSON (NDJSON) over local `AF_UNIX` stream socket supporting two concurrent clients (`role: "engine"` and `role: "ui"`), authenticated with the shared `~/.aethermix/session.token`:

* **Handshake:** `{ "type": "hello", "role": "engine" | "ui", "token": "<session.token>", "protocol_version": "5.1" }`
* **Bridge → Engine & UI:**
  * `bridge_state`: `{ "last_sent_cc": {"ch1_pastor:vol": 96, ...}, "human_seq": {"ch1_pastor:vol": 4, ...}, "bypass_latched": false, "controller_connected": true, "engine_connected": true, "pickup_pending": {"ch5_keys:vol": -1}, "restored_from_disk_at": null }`
  * `human_touch_event`: `{ "channel_id": "ch5_keys", "control": "vol", "human_seq": 5, "picked_up": false, "source": "hardware", "ts_ms": 0 }` *(sent immediately on pre-pickup fader movement so engine freezes glides before crossover)*
  * `human_cc_event`: `{ "channel_id": "ch5_keys", "control": "vol", "cc": 78, "human_seq": 6, "source": "hardware" | "ui", "ts_ms": 0 }`
  * `action_button_event`: `{ "action": "thumbs_up" | "thumbs_down" | "emergency_bypass", "ts_ms": 0 }`
* **Engine → Bridge:**
  * `agent_cc_write`: `{ "channel_id": "ch5_keys", "control": "vol", "cc": 74, "seq": 1042, "ack_human_seq": 6 }` *(dropped immediately if `bypass_latched == true`, `ack_human_seq < current_human_seq`, or within `250 ms` touch guard)*
  * `midi_learn_pulse`: `{ "channel_id": "ch5_keys", "control": "vol" }`
  * `engine_heartbeat`: `{ "ts_ms": 0 }` *(sent at `>= 20 Hz` or on every `50 Hz` loop tick; `> 500 ms` silence trips `engine_connected = false` inside bridge, holding `last_sent_cc` without zeroing)*
* **UI → Bridge (Direct Manual Control, Works Even When Engine Is Dead!):**
  * `ui_manual_cc`: `{ "channel_id": "ch5_keys", "control": "vol", "cc": 70 }`
  * `set_bypass`: `{ "latched": true | false }`
  * `get_bridge_state`: `{}`

---

## 7. Failure-Mode & Resilience Matrix

| Failure / Edge Case | Guaranteed Behavior |
| :--- | :--- |
| **AI engine crashes, hangs, or trips watchdog** | Bridge detects dead engine within `500 ms`, stops expecting agent frames (holds faders, never zeros), and keeps **both** hardware controller passthrough AND SwiftUI direct bridge faders + Bypass live (`< 2 ms`). SwiftUI flashes `DISCONNECTED — MANUAL CONTROL` with Last Known Faders + 1-click restart & `Restore Previous Arming`. |
| **Bridge daemon itself crashes** | `launchd` (`KeepAlive`, `ThrottleInterval: 1`) restarts `midi_bridge_daemon` in `< 2 s`, reloads persisted `last_sent_cc` (`100 ms` debounce) from `~/.aethermix/bridge_state.json`, and re-attaches to OS-owned IAC bus (or verified persistent virtual port). Engine detects `restored_from_disk_at != null`, drops to `SHADOW_OBSERVE`, and marks controls `UNVERIFIED`. |
| **Hardware MIDI controller unplugged mid-service** | Bridge detects USB-MIDI disconnect within `1 s`, holds MainStage faders steady, alerts `CONTROLLER_DISCONNECTED`, serves SwiftUI direct bridge faders as immediate manual fallback (even if engine is down!), and re-attaches in `pickup_catchup` upon replug. |
| **Operator drags fader with mouse in MainStage GUI** | Invisible unless Phase 0 confirms MainStage feedback. Operating rule: mix via MIDI controller or AetherMix UI while armed; disarm channel before mouse-editing MainStage. |
| **Ollama unreachable, slow, or invalid JSON** | `5.0 s` timeout + Ollama JSON Schema `format` + Pydantic per-parameter validation. Tier 1 mixing & deterministic rules continue uninterrupted. |
| **Band stops, pastor speaks from `-20 dB`** | Tier-1 `speech_open` (`150 ms` spoken/prayer/transition, `400 ms` worship, `48 dB/s`) reaches spoken level in **`< 1.0 s`**. |
| **Sermon ends, full band starts song** | `-16 dB` standby baselines + ensemble-gated `instrument_open` (`24 dB/s`, `200 ms` hold, `8 s` auto-release) opens band in **`< 800 ms`** without false-triggering during prayer or solo guitar tuning. |
| **Operator moves volume or reverb control manually** | Passes straight through (`< 2 ms`), sets per-control `15 s` touch lock (only anti-feedback may cut during touch), freezes `learned_offset_db`, **suspends conditional rules on that control**, respects levels above `hard_max_db`, and clears smoothly at `4 dB/s` on phase/segment change. |
| **Real feedback howl vs. sustained synth pad bleed** | Cross-checks `Ch 5–12` taps to reject synth bleed; real narrowband growth `>= 250 ms` cuts `-4 dB` (escalating to `-16 dB`, overriding touch lock with UI banner) and lifts ceiling at `+1 dB/s` after `5 s` clear. |
| **macOS TCC Microphone permission missing** | Bitwise zero-buffer check (`> 2.0 s`) raises `HARDWARE_SILENCE_WARNING` with `Terminal` vs `AetherMix.app` TCC guidance. |

---

## 8. Packaging, macOS Permissions & Volunteer Operations

* **App-Managed Lifecycle (`EngineProcessManager`):**
  * On launch, `AetherMix.app` verifies (and offers 1-click install of) the `com.aethermix.midibridge.plist` `launchd` LaunchAgent, connects directly to `~/.aethermix/bridge.sock`, and checks if `127.0.0.1:8765` is running (`make dev` support); otherwise spawns/monitors the bundled local `uv` Python engine and streams logs into a Diagnostics drawer.
* **macOS TCC Microphone Permissions:** Under `make dev` (launched from Terminal), macOS grants CoreAudio/BlackHole permission to **Terminal.app**; under standalone `AetherMix.app`, macOS grants permission to **`AetherMix.app`** via `NSMicrophoneUsageDescription` in `Info.plist`. `SetupWizardView` detects active launch context and guides the volunteer explicitly.
* **MainStage Exclusive Controller Check (`SetupWizardView`):** Explicitly verifies and prompts the operator to **remove any direct hardware MIDI controller assignments inside MainStage** so ONLY the AetherMix output port drives Concert-level strips (preventing dual-path conflict with pickup and touch logic).
* **Safe-Update Lock:** Engine, model, and app update actions are **hard-blocked while `autonomy_mode == ACTIVE`** (requires `SHADOW_OBSERVE` or `EMERGENCY_BYPASS`).

---

## 9. Phased Implementation Plan, Measurable Gates & Staged Go-Live

### Phase 0 — MainStage Hardware & Contention Spike (1–2 Days, Mandatory Gate Before Phase 1)
Execute `scripts/phase0_spike.py` against your actual MainStage Concert & M4 Max:
* [ ] Confirm whether MainStage sends MIDI feedback CCs on Concert-level Screen Controls (mouse & external CC).
* [ ] Test **`kill -9` on the bridge process + `launchd` (`ThrottleInterval: 1`) restart**: measure time until controller regains control (**target `< 2 s`**) and confirm whether MainStage keeps responding to a recreated CoreMIDI virtual port without re-learning; if not, lock `mainstage_out_mode = "iac_bus"`.
* [ ] Confirm standalone bridge passthrough (`Hardware Controller → bridge → MainStage`) with `< 2 ms` latency (and verify direct controller mapping is removed in MainStage).
* [ ] Measure empirical `dB ↔ CC` curves (`0..127`) for **both** MainStage Volume AND Bus Reverb Send.
* [ ] Confirm pre-fader sends (`Ch 1–12`) to BlackHole remain invariant when Concert faders move.
* [ ] Benchmark candidate Ollama models (`qwen3:14b` / `qwen3:8b` / `qwen2.5:14b`) + 1-thread Background-QoS CLAP alongside your heaviest MainStage concert at live buffer size for **30 minutes** → **Gate: `0` MainStage CPU/CoreAudio overloads**, LLM p95 JSON latency `< 2.0 s`.

### Phase 1 — Standalone Bridge, Contracts, Audio Tap & SHADOW Recorder
* Build `bridge/midi_bridge_daemon.py` + `launchd` plist (`ThrottleInterval: 1`, `100 ms` atomic `bridge_state.json`, dual token-authenticated `engine`/`ui` socket, `human_seq` + `250 ms` in-flight race guard, pre-pickup `human_touch_event`, `500 ms` dead-engine detect), `Makefile`, `scripts/bootstrap_mac.sh`, `core/protocol.py` (v5.1), `schemas/*.json`, `audio_tap.py`, and `scripts/record_shadow_session.py`.
* **Gate:** `scripts/verify_audio_midi.py` confirms hardware + UI direct passthrough survive `kill -9` of engine; headless `pytest` tests prove an `agent_cc_write` arriving `1 ms` after a human move is dropped and pre-pickup movement freezes agent glides.
* **Start Live SHADOW Recording Immediately:** Record 3–4 real church services in `SHADOW` mode right after Phase 1 completes so Phase 2 has real labeled audio.

### Phase 2 — Safety Firewall, Sensor-Fused Perception, Offset Learner & WS Server
* Implement `safety_firewall.py`, `vad_worker.py`, `clap_worker.py`, `scene_classifier.py`, `offset_learner.py`, `rules_manager.py`, `llm_supervisor.py`, `telemetry_store.py`, and authenticated FastAPI WS server (`main.py`).
* **Quantitative Gates (`pytest` + Phase 1 SHADOW recordings):**
  * Speech-open onset (`-20 dB → 0 dB`): **`< 1.0 s`** (`0` missed pastor speech onsets).
  * Instrument-open onset (`-16 dB → -2 dB`): **`< 800 ms`**, with **`0` false triggers** in `PRAYER_UNDERSCORE` or on solo guitar tuning, duck-capped opening (`-6 dB`) during speech-over-intro, and clean `8 s` auto-release.
  * Tier-1 50 Hz loop latency: target `p95 < 5 ms`, hard gate **`p99 <= 25 ms`**; watchdog drops agent writes to `SHADOW` without interrupting bridge passthrough.
  * Anti-feedback: cuts stationary howl `< 400 ms` with escalation; `0` false triggers on synth pad bleed or clean vocals.
  * Learner & human override invariance: human-set volume/reverb moves `0.00 dB` while human offset is active (even with conditional rules active, weight updates, and varying `phi`), and clears smoothly at `4 dB/s` on phase change.
  * Offline classifier accuracy on recorded services: **`>= 85%` macro F1**.
  * Re-verify 30-minute zero-overload stress gate against heaviest MainStage concert.

### Phase 3 — Native SwiftUI Mission Control App
* Build `macos-app/` (`project.yml` + `Package.swift`) with `EngineWebSocketClient`, **`BridgeSocketClient`** (direct virtual faders + bypass), `EngineProcessManager` (LaunchAgent install/verify + safe-update lock), `MeterCanvasView`, per-channel Arm/Sync + 1-click `Restore Previous Arming`, `AgentBrainStreamView`, `RulesManagerView`, `PostServiceReviewView`, `FeedbackControlBar`, and `SetupWizardView` (4-step sequential Soundcheck Calibration + Mute Polarity check).
* **Gate:** `xcodegen` + `xcodebuild`/`swift build` succeeds; 30–50 Hz meters render smoothly; direct bridge faders & Bypass work with engine killed; full WS token auth + `req_id` ack flow verified end-to-end.

### Phase 4 — Rehearsal Harness & 5-Stage Live Go-Live
* Build `scripts/simulate_service.py` (headless + BlackHole playback) and execute staged rollout with explicit promotion & step-back criteria:

| Stage | Scope | Promotion Gate to Next Stage | Step-Back Trigger |
| :--- | :--- | :--- | :--- |
| **Stage 0** | Live `SHADOW OBSERVE` (3–4 services) | Classifier `>= 85%` macro F1, `0` simulated missed pastor onsets, `0` MainStage overloads | Stay in Stage 0 until offline metrics pass |
| **Stage 1** | Rehearsal `ACTIVE` (full band + pastor) | 2 clean full rehearsals, speech/instrument reflexes verified, pickup & `👎` rollback verified | Fix & repeat rehearsal |
| **Stage 2** | Live: arm **`ch5_keys` only** | 2 consecutive clean live services, `<= 2` `👎` rollbacks/service, `0` overloads | Any overload or `> 4` `👎` → return to Stage 1 |
| **Stage 3** | Live: add **Vocals (`ch2–4`) + Band (`ch7–9`)** | 2 consecutive clean live services, `<= 3` `👎` rollbacks/service, `0` false feedback cuts | Any false lead-vocal cut or `> 5` `👎` → fall back to Stage 2 |
| **Stage 4** | Live: add **`ch1_pastor` (Pastor Mic)** | Ongoing normal operation with Freeze Learning available for holiday/special services | Any buried pastor onset `> 1.5 s` or false cut → disarm `ch1_pastor` (Stage 3) |
