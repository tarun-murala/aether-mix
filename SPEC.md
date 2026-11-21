# Project Specification v4: **AetherMix** (Autonomous Agentic Live Sound Engineer for Apple MainStage)

> **Status:** Draft v4 — Pending Phase 0 Hardware Spike  
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
2. **Control Surface Discipline:** All live fader control flows through the **hardware MIDI controller** (or AetherMix's SwiftUI faders) via the standalone `midi_bridge_daemon`. Unless Phase 0 proves MainStage reliably emits GUI feedback CCs, **direct mouse drags inside MainStage's GUI are invisible to AetherMix**—operators must mix via the controller or AetherMix UI while channels are armed.

### Core Non-Negotiable Constraints

1. **100% Local Execution / Zero Cloud Latency:** All inference (VAD, CLAP scene classification, supervised offset learner, supervisory LLM) and CoreAudio/CoreMIDI I/O run locally on Apple Silicon.
2. **Always-On Manual Control (Process Isolation):** The MIDI bridge (`midi_bridge_daemon`) runs as an **independent, zero-ML `launchd`-supervised process** separate from the AI engine. Hardware controller passthrough and the hardware Emergency Bypass button remain live (`< 2 ms`) even if the AI engine crashes, stalls, hits a watchdog trip, or is in `SHADOW_OBSERVE`.
3. **Deterministic Safety Firewall:** No ML model or LLM ever emits raw MIDI directly. Every agent command passes through a deterministic firewall enforcing calibrated dB/CC bounds, speech-open and instrument-open reflexes, drift-free human ephemeral offsets, 5-path asymmetric slew-rate limits, bleed-rejected anti-feedback protection, and strict no-auto-mute protection on spoken mics.
4. **Fail-Safe Live Reliability:**
   * Engine always boots into **`SHADOW OBSERVE`** (agent writes frozen; human passthrough active) with all channels disarmed (`Restore Previous Arming` available in 1 click once ready).
   * Never sends zero/mute CCs on shutdown, crash, or watchdog trip—MainStage safely holds its last fader state while the standalone bridge preserves authoritative `last_sent_cc` and live manual controller access.
5. **Zero MainStage Audio Contention:** Audio ML threads are hard-pinned (`1` thread, Background-QoS subprocess), and live deployment requires passing a 30-minute zero-overload stress gate alongside the heaviest MainStage concert.

---

## 2. Architectural Decision Records (ADRs)

| Area | Decision | Rationale |
| :--- | :--- | :--- |
| **Standalone MIDI Bridge Process** | **`midi_bridge_daemon`** (zero-ML process supervised by `launchd` `KeepAlive`, Unix Domain Socket IPC to engine) | Guarantees an engine crash, GC pause, or watchdog trip **never** kills hardware controller passthrough or Emergency Bypass, and preserves authoritative `last_sent_cc` across engine restarts. |
| **MIDI Topology & Takeover** | `Hardware Controller → midi_bridge_daemon → MainStage` via CoreMIDI Virtual Port (`AetherMix To MainStage`), with `fader_map` and configurable `pickup_catchup` soft-takeover | Works with any physical controller CC layout, prevents non-motorized fader jumps, enables hardware `👍`/`👎`/Bypass buttons while MainStage has macOS focus, and avoids unverified MainStage feedback dependencies. |
| **Per-Channel Arming & Sync** | Master mode (`SHADOW` / `ACTIVE` / `BYPASS`) + per-channel `armed` + `sync_status`, hydrated from `midi_bridge_daemon.last_sent_cc` | Prevents fader jumps on `SHADOW → ACTIVE` or post-crash reconnect, enables 1-click `Restore Previous Arming`, and supports staged live rollout. |
| **Audio Tap Topology** | **Pre-fader Ch 1–12** (perception + internal gain-normalized sum for CLAP) + **Direct Input Ch 13–14** (room mic) + **Post-fader Ch 15–16 Master** (LUFS metering only) | Keeps VAD, RMS, feedback detection, and CLAP strictly invariant to fader moves, preventing phase lock-in loops. |
| **Gain-Normalized Bleed Calibration** | Soundcheck `capture_input_nominals` + `capture_bleed_floor` ("band plays, nobody speaks") → per-mic `bleed_offset_db` | Eliminates raw-gain-staging bugs where a hot DI keys channel makes `mic - band` negative even during clear speech; `0 dB` SNR reliably means *"only bleed"*. |
| **Bidirectional Fast Reflexes** | Tier-1 **`speech_open` (`48 dB/s`)** AND **`instrument_open` (`24 dB/s`)** + raised standby instrument baselines (`-16 dB`) + segment pre-staging | Prevents both buried pastors (`< 1.0 s` onset) and late band entrances (`< 800 ms` from standby) without weakening 4 s phase hysteresis. |
| **Drift-Free Human Offsets & Learning** | **Per-`(phase, channel)` Ridge Offset Regression** (1 sample per operator event) with **frozen learned offset** while a human ephemeral offset is active | Eliminates double-counting and wandering `phi(S_t)` drift while holding a human adjustment; learner updates take effect cleanly once the human offset clears at the next phase/segment change. |
| **Reverb Sends in dB** | Logarithmic **dB + measured `reverb_cc_taper`** (identical to volume faders) | MainStage bus sends are logarithmic dB controls, not linear percentages. |
| **CPU & Memory Isolation** | **Silero VAD (`onnxruntime`, 1 thread)** + **CLAP (`torch` CPU, 1 thread in Background-QoS subprocess)**; Ollama model benchmarked in Phase 0 | Prevents PyTorch thread pools from preempting MainStage's real-time `AudioWorkgroup` threads; validated by a 30-minute zero-overload stress gate. |
| **LLM Rule Safety** | Strict Ollama JSON Schema; auto-apply only single-channel non-mute within `±3.0 dB`, preview required otherwise; continuous stack cap `[-12, +6 dB]` separate from conditional attenuate/mute | Prevents misheard critiques or stacked rules from slamming vocals; keeps LLM advisory-only on automatic phase/safety triggers. |

---

## 3. Technology Stack & Repository Layout

### Stack
* **Standalone MIDI Bridge (`bridge/`):** Python 3.12 stdlib + `python-rtmidi` ONLY (zero NumPy/PyTorch/FastAPI imports, `< 20 MB` RSS, `< 1 ms` latency), supervised by macOS `launchd` (`KeepAlive: true`, `ProcessType: Interactive`), exposing local Unix Domain Socket (`~/.aethermix/bridge.sock`, `0600`).
* **AI & Real-Time Audio Engine (`engine/`):** Python 3.12 via `uv`, FastAPI + WebSocket (`ws://127.0.0.1:8765/ws`, authenticated via `~/.aethermix/session.token`, `0600`), Pydantic v2, `sounddevice` (PortAudio), `numpy`/`scipy`, `onnxruntime` (1-thread CPU Silero VAD), `transformers` + `torch` (1-thread CPU CLAP in Background-QoS subprocess), `httpx` (async Ollama client), `sqlite3` (WAL mode).
* **Native macOS UI (`macos-app/`):** Swift 6 + SwiftUI (macOS 15+), `XcodeGen` (`project.yml`) + `Package.swift`. Quick-chips and keyboard typing are the primary critique UI; Apple `Speech` dictation is optional (rehearsal use).
* **Audio/MIDI Infrastructure:** BlackHole 16ch + 48 kHz macOS Aggregate Device (physical PA interface as clock master, drift correction enabled on BlackHole), CoreMIDI Virtual Ports + Hardware MIDI Controller.

### Directory Layout

```text
aethermix/
├── SPEC.md                           # This engineering specification (v4)
├── Makefile                          # setup, models, test, spike, install-bridge, build-app, dev, simulate
├── scripts/
│   ├── bootstrap_mac.sh              # Homebrew deps, launchd bridge plist, Ollama pull, HF/Silero cache
│   ├── com.aethermix.midibridge.plist # launchd KeepAlive manifest for standalone MIDI bridge
│   ├── phase0_spike.py               # MainStage feedback, volume/reverb taper sweep, pre-fader & 30m overload spike
│   ├── verify_audio_midi.py          # Aggregate/BlackHole RMS + standalone bridge passthrough smoke test
│   ├── record_shadow_session.py      # Records 16ch pre-fader FLAC/WAV + phase annotations in SHADOW mode
│   └── simulate_service.py           # Synthetic multi-stem church service runner (headless + BlackHole)
├── bridge/
│   └── midi_bridge_daemon.py         # Zero-ML standalone MIDI router, fader_map, pickup takeover, HW bypass, last_sent_cc
├── engine/
│   ├── pyproject.toml
│   ├── main.py                       # Orchestrator: Bridge Client, Audio Tap, VAD, CLAP Subprocess, 50Hz Loop, WS Server
│   ├── core/
│   │   ├── protocol.py               # Versioned Pydantic v4 schemas (WS envelopes, Bridge IPC, LLM schema, rules)
│   │   ├── bridge_client.py          # Async Unix-socket client to midi_bridge_daemon (state hydrate + agent writes)
│   │   ├── audio_tap.py              # Lock-free ring buffer, 50Hz RMS/peak/FFT, bleed-normalized SNR, pre-fader CLAP sum, 10Hz LUFS, TCC guard
│   │   ├── vad_worker.py             # 1-thread ONNX Silero VAD (16kHz) + vocal harmonicity/RMS detector
│   │   ├── clap_worker.py            # Isolated Background-QoS subprocess running 1-thread CPU CLAP at 0.5Hz
│   │   ├── scene_classifier.py       # Pre-fader CLAP + quantitative P_telemetry + Order-of-Service fusion + hysteresis
│   │   ├── offset_learner.py         # Per-(phase, channel) Ridge offset regression, decay, versioned snapshots, reset
│   │   ├── llm_supervisor.py         # Async Ollama client (strict JSON schema format, 5s timeout, advisory vs critique)
│   │   ├── safety_firewall.py        # 5-path slew, speech-open, instrument-open, drift-free human offset, bleed-safe feedback guard, watchdog
│   │   ├── rules_manager.py          # Validated rule patching, ±3dB preview gate, continuous vs conditional caps, atomic backups
│   │   └── telemetry_store.py        # SQLite WAL: 1Hz summaries, discrete events, last_known_state, policy_snapshots, review queue
│   ├── schemas/
│   │   ├── mixing_rules.json         # Versioned baselines (vol & reverb dB), LUFS targets, firewall params, learned rules
│   │   └── service_plan.json         # Order of Service, 16ch map, controller fader_map/buttons, bleed calibration, Ollama config
│   ├── storage/
│   │   ├── telemetry.db              # SQLite WAL database (gitignored)
│   │   └── rule_backups/             # Timestamped atomic backups of mixing_rules.json
│   └── tests/
│       ├── mock_io.py                # Headless deterministic Audio + Bridge IPC simulator
│       ├── test_midi_bridge.py       # Standalone passthrough on engine death, HW bypass, pickup takeover, fader_map, USB unplug
│       ├── test_safety_firewall.py   # Speech-open <1s, instrument-open <800ms, 5-path slew, touch-lock priority, human clamp & non-drift, feedback, watchdog
│       ├── test_scene_fusion.py      # Pre-fader sum invariance, normalized bleed SNR, TRANSITION hold-settle, applause/room mic
│       ├── test_offset_learner.py    # 1-sample updates, no double-counting with human offset, freeze queue, decay, reset
│       ├── test_rules_and_llm.py     # ±3dB preview gate, continuous vs conditional caps, schema bounds, default expires_on
│       └── test_end_to_end.py        # Full synthetic service lifecycle + WS auth/RPC verification
└── macos-app/
    ├── project.yml                   # XcodeGen spec (Info.plist TCC keys, entitlements, macOS 15 target)
    ├── Package.swift                 # SwiftPM manifest for CLI build & model tests
    └── Sources/AetherMix/
        ├── App/AetherMixApp.swift    # Lifecycle, managed engine subprocess, token loading
        ├── Models/
        │   ├── ProtocolModels.swift  # Codable mirror of engine/core/protocol.py (v4)
        │   └── MeterStore.swift      # Isolated 30-60Hz meter buffer keyed by channel_ids[]
        ├── Services/
        │   ├── EngineWebSocketClient.swift  # Token auth, req_id RPC acks, heartbeat monitor, last-state cache
        │   └── EngineProcessManager.swift   # Verifies bridge daemon + spawns/monitors local uv engine
        └── Views/
            ├── MainDashboardView.swift      # Status bar, 1-click Restore Arming, crash banner w/ last-known faders
            ├── MeterCanvasView.swift        # Dedicated 30-60Hz SwiftUI Canvas meters + clip/VAD/speech-open LEDs
            ├── MixerTelemetryView.swift     # Per-channel Arm toggle, dual-ring knobs, pickup indicator, human offset & feedback badges
            ├── AgentBrainStreamView.swift   # Filterable feed + Preview/Pending Rule cards ([Apply Session]/[Save Permanent]/[Dismiss])
            ├── RulesManagerView.swift       # Drawer to inspect, toggle, delete permanent rules & reset phase learning
            ├── PostServiceReviewView.swift  # Review & commit/discard queued samples collected under Freeze Learning
            ├── FeedbackControlBar.swift     # 👍/👎 rollback dock, Freeze Learning toggle, chips & text critique
            └── SetupWizardView.swift        # TCC/BlackHole check, Bridge/Controller map, Mute polarity, Nominals + Bleed + Baseline Capture
```

---

## 4. Model Manifest & Automated Provisioning (`scripts/bootstrap_mac.sh`)

1. **System Dependencies (Homebrew):** `blackhole-16ch`, `ollama`, `uv`, `portaudio`, `xcodegen`.
2. **Standalone Bridge Service:** Installs `scripts/com.aethermix.midibridge.plist` to `~/Library/LaunchAgents/` (`KeepAlive: true`, `ProcessType: Interactive`) so the MIDI bridge stays alive across engine restarts.
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
[Hardware MIDI Controller] ──► [midi_bridge_daemon (launchd KeepAlive)] ──► (Virtual "AetherMix To MainStage") ──► [MainStage]
                                           ▲              │
                          (Unix Socket IPC)│              │ Always-on <2ms human passthrough
                                           ▼              │ + HW Emergency Bypass latch
                                   [AetherMix Engine]     │ + Authoritative last_sent_cc[]
```
* **Always-On Human Passthrough:** Maps incoming controller CCs via `midi.fader_map` (`(hw_channel, hw_cc) → (channel_id, control)`) directly to MainStage's target CC in **every** mode (`SHADOW`, `ACTIVE`, `EMERGENCY_BYPASS`, watchdog trip, or engine crash/disconnect). `SHADOW`, `BYPASS`, and watchdog states freeze **agent writes only**.
* **Bridge-Latched Hardware Actions:** Hardware `emergency_bypass` (`CC 104`) latches directly inside `midi_bridge_daemon`, immediately dropping all agent writes from the Unix socket in `< 1 ms` regardless of engine health. Hardware `👍` (`CC 102`) and `👎` (`CC 103`) forward to the engine immediately (works even when MainStage has macOS focus).
* **Authoritative `last_sent_cc` & Soft-Takeover:**
  * The bridge owns `last_sent_cc[(channel_id, control)]` across engine restarts.
  * Configurable takeover mode (`pickup_catchup` default for non-motorized faders: ignores physical fader until it crosses within `±2 CC` of `last_sent_cc`; `immediate` for motorized faders; `relative_encoder` for endless encoders).
  * USB-MIDI Hot-Plug Guard: polls MIDI port list at 1 Hz; if the hardware controller is unplugged mid-service, holds MainStage state, notifies engine (`CONTROLLER_DISCONNECTED`), accepts SwiftUI virtual faders as immediate fallback, and re-attaches cleanly in `pickup_catchup` mode upon replug.
* **Canonical Full-Range `dB ↔ CC` Tapers (0..127, refined in Phase 0):**
  * **Volume Taper:** `[(0, null), (1, -60.0), (16, -36.0), (32, -24.0), (64, -12.0), (96, 0.0), (114, +6.0), (127, +6.0)]`
  * **Reverb Send Taper (dB, not %):** `[(0, null), (1, -60.0), (32, -24.0), (64, -12.0), (96, 0.0), (114, +6.0), (127, +6.0)]`
  * `CC 0` represents `-inf` (off); `CC 1..114` is strictly monotonic (validated at schema load); `CC 115..127` maps cleanly to `+6.0 dB` (so physical controllers sending `115..127` are representable), and inverse `dB → CC` picks the lowest CC (`114`) for `+6.0 dB`.
* **Mute Safety & Polarity Verification:** Automatic mute CCs are hard-forbidden on `spoken_primary`; `SetupWizardView` mandates a **Mute Polarity Verification** step for all mute-enabled channels.
* **Change Deduplication:** Emits MIDI CC only when quantized integer value changes (or during `MIDI Learn Pulse`).

---

### 5.2 Calibration, Initial State Sync & Handover

#### Two-Step Soundcheck Calibration (`SetupWizardView`)
1. **Step A — `capture_input_nominals` (~15 s during soundcheck):** Records each channel's nominal active voice-band (`300–3400 Hz`) RMS (`nominal_vband_rms_db[ch]`) and wideband RMS (`nominal_rms_db[ch]`) while sources play/speak normally.
2. **Step B — `capture_bleed_floor` (~10 s: "Band Plays, Nobody Speaks"):** Measures per-mic acoustic spill from the band onto `Ch 1–4` in gain-normalized units:
   ```text
   norm_vband_db(ch) = rms_300_3400(ch) - nominal_vband_rms_db[ch]
   bleed_offset_db[m] = clamp(median(norm_vband_db(m) - max_{b in band}(norm_vband_db(b))), -35.0, -3.0)
   ```
   *(Default prior if uncalibrated: `-12.0 dB` with a Setup warning badge.)*
3. **Step C — `capture_baseline(phase, quick_derive_others)`:** Snapshots **`midi_bridge_daemon`'s authoritative `last_sent_cc`** (converted via calibrated taper to dB—never trusting un-synced physical non-motorized fader positions!) into `phase_baselines_db[phase]` and `phase_reverb_baselines_db[phase]`, updating `calibrated_at`.

#### Startup, Sync & Fast Post-Crash Re-Arming
1. Engine always boots into `SHADOW_OBSERVE` with all channels disarmed (`armed = false`) and immediately hydrates `current_db` from `midi_bridge_daemon.get_state()`:
   * Channels for which the standalone bridge already holds an observed `last_sent_cc` are immediately `sync_status = "SYNCED"`.
   * On cold boot (bridge also fresh), channels start `UNVERIFIED` until touched on the controller, captured via baseline, or confirmed via `Confirm & Sync` (`2.0 dB/s` initial glide).
2. **1-Click `Restore Previous Arming`:** Persists the prior `armed` bitmask in SQLite `last_known_state`; after an engine restart, the operator can restore all previously armed channels in 1 click (still requiring an explicit switch from `SHADOW_OBSERVE` → `ACTIVE`).

---

### 5.3 Three-Tier Control Loop

#### Tier 1: Real-Time Reflex & Safety Loop (`50 Hz` / `20 ms` tick)

1. **Decoupled Audio Ingestion (`audio_tap.py`, `vad_worker.py`):**
   * PortAudio callback pushes 48 kHz float32 blocks into a lock-free ring buffer (zero allocations, zero ML).
   * `audio_tap.py` computes at `50 Hz`:
     * Per-channel wideband RMS dBFS, voice-band (`300–3400 Hz`) RMS dBFS, true-peak dBFS, clip flags (`>= -0.5 dBFS`), and 1024-point Hann FFT spectra.
     * **Gain- & Bleed-Normalized Speech SNR (canonical metric):**
       ```text
       expected_bleed_norm_db(m) = max(max_{b in band}(norm_vband_db(b)) + bleed_offset_db[m], mic_noise_floor_norm_db[m])
       speech_to_band_snr_db(m)  = norm_vband_db(m) - expected_bleed_norm_db(m)
       ```
       where `band` = channels with role in `{instrument_pad, instrument_guitar, instrument_band}`. `0 dB` = expected bleed only; `+6 dB` = clear speech above bleed.
     * **Sung-Vocal Activity (`Ch 2–4`):** Computed via gain-normalized `200–4000 Hz` RMS + autocorrelation harmonicity salience (`vocal_snr_db >= +3.0 dB`), NOT Silero VAD.
     * **Pre-Fader CLAP Sum:** Builds a gain-normalized unity sum of **pre-fader `Ch 1–12`** into a 3-second rolling buffer for the CLAP subprocess.
     * **Master LUFS (`10 Hz`):** ITU-R BS.1770 momentary (`400 ms`) & short-term (`3 s`) LUFS strictly on **post-fader `Ch 15–16`**.
     * **TCC Zero-Buffer Guard:** Bitwise all-zero input for `> 2.0 s` raises `HARDWARE_SILENCE_WARNING`.
   * `vad_worker.py`: 1-thread CPU ONNX Silero VAD on 16 kHz downsampled `Ch 1` (and spoken `Ch 2`, `32 ms` hop / ~31 Hz).

2. **Drift-Free Supervised Offset Learner (`offset_learner.py`):**
   * Maintains per-`(phase p, channel i)` Ridge regression `(A_{p,i}, b_{p,i})` predicting **target dB offset `y ∈ [-4.0, +4.0]`** from context `phi_i(S_t)`:
     ```text
     phi_i(S_t) = [norm_vband_db_i / 24.0, vocal_to_band_ratio_norm, speech_or_vocal_active_i, master_lufs_error_norm, 1.0]
     live_learned_db[p][i] = clip(w[p][i]^T * phi_i(S_t), -4.0, +4.0)
     ```
     *(In `TRANSITION`, evaluates using the prior macro phase's weights.)*
   * **Anti-Double-Count Freeze Invariant:** While `human_override_active[i] == true`, the channel uses `frozen_learned_db[i]` (captured at touch time) instead of `live_learned_db[p][i]`. Thus neither updated weights `w` nor changing `phi(S_t)` can drift a human-placed fader! Updated weights apply only after the human offset clears.
   * **Exact 1-Sample Updates (queued to SQLite `pending_review_samples` if `freeze_learning == true`):**
     * **Operator Fader Correction (`alpha = 2.0`):** `y = clip(human_db - (baseline_db[p][i] + continuous_rule_sum_db[p][i]), -4.0, +4.0)` paired with 2-second pre-move mean context `phi_bar`.
     * **Thumbs Up (`👍`, `alpha = 1.0`):** 1 sample per armed channel (`y = current_effective_offset_db`, trailing 5 s mean `phi_bar`).
     * **Thumbs Down (`👎`, `alpha = 1.5`):** 1 sample per rolled-back channel (`y = pre_change_offset_db`, pre-change `phi_bar`).
   * **Decay, Snapshots & Reset:** Inter-service forgetting factor (`gamma = 0.98`), SQLite `policy_snapshots`, and `reset_learning(phase?, channel_id?)`.

3. **Target Assembly & Deterministic Safety Firewall (`safety_firewall.py`):**
   * Continuous vs. Conditional rule separation:
     ```text
     continuous_rule_sum_db  = clip(sum(matching vol_db_offset rules), -12.0, +6.0)
     conditional_delta_db    = clip(sum(triggered conditional_attenuate rules), -36.0, 0.0)
     effective_learned_db    = frozen_learned_db[i] if human_override_active[i] else live_learned_db[p][i]
     base_target_db[i]       = phase_base_db[i] + continuous_rule_sum_db + effective_learned_db + human_session_offset_db[i] + conditional_delta_db
     ```
     *(In `TRANSITION`, `phase_base_db[i]` holds the prior macro phase's baseline with a `-4.0 dB` gentle settle on instrument roles.)*
   * Every `20 ms` tick enforces these invariants in strict precedence order:
     1. **Watchdog & Mode Gate:** Tracks tick latency. If `> 5` consecutive ticks exceed `50 ms` or audio stalls `> 500 ms`, drops engine to `SHADOW_OBSERVE` (freezes **agent writes only**; `midi_bridge_daemon` keeps human passthrough alive) and emits `WATCHDOG_DROP_TO_SHADOW`.
     2. **Touch-Lock Precedence & Persistent Human Offset:**
        * Non-echo human move on channel `i` sets `touch_lock_until[i] = now + 15.0 s`, captures `human_anchor_db[i] = human_db`, freezes `frozen_learned_db[i] = live_learned_db[p][i]`, and sets `human_session_offset_db[i] = human_db - (phase_base_db[i] + continuous_rule_sum_db + frozen_learned_db[i])`.
        * **While `touch_lock_until[i]` is active:** **NO reflexes (`speech_open`, `instrument_open`, ducking, phase glides) may move channel `i`**, EXCEPT escalating **Anti-Feedback Cuts** (announced prominently in UI).
        * **After `touch_lock_until[i]` expires:** `human_session_offset_db[i]` holds the operator's exact setting until the next macro phase transition, segment advance, or manual *Release Offset* click.
     3. **Tier-1 `Speech-Open` Reflex (Fast Pastor Onset, `< 1.0 s`):**
        * For `spoken_primary` channels (when not touch-locked), triggers whenever `vad_prob >= 0.70` AND `speech_to_band_snr_db >= +6.0 dB` sustained for:
          * **`150 ms`** in `SPOKEN_WORD`, `PRAYER_UNDERSCORE`, or `TRANSITION`,
          * **`400 ms`** in `WORSHIP_LOW` or `WORSHIP_HIGH` (matches worship ducking hold to prevent 150 ms bleed false-opens; total onset `400 ms + ~420 ms = ~820 ms < 1.0 s`).
        * Target = `phase_baselines_db["SPOKEN_WORD"][i] + human_session_offset_db[i]`, slewed via **`slew_speech_open_up = 48.0 dB/s`**.
        * Holds until phase confirms `SPOKEN_WORD`/`PRAYER_UNDERSCORE` OR sustained speech silence `>= 3.0 s`, then releases at `4.0 dB/s`.
     4. **Tier-1 `Instrument-Open` Reflex (Fast Song Start, `< 800 ms`):**
        * For instrument channels (`instrument_pad`, `instrument_guitar`, `instrument_band`, when not touch-locked) currently below the next expected worship level (`WORSHIP_LOW` or next Order-of-Service worship segment):
        * Whenever `norm_vband_db[i] >= -12.0 dB` sustained for `>= 200 ms` (ignores stray stick-clicks/plug-pops), opens smoothly at **`slew_instrument_open_up = 24.0 dB/s`** (`-16 dB → -2 dB` in `583 ms`), independent of 4 s phase hysteresis!
        * Additionally, when the operator advances `service_step_idx` to a worship segment, instrument targets pre-stage smoothly toward `WORSHIP_LOW` readiness.
     5. **Bleed-Safe, Role-Configurable Auto-Ducking Prior:**
        * Ducked roles configured per phase via `firewall.duck_targets_by_phase` (default: `["instrument_pad"]` in `PRAYER_UNDERSCORE`/`SPOKEN_WORD`/`TRANSITION`; `["instrument_pad", "instrument_guitar", "instrument_band"]` during worship Pastor Interrupt!).
        * Triggers when `Ch 1` (or `Ch 2` in spoken/prayer) meets `vad_prob >= 0.65` & `speech_to_band_snr_db >= +3.0 dB` (`150 ms`) in spoken/prayer/transition, OR when `Ch 1` `speech_open` is active (`>= 400 ms`) during worship.
        * Runs inside the unified 5-path slew limiter (`48 dB/s` attack → `125 ms` for `-6 dB`; `8 dB/s` release → `750 ms` for `+6 dB`).
     6. **Bleed-Rejected, Escalating Anti-Feedback Detector (Overrides Touch Lock!):**
        * Per mic (`feedback_guard == true`, `Ch 1–4`), detects high-Q FFT peaks `> +14 dB` above local median floor (`±10` bins).
        * **Cross-Channel Bleed Check:** Rejects peak at bin `f` if any instrument tap (`Ch 5–12`) has a matching peak (`±1 bin`) within `6 dB` relative prominence.
        * **Growth / Lock-On Criterion:** Requires persistence `>= 250 ms` with net energy growth `>= +2.0 dB` (or sustained `> +18 dB` narrowband lock-on).
        * **Escalating Cut & Gradual Release:** Cuts `-4.0 dB` (`48 dB/s`), escalates `-4.0 dB` every `400 ms` up to `-16.0 dB` if ringing persists, holds sticky `feedback_ceiling_db[i]` until `5.0 s` clear, then lifts at `+1.0 dB/s`.
     7. **Unified 5-Path Float Slew-Rate Limiter & Human-Respecting Clamps:**
        * `slew_protect_down = 48.0 dB/s` (anti-feedback & ducking attack)
        * `slew_speech_open_up = 48.0 dB/s` (`spoken_primary` speech-open reflex)
        * `slew_instrument_open_up = 24.0 dB/s` (instrument song-start reflex)
        * `slew_duck_release_up = 8.0 dB/s` (auto-ducking release)
        * `slew_musical = 4.0 dB/s` (normal glides; `2.0 dB/s` on unverified sync handover)
        * **Human-Respecting Hard Clamps:** Agent-originated moves clamp strictly to `[hard_min_db, hard_max_db]`; if a human deliberately set `human_anchor_db[i] > hard_max_db`, effective upper bound becomes `max(hard_max_db, human_anchor_db[i])` until the human offset clears (anti-feedback ceiling still caps above all).

---

#### Tier 2: Semantic Sensor Fusion & LLM Supervisor (`0.5 Hz` / `2 s` + event triggers)

* **Isolated CLAP Subprocess (`clap_worker.py`):** Spawned (`multiprocessing.spawn`), `torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`, macOS Background QoS (`PRIO_DARWIN_BG` / `os.nice(10)`). Classifies the 3-second rolling **pre-fader sum (`Ch 1–12`)** every `2.0 s`.
* **Quantitative Telemetry Evidence (`P_telemetry`):** Evaluates calibrated `norm_vband_db` (`active >= -18 dB` rel. nominal, `silent <= -30 dB` rel. nominal), `speech_to_band_snr_db`, sung-vocal harmonicity (`Ch 2–4`), and `ch13_room` broadband spectral flatness (`> 0.55` for applause/transitions).
* **Fusion & Hysteresis:** `0.40 * P_clap + 0.40 * P_telemetry + 0.20 * P_service_plan` (or `0.50 / 0.50` if Order-of-Service tracking disabled); requires 2 consecutive `2 s` windows (`4.0 s`) at `>= 0.70` confidence. `TRANSITION` holds prior macro phase with `-4 dB` instrument settle.
* **Async Ollama Supervisor (`llm_supervisor.py`):** Warm-pinned (`keep_alive: "60m"`), strict JSON Schema `format`, `5.0 s` timeout. **Advisory-only** on `phase_transition` and `safety_event` (`proposed_patches` forced empty in code); emits typed patches only on `operator_critique`.

---

#### Tier 3: Human-in-the-Loop Adaptation & Rule Lifecycle (`rules_manager.py`)

* **`👍` / `👎` & Operator Fader Corrections:** All three feed 1 sample per event into `offset_learner.py` (or queue into SQLite `pending_review_samples` when `freeze_learning == true`). `👎` rolls back non-touch-locked armed channels to `T - 5 s` (second press within `5 s` steps back to `T - 15 s` / pre-transition).
* **Natural-Language Critique & Rule Promotion:**
  * Single-channel `vol_db_offset` within `[-3.0, +3.0] dB` (or `reverb_db_offset` within `[-4.0, +4.0] dB`) auto-applies as ephemeral session rule and shows **`[Save Permanent]`** (`promote`) / **`[Dismiss]`** (`dismiss`).
  * Any patch exceeding `±3.0 dB` vol / `±4.0 dB` reverb, any multi-channel patch, or any conditional attenuate/mute requires **`PREVIEW REQUIRED`** approval (**`[Apply Session]`** → `apply_session` | **`[Save Permanent]`** → `promote` | **`[Dismiss]`** → `dismiss`).
  * `expires_on` defaults to `"segment_change"` for `ephemeral_session` rules. Conflicts on same `(phase, channel_id, parameter, condition)` resolve by **most-recent rule wins**. Permanent rules write atomically with backups in `engine/storage/rule_backups/`.

---

## 6. Reference Schemas & IPC Protocols (v4)

### 6.1 `engine/schemas/service_plan.json`

```json
{
  "schema_version": 4,
  "sample_rate": 48000,
  "audio_device_hint": "BlackHole 16ch",
  "ollama_model": "qwen3:14b",
  "ollama_think": false,
  "midi": {
    "hardware_in_port": "X-Touch Mini",
    "takeover_mode": "pickup_catchup",
    "mainstage_out_port": "AetherMix To MainStage",
    "mainstage_feedback_in_port": null,
    "fader_map": [
      {"hw_channel": 1, "hw_cc": 1, "channel_id": "ch1_pastor",  "control": "vol"},
      {"hw_channel": 1, "hw_cc": 2, "channel_id": "ch2_lead_vox","control": "vol"},
      {"hw_channel": 1, "hw_cc": 3, "channel_id": "ch3_bv1",     "control": "vol"},
      {"hw_channel": 1, "hw_cc": 4, "channel_id": "ch4_bv2",     "control": "vol"},
      {"hw_channel": 1, "hw_cc": 5, "channel_id": "ch5_keys",    "control": "vol"},
      {"hw_channel": 1, "hw_cc": 6, "channel_id": "ch7_guitars", "control": "vol"},
      {"hw_channel": 1, "hw_cc": 7, "channel_id": "ch9_drums",   "control": "vol"},
      {"hw_channel": 1, "hw_cc": 11,"channel_id": "ch1_pastor",  "control": "reverb"},
      {"hw_channel": 1, "hw_cc": 12,"channel_id": "ch2_lead_vox","control": "reverb"}
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
      "bleed_offset_db": -16.0,
      "midi_channel": 1,
      "vol_cc": 20,
      "reverb_cc": 30,
      "mute_cc": null,
      "hard_min_db": -60.0,
      "hard_max_db": 3.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0], [127, 6.0]],
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
      "bleed_offset_db": -14.0,
      "midi_channel": 1,
      "vol_cc": 21,
      "reverb_cc": 31,
      "mute_cc": 41,
      "mute_active_high": true,
      "hard_min_db": -60.0,
      "hard_max_db": 3.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0], [127, 6.0]],
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
      "bleed_offset_db": -14.0,
      "midi_channel": 1,
      "vol_cc": 22,
      "mute_cc": 42,
      "mute_active_high": true,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0], [127, 6.0]]
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
      "bleed_offset_db": -14.0,
      "midi_channel": 1,
      "vol_cc": 23,
      "mute_cc": 43,
      "mute_active_high": true,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0], [127, 6.0]]
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
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0], [127, 6.0]]
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
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0], [127, 6.0]]
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
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0], [127, 6.0]]
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

*(Note: `TRANSITION` has no fixed baseline rows—it is a hold & `-4 dB` instrument settle state. Pydantic validates baseline completeness across all 4 macro phases for every controlled channel.)*

```json
{
  "schema_version": 4,
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
    "instrument_open_norm_db": -12.0,
    "instrument_open_hold_ms": 200,
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

### 6.4 WebSocket IPC Protocol (`ws://127.0.0.1:8765/ws`, v4)

* **Auth & Handshake:** Bearer token from `~/.aethermix/session.token` (`0600`) + `hello { protocol_version: "4.0" }`.
* **Envelope:** `{ "type": "<str>", "seq": <int>, "req_id": "<uuid|null>", "ts_ms": <int>, "payload": { ... } }`
* **UI Label → Protocol Action Mapping:**
  * `[Apply Session]` → `resolve_pending_rule(..., "apply_session")`
  * `[Save Permanent]` → `resolve_pending_rule(..., "promote")`
  * `[Dismiss]` → `resolve_pending_rule(..., "dismiss")`
* **Key RPCs (all return `ack` / `error`):** `set_mode`, `set_channel_armed`, `restore_previous_arming`, `sync_channel`, `release_human_offset`, `set_freeze_learning`, `override_phase`, `advance_service_step`, `feedback_thumb`, `submit_critique`, `resolve_pending_rule`, `toggle_rule`, `delete_rule`, `reset_learning`, `commit_queued_reviews`, `capture_input_nominals`, `capture_bleed_floor`, `capture_baseline`, `midi_learn_pulse`, `verify_mute_polarity`, `manual_channel_move`.

---

## 7. Failure-Mode & Resilience Matrix

| Failure / Edge Case | Guaranteed Behavior |
| :--- | :--- |
| **AI engine crashes, hangs, or trips watchdog** | `midi_bridge_daemon` stays alive (`launchd KeepAlive`), keeping `< 2 ms` hardware controller passthrough & HW Emergency Bypass working continuously. Agent writes stop; MainStage holds faders; SwiftUI flashes `DISCONNECTED — MANUAL CONTROL` showing cached **Last Known Faders** + 1-click restart & `Restore Previous Arming`. |
| **Hardware MIDI controller unplugged mid-service** | Bridge detects USB-MIDI disconnect within `1 s`, holds MainStage faders steady, alerts `CONTROLLER_DISCONNECTED`, allows SwiftUI virtual faders as immediate manual control, and auto-reattaches in `pickup_catchup` mode upon replug. |
| **Operator drags fader with mouse inside MainStage GUI** | If Phase 0 shows no MainStage feedback port, mouse drags are invisible to AetherMix. Operating rule: mix via MIDI controller or AetherMix UI while armed; disarm channel before mouse-editing MainStage. |
| **Ollama unreachable, slow, or emits invalid JSON** | `5.0 s` timeout + Ollama JSON Schema `format` + Pydantic per-parameter validation. Tier 1 mixing, reflexes, and deterministic rules run uninterrupted; degraded-LLM badge shown in UI. |
| **Band stops, pastor starts speaking from `-20 dB`** | Tier-1 `speech_open` reflex raises pastor mic at `48 dB/s` (`150 ms` hold in spoken/prayer/transition, `400 ms` in worship) → full spoken level in **`< 1.0 s`** (`~570–820 ms`). |
| **Sermon ends, band kicks straight into loud song** | Instrument standby baselines (`-16 dB`) + Tier-1 `instrument_open` reflex (`24 dB/s`, `200 ms` hold) bring band up in **`< 800 ms`** (plus segment pre-staging when operator advances Order of Service). |
| **Operator adjusts fader manually (even above `hard_max_db`)** | Passed straight through to MainStage (`< 2 ms`); sets `15 s` touch lock (only anti-feedback may cut during touch), freezes `learned_offset_db` (zero drift!), holds human level above `hard_max_db` until phase/segment change, logs 1 high-weight supervised sample. |
| **Sustained synth/keys pad bleeds into vocal mic** | Anti-feedback cross-checks FFT peak bin against `Ch 5–12` instrument taps; matching peaks are rejected as bleed. |
| **Real acoustic feedback howl develops** | Narrowband growth `>= 250 ms` triggers `-4 dB` cut (`48 dB/s`, even during touch lock with UI banner), escalates every `400 ms` up to `-16 dB`, holds sticky ceiling for `5 s` clear, then lifts at `+1 dB/s`. |
| **macOS TCC Microphone permission missing** | Bitwise zero-buffer check (`> 2.0 s`) raises `HARDWARE_SILENCE_WARNING` with `Terminal` vs `AetherMix.app` TCC guidance. |

---

## 8. Phased Implementation Plan, Measurable Gates & Staged Go-Live

### Phase 0 — MainStage Hardware & Contention Spike (1–2 Days, Mandatory Gate Before Phase 1)
Execute `scripts/phase0_spike.py` against your actual MainStage Concert & M4 Max:
* [ ] Confirm whether MainStage sends MIDI feedback CCs on Concert-level Screen Controls (mouse & external CC).
* [ ] Confirm **standalone bridge MIDI-in-the-middle** routing (`Hardware Controller → midi_bridge_daemon → MainStage`) with `< 2 ms` passthrough.
* [ ] Measure empirical `dB ↔ CC` curves for **both** MainStage channel strip Volume AND Bus Reverb Send (`0..127`).
* [ ] Confirm pre-fader sends (`Ch 1–12`) to BlackHole remain invariant when Concert faders move.
* [ ] Benchmark candidate Ollama models (`qwen3:14b` / `qwen3:8b` / `qwen2.5:14b`) + 1-thread Background-QoS CLAP alongside your heaviest MainStage concert at live buffer size for **30 minutes** → **Gate: `0` MainStage CPU/CoreAudio overloads**, LLM p95 JSON latency `< 2.0 s`.

### Phase 1 — Standalone Bridge, Contracts, Audio Tap & SHADOW Recorder
* Build `bridge/midi_bridge_daemon.py` + `launchd` plist, `Makefile`, `scripts/bootstrap_mac.sh`, `core/protocol.py` (v4), `schemas/*.json`, `audio_tap.py` (lock-free ring buffer, bleed-normalized SNR, pre-fader CLAP sum, LUFS, TCC guard), and `scripts/record_shadow_session.py`.
* **Gate:** `scripts/verify_audio_midi.py` confirms live passthrough survives `kill -9` of engine; headless `pytest` bridge & schema tests pass.
* **Start Live SHADOW Recording Immediately:** Begin recording 3–4 real church services in `SHADOW` mode right after Phase 1 completes so Phase 2 has real labeled audio!

### Phase 2 — Safety Firewall, Sensor-Fused Perception, Offset Learner & WS Server
* Implement `safety_firewall.py`, `vad_worker.py`, `clap_worker.py`, `scene_classifier.py`, `offset_learner.py`, `rules_manager.py`, `llm_supervisor.py`, `telemetry_store.py`, and authenticated FastAPI WS server (`main.py`).
* **Quantitative Gates (`pytest` + Phase 1 SHADOW recordings):**
  * Speech-open onset (`-20 dB → 0 dB`): **`< 1.0 s`** (`0` missed pastor speech onsets).
  * Instrument-open onset (`-16 dB → -2 dB`): **`< 800 ms`**.
  * Tier-1 50 Hz loop latency: target `p95 < 5 ms`, hard gate **`p99 <= 25 ms`**; watchdog drops agent writes to `SHADOW` without interrupting bridge human passthrough.
  * Anti-feedback: cuts stationary howl `< 400 ms` with escalation; `0` false triggers on sustained synth pad bleed or clean vocals.
  * Learner non-drift: human-set fader moves `0.00 dB` while human offset is active despite weight updates and varying `phi(S_t)`.
  * Offline classifier accuracy on recorded services: **`>= 85%` macro F1**.
  * Re-verify 30-minute zero-overload stress gate against heaviest MainStage concert.

### Phase 3 — Native SwiftUI Mission Control App
* Build `macos-app/` (`project.yml` + `Package.swift`) with `MeterCanvasView`, per-channel Arm/Sync + 1-click `Restore Previous Arming`, `AgentBrainStreamView`, `RulesManagerView`, `PostServiceReviewView`, `FeedbackControlBar`, `SetupWizardView` (including `capture_input_nominals`, `capture_bleed_floor`, `capture_baseline`, and Mute Polarity check), and `DISCONNECTED` Last-Known-Faders banner.
* **Gate:** `xcodegen` + `xcodebuild`/`swift build` succeeds; 30–50 Hz meters render smoothly; boots in `SHADOW_OBSERVE` with channels disarmed; full WS token auth + `req_id` ack flow verified end-to-end.

### Phase 4 — Rehearsal Harness & 5-Stage Live Go-Live
* Build `scripts/simulate_service.py` (headless + BlackHole playback) and execute staged rollout with explicit promotion & rollback criteria:

| Stage | Scope | Promotion Gate to Next Stage | Step-Back Trigger |
| :--- | :--- | :--- | :--- |
| **Stage 0** | Live `SHADOW OBSERVE` (3–4 services) | Classifier `>= 85%` macro F1, `0` simulated missed pastor onsets, `0` MainStage overloads | Stay in Stage 0 until offline metrics pass |
| **Stage 1** | Rehearsal `ACTIVE` (full band + pastor) | 2 clean full rehearsals, speech/instrument reflexes verified, pickup & `👎` rollback verified | Fix & repeat rehearsal |
| **Stage 2** | Live: arm **`ch5_keys` only** | 2 consecutive clean live services, `<= 2` `👎` rollbacks/service, `0` overloads | Any overload or `> 4` `👎` → return to Stage 1 |
| **Stage 3** | Live: add **Vocals (`ch2–4`) + Band (`ch7–9`)** | 2 consecutive clean live services, `<= 3` `👎` rollbacks/service, `0` false feedback cuts | Any false lead-vocal cut or `> 5` `👎` → fall back to Stage 2 |
| **Stage 4** | Live: add **`ch1_pastor` (Pastor Mic)** | Ongoing normal operation with Freeze Learning available for holiday/special services | Any buried pastor onset `> 1.5 s` or false cut → disarm `ch1_pastor` (Stage 3) |
