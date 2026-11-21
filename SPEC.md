# Project Specification v3: **AetherMix** (Autonomous Agentic Live Sound Engineer for Apple MainStage)

> **Status:** Draft v3 — Pending Phase 0 Hardware Spike  
> **Target Hardware:** Apple Silicon macOS 15+ (optimized for M4 Max, 128 GB Unified Memory)  
> **Host Integration:** Apple MainStage (Concert-Level Audio Channel Strips)

---

## 1. Executive Summary & System Goals

Build **AetherMix**, a native macOS application and local multi-model AI daemon that runs alongside **Apple MainStage** to autonomously mix live multi-channel audio (pulpit/spoken mics, vocals, keys/pads, guitars, drums/bass/tracks) in real time across dynamic church service phases:

* `SPOKEN_WORD`
* `PRAYER_UNDERSCORE`
* `WORSHIP_LOW`
* `WORSHIP_HIGH`
* `TRANSITION` *(hold / gentle-settle state)*

AetherMix sits **inline between the physical MIDI controller (or SwiftUI virtual faders) and MainStage**, maintaining an authoritative source of truth for fader state. It boots safely into **`SHADOW OBSERVE`**, supports **per-channel arming** for staged live rollouts, reacts to speech onset in **`< 1.0 s`** via a deterministic Tier-1 reflex, explains every decision in a live **Agent Brain** UI, and adapts safely from operator fader corrections, `👍`/`👎` feedback, and validated natural-language rules.

### Core Non-Negotiable Constraints

1. **100% Local Execution / Zero Cloud Latency:** All inference (VAD, CLAP scene classification, supervised offset learner, supervisory LLM) and CoreAudio/CoreMIDI I/O run strictly locally on Apple Silicon.
2. **Deterministic Safety Firewall:** No ML model or LLM ever emits raw MIDI directly. Every command passes through a deterministic firewall enforcing calibrated dB/CC bounds, speech-open reflex protection, human ephemeral offsets, 4-path asymmetric slew-rate limits, bleed-rejected anti-feedback protection, and strict no-auto-mute protection on spoken mics.
3. **Fail-Safe Live Reliability:**
   * Always boots into **`SHADOW OBSERVE`** (read-only) with all channels disarmed until explicitly armed/synced.
   * Never sends zero/mute CCs on shutdown, crash, or watchdog trip—MainStage safely holds its last fader state while SwiftUI flashes `DISCONNECTED — MANUAL CONTROL` along with the last known channel fader values.
4. **Zero MainStage Audio Contention:** Audio ML threads are hard-pinned (`1` thread, background QoS subprocess), and live deployment requires passing a 30-minute zero-overload stress gate alongside the heaviest MainStage concert.

---

## 2. Architectural Decision Records (ADRs)

| Area | Decision | Rationale |
| :--- | :--- | :--- |
| **MIDI Topology** | **AetherMix-in-the-Middle:** `Hardware Controller → AetherMix → MainStage` (via native CoreMIDI Virtual Port `AetherMix To MainStage`), with optional secondary feedback port `AetherMix From MainStage` if Phase 0 proves MainStage emits screen-control CCs | Guarantees authoritative fader state, deterministic touch detection, `< 2 ms` human passthrough, hardware `👍`/`👎`/Bypass buttons even when MainStage has macOS focus, and eliminates echo loops/pickup jumps (via configurable soft-takeover). |
| **Per-Channel Arming & Handover** | Master mode (`SHADOW` / `ACTIVE` / `BYPASS`) + per-channel `armed` switch + explicit state sync handover | Prevents fader jumps on `SHADOW → ACTIVE` or post-crash reconnect, and enables safe staged live rollout (Keys/Pad first → Vocals/Band → Pastor mic last). |
| **Audio Tap Topology** | **Pre-fader Ch 1–12** (perception + **internal unity-gain sum for CLAP**) + optional **Ch 13–14 Room Mic** + **Post-fader Ch 15–16 Master** (LUFS metering only) | Keeps VAD, RMS, feedback detection, *and* CLAP completely invariant to fader moves, preventing phase lock-in loops and keeping attenuated mics audible to perception. |
| **Speech Onset Reflex** | **Tier-1 VAD `speech_open` reflex** (`48 dB/s` fast up-slew, independent of CLAP phase hysteresis) | Eliminates the 10–15 s buried-pastor failure mode; brings a standby pastor mic (`-20 dB`) to `0 dB` in `< 600 ms` while phase hysteresis confirms over 4 s. |
| **Online Adaptation** | **Phase- & Channel-Scoped Ridge Offset Regression** (supervised preference learning, 1 sample per operator event) | Replaces mis-specified bandit/SAC formulations: trains directly on target dB offsets (primarily operator fader moves + 1 sample per `👍`/`👎`), never explores randomly live, never swamps L2 regularization, and supports versioning/decay/reset. |
| **Human Override Persistence** | **Touch debounce (`15 s` write block) + Human Ephemeral Offset** held until next phase transition or segment advance | Prevents the agent from undoing an operator's manual fader move 15 s later and fighting the sound engineer mid-song. |
| **CPU & Memory Isolation** | **Silero VAD (`onnxruntime`, 1 thread)** + **CLAP (`torch` CPU, `1` thread in isolated Background-QoS subprocess)**; Ollama model benchmarked in Phase 0 | Prevents PyTorch thread pools from preempting MainStage's real-time `AudioWorkgroup` threads and enforces a hard zero-overload MainStage gate under simultaneous LLM + CLAP load. |
| **LLM Rule Safety** | **Strict Ollama JSON Schema**; auto-apply only within `±3.0 dB` (non-mute), preview required otherwise; hard stack cap `[-12, +6 dB]` | Prevents misheard critiques or stacked rules from slamming vocals mid-song; keeps LLM strictly advisory on automatic phase/safety triggers. |
| **Internal Control Units** | **Float dB internally** → strictly monotonic calibrated `dB ↔ 7-bit CC` curve (`CC 0 = -inf`) | Eliminates 7-bit EMA rounding stalls and accurately maps logarithmic MainStage faders; real curve measured empirically in Phase 0. |
| **SwiftUI Build System** | **SwiftPM (`Package.swift`) + `XcodeGen` (`project.yml`)** | Reproducible CLI/Makefile `.app` bundle builds with required macOS TCC `Info.plist` entries. |

---

## 3. Technology Stack & Repository Layout

### Stack
* **Native macOS UI (`macos-app/`):** Swift 6 + SwiftUI (macOS 15+), `XcodeGen` (`project.yml`) + `Package.swift`. Quick-chips and keyboard typing are the primary critique UI; Apple `Speech` dictation is optional (rehearsal/quiet-booth use).
* **AI & Real-Time Audio Daemon (`engine/`):** Python 3.12 via `uv`, FastAPI + WebSocket (`ws://127.0.0.1:8765/ws`, authenticated via local `0600` bearer token `~/.aethermix/session.token`), Pydantic v2, `sounddevice` (PortAudio), `python-rtmidi` (CoreMIDI), `numpy`/`scipy`, `onnxruntime` (1-thread CPU Silero VAD), `transformers` + `torch` (1-thread CPU CLAP in Background-QoS subprocess), `httpx` (async Ollama client), `sqlite3` (WAL mode).
* **Audio/MIDI Infrastructure:** BlackHole 16ch + 48 kHz macOS Aggregate Device (physical PA interface as clock master, drift correction enabled on BlackHole), CoreMIDI Virtual Ports + Hardware Controller MIDI input.

### Directory Layout

```text
aethermix/
├── SPEC.md                           # This engineering specification (v3)
├── Makefile                          # setup, models, test, spike, build-app, dev, simulate
├── scripts/
│   ├── bootstrap_mac.sh              # Homebrew deps, Ollama model pull, HF/Silero model cache
│   ├── phase0_spike.py               # MainStage MIDI feedback, taper sweep, pre-fader & overload stress spike
│   ├── verify_audio_midi.py          # Aggregate/BlackHole RMS + MIDI-in-the-middle smoke test
│   ├── record_shadow_session.py      # Records 16ch pre-fader FLAC/WAV + phase annotations in SHADOW mode
│   └── simulate_service.py           # Synthetic multi-stem church service runner (headless + BlackHole)
├── engine/
│   ├── pyproject.toml
│   ├── main.py                       # Orchestrator: Audio Tap, VAD Worker, CLAP Subprocess, 50Hz Loop, WS Server
│   ├── core/
│   │   ├── protocol.py               # Versioned Pydantic v3 schemas (WS envelopes, RPC acks, LLM schema, rules)
│   │   ├── audio_tap.py              # Lock-free ring buffer, 50Hz RMS/peak/FFT, pre-fader CLAP sum, 10Hz LUFS, TCC guard
│   │   ├── vad_worker.py             # 1-thread ONNX Silero VAD (16kHz) + 300-3400Hz speech_to_band_snr + vocal activity
│   │   ├── clap_worker.py            # Isolated Background-QoS subprocess running 1-thread CPU CLAP at 0.5Hz
│   │   ├── scene_classifier.py       # Pre-fader CLAP + quantitative P_telemetry + Order-of-Service fusion + hysteresis
│   │   ├── offset_learner.py         # Per-(phase, channel) Ridge offset regression, decay, versioned snapshots, reset
│   │   ├── llm_supervisor.py         # Async Ollama client (strict JSON schema format, 5s timeout, advisory vs critique)
│   │   ├── safety_firewall.py        # 4-path slew, speech-open reflex, human ephemeral offset, bleed-safe feedback guard, watchdog
│   │   ├── midi_bridge.py            # MIDI-in-the-middle router, soft-takeover pickup, echo filter, HW 👍/👎 buttons, dedup
│   │   ├── rules_manager.py          # Validated rule patching, ±3dB auto/preview gate, stack cap, conflict resolution, atomic backups
│   │   └── telemetry_store.py        # SQLite WAL: 1Hz summaries, discrete events, last_known_state, policy_snapshots, review queue
│   ├── schemas/
│   │   ├── mixing_rules.json         # Versioned baselines, LUFS targets, firewall params, learned rules
│   │   └── service_plan.json         # Order of Service, 16ch map, MIDI controller/button bindings, Ollama model config
│   ├── storage/
│   │   ├── telemetry.db              # SQLite WAL database (gitignored)
│   │   └── rule_backups/             # Timestamped atomic backups of mixing_rules.json
│   └── tests/
│       ├── mock_io.py                # Headless deterministic Audio + MIDI simulator
│       ├── test_safety_firewall.py   # Bounds, speech-open <1s, 4-path slew, human offset, bleed-safe feedback, watchdog, rollback
│       ├── test_scene_fusion.py      # Pre-fader sum invariance, hysteresis, applause/room mic, vocal harmonicity
│       ├── test_offset_learner.py    # 1-sample updates, human-target regression, decay, versioning, phase reset
│       ├── test_rules_and_llm.py     # ±3dB preview gate, stack cap, conditional rules, schema rejection
│       └── test_end_to_end.py        # Full synthetic service lifecycle + WS auth/RPC verification
└── macos-app/
    ├── project.yml                   # XcodeGen spec (Info.plist TCC keys, entitlements, macOS 15 target)
    ├── Package.swift                 # SwiftPM manifest for CLI build & model tests
    └── Sources/AetherMix/
        ├── App/AetherMixApp.swift    # Lifecycle, managed engine subprocess, token loading
        ├── Models/
        │   ├── ProtocolModels.swift  # Codable mirror of engine/core/protocol.py (v3)
        │   └── MeterStore.swift      # Isolated 30-60Hz meter buffer keyed by channel_ids[]
        ├── Services/
        │   ├── EngineWebSocketClient.swift  # Token auth, req_id RPC acks, heartbeat monitor, last-state cache
        │   └── EngineProcessManager.swift   # Spawns/monitors local uv engine + Diagnostics log buffer
        └── Views/
            ├── MainDashboardView.swift      # Status bar, arm/sync controls, crash banner w/ last-known faders
            ├── MeterCanvasView.swift        # Dedicated 30-60Hz SwiftUI Canvas meters + clip/VAD/speech-open LEDs
            ├── MixerTelemetryView.swift     # Per-channel Arm toggle, dual-ring knobs, human offset & feedback badges
            ├── AgentBrainStreamView.swift   # Filterable feed + Preview/Pending Rule cards ([Apply Session]/[Save Permanent]/[Dismiss])
            ├── RulesManagerView.swift       # Drawer to inspect, toggle, delete permanent rules & reset phase learning
            ├── PostServiceReviewView.swift  # Review & commit/discard queued samples collected under Freeze Learning
            ├── FeedbackControlBar.swift     # 👍/👎 rollback dock, Freeze Learning toggle, chips & text critique
            └── SetupWizardView.swift        # TCC/BlackHole check, MIDI-in-the-middle + Mute polarity check, MIDI Learn, Baseline Capture
```

---

## 4. Model Manifest & Automated Provisioning (`scripts/bootstrap_mac.sh`)

1. **System Dependencies (Homebrew):** `blackhole-16ch`, `ollama`, `uv`, `portaudio`, `xcodegen`.
2. **Supervisory Reasoning LLM (Ollama — configurable in `service_plan.json`):**
   * Default primary candidate: benchmark current local structured-JSON models in Phase 0 (e.g., `qwen3:14b` with thinking disabled / `qwen3:8b` / `qwen2.5:14b`) and select the fastest model that passes JSON-schema reliability + zero MainStage overload.
   * Optional offline/post-service analysis model (`--with-32b` flag): `qwen3:32b` or `qwen2.5:32b`.
3. **Semantic Scene Classifier (`~/.aethermix/models/clap/`):**
   * `laion/clap-htsat-unfused` via Hugging Face `transformers` (1-thread CPU in Background-QoS subprocess).
   * Prompts:
     * `"pastor or speaker talking alone without music"` → `SPOKEN_WORD`
     * `"soft ambient keyboard pad underneath spoken prayer"` → `PRAYER_UNDERSCORE`
     * `"acoustic guitar or piano gentle worship verse"` → `WORSHIP_LOW`
     * `"full church band with drums electric guitar and loud singing"` → `WORSHIP_HIGH`
     * `"silence applause or stage transition between songs"` → `TRANSITION`
4. **Voice Activity Detection (`~/.aethermix/models/silero_vad.onnx`):**
   * `silero-vad` ONNX executed via `onnxruntime` (`intra_op_num_threads=1`, `inter_op_num_threads=1`) at 16 kHz / 512-sample chunks (`32 ms`).

---

## 5. Detailed Subsystem Architecture

### 5.1 Audio & MIDI Routing Specification

#### BlackHole 16ch Tap Map (`service_plan.json`)
MainStage drives a **48 kHz macOS Aggregate Device** combining the physical PA interface (clock master) and **BlackHole 16ch** (drift correction enabled). All AetherMix-controlled strips reside at MainStage's **Concert Level**.

| BlackHole Ch | Tap Type | Logical Strip (`channel_id`) | Role | Controlled MIDI Params (Ch 1 default) |
| :--- | :--- | :--- | :--- | :--- |
| `Ch 1` (mono) | Pre-Fader | `ch1_pastor` (Pastor / Pulpit) | `spoken_primary` | `CC 20` Vol, `CC 30` Reverb Send *(auto hard-mute forbidden)* |
| `Ch 2` (mono) | Pre-Fader | `ch2_lead_vox` (Worship Leader) | `vocal_lead` | `CC 21` Vol, `CC 31` Reverb Send, `CC 41` Mute |
| `Ch 3` (mono) | Pre-Fader | `ch3_bv1` (Backup Vocal 1) | `vocal_bg` | `CC 22` Vol, `CC 42` Mute |
| `Ch 4` (mono) | Pre-Fader | `ch4_bv2` (Backup Vocal 2) | `vocal_bg` | `CC 23` Vol, `CC 43` Mute |
| `Ch 5–6` (stereo) | Pre-Fader | `ch5_keys` (Keys / Pad) | `instrument_pad` | `CC 24` Vol |
| `Ch 7–8` (stereo) | Pre-Fader | `ch7_guitars` (Guitars) | `instrument_guitar` | `CC 25` Vol |
| `Ch 9–12` (bus) | Pre-Fader | `ch9_drums` (Drums / Bass / Tracks) | `instrument_band` | `CC 26` Vol |
| `Ch 13–14` (stereo) | Pre-Fader | `ch13_room` (Room Ambience, optional) | `room_ambience` | Telemetry only (applause / transition / room ring context) |
| `Ch 15–16` (stereo) | **Post-Fader** | `master_out` (Stereo Master Mix) | `master_meter` | **LUFS & clip telemetry only** *(never fed to CLAP)* |

*(Note: Mid-EQ Dip and Master Limiter macros are explicitly deferred to v4 so every mapped control in v3 has defined semantics, taper, and rules.)*

#### MIDI-in-the-Middle Topology, Soft-Takeover & Calibration
* **Primary Topology:**
  ```text
  [Hardware MIDI Controller] ──► (MIDI In) [AetherMix midi_bridge] ──► (CoreMIDI Virtual "AetherMix To MainStage") ──► [MainStage]
                                                   ▲
                                   (Optional: "AetherMix From MainStage" if Phase 0 confirms GUI feedback)
  ```
* **Direct Human Passthrough (`< 2 ms`):** Incoming CCs from the hardware controller pass through to MainStage immediately (subject to takeover mode), updating AetherMix's authoritative `actual_cc[i]` / `current_db[i]`.
* **Controller Takeover Mode (configurable per device):**
  * `pickup_catchup` *(default for non-motorized faders)*: If the physical fader lags/leads `actual_cc[i]` after an agent move, incoming CCs are ignored until the physical knob crosses within `±2 CC` of `actual_cc[i]` (preventing abrupt volume jumps), with a visual pickup arrow in SwiftUI.
  * `immediate`: For motorized faders or touchscreens.
  * `relative_encoder`: For endless encoders emitting `+1/-1` deltas.
* **Hardware Controller Action Buttons (works when MainStage has macOS focus!):** Configurable MIDI CC/Note bindings in `service_plan.json` for `👍 Good Mix` (e.g., `CC 102`), `👎 Undo & Penalize` (`CC 103`), and `Emergency Bypass` (`CC 104`).
* **Canonical `dB ↔ CC` Curve (7-point strictly monotonic default, refined in Phase 0):**
  * `[(0, null), (1, -60.0), (16, -36.0), (32, -24.0), (64, -12.0), (96, 0.0), (114, +6.0)]`
  * `CC 0` represents `-inf` (off); `CC 1..114` is strictly monotonic piecewise-linear (validated at schema load); values above `hard_max_db` clamp before mapping, and inverse `dB → CC` picks the lowest CC on any flat top plateau.
  * **Reverb Sends (`CC 30`, `CC 31`):** Linear `0.0–100.0%` mapped to `CC 0..127` (with per-phase baselines and `reverb_pct_offset` rules), curve verified in Phase 0.
* **Mute Safety & Polarity Verification:**
  * Hard-forbid automatic mute CCs on any `spoken_primary` channel (`safety_firewall.py` invariant).
  * `SetupWizardView` includes a mandatory **Mute Polarity Verification** step (sends mute test and asks operator to confirm MainStage's Mute button lit up).
* **Change Deduplication:** Transmit MIDI CC only when the quantized integer value changes (or during `MIDI Learn Pulse`).

---

### 5.2 Initial State Sync & `SHADOW → ACTIVE` Handover

1. **Startup State:** Engine always boots in `SHADOW_OBSERVE` with `armed = false` on every channel. Last known fader positions from the previous session are loaded from SQLite `last_known_state` and flagged `sync_status = "UNVERIFIED"`.
2. **Becoming `SYNCED`:** A channel transitions to `sync_status = "SYNCED"` when any of the following occurs:
   * Operator moves that channel's hardware fader (or SwiftUI fader in MIDI-in-the-middle mode),
   * Operator clicks **`Capture Soundcheck Baseline`** (confirming current MainStage/controller state), or
   * Operator clicks **`Confirm & Sync Channel`** (adopts last-known or assumed baseline with a slow `2.0 dB/s` initial takeover ramp displayed in the UI).
3. **Arming Guard:** Switching Master Mode to `ACTIVE` only drives channels that are **both `armed == true` AND `sync_status == "SYNCED"`**. Unsynced channels remain safely in hold (`UNSYNCED — HOLD`) until synced.

---

### 5.3 Three-Tier Control Loop

#### Tier 1: Real-Time Reflex & Safety Loop (`50 Hz` / `20 ms` tick)

1. **Decoupled Audio Ingestion (`audio_tap.py`, `vad_worker.py`):**
   * PortAudio callback writes 48 kHz float32 blocks to a lock-free ring buffer (zero allocations, zero ML).
   * `audio_tap.py` computes at `50 Hz`:
     * Per-channel wideband RMS dBFS, true-peak dBFS, clip flags (`>= -0.5 dBFS`), and 1024-point Hann FFT spectra.
     * **Voice-band (`300–3400 Hz`) RMS** per channel, defining one canonical bleed metric:
       `speech_to_band_snr_db(ch) = rms_300_3400(ch) - max(rms_300_3400(b) for b in band_channels)`
       where `band_channels` are all channels with role in `{instrument_pad, instrument_guitar, instrument_band}`.
     * **Sung-Vocal Activity (`Ch 2–4`):** Computed via `200–4000 Hz` RMS + autocorrelation/harmonicity salience above bleed floor (`vocal_snr_db >= +3.0 dB`), NOT Silero VAD (since Silero VAD is unreliable on singing).
     * **Pre-Fader CLAP Sum:** Builds a gain-trimmed unity-gain mono sum of **pre-fader `Ch 1–12`** (normalized using soundcheck `nominal_rms_db` trims) into a 3-second rolling buffer for the CLAP subprocess.
     * **Master LUFS (`10 Hz`):** ITU-R BS.1770 momentary (`400 ms`) and short-term (`3 s`) LUFS strictly on **post-fader `Ch 15–16`**.
     * **TCC Zero-Buffer Guard:** If all configured input channels read bitwise `0.0` for `> 2.0 s`, raises `HARDWARE_SILENCE_WARNING`.
   * `vad_worker.py` runs in a 1-thread CPU worker downsampling `Ch 1` (and spoken `Ch 2`) to 16 kHz and running Silero VAD ONNX (`512` samples / `32 ms` hop / ~31 Hz).

2. **Supervised Offset Learner (`offset_learner.py`):**
   * Computes bounded learned offset per `(phase p, channel i)`:
     ```text
     learned_offset_db[p][i] = clip(w[p][i]^T * phi_i(S_t), -4.0, +4.0)
     ```
   * **Feature Vector `phi_i(S_t)`:** `[rms_rel_nominal_i, vocal_to_band_ratio_norm, speech_or_vocal_active_i, master_lufs_error_norm, 1.0]` where `master_lufs_error_norm = clip((master_lufs - target_lufs[p]) / 12.0, -1.0, 1.0)`.
   * **Exact Ridge Regression on Target dB Offsets (1 sample per operator event!):**
     * Per `(phase p, channel i)` maintains `(A_{p,i} = lambda * I + sum alpha_k * phi_k phi_k^T, b_{p,i} = sum alpha_k * y_k * phi_k)` with `lambda = 10.0`, initialized to `w = 0`.
     * **Operator Fader Correction (`alpha = 2.0`):** Target `y = clip(human_db - (baseline_db[p][i] + rule_sum_db[p][i]), -4.0, +4.0)` using the 2-second mean context `phi_bar` preceding the move.
     * **Thumbs Up (`👍`, `alpha = 1.0`):** Inserts **1 sample** per armed channel with `y = current_effective_offset_db` and trailing 5-second mean context `phi_bar`.
     * **Thumbs Down (`👎`, `alpha = 1.5`):** Inserts **1 sample** per rolled-back channel with `y = pre_change_offset_db` and pre-change mean context `phi_bar`.
     * **Session Decay, Snapshots & Reset:** Applies gentle forgetting factor (`gamma = 0.98`) between services, persists versioned snapshots to SQLite `policy_snapshots`, and supports `reset_learning(phase?, channel_id?)`.

3. **Target Assembly & Deterministic Safety Firewall (`safety_firewall.py`):**
   * Base target assembly for channel `i` in active phase `p`:
     ```text
     rule_sum_db = clip(sum(matching_active_rules_db(p, i, S_t)), -12.0, +6.0)
     raw_target_db[i] = baseline_db[p][i]
                      + rule_sum_db
                      + learned_offset_db[p][i]
                      + human_session_offset_db[i]
     ```
   * Every `20 ms` tick enforces these deterministic invariants in strict order:
     1. **Watchdog & Mode Gate:** Tracks tick latency/jitter. If `> 5` consecutive ticks exceed `50 ms` or audio stalls `> 500 ms`, trips `WATCHDOG_DROP_TO_SHADOW`, switches to `SHADOW_OBSERVE`, and freezes MIDI output (never zeros). In `SHADOW_OBSERVE` or `EMERGENCY_BYPASS`, streams computed targets to UI only; emits zero MIDI.
     2. **Human Touch Debounce & Ephemeral Offset:**
        * Non-echo human move on channel `i` sets `touch_lock_until[i] = now + 15.0 s` (blocks agent writes while hand is on fader) AND computes `human_session_offset_db[i] = human_db - (baseline_db + rule_sum_db + learned_offset_db)`.
        * After the `15 s` touch lock expires, `human_session_offset_db[i]` **persists for the remainder of the current phase / service segment** (never fighting the operator!) and clears smoothly via musical slew (`4 dB/s`) only on the next macro phase transition, segment advance, or manual *Release Offset* click.
     3. **Tier-1 `Speech-Open` Reflex (Fast Pastor Onset, `< 1.0 s`):**
        * For any `spoken_primary` channel (`Ch 1`), whenever `vad_prob >= 0.70` AND `speech_to_band_snr_db >= +6.0 dB` sustained for `>= 150 ms`:
          * Sets `speech_open_active[i] = true` (independent of CLAP/phase!), raising target to `spoken_open_db` (default `phase_baselines_db["SPOKEN_WORD"][i]`, e.g., `0.0 dB`) via **fast up-slew (`48.0 dB/s`)** (`-20 dB → 0 dB` in `417 ms`; total onset `< 600 ms`).
          * Holds open until either macro phase confirms `SPOKEN_WORD`/`PRAYER_UNDERSCORE` OR sustained speech silence `>= 3.0 s` (then releases at musical `4.0 dB/s`).
     4. **Bleed-Safe Auto-Ducking Prior:**
        * Engages ducking on `instrument_pad` (`Keys/Pad`, default `-6.0 dB` overridable via rule) when:
          * Active phase is `PRAYER_UNDERSCORE`, `SPOKEN_WORD`, or `TRANSITION`, AND `Ch 1` or `Ch 2` has `vad_prob >= 0.65` AND `speech_to_band_snr_db >= +3.0 dB` (`>= 150 ms`), OR
          * Active phase is `WORSHIP_LOW` or `WORSHIP_HIGH`, AND `Ch 1` (`spoken_primary`) triggers `speech_open_active` (`speech_to_band_snr_db >= +6.0 dB`) sustained for `>= 400 ms`.
        * Duck envelope runs **inside** the unified 4-path slew limiter (attack uses `slew_protect_down = 48.0 dB/s` → `125 ms` for `-6 dB`; release uses `slew_duck_release_up = 8.0 dB/s` → `750 ms` for `+6 dB`).
     5. **Bleed-Rejected, Escalating Anti-Feedback Detector:**
        * Monitors open mic channels (`feedback_guard == true`, `Ch 1–4`).
        * Detects narrowband FFT peaks (`> +14 dB` above local median floor within `±10` bins, high Q).
        * **Cross-Channel Bleed Check:** Rejects peak at bin `f` if any instrument tap (`Ch 5–12`) has a matching peak (`±1 bin`) within `6 dB` relative prominence (prevents sustained synth pads bleeding into mics from false-triggering).
        * **Growth / Lock-On Criterion:** Triggers only if bin `f` (`±1 bin`) persists for `>= 250 ms` with net energy growth `>= +2.0 dB` (or sustained narrowband prominence `> +18 dB`).
        * **Escalating Action & Gradual Release:** First trigger cuts `-4.0 dB` (at `48.0 dB/s`) and sets sticky ceiling `feedback_ceiling_db[i]`. If resonance persists after `400 ms`, escalates additional `-4.0 dB` cuts (up to max `-16.0 dB` + urgent UI alarm). After resonance clears for `5.0 s`, lifts ceiling smoothly at `+1.0 dB/s`.
     6. **Unified 4-Path Float Slew-Rate Limiter & Hard Clamps:**
        * `slew_protect_down = 48.0 dB/s` (anti-feedback cuts & auto-ducking attack)
        * `slew_speech_open_up = 48.0 dB/s` (Tier-1 `spoken_primary` speech-open reflex)
        * `slew_duck_release_up = 8.0 dB/s` (auto-ducking release)
        * `slew_musical = 4.0 dB/s` (macro phase transitions & normal mix glides; `2.0 dB/s` on initial sync handover)
        * Hard clamp to `[hard_min_db, hard_max_db]` and `<= feedback_ceiling_db[i]`, then quantize via calibrated taper and emit only on integer CC change.

---

#### Tier 2: Semantic Sensor Fusion & LLM Supervisor (`0.5 Hz` / `2 s` + event triggers)

* **Isolated CLAP Subprocess (`clap_worker.py`):**
  * Spawned via `multiprocessing.get_context("spawn")` with `torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`, and macOS Background/Utility QoS (`PRIO_DARWIN_BG` / `os.nice(10)`).
  * Classifies the **3-second rolling pre-fader sum** (`Ch 1–12`) every `2.0 s`, eliminating post-fader phase lock-in.
* **Quantitative Pre-Fader Telemetry Evidence (`P_telemetry` in `scene_classifier.py`):**
  * Uses concrete calibrated thresholds relative to channel `nominal_rms_db` (default fallback: `active >= -36 dBFS`, `silent <= -48 dBFS`):
    * `pastor_speaking`: `ch1_pastor` `vad_prob >= 0.65` and `speech_to_band_snr_db >= +3.0 dB`
    * `vocals_singing`: any of `ch2–4` vocal-band RMS active and harmonicity/SNR above bleed floor
    * `keys_active`: `ch5_keys` RMS active; `band_active`: `ch7_guitars` or `ch9_drums` RMS active
    * `applause_detected`: `ch13_room` (if mapped) or pre-fader sum broadband spectral flatness `> 0.55` with low pitch salience
  * Soft-scores each phase cleanly from these signals (`SPOKEN_WORD`, `PRAYER_UNDERSCORE`, `WORSHIP_LOW`, `WORSHIP_HIGH`, `TRANSITION`).
* **Fusion & Hysteresis:**
  * With Order-of-Service tracking enabled: `P_fused = normalize(0.40 * P_clap + 0.40 * P_telemetry + 0.20 * P_service_plan)` (`0.50 / 0.50` if disabled).
  * `service_step_idx` advances only on explicit operator action (UI can display a non-intrusive *"Advance to Sermon?"* prompt when detected phase matches next segment for `> 10 s`).
  * **Hysteresis:** Requires **2 consecutive matching `2 s` windows (`4.0 s`)** with `confidence >= 0.70` to switch macro phase.
  * **`TRANSITION` Semantics:** Acts as a **Hold / Gentle Settle** state (softly trims active instrument channels `-4 dB` from prior phase while holding vocal/pastor readiness) rather than snapping to arbitrary muted baselines between back-to-back songs.
* **Async Ollama Supervisor (`llm_supervisor.py`):**
  * Warm-loaded with `keep_alive: "60m"`, strict Ollama `format` JSON Schema enforcement, and `5.0 s` timeout.
  * **Strict Trigger Separation:**
    * On `phase_transition` or `safety_event`: **Advisory/Explanation only** (`proposed_patches` ignored in code).
    * On `operator_critique`: Returns explanation + typed `proposed_patches[]` routed through `rules_manager.py`.

---

#### Tier 3: Human-in-the-Loop Adaptation & Rule Lifecycle (`rules_manager.py`)

* **Thumbs Up (`👍`, Hardware MIDI button or `Cmd+Shift+Up`):**
  * Inserts **1 sample** per armed channel into `offset_learner.py` (or queues to `pending_review_samples` if `freeze_learning == true`).
* **Thumbs Down (`👎`, Hardware MIDI button or `Cmd+Shift+Down`):**
  * **Smart Rollback:** Restores all *non-touch-locked* armed channels to their snapshot at `T - 5 s` (and holds `10 s`); a second `👎` within `5 s` steps back to `T - 15 s` or the last pre-transition snapshot. Inserts **1 corrective sample** per rolled-back channel (or queues under `freeze_learning`).
* **Natural-Language Critique & Two-Tier Rule Confirmation:**
  * Primary input: Quick-chip presets + keyboard typing (optional macOS Speech dictation for rehearsals).
  * Supports full rule parameters:
    * `vol_db_offset` (`[-12.0, +6.0] dB`)
    * `reverb_pct_offset` (`[-50.0, +30.0] %`)
    * `duck_depth_db` (`[-12.0, 0.0] dB`)
    * `conditional_mute` / `conditional_attenuate` (e.g., `when ch1_pastor speech_open → attenuate ch3_bv1, ch4_bv2 by -24 dB` or mute non-`spoken_primary` channels)
  * **Safety Gate (`±3 dB` Preview Threshold):**
    * Single-channel `vol_db_offset` within `[-3.0, +3.0] dB` (or `reverb_pct_offset` within `[-15%, +15%]`) applies immediately as an ephemeral session adjustment AND shows a card with **`[Save Permanent]`** / **`[Dismiss]`** (never reusing `👍`).
    * Any patch exceeding `±3.0 dB`, any multi-channel patch, or any mute rule **does not auto-apply**—it appears as a **`PREVIEW REQUIRED`** card with **`[Apply Session]`** / **`[Save Permanent]`** / **`[Dismiss]`**.
  * **Ephemeral Expiry, Stack Cap & Conflict Resolution:**
    * `ephemeral_session` rules specify `expires_on`: `"phase_change"` | `"segment_change"` | `"end_of_service"` (default `"segment_change"`).
    * Total summed `vol_db_offset` across all active rules per channel is hard-capped to `[-12.0, +6.0] dB`.
    * Duplicate/conflicting rules targeting the same `(phase, channel_id, parameter, condition)` resolve via **most-recently-confirmed rule replaces older rule**.
    * Confirmed permanent rules write atomically (`tmp` + `os.replace`) to `mixing_rules.json` with timestamped backups in `engine/storage/rule_backups/`.

---

## 6. Reference Schemas & WebSocket Protocol (v3)

### 6.1 `engine/schemas/service_plan.json`

```json
{
  "schema_version": 3,
  "sample_rate": 48000,
  "audio_device_hint": "BlackHole 16ch",
  "ollama_model": "qwen3:14b",
  "ollama_think": false,
  "midi": {
    "hardware_in_port": "X-Touch Mini",
    "takeover_mode": "pickup_catchup",
    "mainstage_out_port": "AetherMix To MainStage",
    "mainstage_feedback_in_port": null,
    "action_bindings": {
      "thumbs_up": {"channel": 1, "cc": 102},
      "thumbs_down": {"channel": 1, "cc": 103},
      "emergency_bypass": {"channel": 1, "cc": 104}
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
      "midi_channel": 1,
      "vol_cc": 20,
      "reverb_cc": 30,
      "mute_cc": null,
      "hard_min_db": -60.0,
      "hard_max_db": 3.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0]]
    },
    {
      "id": "ch2_lead_vox",
      "label": "Lead Vocal",
      "tap_channels": [2],
      "role": "vocal_lead",
      "vad_enabled": true,
      "feedback_guard": true,
      "nominal_rms_db": -18.0,
      "midi_channel": 1,
      "vol_cc": 21,
      "reverb_cc": 31,
      "mute_cc": 41,
      "mute_active_high": true,
      "hard_min_db": -60.0,
      "hard_max_db": 3.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0]]
    },
    {
      "id": "ch3_bv1",
      "label": "Backup Vocal 1",
      "tap_channels": [3],
      "role": "vocal_bg",
      "vad_enabled": false,
      "feedback_guard": true,
      "nominal_rms_db": -20.0,
      "midi_channel": 1,
      "vol_cc": 22,
      "mute_cc": 42,
      "mute_active_high": true,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0]]
    },
    {
      "id": "ch4_bv2",
      "label": "Backup Vocal 2",
      "tap_channels": [4],
      "role": "vocal_bg",
      "vad_enabled": false,
      "feedback_guard": true,
      "nominal_rms_db": -20.0,
      "midi_channel": 1,
      "vol_cc": 23,
      "mute_cc": 43,
      "mute_active_high": true,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0]]
    },
    {
      "id": "ch5_keys",
      "label": "Keys / Pad",
      "tap_channels": [5, 6],
      "role": "instrument_pad",
      "vad_enabled": false,
      "feedback_guard": false,
      "nominal_rms_db": -16.0,
      "midi_channel": 1,
      "vol_cc": 24,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0]]
    },
    {
      "id": "ch7_guitars",
      "label": "Guitars",
      "tap_channels": [7, 8],
      "role": "instrument_guitar",
      "vad_enabled": false,
      "feedback_guard": false,
      "nominal_rms_db": -16.0,
      "midi_channel": 1,
      "vol_cc": 25,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0]]
    },
    {
      "id": "ch9_drums",
      "label": "Drums / Bass / Tracks",
      "tap_channels": [9, 10, 11, 12],
      "role": "instrument_band",
      "vad_enabled": false,
      "feedback_guard": false,
      "nominal_rms_db": -14.0,
      "midi_channel": 1,
      "vol_cc": 26,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, null], [1, -60.0], [16, -36.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0]]
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

*(Pydantic enforces at load time that every controlled channel in `service_plan.json` has an entry in every phase baseline.)*

```json
{
  "schema_version": 3,
  "calibrated_at": null,
  "firewall": {
    "touch_lock_seconds": 15.0,
    "echo_suppress_ms": 200,
    "speech_open_vad_min": 0.70,
    "speech_open_snr_db": 6.0,
    "speech_open_hold_ms": 150,
    "speech_open_release_silence_s": 3.0,
    "duck_depth_db": -6.0,
    "duck_snr_spoken_db": 3.0,
    "duck_snr_worship_db": 6.0,
    "duck_worship_hold_ms": 400,
    "slew_protect_down_db_per_sec": 48.0,
    "slew_speech_open_up_db_per_sec": 48.0,
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
    "SPOKEN_WORD": -22.0,
    "PRAYER_UNDERSCORE": -20.0,
    "WORSHIP_LOW": -17.0,
    "WORSHIP_HIGH": -13.0,
    "TRANSITION": -24.0
  },
  "phase_baselines_db": {
    "SPOKEN_WORD":       {"ch1_pastor": 0.0,   "ch2_lead_vox": -24.0, "ch3_bv1": -36.0, "ch4_bv2": -36.0, "ch5_keys": -36.0, "ch7_guitars": -48.0, "ch9_drums": -48.0},
    "PRAYER_UNDERSCORE": {"ch1_pastor": 0.0,   "ch2_lead_vox": -18.0, "ch3_bv1": -36.0, "ch4_bv2": -36.0, "ch5_keys": -10.0, "ch7_guitars": -36.0, "ch9_drums": -48.0},
    "WORSHIP_LOW":       {"ch1_pastor": -18.0, "ch2_lead_vox": 0.0,   "ch3_bv1": -8.0,  "ch4_bv2": -8.0,  "ch5_keys": -4.0,  "ch7_guitars": -6.0,  "ch9_drums": -14.0},
    "WORSHIP_HIGH":      {"ch1_pastor": -20.0, "ch2_lead_vox": 0.0,   "ch3_bv1": -5.0,  "ch4_bv2": -5.0,  "ch5_keys": -3.0,  "ch7_guitars": -4.0,  "ch9_drums": -2.0},
    "TRANSITION":        {"ch1_pastor": -6.0,  "ch2_lead_vox": -10.0, "ch3_bv1": -18.0, "ch4_bv2": -18.0, "ch5_keys": -14.0, "ch7_guitars": -20.0, "ch9_drums": -24.0}
  },
  "phase_reverb_baselines_pct": {
    "SPOKEN_WORD":       {"ch1_pastor": 5.0,  "ch2_lead_vox": 10.0},
    "PRAYER_UNDERSCORE": {"ch1_pastor": 12.0, "ch2_lead_vox": 25.0},
    "WORSHIP_LOW":       {"ch1_pastor": 8.0,  "ch2_lead_vox": 35.0},
    "WORSHIP_HIGH":      {"ch1_pastor": 5.0,  "ch2_lead_vox": 40.0},
    "TRANSITION":        {"ch1_pastor": 8.0,  "ch2_lead_vox": 20.0}
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

Passed directly to Ollama's `format` parameter and validated by Pydantic:

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
          "expires_on": {"type": "string", "enum": ["phase_change", "segment_change", "end_of_service"]},
          "phase": {"type": "string", "enum": ["ANY", "SPOKEN_WORD", "PRAYER_UNDERSCORE", "WORSHIP_LOW", "WORSHIP_HIGH", "TRANSITION"]},
          "channel_id": {"type": "string"},
          "parameter": {"type": "string", "enum": ["vol_db_offset", "reverb_pct_offset", "duck_depth_db", "conditional_attenuate", "conditional_mute"]},
          "value": {"type": "number"},
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

*(Note: When `trigger_type != "operator_critique"`, `llm_supervisor.py` forces `proposed_patches = []` regardless of LLM output.)*

### 6.4 WebSocket IPC Protocol (`ws://127.0.0.1:8765/ws`, v3)

* **Authentication & Handshake:** Client connects with `Authorization: Bearer <token>` (read from `~/.aethermix/session.token`, `0600` permissions) and exchanges `hello { protocol_version: "3.0" }`.
* **Envelope:** `{ "type": "<str>", "seq": <int>, "req_id": "<uuid|null>", "ts_ms": <int>, "payload": { ... } }`
* **Server → Client:**
  * `ack` / `error`: Correlated via `req_id` for every client RPC command.
  * `state_snapshot`: Includes `channel_ids[]` (canonical array order for `meter_frame`), `autonomy_mode`, `freeze_learning`, `active_phase`, `phase_confidence`, `hysteresis_progress`, `service_step_idx`, `channels_state` (`armed`, `sync_status`, `current_db`, `target_db`, `actual_cc`, `target_cc`, `reverb_pct`, `touch_lock_s`, `human_offset_db`, `feedback_ceiling_db`, `speech_open`), `active_rules[]`, `pending_rules[]`, `queued_review_count`, `hardware_warnings[]`.
  * `meter_frame` (`30–50 Hz`): `{ "rms_db": [...], "peak_db": [...], "clip": [...], "vad": [...], "master_lufs": -16.4, "target_cc": [...], "actual_cc": [...] }` aligned strictly to `state_snapshot.channel_ids[]`, routed exclusively to `MeterStore`.
  * `brain_event`: `{ "id", "category": "PERCEPTION|ACTION|SAFETY|LLM|RLHF", "summary", "detail", "confidence", "ts_ms" }`.
  * `heartbeat` (`1 Hz`): `{ "uptime_s", "loop_p95_ms", "loop_max_ms", "audio_ok", "midi_ok", "clap_ok", "ollama_ok" }`.
* **Client → Server (all return `ack` / `error`):**
  * `set_mode`, `set_channel_armed`, `sync_channel` / `sync_all_channels`, `release_human_offset`
  * `set_freeze_learning`, `override_phase`, `advance_service_step`
  * `feedback_thumb` (`"up"` | `"down"`), `submit_critique`, `resolve_pending_rule` (`rule_id`, `"apply_session"` | `"promote"` | `"dismiss"`)
  * `toggle_rule`, `delete_rule`, `reset_learning` (`phase?`, `channel_id?`), `commit_queued_reviews`
  * `capture_baseline` (`phase`, `quick_derive_others: bool`), `capture_input_nominals`
  * `midi_learn_pulse` (`channel_id`, `control`), `verify_mute_polarity`, `manual_channel_move`

---

## 7. Failure-Mode & Resilience Matrix

| Failure / Edge Case | Guaranteed Behavior |
| :--- | :--- |
| **Daemon crashes or restarts mid-service** | Never sends zero/mute CCs; MainStage holds current faders. SwiftUI flashes `DISCONNECTED — MANUAL CONTROL` displaying cached **Last Known Fader Positions** + 1-click restart. Reconnect boots in `SHADOW_OBSERVE` with channels disarmed/unsynced. |
| **Band stops, pastor starts speaking from `-20 dB`** | Tier-1 `speech_open` reflex raises pastor mic at `48 dB/s` within `150 ms` debounce → full spoken level in **`< 600 ms`** (independent of 4 s CLAP phase hysteresis). |
| **Operator adjusts fader manually mid-song** | Passed straight through to MainStage (`< 2 ms`); sets `15 s` touch debounce + **persistent ephemeral offset** held until next phase/segment change; logs 1 high-weight supervised target. |
| **Non-motorized controller fader out of sync with agent** | `pickup_catchup` mode waits until physical knob crosses `±2 CC` of `actual_cc` before taking over, preventing abrupt volume jumps. |
| **Sustained synth/keys pad bleeds into vocal mic** | Anti-feedback cross-checks FFT peak bin against `Ch 5–12` instrument taps; matching peaks are rejected as bleed, not feedback. |
| **Real acoustic feedback howl develops** | Narrowband growth `>= 250 ms` triggers `-4 dB` cut at `48 dB/s`, escalates every `400 ms` up to `-16 dB` if ringing persists, holds sticky ceiling for `5 s` clear, then lifts gradually at `+1 dB/s`. |
| **Misheard / large LLM critique (`"lead vocals down"` `-12 dB`)** | Any patch `> ±3 dB`, multi-channel, or mute enters **`PREVIEW REQUIRED`** card first—never auto-applies mid-song. Total rule sum capped at `[-12, +6 dB]`. |
| **Tier-1 loop jitter / stall (`> 50 ms` × 5 ticks)** | Watchdog drops immediately to `SHADOW_OBSERVE`, freezes MIDI output, logs `WATCHDOG_DROP_TO_SHADOW`, and alerts operator. |
| **macOS TCC Microphone permission missing** | Bitwise zero-buffer check (`> 2.0 s`) raises `HARDWARE_SILENCE_WARNING` with TCC diagnostic checklist (`Terminal` vs `AetherMix.app`). |
| **MainStage holds focus during live set** | Hardware controller MIDI buttons / footswitch trigger `👍`, `👎`, and `Emergency Bypass` directly inside `midi_bridge.py` regardless of active macOS window. |

---

## 8. Packaging, macOS Permissions & Volunteer Operations

* **App-Managed Daemon (`EngineProcessManager`):** `AetherMix.app` checks `127.0.0.1:8765` first (supporting `make dev`); otherwise launches the bundled local `uv`/Python engine environment, generates/loads `~/.aethermix/session.token`, and exposes live engine logs in a Diagnostics drawer.
* **macOS TCC Microphone Permission Note:** Under `make dev` (launched from Terminal), macOS grants CoreAudio/BlackHole capture permission to **Terminal.app**; under standalone `AetherMix.app`, macOS grants permission to **`AetherMix.app`** via `NSMicrophoneUsageDescription` in `Info.plist`. `SetupWizardView` detects active launch context and guides the volunteer explicitly.

---

## 9. Phased Implementation Plan, Measurable Gates & Staged Go-Live

### Phase 0 — MainStage Hardware & Contention Spike (1–2 Days, Mandatory Gate Before Phase 1)
Execute `scripts/phase0_spike.py` against your actual MainStage Concert & M4 Max:
* [ ] Verify whether MainStage sends MIDI feedback CCs on Concert-level Screen Controls (mouse drag & external CC).
* [ ] Confirm **AetherMix-in-the-middle** routing (`Hardware Controller → AetherMix → MainStage`) with `< 2 ms` passthrough and zero double-triggering.
* [ ] Empirically measure real `dB ↔ CC` curves for MainStage channel strip Volume and Bus Reverb Send.
* [ ] Confirm pre-fader sends (`Ch 1–12`) to BlackHole remain invariant when Concert faders move.
* [ ] Benchmark candidate Ollama models (`qwen3:14b` / `qwen3:8b` / `qwen2.5:14b`) + isolated 1-thread CPU CLAP running simultaneously with your heaviest MainStage concert at live buffer size for **30 minutes** → **Gate: `0` MainStage CPU/CoreAudio overloads**, LLM p95 JSON latency `< 2.0 s`.

### Phase 1 — Scaffolding, Contracts & MIDI/Audio Bridge
* Build repo layout, `Makefile`, `scripts/bootstrap_mac.sh`, `core/protocol.py` (v3), `schemas/*.json`, `midi_bridge.py` (MIDI-in-the-middle, soft-takeover, hardware action buttons, deduplication), `audio_tap.py` (lock-free ring buffer, pre-fader CLAP sum, LUFS, TCC guard), and `scripts/record_shadow_session.py`.
* **Gate:** `scripts/verify_audio_midi.py` passes live; headless `pytest` mock I/O & schema cross-validation tests pass.

### Phase 2 — Safety Firewall, Sensor-Fused Perception, Offset Learner & WS Server
* Implement `safety_firewall.py`, `vad_worker.py`, `clap_worker.py` (Background-QoS subprocess), `scene_classifier.py`, `offset_learner.py`, `rules_manager.py`, `llm_supervisor.py`, `telemetry_store.py`, and authenticated FastAPI WS server (`main.py`).
* **Quantitative Gates (`pytest` + real SHADOW recordings):**
  * Speech-open onset (`-20 dB → 0 dB`): **`< 1.0 s`** (`0` missed pastor speech onsets).
  * Tier-1 50 Hz loop jitter: `p95 < 5 ms`, `max < 20 ms`, watchdog drops cleanly to `SHADOW` on injected stall.
  * Anti-feedback: cuts stationary howl within `400 ms` with escalating ceiling; `0` false triggers on sustained synth pad bleed or clean vocals.
  * Offset learner: 1 sample per `👍`/`👎`/human move; weights bounded `[-4, +4 dB]`; phase reset verified.
  * Rules/LLM: `> ±3 dB` and mute patches require preview; rule stack capped at `[-12, +6 dB]`; malformed LLM payloads rejected.
  * Re-verify 30-minute zero-overload stress gate against heaviest MainStage concert.

### Phase 3 — Native SwiftUI Mission Control App
* Build `macos-app/` (`project.yml` + `Package.swift`) including `MeterCanvasView`, per-channel Arm/Sync UI, `AgentBrainStreamView`, `RulesManagerView`, `PostServiceReviewView`, `FeedbackControlBar`, `SetupWizardView` (with Mute Polarity check & Baseline Capture), and `DISCONNECTED` last-known-faders banner.
* **Gate:** `xcodegen` + `xcodebuild`/`swift build` succeeds; 30–50 Hz meters render smoothly; boots in `SHADOW_OBSERVE` with channels disarmed; full WS token auth + `req_id` ack flow verified end-to-end.

### Phase 4 — Rehearsal Harness, Real-Service SHADOW Validation & Staged Go-Live
* Build `scripts/simulate_service.py` (headless + BlackHole playback) and run offline replay benchmarks on **3–4 real church services recorded in `SHADOW` mode** (`scripts/record_shadow_session.py`).
* **Offline Real-Audio Gate:** Phase classifier achieves **`>= 85%` macro F1** across recorded services before any live `ACTIVE` use.
* **5-Stage Live Rollout Plan:**
  1. **Live `SHADOW OBSERVE`** (3–4 services): record 16ch pre-fader audio + tune classifier, bleed thresholds, and feedback guard offline.
  2. **Rehearsal `ACTIVE`** (full band & pastor): validate speech-open reflex, pickup takeover, and `👍`/`👎` rollback live.
  3. **Live Stage 1:** Arm **`ch5_keys` (Keys/Pad) only** during live service (lowest-risk channel).
  4. **Live Stage 2:** Arm **Vocals (`ch2–4`) + Band Bus (`ch7–9`)**.
  5. **Live Stage 3:** Arm **`ch1_pastor` (Pastor Mic)** last, after speech-open reflex & bleed SNR gates are proven across live rehearsals.
