# Project Specification v2: **AetherMix** (Autonomous Agentic Live Sound Engineer for Apple MainStage)

> **Status:** Implementation-Ready Holistic Specification (v2)  
> **Target Hardware:** Apple Silicon macOS 15+ (optimized for M4 Max, 128 GB Unified Memory)  
> **Host Integration:** Apple MainStage (Concert-Level Audio Channel Strips)

---

## 1. Executive Summary & System Goals

Build **AetherMix**, a native macOS application and local multi-model AI daemon that runs alongside **Apple MainStage** to autonomously mix live multi-channel audio (pulpit/spoken mics, vocals, keys/pads, guitars, drums/bass/tracks) in real time across dynamic church service phases:

* `SPOKEN_WORD`
* `PRAYER_UNDERSCORE`
* `WORSHIP_LOW`
* `WORSHIP_HIGH`
* `TRANSITION`

AetherMix operates out-of-the-box after a 60-second **Soundcheck Baseline Capture**, explains every perception and action in a live **Agent Brain** observability UI, accepts real-time **Thumbs Up (`👍`) / Thumbs Down (`👎`)** rewards and natural-language operator critiques, and continuously refines its mixing policy across rehearsals and live services without compromising live-audio safety.

### Core Non-Negotiable Constraints

1. **100% Local Execution / Zero Cloud Latency:** All inference (VAD, CLAP scene classification, contextual bandit, supervisory LLM) and CoreAudio/CoreMIDI I/O run strictly locally on Apple Silicon.
2. **Deterministic Safety Firewall:** No ML model or LLM ever emits raw unbounded MIDI values. Every target passes through a non-LLM deterministic firewall enforcing calibrated dB/CC bounds, MIDI echo suppression, immediate human touch override, asymmetric musical slew-rate smoothing, and stationary narrowband anti-feedback protection.
3. **Fail-Safe Live Reliability:**
   * Always boots into **`SHADOW OBSERVE`** (read-only) mode so an accidental launch or mid-service daemon restart never slams faders to startup defaults.
   * Never sends zero/mute CCs on shutdown or crash—MainStage safely holds its last fader state while SwiftUI flashes a high-contrast `DISCONNECTED — MANUAL CONTROL` banner.
4. **Day-1 Autonomy + Safe RLHF Adaptation:** Works immediately via acoustic DSP priors + soundcheck baseline calibration, while learning bounded, phase-scoped adjustments via `👍`/`👎` rewards and operator-confirmed natural-language rules.

---

## 2. Key Architectural Decisions & Rationale (ADRs)

| Area | Decision | Rationale |
| :--- | :--- | :--- |
| **MIDI Transport** | **Programmatic CoreMIDI Virtual Ports** (`python-rtmidi`) with optional IAC Driver fallback | `bootstrap_mac.sh` cannot reliably provision named IAC buses via CLI. Virtual ports (`AetherMix To/From MainStage`) appear automatically in MainStage with zero Audio MIDI Setup friction. |
| **Audio Tap Topology** | **Pre-fader Ch 1–12 + Post-fader Stereo Master Ch 15–16** over BlackHole 16ch + macOS Aggregate Device | Tapping post-fader channel strips creates a broken control loop (ducking/muting a mic blinds VAD & RMS). Pre-fader direct taps keep perception invariant to fader moves; post-fader master tap feeds CLAP & LUFS loudness monitoring. |
| **MainStage Host Setup** | **Concert-Level Channel Strips** | Per-Patch channel strips reset faders whenever MainStage switches songs/patches. Concert-level strips guarantee continuous AetherMix control across the entire service. |
| **Online RL Policy** | **Phase-Scoped Ridge Contextual Bandit** (NumPy/SciPy) instead of neural SAC | Live services produce sparse feedback (~10–30 `👍`/`👎` clicks/week). Neural SAC is sample-inefficient and risks unstable extrapolation; an L2-regularized linear bandit over interpretable offsets updates in `<1 ms`, is 100% explainable, and cannot diverge. |
| **Compute Isolation** | **Silero VAD (ONNX) + CLAP (`transformers`) pinned to CPU**; Ollama on Metal GPU | Running VAD (31 Hz) and CLAP (0.5 Hz) on M4 Max CPU uses `<5%` of one P-core, leaving Metal GPU bandwidth free for MainStage instruments/plugins and Ollama LLM inference. |
| **Live Supervisory LLM** | **`qwen2.5:14b`** primary (`keep_alive: "60m"`, Ollama JSON Schema `format`); optional `qwen2.5:32b` for deep/post-service analysis | Fast (`<1 s`) structured JSON responses without `<think>` latency stalls (unlike `deepseek-r1:32b`) and lower GPU contention with MainStage sample libraries (30–60 GB RAM). |
| **Internal Control Units** | **Float dB internally** → calibrated piecewise-linear `dB ↔ 7-bit CC` lookup at MIDI boundary | MainStage faders are logarithmic; operating in float dB prevents 7-bit EMA rounding stalls (`0.25 * delta < 0.5`) and makes acoustic rules (`-6 dB`) accurate across the fader range. |
| **SwiftUI Build System** | **SwiftPM (`Package.swift`) + `XcodeGen` (`project.yml`)** | Avoids brittle hand-edited `.pbxproj` files while generating a proper signed macOS `.app` bundle with required TCC `Info.plist` usage descriptions. |

---

## 3. Technology Stack & Repository Layout

### Stack
* **Native macOS UI (`macos-app/`):** Swift 6 + SwiftUI (macOS 15+), `XcodeGen` (`project.yml`) + `Package.swift`, Apple `Speech` framework for push-to-talk critique dictation.
* **AI & Real-Time Audio Daemon (`engine/`):** Python 3.12 via `uv`, FastAPI + WebSocket (`ws://127.0.0.1:8765/ws`), Pydantic v2, `sounddevice` (PortAudio), `python-rtmidi` (CoreMIDI), `numpy`/`scipy`, `onnxruntime` (Silero VAD), `transformers` + `torch` CPU (`laion/clap-htsat-unfused`), `httpx` (Ollama async client), `sqlite3` (WAL mode).
* **Audio/MIDI Infrastructure:** BlackHole 16ch + macOS Aggregate Device (48 kHz locked clock, drift correction enabled), native CoreMIDI Virtual Endpoints.

### Directory Layout

```text
aethermix/
├── SPEC.md                           # This engineering specification
├── Makefile                          # setup, models, test, build-app, dev, simulate
├── scripts/
│   ├── bootstrap_mac.sh              # Homebrew deps, Ollama warm pull, HF/Silero model cache
│   ├── verify_audio_midi.py          # Aggregate/BlackHole RMS + CoreMIDI loopback smoke test
│   └── simulate_service.py           # Synthetic multi-stem church service runner (headless + BlackHole)
├── engine/
│   ├── pyproject.toml
│   ├── main.py                       # Orchestrator: Audio Tap, VAD Worker, 50Hz Loop, CLAP/LLM, WS Server
│   ├── core/
│   │   ├── protocol.py               # Shared Pydantic schemas (WS envelopes, rules, service plan, events)
│   │   ├── audio_tap.py              # Lock-free ring buffer, 50Hz RMS/peak, 10Hz LUFS, TCC zero-buffer guard
│   │   ├── vad_worker.py             # Decoupled 16kHz downsampled Silero VAD (ONNX) + bleed SNR gate
│   │   ├── scene_classifier.py       # 0.5Hz CLAP + deterministic channel telemetry sensor fusion + hysteresis
│   │   ├── rl_policy.py              # Phase-scoped Ridge Contextual Bandit + rolling Replay Buffer
│   │   ├── llm_supervisor.py         # Async Ollama client (JSON schema format, timeout, fallback)
│   │   ├── safety_firewall.py        # Calibrated CC<->dB, echo suppression, touch lock, asymmetric slew, anti-feedback
│   │   ├── midi_bridge.py            # CoreMIDI virtual endpoints (IAC fallback) + change deduplication
│   │   ├── rules_manager.py          # Atomic JSON patch validation, ephemeral vs. pending vs. permanent rules
│   │   └── telemetry_store.py        # SQLite (WAL mode) async 1Hz summaries + discrete event & replay logs
│   ├── schemas/
│   │   ├── mixing_rules.json         # Versioned learned rules, phase baselines, CC taper calibration
│   │   └── service_plan.json         # Order of Service segments & 16ch pre-/post-fader channel map
│   ├── storage/
│   │   └── telemetry.db              # SQLite WAL database (gitignored)
│   └── tests/
│       ├── mock_io.py                # Headless deterministic Audio + MIDI simulator for CI/unit tests
│       ├── test_safety_firewall.py   # Bounds, echo suppression, touch override, slew, anti-feedback, rollback
│       ├── test_scene_fusion.py      # Hysteresis, stage-bleed rejection, CLAP+telemetry fusion
│       ├── test_rl_and_rules.py      # Phase-scoped bandit updates, Pydantic patch validation, staging
│       └── test_end_to_end.py        # Full synthetic service lifecycle without BlackHole hardware
└── macos-app/
    ├── project.yml                   # XcodeGen spec (Info.plist TCC keys, entitlements, macOS 15 target)
    ├── Package.swift                 # SwiftPM manifest for CLI build & model unit tests
    └── Sources/AetherMix/
        ├── App/AetherMixApp.swift    # Lifecycle, optional managed engine subprocess, global shortcuts
        ├── Models/
        │   ├── ProtocolModels.swift  # Codable mirror of engine/core/protocol.py
        │   └── MeterStore.swift      # Isolated high-frequency meter buffer for Canvas rendering
        ├── Services/
        │   ├── EngineWebSocketClient.swift  # Reconnect, heartbeat monitor, command RPC
        │   └── EngineProcessManager.swift   # Launches/monitors local uv engine if not already running
        └── Views/
            ├── MainDashboardView.swift      # Status bar, 3-pane layout, disconnect/bypass banners
            ├── MeterCanvasView.swift        # Dedicated 30-60Hz SwiftUI Canvas meters + clip/VAD LEDs
            ├── MixerTelemetryView.swift     # Dual-ring target vs. actual CC knobs + human-lock badges
            ├── AgentBrainStreamView.swift   # Live filterable perception/action/LLM stream + pending rule cards
            ├── FeedbackControlBar.swift     # 👍/👎 rollback dock, freeze-learning toggle, dictation & chips
            └── SetupWizardView.swift        # Aggregate Device check, Concert-level guide, MIDI Learn Pulse, Baseline Capture
```

---

## 4. Model Manifest & Automated Provisioning (`scripts/bootstrap_mac.sh`)

1. **System Dependencies (Homebrew):**
   * `blackhole-16ch`, `ollama`, `uv`, `portaudio`, `xcodegen`.
2. **Supervisory Reasoning LLM (Ollama):**
   * Primary live model: `ollama pull qwen2.5:14b`
   * Optional deep/post-service model (`--with-32b` flag): `ollama pull qwen2.5:32b`
3. **Semantic Audio Scene Classifier (cached under `~/.aethermix/models/clap/`):**
   * `laion/clap-htsat-unfused` via Hugging Face `transformers` (CPU inference).
   * Zero-shot text prompts:
     * `"pastor or speaker talking alone without music"` → `SPOKEN_WORD`
     * `"soft ambient keyboard pad underneath spoken prayer"` → `PRAYER_UNDERSCORE`
     * `"acoustic guitar or piano gentle worship verse"` → `WORSHIP_LOW`
     * `"full church band with drums electric guitar and loud singing"` → `WORSHIP_HIGH`
     * `"silence applause or stage transition between songs"` → `TRANSITION`
4. **Voice Activity Detection (cached under `~/.aethermix/models/silero_vad.onnx`):**
   * `silero-vad` ONNX model executed via `onnxruntime` on 16 kHz / 512-sample chunks (`32 ms`).

---

## 5. Detailed Subsystem Architecture

### 5.1 Audio & MIDI Routing Specification

#### BlackHole 16ch Tap Map (`service_plan.json`)
MainStage drives a **48 kHz macOS Aggregate Device** combining the physical PA interface (clock master) and **BlackHole 16ch** (drift correction enabled). Live strips reside at MainStage's **Concert Level**, with **pre-fader sends/direct outs** routed to BlackHole Ch 1–12 and the **post-fader stereo master bus** routed to BlackHole Ch 15–16:

| BlackHole Channels | Tap Type | Logical Strip | MIDI Controls (Channel 1 default) |
| :--- | :--- | :--- | :--- |
| `Ch 1` (mono) | Pre-Fader | Pastor / Pulpit Mic | `CC 20` Vol, `CC 30` Reverb Send, `CC 40` Mute |
| `Ch 2` (mono) | Pre-Fader | Worship Leader Vocal | `CC 21` Vol, `CC 31` Reverb Send, `CC 41` Mute |
| `Ch 3` (mono) | Pre-Fader | Backup Vocal 1 | `CC 22` Vol, `CC 42` Mute |
| `Ch 4` (mono) | Pre-Fader | Backup Vocal 2 | `CC 23` Vol, `CC 43` Mute |
| `Ch 5–6` (stereo) | Pre-Fader | Keys / Pad Stereo | `CC 24` Vol, `CC 34` Mid-EQ Dip |
| `Ch 7–8` (stereo) | Pre-Fader | Acoustic / Electric Guitars | `CC 25` Vol |
| `Ch 9–12` (bus) | Pre-Fader | Drums / Bass / Tracks Bus | `CC 26` Vol |
| `Ch 13–14` | Aux / Optional | Room Ambience / Cue (never to PA) | Telemetry only |
| `Ch 15–16` (stereo) | **Post-Fader** | Stereo Master Mix Out | `CC 27` Master Limiter / Macro (optional) |

#### MIDI Calibration, Soft-Mutes & Deduplication
* **Calibrated `dB ↔ CC` Piecewise Taper:** Default MainStage logarithmic curve per channel (configurable):
  * `[(0, -60.0), (16, -36.0), (32, -24.0), (64, -12.0), (96, 0.0), (114, +6.0), (127, +6.0)]`
  * All DSP, ducking, bandit, and rule math executes in **float dB** and quantizes to 7-bit MIDI `0..127` only at `midi_bridge.py`.
* **Mute Polarity & Soft-Mute Preference:** Default mute polarity `>= 64 = Muted`, `< 64 = Unmuted` (configurable). Automated actions prefer smooth volume attenuation (e.g., `-24 dB` glide) over hard mute toggles to avoid clicks and cut reverb tails, reserving hard mute CCs for explicit operator rules (e.g., *"Mute BVs when pastor speaks"*).
* **Change Deduplication:** Transmit a MIDI CC message only when the quantized integer value differs from the last transmitted value on that CC (or during explicit `MIDI Learn Pulse`), preventing MainStage MIDI input flooding.

---

### 5.2 Three-Tier Control Loop

#### Tier 1: Real-Time Reflex & Safety Loop (`50 Hz` / `20 ms` cycle)

1. **Decoupled Audio Ingestion (`audio_tap.py` & `vad_worker.py`):**
   * PortAudio callback writes raw 48 kHz float32 blocks into a lock-free ring buffer (zero allocations, zero ML).
   * `audio_tap.py` computes at `50 Hz`: per-channel RMS dBFS, true-peak dBFS, clip flags (`>= -0.5 dBFS`), and 1024-point Hann FFT magnitude spectra (for anti-feedback). Computes ITU-R BS.1770 momentary (`400 ms`) and short-term (`3 s`) LUFS on Ch 15–16 at `10 Hz`.
   * **TCC Zero-Buffer Guard:** If all configured input channels read exact bitwise `0.0` for `>2.0 s` while audio stream is open, raise `HARDWARE_SILENCE_WARNING` in `state_snapshot` (flags missing macOS Microphone permission or misconfigured Aggregate Device).
   * `vad_worker.py` runs in a dedicated background thread, downsampling `Ch 1–4` (`48 kHz → 16 kHz`) and running Silero VAD ONNX (`512` samples / `32 ms` hop / ~31 Hz), publishing thread-safe `vad_prob[ch]` and `speech_to_band_snr_db[ch]`.

2. **Contextual Bandit Policy (`rl_policy.py`):**
   * For each controlled channel `i` in active phase `p`:
     ```text
     raw_target_db[i] = baseline_db[p][i]
                      + sum(active_rule_offsets_db[p][i])
                      + clip(w[p][i]^T * phi(S_t), -4.0, +4.0)
     ```
   * **Feature Vector `phi(S_t)`:** Interpretable, normalized context `[rms_norm_i, vocal_to_band_ratio_norm, vad_active_i, master_lufs_error_norm, 1.0]`.
   * **Phase Isolation & Regularization:** Each service phase `p` maintains its own ridge regression matrix `(A_p = lambda * I + sum phi*phi^T, b_p = sum r * phi)` with strong L2 prior (`lambda = 10.0`), initialized to `0` on Day 1. Updates execute in `<1 ms` on `👍`/`👎` events.

3. **Deterministic Safety Firewall (`safety_firewall.py`):**
   Every `20 ms` tick applies these invariants in strict order:
   1. **Mode Gate:** If `SHADOW_OBSERVE` or `EMERGENCY_BYPASS`, compute and stream targets for UI visualization, **emit zero MIDI**.
   2. **MIDI Echo Suppression:** Maintain a rolling table of outbound `(cc, sent_val, ts_ms)`. Incoming CCs on `AetherMix From MainStage` matching an outbound write within `±1 CC` and `<= 200 ms` are ignored as MainStage feedback echoes.
   3. **Touch-Sense Human Override:** Any non-echo incoming CC on channel `i` immediately sets `human_lock_until[i] = now + 15.0 s`, snaps internal float state `current_db[i]` to the human's CC value, blocks all agent writes to channel `i` until expiry, and logs the human adjustment as a supervised preference event.
   4. **Bleed-Safe Auto-Ducking Prior:**
      * Active during `PRAYER_UNDERSCORE`, `SPOKEN_WORD`, and `TRANSITION` when `Ch 1 (Pastor)` or `Ch 2 (Leader)` has `vad_prob >= 0.65` AND `speech_to_band_snr_db >= +3.0 dB` (rejects stage bleed).
      * **Pastor Interrupt Exception:** During `WORSHIP_LOW`/`WORSHIP_HIGH`, auto-ducking triggers *only* if `Ch 1` is unmuted, `vad_prob >= 0.75`, and `Ch 1 RMS - Band Bus RMS >= +6.0 dB` sustained for `>= 400 ms`.
      * Smoothly ducks `Keys/Pad` by `-6.0 dB` (overridable via rules) with `150 ms` attack and `800 ms` release.
   5. **Stationary Narrowband Anti-Feedback Detector:**
      * Per open mic channel (`Ch 1–4`), detects narrowband FFT peaks (`> +14 dB` above local median spectral floor within `±10` bins, high Q).
      * Triggers **only** if the exact same peak bin (`±1 bin`) persists with non-decreasing magnitude for `>= 250 ms` (prevents false triggers on singing, guitars, or synth notes).
      * **Action:** Immediately pulls channel target down `-4.0 dB`, clamps a sticky ceiling `max_allowed_db[i] = current_db[i] - 4.0`, emits a `SAFETY` Brain Event, and holds ceiling until resonance clears for `5.0 s`.
   6. **Asymmetric Float Slew-Rate Limiter & Hard Clamps:**
      * Enforces per-tick max rate in dB/sec:
        * Fast downward protection (anti-feedback / ducking attack): up to `36.0 dB/s`.
        * Normal musical glides & phase transitions: `4.0 dB/s` (smooth ~1.5–3.0 s fader travel).
      * Clamps strictly within per-channel `[hard_min_db, hard_max_db]` (e.g., `[-60.0, +3.0 dB]`).
      * Converts final `current_db[i]` via calibrated curve to 7-bit CC and sends only on integer change.

---

#### Tier 2: Semantic Sensor Fusion & LLM Supervisor (`0.5 Hz` / `2 s` + event triggers)

* **Sensor-Fused Phase Classifier (`scene_classifier.py`):**
  * Runs CPU CLAP on the rolling 3-second **post-fader stereo master** window (Ch 15–16) every `2.0 s`.
  * Computes a deterministic **Pre-Fader Channel Evidence Distribution** `P_telemetry(phase)` using live channel states:
    * `Pastor VAD high + Keys active + Drums/Guitars silent` → strong prior for `PRAYER_UNDERSCORE`.
    * `Pastor VAD high + All instruments silent` → strong prior for `SPOKEN_WORD`.
    * `Drums + Guitars + Vocals active + High Master LUFS` → strong prior for `WORSHIP_HIGH`.
    * `Keys/Acoustic active + Drums silent + Vocals active` → strong prior for `WORSHIP_LOW`.
    * `All channels near noise floor or broadband unpitched applause` → strong prior for `TRANSITION`.
  * Fuses posteriors: `P_fused = normalize(0.45 * P_clap + 0.45 * P_telemetry + 0.10 * P_service_plan_prior)` (weights configurable).
  * **Hysteresis:** Requires **2 consecutive matching classifications (4.0 s)** with `confidence >= 0.70` before switching `active_phase`. Manual operator phase overrides bypass hysteresis and pin for a configurable hold window.
* **Async Ollama Supervisor (`llm_supervisor.py`):**
  * On startup, sends a lightweight warm-up request to Ollama (`qwen2.5:14b`, `keep_alive: "60m"`) so the model stays resident in unified memory throughout the service.
  * Triggered asynchronously on: (a) macro phase transition, (b) anti-feedback clamp, (c) operator natural-language critique.
  * Uses Ollama's native `format` JSON Schema parameter to guarantee valid structured JSON output, with a strict `5.0 s` timeout. If Ollama is offline/slow, Tier 1 and deterministic rules continue operating with zero interruption and a degraded-LLM badge appears in the UI.

---

#### Tier 3: Human-in-the-Loop RLHF & Rule Lifecycle (`rules_manager.py`)

* **Thumbs Up (`👍`, `Cmd+Shift+Up`):**
  * Captures trailing `10 s` of `(context, action)` tuples for the **current phase** into `positive_replay_buffer`.
  * Performs an immediate ridge bandit update on the current phase's weights (unless **Service Lock / Freeze Learning** is enabled, in which case samples are stored in SQLite for post-service review).
* **Thumbs Down (`👎`, `Cmd+Shift+Down`):**
  * **Smart Rollback:** Restores all *non-human-locked* channels to their exact state from `T - 5 s` (stored in the 60-second in-memory ring buffer) and holds for `10 s`. Pressing `👎` a second time within `5 s` steps back further to `T - 15 s` (or the last pre-transition snapshot) instead of oscillating.
  * Logs negative reward (`r = -1.0`) for the rejected action vector in the current phase.
* **Natural-Language Critique & Two-Stage Rule Promotion:**
  * Operator critique (text or macOS Speech dictation) + current telemetry snapshot goes to `llm_supervisor.py`.
  * LLM emits a typed JSON rule patch with explicit scope:
    * `ephemeral_session`: applies immediately for the current segment/service without mutating `mixing_rules.json`.
    * `candidate_permanent`: applies immediately AND appears as a **Pending Rule Card** in the Agent Brain UI with **`Keep Permanently (👍)`** / **`Revert (✕)`** buttons.
  * **Validation & Atomic Persistence:** Every patch passes Pydantic validation (valid `phase`, valid `channel_id`, `delta_db` clamped to `[-12.0, +6.0]`, never exceeds `hard_max_db`). Confirmed rules write atomically (`tmp` + `os.replace`) to `mixing_rules.json` with timestamped rollback snapshots in `engine/storage/rule_backups/`.

---

## 6. Reference Schemas & WebSocket Protocol

### 6.1 `engine/schemas/service_plan.json` (Example)

```json
{
  "schema_version": 2,
  "sample_rate": 48000,
  "audio_device_hint": "BlackHole 16ch",
  "midi_mode": "virtual_ports",
  "midi_out_port": "AetherMix To MainStage",
  "midi_in_port": "AetherMix From MainStage",
  "master_tap_channels": [15, 16],
  "channels": [
    {
      "id": "ch1_pastor",
      "label": "Pastor Mic",
      "tap_channels": [1],
      "role": "spoken_primary",
      "vad_enabled": true,
      "feedback_guard": true,
      "midi_channel": 1,
      "vol_cc": 20,
      "reverb_cc": 30,
      "mute_cc": 40,
      "mute_active_high": true,
      "hard_min_db": -60.0,
      "hard_max_db": 3.0,
      "cc_taper": [[0, -60.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0]]
    },
    {
      "id": "ch5_keys",
      "label": "Keys / Pad",
      "tap_channels": [5, 6],
      "role": "instrument_pad",
      "vad_enabled": false,
      "feedback_guard": false,
      "midi_channel": 1,
      "vol_cc": 24,
      "eq_dip_cc": 34,
      "hard_min_db": -60.0,
      "hard_max_db": 0.0,
      "cc_taper": [[0, -60.0], [32, -24.0], [64, -12.0], [96, 0.0], [114, 6.0]]
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

### 6.2 `engine/schemas/mixing_rules.json` (Example)

```json
{
  "schema_version": 2,
  "calibrated_at": null,
  "firewall": {
    "human_lock_seconds": 15.0,
    "echo_suppress_ms": 200,
    "duck_depth_db": -6.0,
    "duck_attack_ms": 150,
    "duck_release_ms": 800,
    "slew_musical_db_per_sec": 4.0,
    "slew_protect_db_per_sec": 36.0,
    "feedback_peak_above_floor_db": 14.0,
    "feedback_hold_ms": 250,
    "feedback_cut_db": -4.0
  },
  "phase_baselines_db": {
    "SPOKEN_WORD":       {"ch1_pastor": 0.0,  "ch2_lead_vox": -24.0, "ch5_keys": -36.0, "ch9_drums": -48.0},
    "PRAYER_UNDERSCORE": {"ch1_pastor": 0.0,  "ch2_lead_vox": -18.0, "ch5_keys": -10.0, "ch9_drums": -48.0},
    "WORSHIP_LOW":       {"ch1_pastor": -24.0,"ch2_lead_vox": 0.0,   "ch5_keys": -4.0,  "ch9_drums": -12.0},
    "WORSHIP_HIGH":      {"ch1_pastor": -36.0,"ch2_lead_vox": 0.0,   "ch5_keys": -3.0,  "ch9_drums": -2.0},
    "TRANSITION":        {"ch1_pastor": -6.0, "ch2_lead_vox": -12.0, "ch5_keys": -14.0, "ch9_drums": -24.0}
  },
  "learned_rules": [
    {
      "id": "rule_001",
      "description": "Keep keys softer under prayer",
      "phase": "PRAYER_UNDERSCORE",
      "channel_id": "ch5_keys",
      "parameter": "vol_db_offset",
      "delta_db": -3.0,
      "enabled": true,
      "source": "operator_confirmed"
    }
  ]
}
```

### 6.3 WebSocket IPC Protocol (`ws://127.0.0.1:8765/ws`)

Envelope: `{ "type": "<str>", "seq": <int>, "ts_ms": <int>, "payload": { ... } }`

* **Server → Client:**
  * `meter_frame` (`30–50 Hz`): `{ "rms_db": [...], "peak_db": [...], "clip": [...], "vad": [...], "master_lufs": -16.4, "target_cc": [...], "actual_cc": [...] }` — routed exclusively to `MeterStore` for `SwiftUI Canvas` rendering (avoids full-view diffing).
  * `state_snapshot` (on connect + every state delta): `{ "autonomy_mode": "SHADOW_OBSERVE", "freeze_learning": false, "active_phase": "PRAYER_UNDERSCORE", "phase_confidence": 0.94, "hysteresis_progress": 1.0, "service_step_idx": 2, "human_locks": {"ch5_keys": 11.2}, "feedback_clamps": {}, "pending_rules": [...], "hardware_warnings": [] }`
  * `brain_event`: `{ "id": "...", "category": "PERCEPTION|ACTION|SAFETY|LLM|RLHF", "summary": "...", "detail": "...", "confidence": 0.92, "ts_ms": ... }`
  * `heartbeat` (`1 Hz`): `{ "uptime_s": 1240, "loop_jitter_ms": 0.8, "audio_ok": true, "midi_ok": true, "ollama_ok": true }`
* **Client → Server:**
  * `set_mode` (`SHADOW_OBSERVE` | `ACTIVE` | `EMERGENCY_BYPASS`)
  * `set_freeze_learning` (`bool`)
  * `override_phase` (`phase`, `hold_seconds`)
  * `advance_service_step` (`step_idx`)
  * `feedback_thumb` (`"up"` | `"down"`)
  * `submit_critique` (`text`)
  * `resolve_pending_rule` (`rule_id`, `"promote"` | `"dismiss"`)
  * `capture_baseline` (`phase`)
  * `midi_learn_pulse` (`channel_id`, `control`: `"vol"` | `"reverb"` | `"mute"`)

### 6.4 Persistence (`telemetry_store.py`)
* SQLite in **WAL mode** with an asynchronous background writer queue.
* Rolling `60 s` full-resolution (`50 Hz`) state ring buffer stays in RAM (powers instant `👎` rollback and `👍` 10-second replay window capture).
* Persists **1 Hz downsampled summaries** + exact discrete records (`phase_transitions`, `human_overrides`, `safety_events`, `feedback_events`, `rule_mutations`) to `engine/storage/telemetry.db`.

---

## 7. Failure-Mode & Resilience Matrix

| Failure / Edge Case | Expected System Behavior |
| :--- | :--- |
| **Python daemon crashes or restarts mid-service** | Never sends zero/mute CCs; MainStage holds current faders. SwiftUI flashes `DISCONNECTED — MANUAL CONTROL` with 1-click restart. On reconnect, engine boots in `SHADOW_OBSERVE` until operator re-enables `ACTIVE`. |
| **MainStage echoes back AetherMix MIDI CCs** | Echo suppression (`±1 CC`, `<= 200 ms`) filters MainStage loopback so Touch-Sense Human Override never self-triggers. |
| **macOS Microphone TCC permission missing / broken BlackHole routing** | `audio_tap.py` detects bitwise all-zero frames for `>2.0 s` and surfaces `HARDWARE_SILENCE_WARNING` banner instead of silently misclassifying as `TRANSITION`. |
| **Loud drums/band bleed into Pastor mic during `WORSHIP_HIGH`** | Auto-ducking is phase-gated and requires `speech_to_band_snr_db >= +6 dB` for `>= 400 ms` during worship, preventing accidental mid-chorus band ducking. |
| **Sustained solo vocal / synth note vs. real mic feedback** | Anti-feedback requires narrowband high-Q bin stationarity + non-decreasing energy for `>= 250 ms`. Real howl triggers immediate `-4 dB` cut + sticky ceiling; musical notes pass untouched. |
| **Ollama unreachable, slow, or emits malformed JSON** | `5.0 s` timeout + Ollama JSON Schema `format` + Pydantic validation. Tier 1 mixing & deterministic rules run uninterrupted; malformed patches are rejected and logged. |
| **Operator presses `👎` twice in a row or after grabbing a fader** | Rollback skips all human-locked channels; second `👎` within `5 s` steps further back (`T - 15 s` / pre-transition snapshot) rather than oscillating. |
| **High-stakes Sunday (e.g., Easter) where zero drift is desired** | Operator toggles **Service Lock (Freeze Learning)**—auto-ducking, rules, and `👎` rollback stay active while RL weight updates are queued offline for post-service review. |

---

## 8. Native macOS App UI Specification (SwiftUI, min `1280×800`)

Dark-mode **"Live Sound Mission Control"** macOS window:

1. **Top Status, Safety & Service Timeline Bar:**
   * Active Phase pill + confidence badge (`● PRAYER_UNDERSCORE · 94%`) + hysteresis progress bar.
   * **Master Autonomy Segmented Control:** `SHADOW OBSERVE` (boot default) | `ACTIVE` | `EMERGENCY BYPASS` (`Cmd+.`).
   * **Service Lock (Freeze Learning)** toggle + Health/Heartbeat LED + `DISCONNECTED — MANUAL CONTROL` emergency banner.
   * Interactive **Order of Service Stepper** + **Capture Soundcheck Baseline** button.
2. **Left Pane — Live Channel Telemetry & Dual-Ring Knob Matrix:**
   * Dedicated `MeterCanvasView` (`SwiftUI Canvas`) rendering `30–60 Hz` dBFS bars, peak-hold ticks, red clip indicators, and green VAD LEDs from `MeterStore`.
   * Dual-ring virtual knobs showing **Agent Target CC** (cyan arc) vs. **Actual MainStage CC** (white needle), live dB readout, **Human Lock countdown badge**, and **Feedback Ceiling indicator**.
3. **Center Pane — "Agent Brain" Live Observability Stream:**
   * Filterable real-time feed (`All`, `Transitions`, `Safety`, `LLM`, `RLHF`) showing perception rationale, dB/CC changes, bandit updates, and interactive **Pending Rule Cards** (`[Promote Permanent]` / `[Revert]`).
4. **Bottom Dock — Tactile RLHF & Critique Console:**
   * Oversized **👍 Good Mix** (`Cmd+Shift+Up`) and **👎 Undo & Penalize** (`Cmd+Shift+Down`) buttons (shortcuts suppressed when text field has focus) with visual rollback toast.
   * Natural-language critique input with macOS `Speech` push-to-talk dictation and quick-chip presets (*"Vocals buried"*, *"Keys too loud"*, *"Too much reverb"*, *"Mute BVs"*).
5. **Setup & MainStage Onboarding Drawer (`SetupWizardView`):**
   * Step 1: Live **Microphone TCC & BlackHole 16ch / Aggregate Device** verification (pre-fader Ch 1–12 + post-fader Ch 15–16 signal check).
   * Step 2: **CoreMIDI Virtual Ports** status + MainStage **Concert-Level Strip** checklist.
   * Step 3: Per-channel **"MIDI Learn Pulse"** buttons (sends an isolated CC sweep for 1-click MainStage Learn) + **Capture Soundcheck Baseline**.
6. **App-Managed Daemon Support (`EngineProcessManager`):**
   * Checks if `127.0.0.1:8765` is already running (supports `make dev`); otherwise can spawn and supervise the local `uv` Python daemon and expose stderr/stdout in a Diagnostics drawer.

---

## 9. Phased Implementation Plan & Verification Gates

* **Phase 1 — Scaffolding, Contracts & Hardware Bridge:**
  * Initialize repo structure, `Makefile`, `engine/pyproject.toml`, `scripts/bootstrap_mac.sh`, `core/protocol.py`, `schemas/*.json`, `midi_bridge.py` (CoreMIDI virtual endpoints + echo filter + deduplication), and `audio_tap.py` (lock-free ring buffer + TCC zero-buffer guard).
  * **Gate:** `scripts/verify_audio_midi.py` verifies live RMS/LUFS & virtual MIDI loopback; headless `pytest` mock I/O tests pass.
* **Phase 2 — Safety Firewall, Sensor-Fused AI Engine & WS Server:**
  * Implement `safety_firewall.py`, `vad_worker.py`, `scene_classifier.py`, `rl_policy.py`, `rules_manager.py`, `llm_supervisor.py`, `telemetry_store.py`, and FastAPI WebSocket server (`main.py`).
  * **Gate:** `pytest` proves: (1) bounds & asymmetric slew enforcement, (2) zero self-lockout under echoed MIDI, (3) immediate human touch yield, (4) anti-feedback cuts stationary howl without false-triggering on musical tones, (5) `👎` rollback respects human locks & supports double-step undo, and (6) malformed/out-of-bounds LLM patches are rejected.
* **Phase 3 — Native SwiftUI Mission Control App:**
  * Build `macos-app/` via `project.yml` + `Package.swift` with `EngineWebSocketClient`, `EngineProcessManager`, `MeterCanvasView`, `MixerTelemetryView`, `AgentBrainStreamView`, `FeedbackControlBar`, and `SetupWizardView`.
  * **Gate:** `xcodegen` + `xcodebuild`/`swift build` succeeds cleanly; app connects to live daemon, renders 30–50 Hz meters smoothly, boots in `SHADOW OBSERVE`, and completes `👍`/`👎` rollback and critique flows end-to-end.
* **Phase 4 — Synthetic Service Rehearsal Harness & End-to-End Validation:**
  * Build `scripts/simulate_service.py` (both **headless mode** and **BlackHole 16ch live playback mode**) replaying realistic multi-stem transitions (`SPOKEN_WORD` → `WORSHIP_LOW` → `WORSHIP_HIGH` → `PRAYER_UNDERSCORE` with pastor speech, keys pad, stage bleed, and simulated feedback ring).
  * **Gate:** End-to-end run verifies hysteresis, bleed-safe auto-ducking, anti-feedback clamp, LLM Brain Stream output, rule promotion, and `👎` rollback.
