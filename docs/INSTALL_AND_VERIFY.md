# AetherMix — Complete New-Mac Installation, Ollama/Qwen & Verification Guide

Use this guide to install **AetherMix** on a fresh Apple Silicon Mac (`M1/M2/M3/M4`, macOS 15+), install all audio/MIDI/AI prerequisites (including **BlackHole 16ch** and **Ollama + Qwen**), verify every dependency automatically, and test real-time mixing in **Apple MainStage** using the packaged 5-tune multi-instrument Sound Pack.

---

## What Is Inside `AetherMix-0.5.1-arm64.dmg`?

Instead of a heavy Electron wrapper, AetherMix packages a **native Swift 6 / SwiftUI macOS app** (`AetherMix.app`) with 60 fps Metal/Canvas metering and all runtime assets bundled inside `AetherMix.app/Contents/Resources/runtime/`:

- **`AetherMix.app`** — Native Mission Control desktop app (drag into `Applications`)
- **Bundled Python AI Engine & Standalone CoreMIDI Bridge** (`bridge/` + `engine/`)
- **Bundled Silero VAD ONNX neural model** (`models/silero_vad.onnx`, works offline immediately)
- **Bundled 5-Tune Multi-Instrument Sound Pack** (`assets/soundpack/`, 37 studio OGG stems + real speech + applause)
- **One-Click Terminal Helpers right inside the DMG:**
  1. `1 - Install Prerequisites & Ollama.command`
  2. `2 - Run Setup Verification Doctor.command`
  3. `3 - Start Live MainStage Sound Pack.command`

---

## Step 1 — Install `AetherMix.app` on the Target Mac

1. Copy **`AetherMix-0.5.1-arm64.dmg`** to the target Apple Silicon Mac and double-click to mount it.
2. Drag **`AetherMix.app`** onto the **`Applications`** folder icon inside the DMG window.
3. *(First launch on another Mac — ad-hoc signed build)*: Right-click `/Applications/AetherMix.app` → click **Open** (or run `xattr -dr com.apple.quarantine /Applications/AetherMix.app` in Terminal).

---

## Step 2 — Install Prerequisites (Homebrew, BlackHole 16ch, Python Runtime, Ollama + Qwen)

Double-click **`1 - Install Prerequisites & Ollama.command`** inside the DMG (or run from Terminal):

```bash
/Applications/AetherMix.app/Contents/Resources/runtime/scripts/install_prereqs.sh
```

> **Note:** If Homebrew is not installed on the target Mac yet, install it first:
> ```bash
> /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
> ```

### What the Installer Sets Up Automatically
1. **Homebrew packages:** `uv`, `portaudio`, `ollama`, and **`blackhole-16ch`** (16-channel virtual CoreAudio bus).
2. **Portable Python 3.11 runtime** in `~/.aethermix/venv` (with `fastapi`, `sounddevice`, `soundfile`, `python-rtmidi`, `onnxruntime`, `scipy`, `numpy`).
3. **Silero VAD ONNX** copied from the DMG bundle into `~/.aethermix/models/silero_vad.onnx`.
4. **Ollama + Qwen model:**
   - Automatically starts the local Ollama daemon (`http://127.0.0.1:11434`).
   - Pulls **`qwen3:14b`** on Macs with **24 GB+ Unified RAM** (or **`qwen3:8b`** on 16 GB Macs).
   - To choose a specific Qwen model manually, run:
     ```bash
     /Applications/AetherMix.app/Contents/Resources/runtime/scripts/install_prereqs.sh --model qwen3:14b
     # or for 16 GB Macs:
     /Applications/AetherMix.app/Contents/Resources/runtime/scripts/install_prereqs.sh --model qwen3:8b
     # or optional larger offline analysis model:
     ollama pull qwen3:32b
     ```
5. **Standalone CoreMIDI Bridge LaunchAgent** (`~/Library/LaunchAgents/com.aethermix.midibridge.plist` with `KeepAlive=true` and `ThrottleInterval=1`).
6. **Writable user config & security token** in `~/.aethermix/config/` and `~/.aethermix/session.token` (`0600`).

If BlackHole 16ch was freshly installed and doesn't show up in CoreAudio yet, reload CoreAudio once:
```bash
sudo killall coreaudiod
```

---

## Step 3 — Verify Every Component Is Installed & Working (Verification Doctor)

Double-click **`2 - Run Setup Verification Doctor.command`** inside the DMG (or run in Terminal):

```bash
python3 /Applications/AetherMix.app/Contents/Resources/runtime/scripts/check_setup.py --test-ollama
```
*(In a git clone, run `make check-setup`.)*

### Checklist Verified Automatically by the Doctor
| # | Check | What It Verifies |
|---|---|---|
| 1 | **Apple Silicon & macOS** | Confirms `arm64`, macOS 15+, and Unified RAM sizing |
| 2 | **Core CLI Tools** | Confirms `brew`, `uv`, and `ollama` are installed |
| 3 | **AetherMix App & Runtime** | Confirms `AetherMix.app` and bundled `bridge/` + `engine/` |
| 4 | **Python 3.11+ Environment** | Imports all 11 audio/MIDI/ML/web packages cleanly |
| 5 | **BlackHole 16ch Driver** | Confirms HAL driver + live `16 in / 16 out @ 48 kHz` CoreAudio device |
| 6 | **Apple MainStage** | Detects `/Applications/MainStage.app` and version |
| 7 | **CoreMIDI & Bridge LaunchAgent** | Probes real virtual CoreMIDI port creation + LaunchAgent health |
| 8 | **Silero VAD ONNX** | Loads `~/.aethermix/models/silero_vad.onnx` and executes a live inference test |
| 9 | **Ollama + Qwen** | Connects to `127.0.0.1:11434`, verifies `qwen3:14b`/`qwen3:8b`, and runs a live structured JSON generation test |
| 10 | **Packaged 5-Tune Sound Pack** | Verifies `manifest.json` + all 37 OGG Vorbis stems |
| 11 | **Session Token & Permissions** | Verifies `~/.aethermix/session.token` exists with `0600` permissions |

### Manual Ollama + Qwen Quick Test (Optional)
```bash
# Confirm Ollama daemon & installed Qwen models:
ollama list

# Quick live response check:
ollama run qwen3:14b "Reply with OK if ready."
```

---

## Step 4 — Grant macOS Microphone Permission (TCC)

macOS blocks CoreAudio input taps (returning all zeros) until Microphone permission is granted:
1. Open **System Settings → Privacy & Security → Microphone**.
2. Enable **AetherMix** (and **Terminal** if you launch CLI scripts).
3. If audio meters stay flat at `-120 dBFS`, restart AetherMix after toggling the permission switch.

---

## Step 5 — Real-Time MainStage Test with the Packaged 5-Tune Sound Pack

1. Double-click **`3 - Start Live MainStage Sound Pack.command`** (or run):
   ```bash
   ~/.aethermix/venv/bin/python /Applications/AetherMix.app/Contents/Resources/runtime/scripts/play_mainstage_pack.py
   ```
   This streams all 5 multi-instrument tunes + real speech + applause across **BlackHole 16ch (Ch 1–14)**, opens the live CoreMIDI virtual port **`AetherMix To MainStage`**, and starts the AI engine on `ws://127.0.0.1:8765/ws`.
2. Open **`/Applications/AetherMix.app`** — you will see live 50 Hz audio meters, real-time phase detection, and the AI Brain Stream.
3. Open **Apple MainStage** → **Settings → Audio**:
   - **Audio Input:** `BlackHole 16ch`
   - **Audio Output:** `MacBook Pro Speakers` (or your PA audio interface)
4. Create Audio Channel Strips for **Input 1** (Pastor), **2** (Lead Vox), **3** (BV 1), **4** (BV 2), **5–6** (Keys), **7–8** (Guitars), **9–10** (Drums), **11–12** (Bass), and map faders to **`AetherMix To MainStage` (`CC 20–26`, `CC 30–31`)** using MainStage's **Assign & Map** (or AetherMix's **Setup Wizard → MIDI Learn Pulse**).
5. Switch AetherMix from **SHADOW** to **ACTIVE** and watch MainStage's faders mix the 5 live songs, duck the band during pastor speech, and protect vocal mics automatically!

---

## Building a Fresh DMG From Source (Developers)

```bash
make dmg
# Output: dist/AetherMix-0.5.1-arm64.dmg
```
