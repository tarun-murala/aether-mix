# Plug-and-Play Apple MainStage Live Testing Guide

AetherMix packages **5 distinct multi-instrument studio tunes** (plus real recorded human speech and stereo room applause) directly inside [`assets/soundpack/`](../assets/soundpack/manifest.json)—no external stem downloads required.

## Packaged Multi-Instrument Tunes

| # | Title | Style / Instruments | Tempo & Key | Loop |
|---|---|---|---|---|
| 1 | **Morning Grace** | Warm Worship Ballad — Grand Piano, Warm Pad, Fingerpicked Steel Acoustic, Clean Electric Swells, Ballad Kit, Fingered Bass, 3-Part Vocals | 72 BPM · G Major | 26.7 s |
| 2 | **Anthem Rise** | Modern Pop-Rock — Bright Piano, Synth Pad, Dotted-8th Delay Electric Riff, Overdriven Rhythm Guitars, Full Rock Kit, Picked Bass, Anthemic Vocals | 124 BPM · D Major | 31.0 s |
| 3 | **Sunday Soul** | Gospel Groove — Fender Rhodes EP, Hammond B3 Organ, Funky Jazz Clean Guitar, Ghost-Note Gospel Drums, Walking Bass, Gospel Choir | 92 BPM · Bb Major | 20.9 s |
| 4 | **Acoustic Campfire** | Unplugged Folk — Soft Piano, Warm Strings, Strummed Steel Guitar, Nylon Counter-Melody, Shaker/Tambourine/Rimshot, Upright Double Bass | 108 BPM · C Major | 26.7 s |
| 5 | **Still Waters** | Ambient Prayer & Post-Rock Swell — Lush Atmospheric Pads, Felt Piano, Echoing Clean Electric, Cinematic Floor Toms, Fretless Bass, Ethereal Oohs | 66 BPM · E Minor | 29.1 s |

---

## 16-Channel MainStage Audio & MIDI Map

| BlackHole 16ch Input | MainStage Channel Strip | AetherMix ID | Volume CC (Ch 1) | Reverb CC | Mute CC |
|---|---|---|---|---|---|
| **Input 1** (Mono) | Pastor Mic (Real Speech) | `ch1_pastor` | `CC 20` | `CC 30` | — |
| **Input 2** (Mono) | Lead Vocal | `ch2_lead_vox` | `CC 21` | `CC 31` | `CC 41` |
| **Input 3** (Mono) | Backup Vocal 1 | `ch3_bv1` | `CC 22` | — | `CC 42` |
| **Input 4** (Mono) | Backup Vocal 2 | `ch4_bv2` | `CC 23` | — | `CC 43` |
| **Input 5–6** (Stereo) | Keys / Piano / Rhodes / Organ / Pad | `ch5_keys` | `CC 24` | — | — |
| **Input 7–8** (Stereo) | Acoustic & Electric Guitars | `ch7_guitars` | `CC 25` | — | — |
| **Input 9–10** (Stereo) | Drums / Percussion | `ch9_drums` | `CC 26` | — | — |
| **Input 11–12** (Stereo) | Bass (Electric / Upright / Fretless) | `ch9_drums` | `CC 26` | — | — |
| **Input 13–14** (Stereo) | Room Mics / Applause (Direct Tap) | `room_tap` | — | — | — |
| **Input 15–16** (Stereo) | Master Return Tap (Optional LUFS) | `master_tap` | — | — | — |

---

## Quick Start (60 Seconds)

### 1. Start the Live 16-Channel Stream + CoreMIDI Bridge + Engine
```bash
# Full live church service flow cycling through all 5 tunes + speech + applause:
make live-mainstage

# Or jump directly to a specific tune (1..5):
engine/.venv/bin/python scripts/play_mainstage_pack.py --tune 2

# Or interactive keyboard jukebox (stream-only while running the AetherMix app):
engine/.venv/bin/python scripts/play_mainstage_pack.py --player-only --mode jukebox --interactive
```

*(Want to preview the tunes immediately on your Mac speakers without MainStage? Run `engine/.venv/bin/python scripts/play_mainstage_pack.py --audition --mode jukebox`!)*

### 2. Plug into Apple MainStage
1. Open **MainStage** → **Settings → Audio**:
   - **Audio Input:** `BlackHole 16ch`
   - **Audio Output:** `MacBook Pro Speakers` (or your studio monitors / PA interface)
2. Create Concert-Level **Audio** channel strips for Inputs `1`, `2`, `3`, `4`, `5-6`, `7-8`, `9-10`, `11-12` (enable **Input Monitoring** `I` on each strip).
3. In MainStage **Layout** mode, select each fader → click **Assign & Map** → in the **AetherMix** macOS app click **Setup Wizard → MIDI Learn Pulse** (or arm **Active** mode) so MainStage locks onto `AetherMix To MainStage` (`CC 20–26`, `CC 30–31`).

### 3. Watch & Control Real-Time AI Mixing
Launch the native macOS UI:
```bash
macos-app/.build/debug/AetherMix
```
Switch from **SHADOW** to **ACTIVE** to watch MainStage's faders glide smoothly in real time as the tunes change between Acoustic Worship, Full-Band Pop-Rock, Gospel Groove, Pastor Speech Ducking, Sermon Solo, and Applause!
