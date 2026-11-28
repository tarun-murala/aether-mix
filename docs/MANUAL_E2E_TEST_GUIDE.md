<!-- Manual end-to-end test guide. Source: Claude Doc https://claude.ai/artifact/E5w9KP4cxBBt8EgqAuzFHG (exported 2026-11-28). -->
# Aether-Mix manual end-to-end test guide

## Overview

This guide walks you through one full manual pass of Aether-Mix on your Mac: the packaged sound pack plays into MainStage, the engine listens, and you watch MainStage faders move in response. Plan on about 45 minutes for the first pass and 15 minutes for each pass after that.

There are two levels. Do **Level 1** first; only move to Level 2 once every Level 1 scenario passes.

| Level | What you prove | Audio routing | Time |
| --- | --- | --- | --- |
| 1. Open loop | Phase detection, ducking, instrument open, feedback cut, human override all drive MainStage faders | BlackHole 16ch only | 30–45 min |
| 2. Closed loop | The engine hears what MainStage actually outputs (fader moves change what it measures) | AetherMix-Live aggregate (BlackHole 16ch + 64ch) | +30 min |

In Level 1 the engine reads the raw stems on BlackHole 16ch, so it never hears MainStage's faders and the master tap (15–16) is silent. That is expected, not a bug.

**What you need**

- The repo at `~/git/aether-mix` with `engine/.venv` built (`make setup`) and the app built (`macos-app/.build/release/AetherMix` exists).
- MainStage, BlackHole 16ch, Ollama with `qwen3:14b` (or `qwen3:8b`), and the bridge LaunchAgent installed (`make install-bridge`).
- BlackHole 64ch for Level 2 only (`brew install blackhole-64ch`).
- Headphones or speakers on your Mac output.

**Before you start:** make sure nothing else (an eval run, a second player) is streaming to BlackHole 16ch, because two players at once make every result meaningless. If your working tree has uncommitted changes to `assets/soundpack/01_morning_grace/*.ogg`, note that local edits to the pack have caused false anti-feedback cuts on tune 3 before. If you see that, run `git stash` on those files or test with the committed pack.

## One-time setup

Do this once per Mac. Skip any step you already did and that still checks out.

### Audio MIDI Setup (Applications → Utilities → Audio MIDI Setup)

1. Select **BlackHole 16ch**. Set Format to **48,000 Hz**. It should show 16 in / 16 out.
2. (Level 2 only) Select **BlackHole 64ch** and set it to **48,000 Hz**.
3. (Level 2 only) Click **+** → **Create Aggregate Device**, rename it exactly `AetherMix-Live`. Tick the sub-devices in this order: your speakers or audio interface first, then BlackHole 16ch, then BlackHole 64ch. Set **Clock Source** to the speakers or interface. Tick **Drift Correction** on both BlackHole rows. Set the format to 48 kHz.
4. Write down the aggregate's channel numbering. It follows the sub-device order, so BlackHole 16ch and 64ch channels start after your speaker channels (for MacBook speakers, 16ch is inputs 1–16 because speakers have no inputs, but outputs shift by 2).
5. If BlackHole does not show up at all, run `sudo killall coreaudiod` and reopen Audio MIDI Setup.

You do **not** need the IAC Driver. The bridge creates its own virtual MIDI port, `AetherMix To MainStage`, whenever it is running.

### Permissions

1. System Settings → Privacy & Security → **Microphone**: enable **AetherMix** and **Terminal** (or iTerm). Without this, every meter sits at −120 dBFS.
2. Restart the app after changing the toggle.

### Verify the install

In Terminal, from `~/git/aether-mix`:

```bash
make check-setup                 # 11 checks; all should pass, including Ollama JSON test
launchctl print gui/$(id -u)/com.aethermix.midibridge | grep state   # expect: state = running
ollama list                      # expect qwen3:14b or qwen3:8b
engine/.venv/bin/python scripts/run_eval.py routing-check   # Level 2: OK for 16ch, 64ch and AetherMix-Live
```

If `make check-setup` fails, fix that item first using [Troubleshooting](#troubleshooting) before going further.

## MainStage concert setup

You build one concert with 8 audio channel strips at **Concert level**, each fed by a BlackHole 16ch input and each fader mapped to a CC on `AetherMix To MainStage`. Save it as `AetherMix Test.concert` so you only do this once. The bridge must be running while you map, or the port won't appear.

1. MainStage → Settings → Audio. Level 1: **Input** `BlackHole 16ch`, **Output** your speakers. Level 2: input and output `AetherMix-Live`, buffer 128. Sample rate 48 kHz.
2. File → New Concert → an empty template. Delete any default instrument strips.
3. At the concert level (click the concert name in the Concerts list), add the 8 **Audio** channel strips in the table below. On each, set the input and turn **Input Monitoring** (the `I` button) on.
4. Leave strip 13–14 (room) unmapped. It is only for Level 2.

| # | Strip name | Input (BlackHole 16ch) | Mono/Stereo | What plays on it | Engine channel | Volume CC (MIDI ch 1) | Reverb CC | Mute CC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Pastor | 1 | Mono | Real recorded speech (male and female) | `ch1_pastor` | 20 | 30 | none (auto-mute is forbidden) |
| 2 | Lead Vox | 2 | Mono | Lead vocal stem; also the 2.48 kHz feedback test tone | `ch2_lead_vox` | 21 | 31 | 41 |
| 3 | BV 1 | 3 | Mono | Backup vocal 1 | `ch3_bv1` | 22 | none | 42 |
| 4 | BV 2 | 4 | Mono | Backup vocal 2 | `ch4_bv2` | 23 | none | 43 |
| 5 | Keys | 5–6 | Stereo | Piano, Rhodes, organ, pads | `ch5_keys` | 24 | none | none |
| 6 | Guitars | 7–8 | Stereo | Acoustic and electric guitars | `ch7_guitars` | 25 | none | none |
| 7 | Drums | 9–10 | Stereo | Drums and percussion | `ch9_drums` | 26 | none | none |
| 8 | Bass | 11–12 | Stereo | Bass | `ch9_drums` (shared) | 26 (same as Drums) | none | none |
| – | Room (Level 2) | 13–14 | Stereo | Congregation applause and room tone | `room_tap` | none | none | none |

Drums and Bass are one engine channel, so both MainStage faders map to CC 26 and move together. That is by design.

### Map the faders

1. In MainStage, remove every direct assignment from any hardware controller (X-Touch Mini or similar). The controller must only talk to the bridge, never to MainStage.
2. Start the bridge and app first (see Launch order, steps 1–4), then come back here.
3. In **Layout** mode, select a strip's volume fader → **Assign & Map** → **Learn**.
4. In the Aether-Mix app, open **Setup → step 4 (MIDI Learn)** and click **Vol CC 20** (or the matching CC). MainStage should learn `AetherMix To MainStage, ch 1, CC 20`.
5. Repeat for each row. For Pastor and Lead Vox reverb, map the reverb send knob with **Rev CC 30** and **Rev CC 31**.
6. For BV/Lead mutes, map the mute button to CC 41–43, then on the same wizard step click **Verify** to record whether 127 or 0 means mute.
7. Back on **Setup → step 3**, tick the confirmation box and click **Auto-Verify**. Expect a pass with no direct hardware assignments found.
8. Save the concert.

### Level 2 extra routing (closed loop)

1. On every one of the 8 strips, set **Send 1** to **Pre fader** at 0 dB into its own aux, and route that aux to the matching BlackHole 64ch channel (strip input 1 → 64ch 1, inputs 5–6 → 64ch 5–6, and so on up to 12).
2. Add the Room strip on input 13–14, output straight to BlackHole 64ch 13–14, no CC mapped.
3. On every strip, set **Send 2** to **Post fader** at 0 dB into one shared **Master Tap** aux on BlackHole 64ch 15–16 (include the reverb return). Keep the main output strip flat at 0 dB.
4. Run `engine/.venv/bin/python scripts/run_eval.py closedloop-plan` once. It writes `engine/schemas/service_plan.closedloop.json`, which tells the engine to listen on BlackHole 64ch.

## Launch order

Start things in this order every time. Each step has a check; don't move on until it passes.

**Use the player in `--player-only` mode.** `make live-mainstage` and `make dev` each start their own bridge, which takes over `~/.aethermix/bridge.sock` from the LaunchAgent bridge and can leave two `AetherMix To MainStage` ports. With the LaunchAgent running, let the app own the engine and run the player audio-only.

1. **Ollama.** Open the Ollama app, or run `ollama serve` in its own Terminal tab. Check: `curl -s 127.0.0.1:11434/api/tags` lists `qwen3:14b`.
2. **MIDI bridge daemon.** It starts on login through launchd. Check: `launchctl print gui/$(id -u)/com.aethermix.midibridge | grep state` says `running`, and `tail -f ~/.aethermix/bridge.stderr.log` shows the virtual port opened. If not, run `make install-bridge`.
3. **MainStage.** Open `AetherMix Test.concert`. Check: Settings → MIDI lists `AetherMix To MainStage` as an input, and the audio input is BlackHole 16ch (Level 1) or AetherMix-Live (Level 2).
4. **Engine (Level 2 only).** In a Terminal tab: `engine/.venv/bin/python engine/main.py --service-plan engine/schemas/service_plan.closedloop.json`. Check: the log says it is listening on `127.0.0.1:8765`. In Level 1, skip this; the app starts the engine for you.
5. **Aether-Mix app.** Run `macos-app/.build/release/AetherMix` from the repo root. Check: Setup → step 1 (System Check) shows engine, bridge and MainStage MIDI link green. The diagnostics drawer says `managedSubprocess` (Level 1) or attached to a running engine (Level 2). Mode is **SHADOW** and every channel shows **SAFE**.
6. **Audio player.** In a new Terminal tab:

   ```bash
   engine/.venv/bin/python scripts/play_mainstage_pack.py --player-only --mode jukebox --interactive
   ```

   Check: MainStage input meters move on strips 1–8 and the app's meters move. The keys are below.

| Key | What it does | Use it for |
| --- | --- | --- |
| `1`–`5` | Switch to tune 1 Morning Grace, 2 Anthem Rise, 3 Sunday Soul, 4 Acoustic Campfire, 5 Still Waters | Worship phases |
| `b` | Toggle acoustic only (WORSHIP_LOW) vs full band (WORSHIP_HIGH) | Phase changes |
| `p` | Toggle pastor speech on ch 1 over whatever is playing | Ducking |
| `s` | Sermon solo: pastor only, band silent | Spoken word, then instrument open |
| `a` | 5 s of applause on ch 13–14 | Transition |
| `f` | 2.5 s, 2.48 kHz howl on Lead Vox (ch 2) | Feedback |
| `0` | Back to auto-cycle | Soak test |
| `q` | Quit |  |

For a hands-off run, drop `--mode jukebox --interactive` and the player cycles an 8-scene service of about 2 minutes: prayer underscore, acoustic worship, full band, pastor interrupt over full band, gospel, applause, sermon solo, folk.

### Arm and go ACTIVE

1. Let one tune play for 10 s in SHADOW. The phase pill should read `WORSHIP HI` or `WORSHIP LO`, but MainStage faders must **not** move. If they move in SHADOW, stop: that is a bug.
2. In the mixer panel, click **SAFE** on each of the 7 channels so it reads **ARMED**.
3. Click **SYNC?** on each channel. It turns **SYNCED** and the MainStage fader glides at 2 dB/s to the engine's value.
4. Click **ACTIVE** in the mode switch at the top. You are now ready for the scenarios.

## Test scenarios

Run these in order with the player's Terminal tab focused, so the keys reach it. Watch three places: the MainStage faders, the app's phase pill and mixer panel, and the Brain Stream. Fader values below are the engine's phase baselines from `engine/schemas/mixing_rules.json`; MainStage shows them on its own taper, so expect them within about 1 dB.

### T1. Worship phases

1. Press `1` (Morning Grace), then `b` for acoustic only.
2. Wait up to 10 s for the hysteresis bar under the phase pill to fill.
3. Press `2` (Anthem Rise), then `b` again for full band.

**Expect:** phase `WORSHIP LO` after step 2, with Lead 0, BV −8, Keys −4, Guitars −6, Drums/Bass −12, Pastor −18 dB. After step 3, `WORSHIP HI` with Drums/Bass −2, Keys −3, Guitars −4, BV −5 dB. Faders glide at about 4 dB/s, never jump.

### T2. Sermon ducking

1. With tune 2 playing in WORSHIP HI, press `p` (pastor on over the band).
2. Wait 10 s, then press `p` again.
3. Press `s` (sermon solo) and wait 15 s.

**Expect:** within about 0.4 s of step 1, Brain Stream shows `Speech-open reflex engaged on ch1_pastor` and the Pastor fader rises fast (48 dB/s). Keys, Guitars and Drums/Bass duck by about 6 dB. About 3 s after step 2 the pastor releases and the band comes back at about 8 dB/s. In step 3 the phase moves to `SPOKEN` with Pastor 0, Lead −20, BV −24, Keys/Guitars/Drums −16 dB. Pastor is never muted.

### T3. Instrument open

1. Stay in sermon solo (T2 step 3) for at least 3 s so the band is fully silent.
2. Press `2` to bring the full band back in at once.
3. Repeat with `4` (Acoustic Campfire).
4. Negative check: press `5` (Still Waters) and `p` to get the prayer underscore (pad plus pastor), then press `2`.

**Expect:** within about 0.2–1 s of step 2, Brain Stream shows `Instrument-open reflex engaged (...)` and the band faders rise at about 24 dB/s, so the music isn't buried under sermon levels. The reflex either hands over to WORSHIP once the phase changes or shows `Instrument-open reflex auto-released after 8s guard`. In step 4 there must be **no** instrument-open event while the phase is PRAYER UNDERSCORE. Note any open that fires on its own during the sermon (a known risk from a single drum or bass hit).

### T4. Feedback

1. Press `1` and let it settle in WORSHIP LO.
2. Press `f` once (2.5 s howl on Lead Vox).
3. Press `f` three times in a row, about 2 s apart.
4. Negative check: press `3` (Sunday Soul) and let it play 60 s.

**Expect:** within about 0.25 s of the howl, a SAFETY event on `ch2_lead_vox` naming the bin and a `ceiling=` value, and the Lead Vox fader drops 4 dB. Repeated howls step the cut down by 4 dB each time to a maximum of −16 dB. About 5 s after the last howl the fader climbs back at about 1 dB/s. Step 4 must produce **no** feedback cuts on any channel, especially BV 2.

### T5. Human override

1. In WORSHIP HI, drag the Lead Vox fader in the app's mixer panel up about 4 dB (or move it on the X-Touch Mini if connected).
2. Within 15 s, press `p` to trigger a pastor duck.
3. After 20 s, click **Release Offset** on Lead Vox.
4. Repeat step 1, then press `f` while the touch lock is on.

**Expect:** MainStage follows your move and the engine stops writing that fader for 15 s (touch lock). Other channels still duck in step 2. Your +4 dB offset is held after the lock expires, until a phase change or step 3, where Lead Vox glides back at about 4 dB/s. In step 4 the feedback cut still happens: anti-feedback overrides your touch lock.

### T6. Phase override

1. Click the phase pill → **Override → Spoken Word** while a tune plays.
2. Then **Override → Worship High**.

**Expect:** faders glide to the SPOKEN baselines (T2) and then the WORSHIP HI baselines (T1), regardless of what the classifier hears.

### T7. Undo and critique

1. After a fader move you dislike, press **⌘⇧↓** once. Then press it twice within 5 s.
2. Press **⌘⇧↑** on a mix you like.
3. In the Brain Stream critique box type `lead vocal is too quiet in the chorus` and click **Send**. Then send `turn the drums up 10 dB`.

**Expect:** one press rolls faders back to where they were 5 s ago; two presses go back 15 s (or to before the last transition) and hold for 10 s, skipping touch-locked faders. The first critique returns a valid rule and auto-applies (within ±3 dB). The second exceeds ±3 dB, so it must land in **Review** for approval instead of applying.

### T8. Safety and resilience

1. Click **BYPASS**. Then try **ACTIVE** directly.
2. Click **SHADOW**, re-arm and go **ACTIVE**.
3. In Terminal: `pkill -9 -f midi_bridge_daemon.py`.
4. Quit the app (which stops its engine), then relaunch it.

**Expect:** in BYPASS the engine stops writing and faders hold. ACTIVE is blocked from BYPASS; you must pass through SHADOW. After step 3, launchd restarts the bridge in under 2 s, the engine drops to SHADOW and channels show `SYNC?` again. After step 4 the app comes back in SHADOW with channels SAFE, and **Restore Arming** brings back your arming. At no point does any MainStage fader go to zero.

### T9. Soak

1. Press `0` for auto-cycle and leave it ACTIVE for 30 minutes.

**Expect:** no `WATCHDOG_DROP_TO_SHADOW` events, no stuck faders, no CoreAudio overloads in MainStage's CPU meter.

### T10. Closed loop (Level 2 only)

1. With the closed-loop engine running and ACTIVE: `engine/.venv/bin/python scripts/run_eval.py prefader-check`.
2. Then a scored run: `engine/.venv/bin/python scripts/run_eval.py live svc1 --profile quick --allow-synthetic --no-fail`.

**Expect:** step 1 passes, which means pre-fader taps 1–12 move no more than 0.1 dB while the master tap 15–16 follows the fader by at least 20 dB. Step 2 writes a run folder under `recordings/eval_runs/` with a gate report; the app's master LUFS reading now changes when faders move.

## Troubleshooting

When a scenario fails, work down the signal path one hop at a time instead of restarting everything. Each hop has a quick check.

### The loop

1. **Write down** the scenario, the step, what you expected and what happened (use the log template below).
2. **Find the broken hop** with the table below, starting from the player and moving toward MainStage: player → BlackHole → engine meters → phase → Brain Stream event → bridge → MainStage fader.
3. **Fix only that hop**, then rerun only the failed scenario from its step 1.
4. If it passes, rerun the scenario before it too (fixes sometimes break the previous one), then move on.
5. If the same scenario fails three times with a different fix each time, stop and save the evidence: `~/.aethermix/bridge.stderr.log`, the app's diagnostics drawer logs, the engine `telemetry.db`, and a screen recording. Attach those to the bug report.
6. When all of T1–T9 pass in one session, do one clean restart (quit everything, follow Launch order from the top) and run T1, T2, T3 and T4 again. That is your "it works" bar for Level 1.

### Symptom → check → fix

| Symptom | Check | Fix |
| --- | --- | --- |
| Player errors `device matching 'BlackHole 16ch' ... not found` | Audio MIDI Setup lists BlackHole 16ch at 48 kHz | `brew reinstall --cask blackhole-16ch`, then `sudo killall coreaudiod` |
| MainStage strips show no input | Strip input numbers match the table; Input Monitoring `I` is on; MainStage audio input is BlackHole 16ch | Fix the strip input or the device in Settings → Audio |
| App meters flat at −120 dBFS | Microphone permission for AetherMix and Terminal | Enable it, quit and relaunch the app |
| Phase pill stuck or wrong | Confidence %, hysteresis bar filling | Wait 10 s; if still wrong, note it as a classifier bug and use Override to keep testing |
| Brain Stream shows events, MainStage faders don't move | Mode is ACTIVE; channel ARMED and SYNCED; Setup step 1 shows bridge green | Arm and SYNC; `launchctl kickstart -k gui/$(id -u)/com.aethermix.midibridge` |
| Faders move but the wrong strip moves | MainStage Assign & Map shows the CC from the strip table | Re-learn that fader with MIDI Learn Pulse |
| Two `AetherMix To MainStage` ports in MainStage | `pgrep -fl midi_bridge_daemon` shows more than one process | Stop `make dev` / `make live-mainstage`; keep only the LaunchAgent bridge |
| Faders jump or fight you | Hardware controller still mapped directly in MainStage | Remove direct assignments, rerun Auto-Verify |
| App says engine stopped | Diagnostics drawer stderr; port 8765 already in use (`lsof -i :8765`) | Kill the stray engine, relaunch the app |
| Feedback cut on tune 3 BV 2 with no howl | `git status` shows modified `.ogg` files | Test with the committed sound pack |
| Engine drops to SHADOW on its own | Brain Stream shows `WATCHDOG_DROP_TO_SHADOW` | Close heavy apps, raise MainStage buffer to 256, re-arm |
| Critique never answers | `ollama list`; `curl -s 127.0.0.1:11434/api/tags` | Start Ollama; `ollama pull qwen3:14b` |
| Level 2: engine hears nothing | `run_eval.py routing-check`; Send 1 is Pre fader into BlackHole 64ch | Fix aux outputs; engine must run with `service_plan.closedloop.json` |

Useful live views while you test:

```bash
tail -f ~/.aethermix/bridge.stderr.log                        # every CC the bridge sends
log stream --predicate 'process == "MainStage"' | grep -i overload   # CoreAudio overloads
engine/.venv/bin/python scripts/verify_audio_midi.py          # scripted audio and MIDI check
```

## Results log

Fill one row per scenario each session. Put the exact Brain Stream text or fader value in Notes when something fails.

| Scenario | Level | Result | Notes |
| --- | --- | --- | --- |
| T1 Worship phases | 1 | Not run |  |
| T2 Sermon ducking | 1 | Not run |  |
| T3 Instrument open | 1 | Not run |  |
| T4 Feedback | 1 | Not run |  |
| T5 Human override | 1 | Not run |  |
| T6 Phase override | 1 | Not run |  |
| T7 Undo and critique | 1 | Not run |  |
| T8 Safety and resilience | 1 | Not run |  |
| T9 Soak (30 min) | 1 | Not run |  |
| T10 Closed loop | 2 | Not run |  |
