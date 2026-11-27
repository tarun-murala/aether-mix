# MainStage Multitrack Eval Spec

Status: v0.1, written against SPEC.md v5.1. Tracks, scripts, manifests and tests live in this repo; the audio does not.

## 1. Purpose

Test AetherMix across many real inputs: 5 different recordings per instrument slot (keys, acoustic guitar,
electric guitar, bass, drums), vocal songs with backing vocals, choir, and sermon (spoken word), then score it against
the SPEC section 9 gates. Each of 5 scripted ~25 minute services (Latin square) plays a different source in every slot, so
every slot is heard 5 ways and a gate passes only if it holds in all 5.

## 2. Three modes

| Mode | What runs | Scored by | Status |
| --- | --- | --- | --- |
| **A+ headless closed loop** | Stems -> `evals` timeline -> the real `AetherMixEngine` 50 Hz pipeline. The post-fader master (taps 15-16) is rebuilt every tick from the engine's own fader state. | `make eval-*` (automatic) | Implemented |
| **B MainStage closed loop** | Player -> BlackHole 16ch -> MainStage -> pre/post taps on BlackHole 64ch -> engine -> bridge -> MainStage CC | Manual gates + the Phase 0 scripts (see the runbook) | Rig steps documented; automatic scoring of live captures not implemented |
| **C physical inputs** | Interface mic/line inputs in the aggregate; a human plays and talks | Manual | Documented |

A+ exercises perception, the safety firewall, the offset learner and the LUFS meter exactly as the live engine does. It does
not exercise MainStage, CoreMIDI, BlackHole, the bridge daemon or real latency. Do not read an A+ pass as a MainStage pass.

## 3. Quick start

```bash
make eval-list                       # services x slots, which sources are on disk
make eval-smoke                      # ~10 s per service on synthetic stand-ins, labelled PROVISIONAL
make eval-fetch ACCEPT_TERMS=1       # download what is automatic (read the license notes in section 5 first)
make eval-audition                   # how each multitrack was split into parts: LISTEN before trusting a pick
make eval-quick                      # 6 min of audio per service (~75 s to run), real audio only (refuses synthetic stand-ins)
make eval-full                       # 26 min of audio per service (~5 min to run)
make eval-test                       # unit and integration tests for the suite itself
```

`run_eval.py run` refuses to score when any slot would use a synthetic stand-in unless `--allow-synthetic` is passed, and
such reports are headed PROVISIONAL with the stand-in parts listed. Reports land in `recordings/eval_runs/<timestamp>/`
(`report.md`, `report.json`, per-service `ticks.npz`, `markers.csv`, tap trims; `--write-audio` adds the AetherMix master,
a fixed static mix and a blind listening sheet).

## 4. Track list (5 per slot)

All Cambridge-MT picks were made from the library index, which lists genre and track counts but no instruments. Every pick is
a candidate until you have auditioned it (`make eval-audition`). Replace any dud with another artist of the same genre by
editing `evals/manifest/sources.yaml`; the tests check that the five sources in a slot stay distinct.

| Slot | Svc 1 | Svc 2 | Svc 3 | Svc 4 | Svc 5 |
| --- | --- | --- | --- | --- | --- |
| Keys / piano (In 5-6, `ch5_keys`) | Jesper Buhl Trio (Acoustic Jazz, 6 tr, 332 MB) | H-owl Project (Jazz Trio, 10 tr, 344 MB) | Studio ROM - PHONOMETRY I: Calm Razor Portrait (Neo-classical Ambient Electronica, 28 tr, 587 MB) | Kai Roger Willumsen - Vintage Synthesizer 70s And 80s Dreams (70s/80s Instrumental Electronica, 31 tr, 436 MB) | MainStage Vintage B3 + pad, played live (user) |
| Acoustic guitar (In 7-8, `ch7_guitars`) | David Youngs - Transience (Intricate Solo Acoustic Guitar, 4 tr, 75 MB) | Jesse Joy (Fingerstyle Acoustic Folk, 19 tr, 131 MB) | Timo Carlier (Acoustic Singer-Songwriter, 9 tr, 170 MB) | Andrew Cole (Acoustic Folk-Pop, 33 tr, 756 MB) | Santi Vega (Intricate Solo Acoustic Guitar, 1 tr, 53 MB) |
| Electric guitar (In 7-8, `ch7_guitars`) | Rod Alexander (Guitar Instrumental, 14 tr, 218 MB) | Neon Hornet (Grungey Blues Rock, 15 tr, 413 MB) | BMorgan&FRIENDS (Christian Rock, 15 tr, 681 MB) | Blues Bastards (Blues Rock, 34 tr, 637 MB) | Hurray For The Riff Raff (Indie Rock, 20 tr, 612 MB) |
| Bass (In 11-12, `ch9_drums`) | Marcus Boeltz (Funk Rock, 15 tr, 264 MB) | G-Bass Project (Downtempo Neo-Soul, 14 tr, 352 MB) | Ghost Note (World Funk, 18 tr, 688 MB) | Phre The Eon (Indie Funk/Rock, 25 tr, 450 MB) | Ikebe Shakedown (Funk, 18 tr, 550 MB) |
| Drums (In 9-10, `ch9_drums`) | Ripe (Funk, 18 tr, 624 MB) | Umphrey's McGee (Jam Rock, 25 tr, 833 MB) | Funky Dawgs Brass Band (New Orleans Funk/R&B, 25 tr, 776 MB) | Big Mean Sound Machine (Afrobeat, 21 tr, 1200 MB) | Alan Evans Trio (Live Funk Rock, 16 tr, 1700 MB) |
| Vocal song: lead + 2 BVs + own band (In 2-4) | Kat Wright (Soul Singer-Songwriter, 24 tr, 713 MB) | Remember Jones (Soul/Pop, 23 tr, 683 MB) | BMorgan&FRIENDS (Christian Rock, 15 tr, 681 MB) | Jay Menon (Intimate Pop Ballad, 23 tr, 357 MB) | Nicki Bluhm & The Gramblers (Live Country Soul, 19 tr, 502 MB) |
| Choir / gospel (BV slots 3-4) | Don Camillo Choir - Kasimir's Adventure (German Unaccompanied Choir, 20 tr, 331 MB) | Öjebokören (Choral musical, 7 tr, 157 MB) | The Sunshine Garcia Band (Reggae/Gospel, 56 tr, 1300 MB) | Mirroman (A Capella Duo, 3 tr, 62 MB) | thelvnguage (Indie Soul, 15 tr, 269 MB) |
| Sermon / spoken (In 1, `ch1_pastor`) | Spurgeon's Sermons, May 1858 (LibriVox) (male reader) | Wesley, Sermons on Several Occasions, First Series (LibriVox) (reader(s)) | LibriSpeech dev-clean female readers, stitched (OpenSLR) | Your church's own recorded sermon (user) | Volunteer speaking into a real mic (announcements, quiet prayer, mic handling) (user) |
| Room / applause (In 13-14, direct) | Packaged applause bed (repo soundpack) (repo) | Freesound: church applause (CC0 only) (Freesound) | Freesound: congregation room tone (CC0 only) (Freesound) | Room tone and claps recorded in your own sanctuary (user) | Freesound: applause with cheering (CC0 only) (Freesound) |

Notes:
- Vocal songs need a male/female lead split (aim for 2+2 of the 5); the index shows no singer gender, so check while auditioning.
- Gospel organ is a gap in the library, so Service 5 keys is a MainStage B3/pad patch played live and recorded.
- Feedback howls are synthesized (2.5 kHz, +3 dB/s from -28 dBFS) so frequency and growth are known exactly.
- Sermon sources: LibriVox readings are calm read speech with no congregation. Service 4 (your own sermon) is the only realistic preaching source.

## 5. Licensing and provenance

| Provider | License | What the suite does |
| --- | --- | --- |
| [Cambridge-MT](https://www.cambridge-mt.com/ms/mtk/) multitracks | No formal license. The [FAQ](https://cambridge-mt.com/ms/mtk-faq) allows classroom and not-for-profit use, does not address software evaluation or AI, and forbids packaging into a commercial product. | `commercial_use: unknown`. Fetch needs `--accept-terms`. Audio stays under `recordings/` (gitignored) and out of the soundpack and DMG. Ask the site owner (contact on the [About page](https://cambridge-mt.com/about)) before any commercial use. |
| LibriVox ([Spurgeon](https://librivox.app/book/1131), [Wesley](https://archive.org/details/sermonsfirstseries_1307_librivox)) | Public domain (US) | Downloaded via archive.org. |
| [LibriSpeech](https://www.openslr.org/12) | CC BY 4.0 | Credit the corpus. Fetch needs `--accept-terms`. |
| [Freesound](https://freesound.org/search/?q=church+applause) | Use CC0 clips only | Manual download into `recordings/stems/eval/<id>/`. |
| Your own recordings | Yours; get the pastor's permission | Manual. |

`fetch` appends every download to `recordings/stems/eval/LEDGER.json` (URL, bytes, SHA-256, license, UTC time).
Backups if a pick disappears: [MUSDB18-HQ](https://zenodo.org/records/3338373) (4 stems per song, non-commercial research only,
cannot separate keys from guitar) and [MedleyDB](https://zenodo.org/record/1438309) (already used by `validate_real_stems.py`).

## 6. Test service script

One shared script; every service uses it with its own sources. Layer positions are fractions of a segment so the shorter
profiles keep the trap order. Source parts loop when a recording is shorter than the segment.

| # | Segment | Ground truth | full / quick / smoke (s) | What plays | Expectations |
| --- | --- | --- | --- | --- | --- |
| 1 | Band warm-up (unscored) | unscored | 135/24/- | drums:drums; bass:bass; keys:keys | probe_instrument_open ['ch9_drums']; probe_instrument_open ['ch9_drums']; no_instrument_open ['ch5_keys'] |
| 2 | Welcome (spoken word) | SPOKEN_WORD | 90/20/10 | sermon:speech | speech_onset |
| 3 | Applause | TRANSITION | 15/15/8 | room:room | - |
| 4 | Song 1 (full band, pastor over the intro) | WORSHIP_HIGH | 240/40/12 | vocal_song:song; sermon:speech | instrument_open ['ch9_drums', 'ch5_keys', 'ch7_guitars']; speech_over_intro_cap |
| 5 | Pastor interrupt over the song | unscored | 15/15/8 | vocal_song:song; sermon:speech | speech_onset; ducking ['ch9_drums', 'ch7_guitars', 'ch5_keys'] |
| 6 | Applause | TRANSITION | 15/15/- | room:room | - |
| 7 | Offertory (solo keys, then solo acoustic guitar) | WORSHIP_LOW | 120/24/- | keys:keys; acoustic:guitar | probe_instrument_open ['ch5_keys']; probe_instrument_open ['ch7_guitars'] |
| 8 | Special music (choir) | WORSHIP_LOW | 90/20/- | choir:choir | no_speech_open |
| 9 | Pastoral prayer over a keys pad | PRAYER_UNDERSCORE | 120/24/- | sermon:speech; keys:keys; electric:guitar | no_instrument_open ['ch5_keys']; no_instrument_open ['ch7_guitars'] |
| 10 | Sermon (spoken word with traps) | SPOKEN_WORD | 480/90/- | sermon:speech; acoustic:guitar; bass:bass; drums:drums; synth:howl; room:room | no_instrument_open ['ch7_guitars']; probe_instrument_open ['ch9_drums']; probe_instrument_open ['ch9_drums']; feedback_cut ch2_lead_vox; human_move ch1_pastor +3 dB |
| 11 | Response song (cold full-band start) | WORSHIP_HIGH | 120/30/- | vocal_song:song | instrument_open ['ch9_drums', 'ch5_keys', 'ch7_guitars'] |
| 12 | Benediction over the fading outro | SPOKEN_WORD | 40/20/- | sermon:speech; vocal_song:song | speech_onset |
| 13 | Closing applause | TRANSITION | 15/15/- | room:room | - |
| 14 | Silence (nothing may fire) | unscored | 60/15/- | silence | no_speech_open; no_instrument_open ['ch5_keys', 'ch7_guitars', 'ch9_drums'] |

Per-run variation: each tap group gets a seeded trim of +/-6 dB (logged in `trims_db.json`) to exercise gain normalization,
and stage bleed (default -22 dB into vocal/spoken mics, `--bleed-db -15` for a stress run) is added from the band taps because
studio multitracks have almost none.

## 7. Gates

Limits mirror SPEC.md section 9 Phase 2 unless the Source column says otherwise. Edit `evals/manifest/gates.yaml` to change one.

| Gate | Limit | Mode | Source |
| --- | --- | --- | --- |
| `phase_macro_f1` | >= 0.85 | headless | SPEC section 9 Phase 2 (offline classifier >= 85% macro F1) |
| `speech_open_latency_s` | <= 1.0 | headless | SPEC section 9 Phase 2 (speech-open < 1.0 s) |
| `speech_open_misses` | == 0.0 | headless | SPEC section 9 Phase 2 (0 missed pastor speech onsets) |
| `instrument_open_latency_s` | <= 0.8 | headless | SPEC section 9 Phase 2 (instrument-open < 800 ms) |
| `false_instrument_opens` | == 0.0 | headless | SPEC section 9 Phase 2 (0 false triggers in prayer or on solo guitar tuning) |
| `speech_over_intro_excess_db` | <= 1.0 | headless | SPEC section 5.3 Tier 1 item 4 (capped at worship_target + duck_depth_db) |
| `duck_drop_db` | >= 4.0 | headless | SPEC section 5.3 Tier 1 item 5 (-6 dB duck depth); 4 dB tolerance matches scripts/simulate_service.py |
| `feedback_cut_latency_s` | <= 0.4 | headless | SPEC section 9 Phase 2 (cuts stationary howl < 400 ms) |
| `false_feedback_cuts` | == 0.0 | headless | SPEC section 9 Phase 2 (0 false triggers on synth pad bleed or clean vocals) |
| `human_override_drift_db` | <= 0.05 | headless | SPEC section 9 Phase 2 (0.00 dB drift while human offset is active) |
| `transition_max_slew_db_s` | <= 8.5 | headless | SPEC section 5.3 slew table (duck release 8 dB/s is the fastest upward path; musical 4 dB/s) |
| `lufs_error_lu` | <= 2.0 (informational) | headless | SPEC mixing_rules.json phase_target_lufs; tolerance chosen for this suite |
| `clarity_gain_db` | >= 5.0 | headless | scripts/validate_real_stems.py clarity gate (>= +5 dB) |
| `no_speech_open_violations` | == 0.0 | headless | docs/EVAL_SPEC.md (choir must not read as speech) |
| `loop_p99_ms` | <= 25.0 | headless | SPEC section 9 Phase 2 (p99 <= 25 ms) |
| `ch9_single_onset_opens` | <= 0.0 (informational) | headless | docs/EVAL_SPEC.md risk 1 |
| `mainstage_prefader_invariance_db` | <= 0.1 | mainstage | SPEC section 9 Phase 0 |
| `mainstage_overloads_30min` | == 0.0 | mainstage | SPEC section 9 Phase 0 and 2 |
| `resilience_checks` | == 0.0 | manual | SPEC section 7 failure matrix |
| `llm_critique_validity` | == 0.0 | manual | SPEC section 6.3 and Phase 0 |
| `listening_score` | >= 4.0 | manual | docs/EVAL_SPEC.md (mean >= 4, no segment <= 2) |

How the headless scorer measures the reflexes (`evals/scoring.py`):
- Onsets are taken from the audio itself (first 20 ms window above -40 dBFS), not the layer start, because real speech and songs rarely start on the beat.
- **speech_open** is the first tick with `speech_open` set and the pastor fader within 1 dB of open, counted only when the fader started below -6 dB.
- **instrument_open** is the first `instrument_open` flag on any listed channel within 3 s, and the fader must rise at least 10 dB within 1 s.
- **Phase F1** scores ticks after a 4.5 s hysteresis settle in labelled segments.
- **Clarity gain** is the pastor-to-band energy ratio on the emulated master minus the same on a unity sum, over ticks where both speech and band layers play.
- Perception runs without CLAP (as `simulate_service.py` does), so phase scores reflect telemetry, VAD and, with `--use-service-plan`, Order-of-Service fusion.

## 8. Execution order

1. `make test` and `make verify`: baseline.
2. `make eval-fetch ACCEPT_TERMS=1`, then place the manual sources (sections 4 and 5) under `recordings/stems/eval/<id>/`.
3. `make eval-audition`: confirm each instrument is present and isolated and vocal leads split male/female; fix `picks:` or swap sources.
4. `make eval-smoke`, then `make eval-quick`: fix perception issues here, where runs are cheap.
5. Phase 0 on the Mac (`make spike`), then the MainStage rig and its pre-fader invariance check ([MAINSTAGE_EVAL_RUNBOOK.md](MAINSTAGE_EVAL_RUNBOOK.md)).
6. Mode B: `run_eval.py play svc1 .. svc5` through MainStage in ACTIVE with all channels armed; record the master tap and use the markers CSV as ground truth.
7. Mode C / fault injection: loop one service 30 minutes on the heaviest concert; `kill -9` engine and bridge; unplug the controller.
8. Blind listening with `--write-audio` output; fill `listening_sheet.csv`.
9. Record 3-4 real services in SHADOW (`scripts/record_shadow_session.py`), as SPEC Stage 0 requires; they replace most of this synthetic corpus as the regression set.

## 9. Known limits and findings to chase

- Source URLs and the Cambridge-MT artist-name matching have not been exercised against the live hosts from this environment; the resolvers are unit-tested on fixtures only. If a fetch fails, download by hand into `recordings/stems/eval/<id>/`.
- Automatic scoring of live MainStage captures (Mode B) is not implemented; the CC log and `telemetry.db` need a parser against the real bridge output.
- Bass and drums share `ch9_drums`, and one onset on it can open the band. The sermon probes record whether it happens (`ch9_single_onset_opens`); if it does, decide whether that is acceptable. That would be a spec change.
- The speech-over-intro cap can be exceeded for about 0.5 s when the pastor starts speaking at the same instant as the band (the instrument-open hold is 200 ms, the speech-open hold 400 ms). The headless smoke run reproduces it; check whether it is real on your recordings.
- LUFS error is informational in headless mode because the emulated master is not level-calibrated.
- The listening score, LLM critique validity, resilience and MainStage overload gates are manual.
