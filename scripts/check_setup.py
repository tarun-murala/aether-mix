#!/usr/bin/env python3
"""AetherMix Pre-Flight Environment & Readiness Verification Doctor.

Verifies every component required to run AetherMix + MainStage + Ollama (Qwen)
on a macOS Apple Silicon machine and prints an actionable PASS/WARN/FAIL report.

Usage:
  python3 scripts/check_setup.py
  python3 scripts/check_setup.py --test-ollama
  python3 scripts/check_setup.py --model qwen3:8b --json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import plistlib
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Auto-re-exec into AetherMix's managed venv when invoked via system python3
_HOME = Path.home()
_AETHERMIX_DIR = _HOME / ".aethermix"
_SCRIPT_DIR = Path(__file__).resolve().parent
_CANDIDATE_ROOTS = [
    _SCRIPT_DIR.parent,
    Path("/Applications/AetherMix.app/Contents/Resources/runtime"),
]
_CANDIDATE_VENV_PYTHONS = [
    _SCRIPT_DIR.parent / "engine" / ".venv" / "bin" / "python",
    _AETHERMIX_DIR / "venv" / "bin" / "python",
]


def _maybe_reexec_in_venv() -> None:
    if os.environ.get("AETHERMIX_DOCTOR_REEXEC") == "1":
        return
    cur = Path(sys.executable).resolve()
    for py in _CANDIDATE_VENV_PYTHONS:
        if py.exists() and os.access(py, os.X_OK):
            if cur != py.resolve():
                env = os.environ.copy()
                env["AETHERMIX_DOCTOR_REEXEC"] = "1"
                os.execve(str(py), [str(py), *sys.argv], env)
            return


_maybe_reexec_in_venv()


@dataclass
class CheckResult:
    id: str
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    fix: str | None = None


def _find_runtime_root() -> Path:
    for root in _CANDIDATE_ROOTS:
        if (root / "engine" / "main.py").exists():
            return root
    return _SCRIPT_DIR.parent


def _resolve_bin(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin", str(_HOME / ".local" / "bin")):
        cand = Path(prefix) / name
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _get_total_ram_gb() -> float | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=3).strip()
        return int(out) / (1024**3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _read_configured_ollama_model(runtime_root: Path, override: str | None) -> str:
    if override:
        return override
    for cand in (
        _AETHERMIX_DIR / "config" / "service_plan.json",
        runtime_root / "engine" / "schemas" / "service_plan.json",
    ):
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                model = data.get("ollama_model")
                if isinstance(model, str) and model.strip():
                    return model.strip()
            except (OSError, json.JSONDecodeError):
                pass
    return "qwen3:14b"


def check_macos_and_hardware() -> CheckResult:
    arch = platform.machine()
    mac_ver = platform.mac_ver()[0] or "unknown"
    ram_gb = _get_total_ram_gb()
    ram_str = f"{ram_gb:.0f} GB Unified RAM" if ram_gb is not None else "RAM unknown"
    major = int(mac_ver.split(".")[0]) if mac_ver and mac_ver.split(".")[0].isdigit() else 0

    if arch != "arm64":
        return CheckResult(
            "hardware",
            "Apple Silicon & macOS",
            "WARN",
            f"Detected {arch} on macOS {mac_ver} ({ram_str}); AetherMix targets Apple Silicon (arm64) macOS 15+.",
            "Run on an Apple Silicon Mac (M1/M2/M3/M4) for full low-latency neural inference.",
        )
    if major < 15:
        return CheckResult(
            "hardware",
            "Apple Silicon & macOS",
            "WARN",
            f"Apple Silicon ({arch}), macOS {mac_ver} ({ram_str}); macOS 15+ recommended.",
            "Upgrade macOS to 15 (Sequoia) or newer.",
        )
    rec_model = "qwen3:14b" if (ram_gb is None or ram_gb >= 24.0) else "qwen3:8b"
    return CheckResult(
        "hardware",
        "Apple Silicon & macOS",
        "PASS",
        f"Apple Silicon ({arch}), macOS {mac_ver}, {ram_str} (recommended Ollama model: {rec_model})",
    )


def check_cli_tools() -> CheckResult:
    tools = {name: _resolve_bin(name) for name in ("brew", "uv", "ollama")}
    missing = [k for k, v in tools.items() if v is None]
    if not missing:
        return CheckResult(
            "cli_tools",
            "Core CLI Tools (brew, uv, ollama)",
            "PASS",
            ", ".join(f"{k}={v}" for k, v in tools.items()),
        )
    return CheckResult(
        "cli_tools",
        "Core CLI Tools (brew, uv, ollama)",
        "FAIL",
        f"Missing: {', '.join(missing)}",
        "Run `./scripts/install_prereqs.sh` or `brew install uv ollama portaudio`.",
    )


def check_app_and_runtime(runtime_root: Path) -> CheckResult:
    app_installed = Path("/Applications/AetherMix.app").exists()
    engine_ok = (runtime_root / "engine" / "main.py").exists()
    bridge_ok = (runtime_root / "bridge" / "midi_bridge_daemon.py").exists()
    if engine_ok and bridge_ok:
        return CheckResult(
            "runtime",
            "AetherMix App & Runtime",
            "PASS",
            f"Runtime verified at {runtime_root} (App in /Applications: {'yes' if app_installed else 'dev mode'})",
        )
    return CheckResult(
        "runtime",
        "AetherMix App & Runtime",
        "FAIL",
        f"engine/main.py or bridge/midi_bridge_daemon.py missing under {runtime_root}",
        "Drag AetherMix.app into /Applications from the DMG or run from the aether-mix repo.",
    )


def check_python_packages() -> CheckResult:
    required = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "scipy",
        "httpx",
        "websockets",
        "sounddevice",
        "soundfile",
        "rtmidi",
        "onnxruntime",
    ]
    missing: list[str] = []
    for mod in required:
        try:
            importlib.import_module(mod)
        except (ImportError, OSError, RuntimeError):
            missing.append(mod)
    py_ver = platform.python_version()
    if not missing:
        return CheckResult(
            "python_env",
            "Python 3.11+ Engine Environment",
            "PASS",
            f"Python {py_ver} ({sys.executable}) — all {len(required)} required audio/MIDI/ML packages import cleanly",
        )
    return CheckResult(
        "python_env",
        "Python 3.11+ Engine Environment",
        "FAIL",
        f"Python {py_ver} ({sys.executable}) missing packages: {', '.join(missing)}",
        "Run `./scripts/install_prereqs.sh` (or `cd engine && uv sync --extra dev --extra audio`).",
    )


def check_blackhole_16ch() -> CheckResult:
    driver_path = Path("/Library/Audio/Plug-Ins/HAL/BlackHole16ch.driver")
    driver_exists = driver_path.exists()
    try:
        import sounddevice as sd

        devs = sd.query_devices()
        bh = [
            d
            for d in devs
            if "blackhole" in str(d.get("name", "")).lower()
            and int(d.get("max_input_channels", 0)) >= 16
            and int(d.get("max_output_channels", 0)) >= 16
        ]
        if bh:
            d = bh[0]
            return CheckResult(
                "blackhole",
                "BlackHole 16ch CoreAudio Driver",
                "PASS",
                f"Detected '{d['name']}' (in={d['max_input_channels']}, out={d['max_output_channels']}, sr={d['default_samplerate']:.0f} Hz)",
            )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if driver_exists:
            return CheckResult(
                "blackhole",
                "BlackHole 16ch CoreAudio Driver",
                "WARN",
                f"Driver exists at {driver_path}, but PortAudio query failed: {exc}",
                "Restart CoreAudio (`sudo killall coreaudiod`) or re-run outside sandbox.",
            )
    if driver_exists:
        return CheckResult(
            "blackhole",
            "BlackHole 16ch CoreAudio Driver",
            "WARN",
            f"Driver bundle exists at {driver_path} but is not visible to CoreAudio yet.",
            "Run `sudo killall coreaudiod` (or reboot macOS) so CoreAudio loads BlackHole 16ch.",
        )
    return CheckResult(
        "blackhole",
        "BlackHole 16ch CoreAudio Driver",
        "FAIL",
        "BlackHole 16ch is not installed.",
        "Run `brew install --cask blackhole-16ch` then `sudo killall coreaudiod`.",
    )


def check_mainstage() -> CheckResult:
    app = Path("/Applications/MainStage.app")
    if not app.exists():
        return CheckResult(
            "mainstage",
            "Apple MainStage",
            "WARN",
            "/Applications/MainStage.app not found (optional if routing to another DAW/mixer).",
            "Install Apple MainStage from the Mac App Store to test real-time MainStage fader control.",
        )
    ver = "installed"
    plist = app / "Contents" / "Info.plist"
    if plist.exists():
        try:
            info = plistlib.loads(plist.read_bytes())
            ver = f"v{info.get('CFBundleShortVersionString', 'unknown')}"
        except (OSError, plistlib.InvalidFileException, ValueError):
            ver = "installed"
    return CheckResult("mainstage", "Apple MainStage", "PASS", f"/Applications/MainStage.app ({ver})")


def check_coremidi_and_bridge() -> CheckResult:
    plist = _HOME / "Library" / "LaunchAgents" / "com.aethermix.midibridge.plist"
    sock = _AETHERMIX_DIR / "bridge.sock"
    midi_ok = False
    midi_err = ""
    # Probe python-rtmidi in a child process so CoreMIDI C++ aborts (-10833) cannot crash the doctor
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import rtmidi; mo = rtmidi.MidiOut(); n = len(mo.get_ports()); "
                    "mo.open_virtual_port('AetherMix Doctor Probe'); mo.close_port(); print(n)"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if probe.returncode == 0 and probe.stdout.strip().isdigit():
            midi_ok = True
            port_summary = f"{int(probe.stdout.strip())} MIDI dest(s)"
        else:
            midi_err = (probe.stderr or probe.stdout or f"exit {probe.returncode}").strip().splitlines()[-1]
            port_summary = f"rtmidi probe failed ({midi_err})"
    except (OSError, subprocess.SubprocessError) as exc:
        midi_err = str(exc)
        port_summary = f"rtmidi probe failed ({midi_err})"

    launchd_loaded = False
    if plist.exists():
        try:
            r = subprocess.run(
                ["launchctl", "print", f"gui/{os.getuid()}/com.aethermix.midibridge"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            launchd_loaded = r.returncode == 0
        except OSError:
            pass

    if midi_ok and plist.exists() and launchd_loaded:
        return CheckResult(
            "coremidi",
            "CoreMIDI & Standalone MIDI Bridge",
            "PASS",
            f"CoreMIDI virtual ports OK ({port_summary}); LaunchAgent loaded; socket={'present' if sock.exists() else 'idle'}",
        )
    if midi_ok:
        return CheckResult(
            "coremidi",
            "CoreMIDI & Standalone MIDI Bridge",
            "WARN",
            f"CoreMIDI virtual ports OK ({port_summary}), but com.aethermix.midibridge LaunchAgent is not loaded yet.",
            "Run `make install-bridge` (or `./scripts/install_prereqs.sh`) or click 'Install Bridge LaunchAgent' in AetherMix.",
        )
    return CheckResult(
        "coremidi",
        "CoreMIDI & Standalone MIDI Bridge",
        "FAIL",
        f"CoreMIDI unavailable: {midi_err}",
        "Reinstall `python-rtmidi` inside the venv or run outside a restricted sandbox.",
    )


def check_silero_vad(runtime_root: Path) -> CheckResult:
    candidates = [
        _AETHERMIX_DIR / "models" / "silero_vad.onnx",
        runtime_root / "models" / "silero_vad.onnx",
    ]
    model_path = next((p for p in candidates if p.exists() and p.stat().st_size > 100_000), None)
    if model_path is None:
        return CheckResult(
            "silero_vad",
            "Silero VAD ONNX Model",
            "FAIL",
            f"silero_vad.onnx not found at {candidates[0]}",
            "Run `./scripts/install_prereqs.sh` or `curl -fsSL https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx -o ~/.aethermix/models/silero_vad.onnx`.",
        )
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        x = np.zeros((1, 576), dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr = np.array(16000, dtype=np.int64)
        out, _ = sess.run(None, {"input": x, "state": state, "sr": sr})
        prob = float(out.reshape(-1)[0])
        return CheckResult(
            "silero_vad",
            "Silero VAD ONNX Model",
            "PASS",
            f"Loaded & ran inference cleanly ({model_path}, {model_path.stat().st_size / 1e6:.2f} MB, silence prob={prob:.4f})",
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return CheckResult(
            "silero_vad",
            "Silero VAD ONNX Model",
            "FAIL",
            f"Model found at {model_path} but ONNX Runtime failed: {exc}",
            "Re-sync venv (`uv sync --extra audio`) and re-download silero_vad.onnx.",
        )


def check_ollama_and_qwen(expected_model: str, *, run_inference: bool) -> CheckResult:
    ollama_bin = _resolve_bin("ollama")
    if ollama_bin is None:
        return CheckResult(
            "ollama",
            f"Ollama & Qwen Model ({expected_model})",
            "FAIL",
            "Ollama binary not found.",
            f"Run `brew install ollama && brew services start ollama && ollama pull {expected_model}`.",
        )

    # Query local Ollama daemon API
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3.0) as resp:
            tags_data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return CheckResult(
            "ollama",
            f"Ollama & Qwen Model ({expected_model})",
            "FAIL",
            f"Ollama installed ({ollama_bin}) but daemon is not reachable on http://127.0.0.1:11434 ({exc}).",
            f"Start Ollama (`brew services start ollama` or `ollama serve`) then run `ollama pull {expected_model}`.",
        )

    models = [str(m.get("name", "")) for m in tags_data.get("models", []) if isinstance(m, dict)]
    has_exact = any(m == expected_model or m.startswith(expected_model + ":") for m in models)
    qwen_models = [m for m in models if "qwen" in m.lower()]

    if not has_exact and not qwen_models:
        return CheckResult(
            "ollama",
            f"Ollama & Qwen Model ({expected_model})",
            "FAIL",
            f"Ollama daemon is running, but no Qwen model is pulled yet (installed: {models or 'none'}).",
            f"Run `ollama pull {expected_model}` (or `ollama pull qwen3:8b` on 16 GB Macs).",
        )

    active_model = expected_model if has_exact else qwen_models[0]
    status = "PASS" if has_exact else "WARN"
    detail = (
        f"Ollama reachable; exact model '{active_model}' installed"
        if has_exact
        else f"Ollama reachable; found '{active_model}' (configured target is '{expected_model}')"
    )
    fix = None if has_exact else f"Run `ollama pull {expected_model}` or set `ollama_model` to `{active_model}` in service_plan.json."

    if run_inference:
        try:
            payload = json.dumps(
                {
                    "model": active_model,
                    "prompt": 'Respond ONLY with valid JSON: {"ready": true}',
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "options": {"num_predict": 24, "temperature": 0.0},
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=45.0) as resp:
                gen = json.loads(resp.read().decode("utf-8"))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            parsed = json.loads(str(gen.get("response", "{}")))
            detail += f"; live JSON inference OK in {dt_ms:.0f} ms ({parsed})"
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as exc:
            return CheckResult(
                "ollama",
                f"Ollama & Qwen Model ({expected_model})",
                "WARN",
                f"{detail}; live inference test failed: {exc}",
                f"Verify `ollama run {active_model} 'hello'` works in Terminal.",
            )

    return CheckResult("ollama", f"Ollama & Qwen Model ({expected_model})", status, detail, fix)


def check_soundpack(runtime_root: Path) -> CheckResult:
    pack_dir = runtime_root / "assets" / "soundpack"
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        return CheckResult(
            "soundpack",
            "Packaged 5-Tune Multi-Instrument Sound Pack",
            "FAIL",
            f"Missing {manifest_path}",
            "Run `make soundpack` or install from the AetherMix DMG.",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tunes = manifest.get("tunes", [])
        missing_files: list[str] = []
        for rel in manifest.get("common", {}).values():
            if not (pack_dir / rel).exists():
                missing_files.append(rel)
        for t in tunes:
            for rel in t.get("stems", {}).values():
                if not (pack_dir / rel).exists():
                    missing_files.append(rel)
        if missing_files:
            return CheckResult(
                "soundpack",
                "Packaged 5-Tune Multi-Instrument Sound Pack",
                "FAIL",
                f"Missing {len(missing_files)} stem file(s): {missing_files[:4]}",
                "Re-run `make soundpack`.",
            )
        total_mb = sum(p.stat().st_size for p in pack_dir.rglob("*.ogg")) / (1024 * 1024)
        return CheckResult(
            "soundpack",
            "Packaged 5-Tune Multi-Instrument Sound Pack",
            "PASS",
            f"{len(tunes)} multi-instrument tunes + speech + applause verified ({total_mb:.2f} MB)",
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return CheckResult(
            "soundpack",
            "Packaged 5-Tune Multi-Instrument Sound Pack",
            "FAIL",
            f"Invalid soundpack manifest: {exc}",
            "Re-run `make soundpack`.",
        )


def check_session_token() -> CheckResult:
    tok = _AETHERMIX_DIR / "session.token"
    if not tok.exists():
        return CheckResult(
            "token",
            "Session Token & Security Permissions",
            "WARN",
            f"{tok} does not exist yet (will be auto-created with 0600 on first engine/bridge start).",
            "Run `./scripts/install_prereqs.sh` or launch AetherMix once.",
        )
    mode = stat.S_IMODE(tok.stat().st_mode)
    if mode != 0o600:
        return CheckResult(
            "token",
            "Session Token & Security Permissions",
            "WARN",
            f"{tok} has mode {oct(mode)} instead of 0o600.",
            f"Run `chmod 600 {tok}`.",
        )
    return CheckResult("token", "Session Token & Security Permissions", "PASS", f"{tok} exists (mode 0600)")


def run_all_checks(*, model_override: str | None = None, test_ollama: bool = False) -> list[CheckResult]:
    runtime_root = _find_runtime_root()
    expected_model = _read_configured_ollama_model(runtime_root, model_override)
    return [
        check_macos_and_hardware(),
        check_cli_tools(),
        check_app_and_runtime(runtime_root),
        check_python_packages(),
        check_blackhole_16ch(),
        check_mainstage(),
        check_coremidi_and_bridge(),
        check_silero_vad(runtime_root),
        check_ollama_and_qwen(expected_model, run_inference=test_ollama),
        check_soundpack(runtime_root),
        check_session_token(),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="AetherMix Environment & Readiness Doctor")
    parser.add_argument("--model", default=None, help="Override expected Ollama model (e.g. qwen3:8b, qwen3:14b)")
    parser.add_argument("--test-ollama", action="store_true", help="Run a live JSON inference smoke test against Ollama")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    args = parser.parse_args()

    results = run_all_checks(model_override=args.model, test_ollama=args.test_ollama)
    passed = sum(1 for r in results if r.status == "PASS")
    warns = sum(1 for r in results if r.status == "WARN")
    fails = sum(1 for r in results if r.status == "FAIL")

    if args.json:
        payload: dict[str, Any] = {
            "summary": {"pass": passed, "warn": warns, "fail": fails, "ready": fails == 0},
            "checks": [asdict(r) for r in results],
        }
        print(json.dumps(payload, indent=2))
        return 0 if fails == 0 else 1

    print("\n=====================================================================")
    print("  AetherMix v5.1 — Machine Setup & Readiness Verification Doctor")
    print("=====================================================================\n")
    for r in results:
        badge = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[r.status]
        print(f"  {badge} {r.name}")
        print(f"         {r.detail}")
        if r.fix:
            print(f"         Fix -> {r.fix}")
    print("\n---------------------------------------------------------------------")
    print(f"  Summary: {passed} PASS · {warns} WARN · {fails} FAIL")
    print("  Tip: Grant macOS Microphone permission when prompted (Settings -> Privacy & Security -> Microphone)")
    print("       for both AetherMix.app (and Terminal if running CLI scripts).")
    print("---------------------------------------------------------------------\n")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
