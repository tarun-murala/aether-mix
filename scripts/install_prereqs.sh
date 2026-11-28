#!/usr/bin/env bash
# AetherMix v5.1 — One-Click macOS Prerequisites, Ollama + Qwen, & Runtime Installer
# Works both from an installed /Applications/AetherMix.app bundle and from a git clone.
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../engine/main.py" ]]; then
  RUNTIME_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "/Applications/AetherMix.app/Contents/Resources/runtime/engine/main.py" ]]; then
  RUNTIME_ROOT="/Applications/AetherMix.app/Contents/Resources/runtime"
else
  echo "ERROR: Could not locate AetherMix runtime. Drag AetherMix.app into /Applications first!" >&2
  exit 1
fi

AETHERMIX_DIR="${HOME}/.aethermix"
VENV_DIR="${AETHERMIX_DIR}/venv"
MODELS_DIR="${AETHERMIX_DIR}/models"
CONFIG_DIR="${AETHERMIX_DIR}/config"
SILERO_DEST="${MODELS_DIR}/silero_vad.onnx"

# Auto-pick Qwen model based on Unified Memory unless user overrides AETHERMIX_OLLAMA_MODEL
TOTAL_MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 34359738368)"
TOTAL_MEM_GB="$(( TOTAL_MEM_BYTES / 1073741824 ))"
DEFAULT_MODEL="qwen3:14b"
if [[ "${TOTAL_MEM_GB}" -lt 22 ]]; then
  DEFAULT_MODEL="qwen3:8b"
fi
OLLAMA_MODEL="${AETHERMIX_OLLAMA_MODEL:-${DEFAULT_MODEL}}"
TEST_OLLAMA=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      OLLAMA_MODEL="$2"
      shift 2
      ;;
    --skip-ollama-test)
      TEST_OLLAMA=0
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

echo "====================================================================="
echo "  AetherMix v5.1 — macOS Prerequisites & Ollama (${OLLAMA_MODEL}) Setup"
echo "====================================================================="
echo "  Runtime:        ${RUNTIME_ROOT}"
echo "  Detected RAM:   ${TOTAL_MEM_GB} GB"
echo "  Ollama Model:   ${OLLAMA_MODEL}"
echo ""

mkdir -p "${AETHERMIX_DIR}" "${MODELS_DIR}" "${CONFIG_DIR}" "${AETHERMIX_DIR}/storage/rule_backups"
chmod 700 "${AETHERMIX_DIR}"

# 1. Homebrew check
if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew is not installed on this Mac." >&2
  echo "Install Homebrew first:" >&2
  echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"' >&2
  exit 1
fi

# 2. Install Homebrew formulae + BlackHole 16ch cask
echo "==> [1/6] Installing Homebrew dependencies (uv, portaudio, ollama, blackhole-16ch)..."
for pkg in uv portaudio ollama; do
  if brew list --formula "${pkg}" >/dev/null 2>&1 || command -v "${pkg}" >/dev/null 2>&1; then
    echo "    [ok] ${pkg} already installed"
  else
    brew install "${pkg}"
  fi
done

if [[ -d "/Library/Audio/Plug-Ins/HAL/BlackHole16ch.driver" ]] || brew list --cask blackhole-16ch >/dev/null 2>&1; then
  echo "    [ok] BlackHole 16ch already installed"
else
  brew install --cask blackhole-16ch
  echo "    [note] If BlackHole 16ch does not appear immediately, run: sudo killall coreaudiod"
fi

# 3. Create / sync portable Python 3.11+ virtual environment in ~/.aethermix/venv
echo "==> [2/6] Provisioning Python 3.11 runtime in ${VENV_DIR}..."
UV_BIN="$(command -v uv)"
if [[ -f "${RUNTIME_ROOT}/engine/.venv/bin/python" && "${RUNTIME_ROOT}" != *".app/"* ]]; then
  echo "    [ok] Using local development repo venv (${RUNTIME_ROOT}/engine/.venv)"
  (cd "${RUNTIME_ROOT}/engine" && "${UV_BIN}" sync --extra dev --extra audio)
  PYTHON_BIN="${RUNTIME_ROOT}/engine/.venv/bin/python"
else
  UV_PROJECT_ENVIRONMENT="${VENV_DIR}" "${UV_BIN}" sync \
    --project "${RUNTIME_ROOT}/engine" \
    --extra audio \
    --python 3.11
  PYTHON_BIN="${VENV_DIR}/bin/python"
fi

# 4. Seed writable user config & 0600 session token
echo "==> [3/6] Seeding ~/.aethermix/config and 0600 session token..."
for cfg in service_plan.json mixing_rules.json; do
  if [[ ! -f "${CONFIG_DIR}/${cfg}" ]]; then
    cp "${RUNTIME_ROOT}/engine/schemas/${cfg}" "${CONFIG_DIR}/${cfg}"
    echo "    [created] ${CONFIG_DIR}/${cfg}"
  else
    echo "    [kept]    ${CONFIG_DIR}/${cfg}"
  fi
done

# Update configured ollama_model in user config if user chose a different model (e.g. qwen3:8b on 16GB Mac)
"${PYTHON_BIN}" - "${CONFIG_DIR}/service_plan.json" "${OLLAMA_MODEL}" <<'PYEOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
model = sys.argv[2]
data = json.loads(p.read_text(encoding="utf-8"))
if data.get("ollama_model") != model:
    data["ollama_model"] = model
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"    [updated] ollama_model -> {model} in {p}")
PYEOF

PYTHONPATH="${RUNTIME_ROOT}" "${PYTHON_BIN}" -c "from engine.core.protocol import ensure_session_token; ensure_session_token()"
echo "    [ok] Session token ready at ${AETHERMIX_DIR}/session.token (0600)"

# 5. Stage Silero VAD ONNX model (bundled in DMG -> zero download if present!)
echo "==> [4/6] Staging Silero VAD ONNX model..."
if [[ -f "${SILERO_DEST}" && "$(stat -f%z "${SILERO_DEST}" 2>/dev/null || echo 0)" -gt 100000 ]]; then
  echo "    [ok] ${SILERO_DEST} already cached"
elif [[ -f "${RUNTIME_ROOT}/models/silero_vad.onnx" ]]; then
  cp "${RUNTIME_ROOT}/models/silero_vad.onnx" "${SILERO_DEST}"
  echo "    [ok] Copied bundled silero_vad.onnx -> ${SILERO_DEST}"
else
  curl -fsSL \
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx" \
    -o "${SILERO_DEST}.tmp"
  mv "${SILERO_DEST}.tmp" "${SILERO_DEST}"
  echo "    [ok] Downloaded silero_vad.onnx -> ${SILERO_DEST}"
fi

# 6. Start Ollama & pull Qwen model
echo "==> [5/6] Starting Ollama service and pulling ${OLLAMA_MODEL}..."
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  if brew services list 2>/dev/null | grep -q "^ollama"; then
    brew services start ollama >/dev/null 2>&1 || true
  fi
  if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    nohup ollama serve >"${AETHERMIX_DIR}/ollama.log" 2>&1 &
  fi
  for _ in $(seq 1 20); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi
ollama pull "${OLLAMA_MODEL}"

# 7. Install standalone CoreMIDI Bridge LaunchAgent
echo "==> [6/6] Installing standalone CoreMIDI Bridge LaunchAgent..."
PLIST_SRC="${RUNTIME_ROOT}/scripts/com.aethermix.midibridge.plist"
PLIST_DEST="${HOME}/Library/LaunchAgents/com.aethermix.midibridge.plist"
mkdir -p "${HOME}/Library/LaunchAgents"
sed \
  -e "s|__REPO_ROOT__|${RUNTIME_ROOT}|g" \
  -e "s|__PYTHON__|${PYTHON_BIN}|g" \
  -e "s|__HOME__|${HOME}|g" \
  "${PLIST_SRC}" > "${PLIST_DEST}"
launchctl bootout "gui/$(id -u)/com.aethermix.midibridge" 2>/dev/null || true
sleep 0.3
launchctl enable "gui/$(id -u)/com.aethermix.midibridge" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "${PLIST_DEST}"
echo "    [ok] LaunchAgent com.aethermix.midibridge installed and started"

echo ""
echo "==> Running Pre-Flight Verification Doctor..."
if [[ "${TEST_OLLAMA}" -eq 1 ]]; then
  "${PYTHON_BIN}" "${RUNTIME_ROOT}/scripts/check_setup.py" --model "${OLLAMA_MODEL}" --test-ollama
else
  "${PYTHON_BIN}" "${RUNTIME_ROOT}/scripts/check_setup.py" --model "${OLLAMA_MODEL}"
fi
