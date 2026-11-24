#!/usr/bin/env bash
# AetherMix v5.1 macOS Provisioning & Bootstrap Script
# Checks/installs Homebrew deps, installs standalone launchd MIDI bridge plist,
# pulls Ollama model(s), and caches Silero VAD + CLAP weights locally.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AETHERMIX_DIR="${HOME}/.aethermix"
MODELS_DIR="${AETHERMIX_DIR}/models"
CLAP_DIR="${MODELS_DIR}/clap"
SILERO_ONNX="${MODELS_DIR}/silero_vad.onnx"

OLLAMA_MODEL="${AETHERMIX_OLLAMA_MODEL:-qwen3:14b}"
WITH_32B=0
SKIP_BREW=0
SKIP_MODELS=0
SKIP_LAUNCHD=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  --model <name>   Ollama supervisory model to pull (default: qwen3:14b)
  --with-32b       Also pull optional offline analysis model (qwen3:32b)
  --skip-brew      Skip Homebrew package checks/installs
  --skip-models    Skip Ollama / Silero / CLAP weight downloads
  --skip-launchd   Skip installing com.aethermix.midibridge.plist LaunchAgent
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      OLLAMA_MODEL="$2"
      shift 2
      ;;
    --with-32b)
      WITH_32B=1
      shift
      ;;
    --skip-brew)
      SKIP_BREW=1
      shift
      ;;
    --skip-models)
      SKIP_MODELS=1
      shift
      ;;
    --skip-launchd)
      SKIP_LAUNCHD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "==> AetherMix v5.1 macOS Bootstrap"
echo "    Repo root: ${REPO_ROOT}"
mkdir -p "${AETHERMIX_DIR}" "${MODELS_DIR}" "${CLAP_DIR}"
chmod 700 "${AETHERMIX_DIR}"

# 1. Check / install Homebrew dependencies
if [[ "${SKIP_BREW}" -eq 0 ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "ERROR: Homebrew is required on macOS. Install from https://brew.sh" >&2
    exit 1
  fi

  echo "==> Checking Homebrew formulae (ollama, uv, portaudio, xcodegen)..."
  for pkg in ollama uv portaudio xcodegen; do
    if brew list --formula "${pkg}" >/dev/null 2>&1; then
      echo "    [ok] ${pkg} already installed"
    else
      echo "    [install] brew install ${pkg}"
      brew install "${pkg}"
    fi
  done

  echo "==> Checking Homebrew cask (blackhole-16ch)..."
  if brew list --cask blackhole-16ch >/dev/null 2>&1; then
    echo "    [ok] blackhole-16ch already installed"
  else
    echo "    [install] brew install --cask blackhole-16ch"
    brew install --cask blackhole-16ch
  fi
fi

# 2. Sync Python virtual environment
echo "==> Syncing Python engine environment via uv..."
if [[ "${SKIP_MODELS}" -eq 0 ]]; then
  (cd "${REPO_ROOT}/engine" && uv sync --extra dev --extra audio --extra ml)
else
  (cd "${REPO_ROOT}/engine" && uv sync --extra dev --extra audio)
fi

PYTHON_BIN="${REPO_ROOT}/engine/.venv/bin/python"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Ensure ~/.aethermix/session.token exists with 0600 permissions
"${PYTHON_BIN}" -c "from engine.core.protocol import ensure_session_token; ensure_session_token()"
echo "    [ok] Session token verified at ${AETHERMIX_DIR}/session.token (0600)"

# 3. Install standalone MIDI bridge LaunchAgent
if [[ "${SKIP_LAUNCHD}" -eq 0 ]]; then
  echo "==> Installing standalone MIDI bridge LaunchAgent via make install-bridge..."
  (cd "${REPO_ROOT}" && make install-bridge)
fi

# 4. Pull Ollama models & cache Silero VAD + CLAP
if [[ "${SKIP_MODELS}" -eq 0 ]]; then
  if command -v ollama >/dev/null 2>&1; then
    echo "==> Pulling Ollama supervisory model: ${OLLAMA_MODEL}..."
    ollama pull "${OLLAMA_MODEL}"
    if [[ "${WITH_32B}" -eq 1 ]]; then
      echo "==> Pulling optional offline post-service model: qwen3:32b..."
      ollama pull "qwen3:32b"
    fi
  else
    echo "WARNING: ollama binary not found; skipping model pull." >&2
  fi

  echo "==> Caching Silero VAD ONNX model -> ${SILERO_ONNX}..."
  if [[ ! -f "${SILERO_ONNX}" ]]; then
    curl -fsSL \
      "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx" \
      -o "${SILERO_ONNX}.tmp"
    mv "${SILERO_ONNX}.tmp" "${SILERO_ONNX}"
    echo "    [ok] Downloaded silero_vad.onnx"
  else
    echo "    [ok] silero_vad.onnx already cached"
  fi

  echo "==> Caching CLAP model (laion/clap-htsat-unfused) -> ${CLAP_DIR}..."
  OMP_NUM_THREADS=1 "${PYTHON_BIN}" - <<PYEOF
import os
from pathlib import Path

clap_dir = Path("${CLAP_DIR}")
try:
    from transformers import ClapModel, ClapProcessor
    model_id = "laion/clap-htsat-unfused"
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id)
    processor.save_pretrained(clap_dir)
    model.save_pretrained(clap_dir)
    print(f"    [ok] Cached {model_id} in {clap_dir}")
except Exception as exc:
    print(f"    [warn] Could not cache CLAP weights ({exc}); engine will fall back cleanly.")
PYEOF
fi

echo "==> AetherMix bootstrap complete!"
echo "    Next steps:"
echo "      1. Configure 48 kHz Aggregate Device (PA interface clock master + BlackHole 16ch drift correction)."
echo "      2. Run Phase 0 spike: make spike"
echo "      3. Run verification:  make verify"
