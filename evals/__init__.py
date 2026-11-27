"""AetherMix multitrack MainStage evaluation suite (see docs/EVAL_SPEC.md)."""

from __future__ import annotations

from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
MANIFEST_DIR = EVALS_DIR / "manifest"
DEFAULT_STEMS_ROOT = REPO_ROOT / "recordings" / "stems" / "eval"
DEFAULT_RUNS_ROOT = REPO_ROOT / "recordings" / "eval_runs"
SAMPLE_RATE = 48000
BLOCK_FRAMES = 960  # 20 ms = 50 Hz engine tick
