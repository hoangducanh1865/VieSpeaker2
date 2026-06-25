#!/bin/bash -l
# ---------------------------------------------------------------------------- #
# VieSpeaker2 — shared job body.
#
# Used by BOTH:
#   * experiment/submit-job.sh   (sbatch / batch job)
#   * experiment/run-interactive.sh (srun / interactive on a dgx node)
# so the two entry points can never drift apart.
#
# It activates the conda env and runs the scenario sweep. Any CLI args are
# passed straight through to run_scenarios.py, e.g.:
#   sbatch experiment/submit-job.sh --smoke
#   ./experiment/run-interactive.sh --only p1_local p3_ahc_ecapa
#
# Override-able via environment variables (all have sane defaults):
#   VIESPEAKER_DIR   repo root        (default: SLURM submit dir, else fixed path)
#   CONDA_BASE       miniconda prefix (default: /home/user14/miniconda3)
#   CONDA_ENV        env name         (default: VieSpeaker2)
#   SKIP_SELFCHECK   set to 1 to skip scripts/selfcheck.py
# ---------------------------------------------------------------------------- #
set -euo pipefail

PROJECT_DIR="${VIESPEAKER_DIR:-${SLURM_SUBMIT_DIR:-/home/user14/anhhd/sv/VieSpeaker2}}"
CONDA_BASE="${CONDA_BASE:-/home/user14/miniconda3}"
CONDA_ENV="${CONDA_ENV:-VieSpeaker2}"

cd "$PROJECT_DIR" || { echo "[FATAL] Project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p experiment/logs

echo "==================================================================="
echo "  VieSpeaker2 job | host=$(hostname) | $(date '+%F %T')"
echo "  PROJECT_DIR=$PROJECT_DIR"
echo "==================================================================="
nvidia-smi || echo "[WARN] nvidia-smi unavailable (no GPU visible?)"

# conda's activate hook touches unset variables; relax 'set -u' only around it.
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
set -u
echo "=== Using python: $(which python) ($(python --version 2>&1)) ==="

if [[ ! -f .env ]]; then
  echo "[WARN] .env not found at $PROJECT_DIR/.env"
  echo "       P1 cloud (precision-2) needs PYANNOTEAI_API_KEY;"
  echo "       P1 local (3.1) needs HUGGINGFACE_ACCESS_TOKEN."
  echo "       Create it from .env.example before running cloud scenarios."
fi

if [[ "${SKIP_SELFCHECK:-0}" != "1" ]]; then
  echo "=== Preflight self-check ==="
  python scripts/selfcheck.py || { echo "[FATAL] Self-check failed (core component)."; exit 1; }
fi

echo "=== Running scenario sweep:  run_scenarios.py $* ==="
python experiment/scenarios/run_scenarios.py "$@"

echo "=== Done. See experiment/<RUNTAG>/REPORT.md ==="
