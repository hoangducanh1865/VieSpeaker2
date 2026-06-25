#!/bin/bash -l
# ---------------------------------------------------------------------------- #
# VieSpeaker2 — run the scenario sweep INTERACTIVELY on a dgx node (Cách B).
#
# Run this ON THE HEAD NODE (bcm-headnode02). It uses `srun` to grab one A100
# and streams logs live to your terminal — good for debugging / short runs.
# (Batch / long runs: use `sbatch experiment/submit-job.sh` instead.)
#
#   cd /home/user14/anhhd/sv/VieSpeaker2
#   ./experiment/run-interactive.sh                 # full sweep, live logs
#   ./experiment/run-interactive.sh --smoke         # quick sanity
#   ./experiment/run-interactive.sh --only p1_local
#
# Tune the request via env vars (defaults shown):
#   GPUS=1 CPUS=8 MEM=64G ./experiment/run-interactive.sh --smoke
#
# Prefer a raw debugging shell on a node (then activate + run by hand)? Use:
#   srun --partition=defq --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --pty bash -l
# ---------------------------------------------------------------------------- #
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$HERE")"
export VIESPEAKER_DIR="$PROJECT_DIR"   # so _job_body.sh cd's to the right place

GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"

echo "[run-interactive] requesting ${GPUS}x A100, ${CPUS} CPU, ${MEM} on partition defq ..."
exec srun \
  --partition=defq \
  --gres="gpu:a100:${GPUS}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --job-name=viespeaker2_int \
  --pty bash -l "$PROJECT_DIR/experiment/_job_body.sh" "$@"
