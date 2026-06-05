#!/bin/bash -l
#SBATCH --job-name=viespeaker2_sweep
#SBATCH --partition=defq
#SBATCH --output=/home/user14/anhhd/sv/VieSpeaker2/experiment/logs/sweep_%j.out
#SBATCH --error=/home/user14/anhhd/sv/VieSpeaker2/experiment/logs/sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10g
#SBATCH --gres=gpu:a100:1

# --------------------------------------------------------------------------- #
# VieSpeaker2 — full scenario sweep on an A100 node (SLURM).
# Submit with:   sbatch experiment/submit-job.sh
# --------------------------------------------------------------------------- #

PROJECT_DIR=/home/user14/anhhd/sv/VieSpeaker2
cd "$PROJECT_DIR" || { echo "Project dir not found: $PROJECT_DIR"; exit 1; }

# Activate conda env (adjust the miniconda path if different on your node).
source /home/user14/miniconda3/bin/activate VieSpeaker2

mkdir -p experiment/logs

echo "=== Host: $(hostname) ==="
nvidia-smi || true

echo "=== Preflight self-check ==="
python scripts/selfcheck.py || { echo "Self-check failed (core)."; exit 1; }

echo "=== Running full scenario sweep ==="
python experiment/scenarios/run_scenarios.py

echo "=== Done. See experiment/<RUNTAG>/REPORT.md ==="
