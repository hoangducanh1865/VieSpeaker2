#!/bin/bash -l
#SBATCH --job-name=viespeaker2_sweep
#SBATCH --partition=defq
#SBATCH --output=experiment/logs/%x_%j.out
#SBATCH --error=experiment/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
# #SBATCH --time=08:00:00        # uncomment to cap walltime (rely on partition default otherwise)
# ---------------------------------------------------------------------------- #
# VieSpeaker2 — scenario sweep on ONE A100 node (SLURM batch job).
#
# IMPORTANT: submit FROM the repo root so the relative log paths above resolve
# and experiment/logs/ already exists:
#
#   cd /home/user14/anhhd/sv/VieSpeaker2
#   sbatch experiment/submit-job.sh                      # full sweep
#   sbatch experiment/submit-job.sh --smoke              # quick sanity (2 samples)
#   sbatch experiment/submit-job.sh --only p1_local p3_ahc_ecapa
#   sbatch experiment/submit-job.sh --samples interview_noise movie
#   sbatch -w dgx02 experiment/submit-job.sh             # pin a node (skip if it is DRAIN)
#
# Watch it:   squeue -u $USER   |   tail -f experiment/logs/viespeaker2_sweep_<jobid>.out
# ---------------------------------------------------------------------------- #

# The compute node starts the batch script with cwd = submit dir. Run the shared
# body (activates conda + runs the sweep), forwarding all args to run_scenarios.py.
exec bash -l "${SLURM_SUBMIT_DIR:-.}/experiment/_job_body.sh" "$@"
