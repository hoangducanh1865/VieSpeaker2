#!/usr/bin/env python3
"""
VieSpeaker2 — Orchestrator

Runs speaker diarization pipelines sequentially:
  Pipeline 1: Audio-only diarization (pyannote)
  Pipeline 2: Audio-visual ASD supplement
  Pipeline 3: Embedding-based cleansing (AHC or CDGCN)

Usage:
    python main.py --pipeline 1 --sample interview_noise
    python main.py --pipeline all --sample interview_noise --method cdgcn
    python main.py --pipeline all   # runs all 6 samples
"""

import os
import sys
import argparse
import subprocess

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")

ALL_SAMPLES = ["drama", "interview_clean", "interview_noise", "movie", "sample_0", "singing"]

DATA_DIR = os.path.join(_ROOT, "data")
TEST_SET_DIR = os.path.join(DATA_DIR, "diarization_test_set")
AUDIO_DIR = os.path.join(TEST_SET_DIR, "audio")
VIDEO_DIR = os.path.join(TEST_SET_DIR, "video")
LABEL_DIR = os.path.join(TEST_SET_DIR, "label")
EXPERIMENTS_DIR = os.path.join(_ROOT, "experiment")

P1_SCRIPT = os.path.join(_SRC, "pipeline", "audio_pipeline", "speaker_diarization.py")
P2_SCRIPT = os.path.join(_SRC, "pipeline", "audio_visual_pipeline", "supplement_pipeline.py")
P3_SCRIPT = os.path.join(_SRC, "pipeline", "clean_pipeline", "clean.py")
EVAL_SCRIPT = os.path.join(_SRC, "evaluation", "evaluation.py")


def _run(cmd: list, label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print("CMD:", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARNING] {label} exited with code {result.returncode}")
    return result.returncode == 0


def run_pipeline_1(sample: str, output_dir: str, model: str, min_seg: float, max_gap: float) -> str:
    audio_path = os.path.join(AUDIO_DIR, f"{sample}.wav")
    os.makedirs(output_dir, exist_ok=True)
    ok = _run(
        [
            sys.executable, P1_SCRIPT,
            "--audio_path", audio_path,
            "--output_dir", output_dir,
            "--model", model,
            "--min_segment_duration", str(min_seg),
            "--max_gap_threshold", str(max_gap),
        ],
        f"Pipeline 1 — Speaker Diarization: {sample}",
    )
    output_path = os.path.join(output_dir, f"{sample}.txt")
    if ok and os.path.exists(output_path):
        print(f"[P1] Output: {output_path}")
    else:
        print(f"[P1] Expected output not found: {output_path}")
    return output_path


def run_pipeline_2(sample: str, sd_diarization: str, asd_model: str, output_dir: str) -> str:
    video_path = os.path.join(VIDEO_DIR, f"{sample}.mp4")
    ok = _run(
        [
            sys.executable, P2_SCRIPT,
            "--video_path", video_path,
            "--sd_diarization", sd_diarization,
            "--model", asd_model,
            "--out_dir", output_dir,
        ],
        f"Pipeline 2 — Audio-Visual ASD ({asd_model}): {sample}",
    )
    output_path = os.path.join(output_dir, sample, "supplemented_diarization.txt")
    if ok and os.path.exists(output_path):
        print(f"[P2] Output: {output_path}")
    else:
        print(f"[P2] Expected output not found: {output_path}")
    return output_path


def run_pipeline_3(
    sample: str, diarization_path: str, method: str, output_dir: str, extra_args: list
) -> str:
    audio_path = os.path.join(AUDIO_DIR, f"{sample}.wav")
    sample_out = os.path.join(output_dir, sample)
    os.makedirs(sample_out, exist_ok=True)
    ok = _run(
        [
            sys.executable, P3_SCRIPT,
            "--method", method,
            "--diarization_path", diarization_path,
            "--audio_path", audio_path,
            "--output_dir", sample_out,
        ] + extra_args,
        f"Pipeline 3 — Cleansing ({method}): {sample}",
    )
    output_path = os.path.join(sample_out, "cleansed_diarization.txt")
    if ok and os.path.exists(output_path):
        print(f"[P3] Output: {output_path}")
    else:
        print(f"[P3] Expected output not found: {output_path}")
    return output_path


def run_evaluation(pipeline_num: int, sample: str, hyp_path: str, ref_path: str):
    if not os.path.exists(hyp_path):
        print(f"[EVAL] Skipping — hypothesis file missing: {hyp_path}")
        return
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    _run(
        [
            sys.executable, EVAL_SCRIPT,
            "--pipeline", str(pipeline_num),
            "--hyp_path", hyp_path,
            "--ref_path", ref_path,
            "--sample_key", sample,
            "--experiments_dir", EXPERIMENTS_DIR,
        ],
        f"Evaluation — Pipeline {pipeline_num}: {sample}",
    )


def _p3_extra_args(args) -> list:
    extra = []
    if args.method == "ahc":
        extra += [
            "--merge_gap_sec", str(args.merge_gap_sec),
            "--min_duration_sec", str(args.min_duration_sec),
            "--threshold", str(args.threshold),
            "--min_cluster_size", str(args.min_cluster_size),
        ]
    else:
        extra += [
            "--k", str(args.k),
            "--resolution", str(args.resolution),
            "--purity_threshold", str(args.purity_threshold),
        ]
    extra += ["--device", args.device]
    return extra


def main():
    parser = argparse.ArgumentParser(description="VieSpeaker2 Orchestrator")
    parser.add_argument(
        "--pipeline", choices=["1", "2", "3", "all"], required=True,
        help="Which pipeline(s) to run",
    )
    parser.add_argument(
        "--sample", default=None,
        help=f"Sample name. One of {ALL_SAMPLES}. Omit to run all samples.",
    )
    # Pipeline 1 options
    parser.add_argument("--p1_model", default="pyannote/speaker-diarization-precision-2",
                        help="Pipeline 1 model. Default: precision-2 (cloud). Alt: pyannote/speaker-diarization-3.1 (local).")
    parser.add_argument("--min_segment_duration", type=float, default=0.5)
    parser.add_argument("--max_gap_threshold", type=float, default=0.5)
    # Pipeline 2 options
    parser.add_argument("--asd_model", choices=["lr_asd", "loconet"], default="lr_asd")
    # Pipeline 3 options
    parser.add_argument("--method", choices=["ahc", "cdgcn"], default="ahc")
    parser.add_argument("--device", default="cuda")
    # AHC params
    parser.add_argument("--merge_gap_sec", type=float, default=0.5)
    parser.add_argument("--min_duration_sec", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="AHC cosine distance threshold (default 0.5)")
    parser.add_argument("--min_cluster_size", type=int, default=3)
    # CDGCN params
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--resolution", type=float, default=0.6)
    parser.add_argument("--purity_threshold", type=float, default=0.8)

    args = parser.parse_args()

    samples = [args.sample] if args.sample else ALL_SAMPLES

    p1_out = os.path.join(DATA_DIR, "diarization")
    p2_out = os.path.join(DATA_DIR, "audio_visual")
    p3_out = os.path.join(DATA_DIR, "clean")

    p3_extra = _p3_extra_args(args)

    for sample in samples:
        print(f"\n{'#'*60}")
        print(f"  SAMPLE: {sample}")
        print(f"{'#'*60}")

        ref_path = os.path.join(LABEL_DIR, f"{sample}.txt")

        if args.pipeline == "1":
            hyp = run_pipeline_1(
                sample, p1_out, args.p1_model,
                args.min_segment_duration, args.max_gap_threshold,
            )
            run_evaluation(1, sample, hyp, ref_path)

        elif args.pipeline == "2":
            sd_path = os.path.join(p1_out, f"{sample}.txt")
            if not os.path.exists(sd_path):
                print(f"[ERROR] Pipeline 1 output not found: {sd_path}")
                print("        Run --pipeline 1 first.")
                continue
            hyp = run_pipeline_2(sample, sd_path, args.asd_model, p2_out)
            run_evaluation(2, sample, hyp, ref_path)

        elif args.pipeline == "3":
            p2_hyp = os.path.join(p2_out, sample, "supplemented_diarization.txt")
            if not os.path.exists(p2_hyp):
                print(f"[ERROR] Pipeline 2 output not found: {p2_hyp}")
                print("        Run --pipeline 2 first.")
                continue
            hyp = run_pipeline_3(sample, p2_hyp, args.method, p3_out, p3_extra)
            run_evaluation(3, sample, hyp, ref_path)

        elif args.pipeline == "all":
            hyp1 = run_pipeline_1(
                sample, p1_out, args.p1_model,
                args.min_segment_duration, args.max_gap_threshold,
            )
            run_evaluation(1, sample, hyp1, ref_path)

            hyp2 = run_pipeline_2(sample, hyp1, args.asd_model, p2_out)
            run_evaluation(2, sample, hyp2, ref_path)

            hyp3 = run_pipeline_3(sample, hyp2, args.method, p3_out, p3_extra)
            run_evaluation(3, sample, hyp3, ref_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
