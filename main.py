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

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
from viespeaker import bootstrap, paths  # noqa: E402
bootstrap.setup()

ALL_SAMPLES = ["drama", "interview_clean", "interview_noise", "movie", "sample_0", "singing"]

# Pipeline OUTPUTS stay in the repo's data/ dir (gitignored). INPUT audio/video
# come from the external assets dir; ground-truth labels stay in the repo.
DATA_DIR = os.path.join(_ROOT, "data")
AUDIO_DIR = str(paths.AUDIO_DIR)
VIDEO_DIR = str(paths.VIDEO_DIR)
LABEL_DIR = str(paths.LABEL_DIR)
EXPERIMENTS_DIR = os.path.join(_ROOT, "experiment")

from viespeaker import pipeline_api  # noqa: E402  (subprocess command construction)


def run_pipeline_1(sample: str, output_dir: str, model: str, min_seg: float, max_gap: float,
                   overlap_policy: str = "keep") -> str:
    audio_path = os.path.join(AUDIO_DIR, f"{sample}.wav")
    out = pipeline_api.run_p1(
        audio_path, output_dir, model, min_seg=min_seg, max_gap=max_gap,
        overlap_policy=overlap_policy, label=f"Pipeline 1 — Speaker Diarization: {sample}",
    )
    output_path = os.path.join(output_dir, f"{sample}.txt")
    print(f"[P1] Output: {output_path}" if out else f"[P1] Expected output not found: {output_path}")
    return output_path


def run_pipeline_2(sample: str, sd_diarization: str, asd_model: str, output_dir: str) -> str:
    video_path = os.path.join(VIDEO_DIR, f"{sample}.mp4")
    out = pipeline_api.run_p2(
        video_path, sd_diarization, asd_model, output_dir,
        label=f"Pipeline 2 — Audio-Visual ASD ({asd_model}): {sample}",
    )
    output_path = os.path.join(output_dir, sample, "supplemented_diarization.txt")
    print(f"[P2] Output: {output_path}" if out else f"[P2] Expected output not found: {output_path}")
    return output_path


def run_pipeline_3(
    sample: str, diarization_path: str, method: str, output_dir: str, extra_args: list
) -> str:
    audio_path = os.path.join(AUDIO_DIR, f"{sample}.wav")
    sample_out = os.path.join(output_dir, sample)

    method_extra = list(extra_args)
    if method == "dover-lap":
        method_extra += ["--diarization_path2", os.path.join(DATA_DIR, "diarization", f"{sample}.txt")]

    out = pipeline_api.run_p3(
        method, diarization_path, audio_path, sample_out, method_extra,
        label=f"Pipeline 3 — Cleansing ({method}): {sample}",
    )
    output_path = os.path.join(sample_out, "cleansed_diarization.txt")
    print(f"[P3] Output: {output_path}" if out else f"[P3] Expected output not found: {output_path}")
    return output_path


def run_evaluation(pipeline_num, sample: str, hyp_path: str, ref_path: str, method: str = "",
                   collar: float = 0.0):
    pipeline_api.evaluate(
        pipeline_num, hyp_path, ref_path, sample, EXPERIMENTS_DIR,
        method=method, collar=collar, label=f"Evaluation — Pipeline {pipeline_num}: {sample}",
    )


def run_fusion_pipeline(sample: str, input_paths: list, output_dir: str) -> str:
    sample_out = os.path.join(output_dir, sample)
    out = pipeline_api.run_fusion(input_paths, sample_out, sample, label=f"Fusion (DOVER-Lap): {sample}")
    output_path = os.path.join(sample_out, "fused_diarization.txt")
    print(f"[FUSION] Output: {output_path}" if out else f"[FUSION] Expected output not found: {output_path}")
    return output_path


def _p3_extra_args(args) -> list:
    extra = []
    if args.method == "ahc":
        extra += [
            "--merge_gap_sec", str(args.merge_gap_sec),
            "--min_duration_sec", str(args.min_duration_sec),
            "--threshold", str(args.threshold),
            "--min_cluster_size", str(args.min_cluster_size),
        ]
    elif args.method == "cdgcn":
        extra += [
            "--k", str(args.k),
            "--resolution", str(args.resolution),
            "--purity_threshold", str(args.purity_threshold),
        ]
    elif args.method == "vbx":
        extra += [
            "--vbx_loop_prob", str(args.vbx_loop_prob),
            "--vbx_fa", str(args.vbx_fa),
            "--vbx_fb", str(args.vbx_fb),
            "--vbx_max_speakers", str(args.vbx_max_speakers),
            "--vbx_lda_dim", str(args.vbx_lda_dim),
            "--vbx_init_smoothing", str(args.vbx_init_smoothing),
        ]
    elif args.method == "nme-sc":
        extra += [
            "--max_speakers", str(args.max_speakers),
            "--max_rp_threshold", str(args.max_rp_threshold),
        ]
    # dover-lap: diarization_path2 is injected in run_pipeline_3
    # embedding backend applies to ahc/cdgcn/nme-sc (ignored by vbx/dover-lap)
    extra += ["--embedding", args.embedding]
    extra += ["--device", args.device]
    return extra


def main():
    parser = argparse.ArgumentParser(description="VieSpeaker2 Orchestrator")
    parser.add_argument(
        "--pipeline", choices=["1", "2", "3", "all", "fusion"], required=True,
        help="Which pipeline(s) to run. 'fusion' = DOVER-Lap fuse of P1 + P2.",
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
    parser.add_argument("--overlap_policy", choices=["keep", "drop", "dominant"], default="keep",
                        help="P1 overlap handling (default: keep — never delete overlap speech).")
    # Pipeline 2 options
    parser.add_argument("--asd_model", choices=["lr_asd", "loconet"], default="lr_asd")
    # Pipeline 3 options
    parser.add_argument("--method", choices=["ahc", "cdgcn", "vbx", "dover-lap", "nme-sc"],
                        default="ahc")
    parser.add_argument("--embedding", default="ecapa",
                        choices=["ecapa", "wespeaker34", "wespeaker293", "campplus", "redimnet"],
                        help="Speaker-embedding backend for ahc/cdgcn/nme-sc P3 methods.")
    parser.add_argument("--collar", type=float, default=0.0,
                        help="Primary DER collar (s). DER at collar=0.25 is always reported too.")
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
    # VBx params
    parser.add_argument("--vbx_loop_prob", type=float, default=0.9)
    parser.add_argument("--vbx_fa", type=float, default=0.3)
    parser.add_argument("--vbx_fb", type=float, default=17.0)
    parser.add_argument("--vbx_max_speakers", type=int, default=10)
    parser.add_argument("--vbx_lda_dim", type=int, default=128)
    parser.add_argument("--vbx_init_smoothing", type=float, default=5.0)
    # NME-SC params
    parser.add_argument("--max_speakers", type=int, default=8)
    parser.add_argument("--max_rp_threshold", type=float, default=0.25)

    args = parser.parse_args()

    samples = [args.sample] if args.sample else ALL_SAMPLES

    p1_out = os.path.join(DATA_DIR, "diarization")
    p2_out = os.path.join(DATA_DIR, "audio_visual")
    p3_out = os.path.join(DATA_DIR, "clean")
    fusion_out = os.path.join(DATA_DIR, "fusion")

    p3_extra = _p3_extra_args(args)

    for sample in samples:
        print(f"\n{'#'*60}")
        print(f"  SAMPLE: {sample}")
        print(f"{'#'*60}")

        ref_path = os.path.join(LABEL_DIR, f"{sample}.txt")

        if args.pipeline == "1":
            hyp = run_pipeline_1(
                sample, p1_out, args.p1_model,
                args.min_segment_duration, args.max_gap_threshold, args.overlap_policy,
            )
            run_evaluation(1, sample, hyp, ref_path, collar=args.collar)

        elif args.pipeline == "2":
            sd_path = os.path.join(p1_out, f"{sample}.txt")
            if not os.path.exists(sd_path):
                print(f"[ERROR] Pipeline 1 output not found: {sd_path}")
                print("        Run --pipeline 1 first.")
                continue
            hyp = run_pipeline_2(sample, sd_path, args.asd_model, p2_out)
            run_evaluation(2, sample, hyp, ref_path, collar=args.collar)

        elif args.pipeline == "3":
            p2_hyp = os.path.join(p2_out, sample, "supplemented_diarization.txt")
            if not os.path.exists(p2_hyp):
                print(f"[ERROR] Pipeline 2 output not found: {p2_hyp}")
                print("        Run --pipeline 2 first.")
                continue
            hyp = run_pipeline_3(sample, p2_hyp, args.method, p3_out, p3_extra)
            run_evaluation(3, sample, hyp, ref_path, method=args.method, collar=args.collar)

        elif args.pipeline == "all":
            hyp1 = run_pipeline_1(
                sample, p1_out, args.p1_model,
                args.min_segment_duration, args.max_gap_threshold, args.overlap_policy,
            )
            run_evaluation(1, sample, hyp1, ref_path, collar=args.collar)

            hyp2 = run_pipeline_2(sample, hyp1, args.asd_model, p2_out)
            run_evaluation(2, sample, hyp2, ref_path, collar=args.collar)

            hyp3 = run_pipeline_3(sample, hyp2, args.method, p3_out, p3_extra)
            run_evaluation(3, sample, hyp3, ref_path, method=args.method, collar=args.collar)

        elif args.pipeline == "fusion":
            # Fuse P1 (audio) + P2 (audio-visual). Reuse cached outputs if present.
            sd_path = os.path.join(p1_out, f"{sample}.txt")
            if not os.path.exists(sd_path):
                sd_path = run_pipeline_1(
                    sample, p1_out, args.p1_model,
                    args.min_segment_duration, args.max_gap_threshold, args.overlap_policy,
                )
                run_evaluation(1, sample, sd_path, ref_path, collar=args.collar)
            p2_hyp = os.path.join(p2_out, sample, "supplemented_diarization.txt")
            if not os.path.exists(p2_hyp):
                p2_hyp = run_pipeline_2(sample, sd_path, args.asd_model, p2_out)
                run_evaluation(2, sample, p2_hyp, ref_path, collar=args.collar)
            fused = run_fusion_pipeline(sample, [sd_path, p2_hyp], fusion_out)
            run_evaluation("fusion", sample, fused, ref_path, collar=args.collar)

    print("\nDone.")


if __name__ == "__main__":
    main()
