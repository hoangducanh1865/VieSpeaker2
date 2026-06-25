"""Thin, reusable wrappers around the pipeline entry scripts.

Centralizes the subprocess-command construction that ``main.py`` and the scenario
runner used to duplicate. Each ``run_*`` returns the expected output path (or
``None`` if it was not produced); stages stay isolated in their own process so
their (conflicting) dependencies never clash.
"""

from __future__ import annotations

import os
import subprocess
import sys

from . import paths

PY = sys.executable

P1_SCRIPT = str(paths.SRC_ROOT / "pipeline" / "audio_pipeline" / "speaker_diarization.py")
P2_SCRIPT = str(paths.SRC_ROOT / "pipeline" / "audio_visual_pipeline" / "supplement_pipeline.py")
P3_SCRIPT = str(paths.SRC_ROOT / "pipeline" / "clean_pipeline" / "clean.py")
FUSION_SCRIPT = str(paths.SRC_ROOT / "pipeline" / "fusion_pipeline" / "fuse.py")
EVAL_SCRIPT = str(paths.SRC_ROOT / "evaluation" / "evaluation.py")


def run(cmd, label: str | None = None) -> bool:
    """Run a subprocess (streaming output). Returns True on exit code 0."""
    if label:
        print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    print("$", " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.run([str(c) for c in cmd]).returncode
    if rc != 0:
        print(f"[rc={rc}] {label or ''}")
    return rc == 0


def run_p1(audio_path, output_dir, model, *, min_seg=0.5, max_gap=0.5,
           overlap_policy="keep", label=None):
    os.makedirs(output_dir, exist_ok=True)
    ok = run([PY, P1_SCRIPT, "--audio_path", audio_path, "--output_dir", output_dir,
              "--model", model, "--min_segment_duration", min_seg,
              "--max_gap_threshold", max_gap, "--overlap_policy", overlap_policy], label)
    out = os.path.join(output_dir, f"{_stem(audio_path)}.txt")
    return out if (ok and os.path.exists(out)) else (out if os.path.exists(out) else None)


def run_p2(video_path, sd_diarization, asd_model, out_dir, *, label=None):
    sample = _stem(video_path)
    run([PY, P2_SCRIPT, "--video_path", video_path, "--sd_diarization", sd_diarization,
         "--model", asd_model, "--out_dir", out_dir], label)
    out = os.path.join(out_dir, sample, "supplemented_diarization.txt")
    return out if os.path.exists(out) else None


def run_p3(method, diarization_path, audio_path, output_dir, extra_args=None, *, label=None):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [PY, P3_SCRIPT, "--method", method, "--diarization_path", diarization_path,
           "--audio_path", audio_path, "--output_dir", output_dir] + list(extra_args or [])
    run(cmd, label)
    out = os.path.join(output_dir, "cleansed_diarization.txt")
    return out if os.path.exists(out) else None


def run_fusion(inputs, output_dir, file_id, *, label=None):
    present = [p for p in inputs if p and os.path.exists(p)]
    os.makedirs(output_dir, exist_ok=True)
    run([PY, FUSION_SCRIPT, "--inputs", *present, "--output_dir", output_dir,
         "--file_id", file_id], label)
    out = os.path.join(output_dir, "fused_diarization.txt")
    return out if os.path.exists(out) else None


def evaluate(pipeline, hyp_path, ref_path, sample_key, experiments_dir, *,
             method="", collar=0.0, label=None):
    if not hyp_path or not os.path.exists(hyp_path):
        print(f"[EVAL] Skipping — hypothesis file missing: {hyp_path}")
        return
    os.makedirs(experiments_dir, exist_ok=True)
    cmd = [PY, EVAL_SCRIPT, "--pipeline", str(pipeline), "--hyp_path", hyp_path,
           "--ref_path", ref_path, "--sample_key", sample_key,
           "--experiments_dir", experiments_dir, "--collar", collar]
    if str(pipeline) == "3" and method:
        cmd += ["--method", method]
    run(cmd, label)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
