#!/usr/bin/env python3
"""Run the curated VieSpeaker2 scenario matrix and produce a self-contained report.

Design goals
------------
* Cheap re-use: each P1 model is computed once; each P2 (face pipeline) is
  computed once per (base-P1, ASD); P3/fusion scenarios reuse those cached
  outputs. This avoids re-paying cloud API credits and the slow face pipeline.
* Resilient: every prerequisite and scenario is wrapped — one failure (e.g. a
  best-effort embedding that needs network) is recorded and the sweep continues.
* Always-current report: experiment/<RUNTAG>/REPORT.md is regenerated after every
  scenario so partial results are immediately usable.

Usage
-----
    python experiment/scenarios/run_scenarios.py                 # full sweep
    python experiment/scenarios/run_scenarios.py --smoke         # quick sanity
    python experiment/scenarios/run_scenarios.py --only p1_cloud p3_vbx
    python experiment/scenarios/run_scenarios.py --samples interview_noise movie
"""

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import traceback

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, _SRC)
sys.path.insert(0, _THIS)
from viespeaker import bootstrap, paths  # noqa: E402
bootstrap.setup()

import scenarios as SC  # noqa: E402
from evaluation import evaluate_diarization_files  # noqa: E402
from make_report import write_report  # noqa: E402

# Pipeline entrypoints + subprocess wrappers (shared with main.py via pipeline_api).
from viespeaker import pipeline_api  # noqa: E402

# Test-set layout
# Inputs (audio/video) come from the external assets dir; labels stay in repo.
AUDIO_DIR = str(paths.AUDIO_DIR)
VIDEO_DIR = str(paths.VIDEO_DIR)
LABEL_DIR = str(paths.LABEL_DIR)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", _ROOT, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# prerequisite computation (cached)
# --------------------------------------------------------------------------- #
def ensure_p1(cache_dir, key, sample):
    """Return path to cached P1 output for (key, sample), computing if missing."""
    out_dir = os.path.join(cache_dir, "p1", key)
    out = os.path.join(out_dir, f"{sample}.txt")
    if os.path.exists(out):
        return out
    return pipeline_api.run_p1(
        os.path.join(AUDIO_DIR, f"{sample}.wav"), out_dir, SC.P1_MODELS[key],
        overlap_policy="keep",
    )


def ensure_p2(cache_dir, p1_key, asd, sample, device):
    """Return path to cached P2 supplemented output, computing if missing."""
    out_dir = os.path.join(cache_dir, "p2", f"{p1_key}_{asd}")
    out = os.path.join(out_dir, sample, "supplemented_diarization.txt")
    if os.path.exists(out):
        return out
    sd = ensure_p1(cache_dir, p1_key, sample)
    if not sd:
        return None
    return pipeline_api.run_p2(
        os.path.join(VIDEO_DIR, f"{sample}.mp4"), sd, asd, out_dir,
    )


# --------------------------------------------------------------------------- #
# per-scenario hypothesis materialisation
# --------------------------------------------------------------------------- #
def build_hyp(scn, sample, cache_dir, scen_dir, device):
    kind = scn["kind"]
    if kind == "p1":
        return ensure_p1(cache_dir, scn["p1"], sample)

    if kind == "p2":
        return ensure_p2(cache_dir, scn.get("p1", SC.BASE_P1), scn["asd"], sample, device)

    if kind == "p3":
        if scn.get("stage_in") == "p2":
            din = ensure_p2(cache_dir, scn.get("p1", SC.BASE_P1), scn["asd"], sample, device)
        else:
            din = ensure_p1(cache_dir, scn.get("p1", SC.BASE_P1), sample)
        if not din:
            return None
        sample_out = os.path.join(scen_dir, sample)
        extra = ["--embedding", scn.get("embedding", "ecapa"), "--device", device]
        if scn["method"] == "dover-lap":
            extra += ["--diarization_path2", ensure_p1(cache_dir, SC.BASE_P1, sample) or din]
        return pipeline_api.run_p3(
            scn["method"], din, os.path.join(AUDIO_DIR, f"{sample}.wav"), sample_out, extra,
        )

    if kind == "fusion":
        inputs = []
        for spec in scn["fusion"]:
            tag, val = spec.split(":")
            p = ensure_p1(cache_dir, val, sample) if tag == "p1" \
                else ensure_p2(cache_dir, SC.BASE_P1, val, sample, device)
            if p:
                inputs.append(p)
        if not inputs:
            return None
        sample_out = os.path.join(scen_dir, sample)
        return pipeline_api.run_fusion(inputs, sample_out, sample)

    raise ValueError(f"Unknown scenario kind: {kind}")


_METRIC_KEYS = ["der", "der_collar025", "false_alarm_pct", "missed_detection_pct",
                "confusion_pct", "purity", "coverage", "f1"]


def evaluate(ref, hyp):
    m = evaluate_diarization_files(ref, hyp, collar=0.0)
    return {k: m[k] for k in _METRIC_KEYS}


def _avg(per_sample):
    rows = [v for v in per_sample.values() if "error" not in v]
    if not rows:
        return {}
    return {k: sum(r[k] for r in rows) / len(rows) for k in _METRIC_KEYS}


# --------------------------------------------------------------------------- #
# main sweep
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="VieSpeaker2 scenario sweep")
    ap.add_argument("--smoke", action="store_true", help="Quick subset of scenarios + samples")
    ap.add_argument("--only", nargs="*", default=None, help="Run only these scenario ids")
    ap.add_argument("--samples", nargs="*", default=None, help="Restrict to these samples")
    ap.add_argument("--runtag", default=None, help="Override run tag (default: date+gitsha)")
    ap.add_argument("--out_root", default=os.path.join(_ROOT, "experiment"))
    args = ap.parse_args()

    runtag = args.runtag or f"{_dt.datetime.now():%Y%m%d_%H%M%S}_{_git_sha()}"
    base = os.path.join(args.out_root, runtag)
    cache_dir = os.path.join(base, "cache")
    scen_root = os.path.join(base, "scenarios")
    os.makedirs(scen_root, exist_ok=True)

    samples = args.samples or (SC.SMOKE_SAMPLES if args.smoke else SC.ALL_SAMPLES)
    scen_list = SC.get_scenarios(only=args.only, smoke=args.smoke)
    device = _device()

    meta = {
        "runtag": runtag,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "device": device,
        "samples": samples,
        "base_p1": SC.BASE_P1,
        "n_scenarios": len(scen_list),
    }
    results = {"meta": meta, "scenarios": {}}
    results_path = os.path.join(base, "results.json")
    report_path = os.path.join(base, "REPORT.md")

    print(f"\n=== VieSpeaker2 sweep '{runtag}' | device={device} | "
          f"{len(scen_list)} scenarios x {len(samples)} samples ===\n")

    for si, scn in enumerate(scen_list, 1):
        sid = scn["id"]
        print(f"\n########## [{si}/{len(scen_list)}] {sid} — {scn['desc']} ##########")
        scen_dir = os.path.join(scen_root, sid)
        os.makedirs(scen_dir, exist_ok=True)

        entry = {
            "desc": scn["desc"], "kind": scn["kind"],
            "config": {k: scn[k] for k in scn if k not in ("desc",)},
            "best_effort": scn.get("best_effort", False),
            "per_sample": {}, "status": "OK", "error": None,
        }

        for sample in samples:
            ref = os.path.join(LABEL_DIR, f"{sample}.txt")
            try:
                hyp = build_hyp(scn, sample, cache_dir, scen_dir, device)
                if not hyp or not os.path.exists(hyp):
                    raise RuntimeError("hypothesis not produced")
                entry["per_sample"][sample] = evaluate(ref, hyp)
                d = entry["per_sample"][sample]["der"]
                print(f"    {sample:16s} DER={d:6.2f}%")
            except Exception as e:  # one sample failing must not kill the sweep
                entry["per_sample"][sample] = {"error": f"{type(e).__name__}: {e}"}
                print(f"    {sample:16s} FAILED: {e}")
                traceback.print_exc()

        ok = sum(1 for v in entry["per_sample"].values() if "error" not in v)
        if ok == 0:
            entry["status"] = "FAILED"
        elif ok < len(samples):
            entry["status"] = "PARTIAL"
        entry["avg"] = _avg(entry["per_sample"])

        results["scenarios"][sid] = entry
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        write_report(results, report_path)
        print(f"  -> status={entry['status']}  "
              f"avg_DER={entry['avg'].get('der', float('nan')):.2f}%  (report updated)")

    print(f"\nDONE. Results: {results_path}\nReport : {report_path}")


if __name__ == "__main__":
    main()
