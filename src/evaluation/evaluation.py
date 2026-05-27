import os
import json
import argparse
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationPurity, DiarizationCoverage


def load_annotation_from_file(file_path, uri="default_audio_id"):
    annotation = Annotation(uri=uri)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file: {file_path}")
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2]
                annotation[Segment(start, end)] = speaker
    return annotation


def measure_dataset_pollution(reference: Annotation, hypothesis: Annotation):
    total_pollution_time = 0.0
    severe_pollution_time = 0.0
    problematic_segments = []
    for hyp_segment, _, hyp_speaker in hypothesis.itertracks(yield_label=True):
        ref_in_this_segment = reference.crop(hyp_segment)
        actual_speakers = ref_in_this_segment.labels()
        if len(actual_speakers) > 1:
            speaker_durations = {}
            for spk in actual_speakers:
                spk_timeline = ref_in_this_segment.label_timeline(spk)
                speaker_durations[spk] = spk_timeline.duration()
            sorted_durations = sorted(speaker_durations.values(), reverse=True)
            dominant_duration = sorted_durations[0]
            total_speech_duration = sum(sorted_durations)
            impurity_duration = total_speech_duration - dominant_duration
            total_pollution_time += impurity_duration
            if impurity_duration > 0.5:
                severe_pollution_time += impurity_duration
            problematic_segments.append({
                "time": f"[{hyp_segment.start:.3f} --> {hyp_segment.end:.3f}]",
                "predicted_as": hyp_speaker,
                "actual_speakers": actual_speakers,
                "pollution_duration": impurity_duration,
            })
    return total_pollution_time, severe_pollution_time, problematic_segments


def measure_coverage(reference: Annotation, hypothesis: Annotation):
    hypothesis_speech_duration = hypothesis.get_timeline().duration()
    video_duration = reference.get_timeline().extent().duration
    if video_duration == 0:
        return 0.0
    return hypothesis_speech_duration / video_duration


def evaluate_diarization_files(ref_file_path, hyp_file_path):
    common_uri = "evaluation_session_01"
    reference = load_annotation_from_file(ref_file_path, uri=common_uri)
    hypothesis = load_annotation_from_file(hyp_file_path, uri=common_uri)
    hypothesis_speech_duration = hypothesis.get_timeline().duration()

    der_metric = DiarizationErrorRate()
    der_score = der_metric(reference, hypothesis)
    components = der_metric.compute_components(reference, hypothesis)

    purity_metric = DiarizationPurity()
    purity_score = float(purity_metric(reference, hypothesis))

    coverage_metric = DiarizationCoverage()
    coverage_score = float(coverage_metric(reference, hypothesis))

    total = components["total"]
    false_alarm_pct = (components["false alarm"] * 100 / total) if total > 0 else 0.0
    missed_detection_pct = (components["missed detection"] * 100 / total) if total > 0 else 0.0
    confusion_pct = (components["confusion"] * 100 / total) if total > 0 else 0.0

    coverage_hyp = measure_coverage(reference, hypothesis)
    pollution_time, severe_pollution_time, _ = measure_dataset_pollution(reference, hypothesis)
    impurity_0 = (pollution_time * 100 / hypothesis_speech_duration) if hypothesis_speech_duration > 0 else 0.0
    impurity_5 = (severe_pollution_time * 100 / hypothesis_speech_duration) if hypothesis_speech_duration > 0 else 0.0

    f1 = (
        2 * purity_score * coverage_score / (purity_score + coverage_score)
        if (purity_score + coverage_score) > 0
        else 0.0
    )

    print(f"--- Evaluation Results ---")
    print(f"Reference File:  {ref_file_path}")
    print(f"Hypothesis File: {hyp_file_path}")
    print(f"Overall DER:     {der_score * 100:.2f}%\n")
    print(f"Coverage (hyp/video):  {coverage_hyp * 100:.2f}%")
    print(f"Purity:                {purity_score * 100:.2f}%")
    print(f"Coverage:              {coverage_score * 100:.2f}%")
    print(f"F1:                    {f1 * 100:.2f}%")
    print("Detailed Components (in seconds):")
    print(f"- Total Speech:      {total:.2f}s (100.00%)")
    print(f"- False Alarms:      {components['false alarm']:.2f}s ({false_alarm_pct:.2f}%)")
    print(f"- Missed Detection:  {components['missed detection']:.2f}s ({missed_detection_pct:.2f}%)")
    print(f"- Speaker Confusion: {components['confusion']:.2f}s ({confusion_pct:.2f}%)")
    print(f"Impurity (threshold=0.0s): {impurity_0:.2f}%")
    print(f"Impurity (threshold=0.5s): {impurity_5:.2f}%")

    return {
        "der": float(der_score) * 100,
        "false_alarm_pct": false_alarm_pct,
        "missed_detection_pct": missed_detection_pct,
        "confusion_pct": confusion_pct,
        "purity": purity_score * 100,
        "coverage": coverage_score * 100,
        "f1": f1 * 100,
        "coverage_hyp": coverage_hyp * 100,
        "impurity_0": impurity_0,
        "impurity_5": impurity_5,
        "pollution": pollution_time,
        "severe_pollution": severe_pollution_time,
        "total": float(total),
        "false_alarm_s": float(components["false alarm"]),
        "missed_detection_s": float(components["missed detection"]),
        "confusion_s": float(components["confusion"]),
    }


# ── Existing bar charts ───────────────────────────────────────────────────────

def _date_tag():
    return datetime.now().strftime("%Y%m%d")


def plot_pipeline1_der_chart(results: dict, experiments_dir: str):
    samples = sorted(results.keys())
    fa = [results[s]["false_alarm_pct"] for s in samples]
    md = [results[s]["missed_detection_pct"] for s in samples]
    conf = [results[s]["confusion_pct"] for s in samples]
    x = range(len(samples))
    fig, ax = plt.subplots(figsize=(max(6, len(samples) * 1.4), 5))
    ax.bar(x, fa, label="False Alarm", color="#4c72b0")
    ax.bar(x, md, bottom=fa, label="Missed Detection", color="#dd8452")
    ax.bar(x, conf, bottom=[f + m for f, m in zip(fa, md)], label="Confusion", color="#55a868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(samples, rotation=20, ha="right")
    ax.set_ylabel("% of reference speech")
    ax.set_title("Pipeline 1 — DER Components per Sample")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(experiments_dir, f"pipeline1_der_{_date_tag()}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


def plot_pipeline2_prf_chart(results: dict, experiments_dir: str):
    samples = sorted(results.keys())
    purity = [results[s]["purity"] for s in samples]
    coverage = [results[s]["coverage"] for s in samples]
    f1 = [results[s]["f1"] for s in samples]
    x = list(range(len(samples)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(samples) * 1.6), 5))
    ax.bar([xi - width for xi in x], purity, width=width, label="Purity", color="#4c72b0")
    ax.bar(x, coverage, width=width, label="Coverage", color="#dd8452")
    ax.bar([xi + width for xi in x], f1, width=width, label="F1", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=20, ha="right")
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    ax.set_title("Pipeline 2 — Purity / Coverage / F1 per Sample")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(experiments_dir, f"pipeline2_prf_{_date_tag()}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


def plot_pipeline3_purity_chart(results_p3: dict, results_p2: dict, experiments_dir: str):
    samples = sorted(results_p3.keys())
    der_after = [results_p3[s]["der"] for s in samples]
    der_before = [results_p2.get(s, {}).get("der", float("nan")) for s in samples]
    x = list(range(len(samples)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(samples) * 1.6), 5))
    ax.bar([xi - width / 2 for xi in x], der_before, width=width, label="DER before (P2)", color="#dd8452")
    ax.bar([xi + width / 2 for xi in x], der_after, width=width, label="DER after (P3)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=20, ha="right")
    ax.set_ylabel("DER %")
    ax.set_title("Pipeline 3 — DER Before vs After Cleansing")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(experiments_dir, f"pipeline3_purity_{_date_tag()}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


# ── New: metrics summary table ────────────────────────────────────────────────

def plot_metrics_table(metrics: dict, sample_key: str, pipeline_num: int, experiments_dir: str):
    """Render a full metrics summary table as a PNG for one sample."""
    rows = [
        ["Overall DER",           f"{metrics['der']:.2f}%"],
        ["Coverage (hyp/video)",  f"{metrics['coverage_hyp']:.2f}%"],
        ["Purity",                f"{metrics['purity']:.2f}%"],
        ["Coverage",              f"{metrics['coverage']:.2f}%"],
        ["F1",                    f"{metrics['f1']:.2f}%"],
        ["Total Speech",          f"{metrics['total']:.2f}s"],
        ["False Alarms",          f"{metrics['false_alarm_s']:.2f}s  ({metrics['false_alarm_pct']:.2f}%)"],
        ["Missed Detection",      f"{metrics['missed_detection_s']:.2f}s  ({metrics['missed_detection_pct']:.2f}%)"],
        ["Speaker Confusion",     f"{metrics['confusion_s']:.2f}s  ({metrics['confusion_pct']:.2f}%)"],
        ["Impurity (≥ 0.0 s)",    f"{metrics['impurity_0']:.2f}%"],
        ["Impurity (≥ 0.5 s)",    f"{metrics['impurity_5']:.2f}%"],
    ]

    n_rows = len(rows)
    fig_h = 0.45 * n_rows + 1.2
    fig, ax = plt.subplots(figsize=(7, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.55)

    # Style header
    for col in range(2):
        cell = tbl[0, col]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for row in range(1, n_rows + 1):
        bg = "#eaf0fb" if row % 2 == 0 else "white"
        for col in range(2):
            tbl[row, col].set_facecolor(bg)
        # Highlight DER row
        if rows[row - 1][0] == "Overall DER":
            for col in range(2):
                tbl[row, col].set_facecolor("#fdecea")
                tbl[row, col].set_text_props(fontweight="bold")

    ax.set_title(
        f"Pipeline {pipeline_num} — {sample_key}",
        fontsize=13, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    out_path = os.path.join(
        experiments_dir, f"pipeline{pipeline_num}_table_{sample_key}_{_date_tag()}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


# ── New: cross-pipeline line chart ───────────────────────────────────────────

def plot_pipeline_comparison_line(experiments_dir: str):
    """Line chart comparing DER, F1, Purity, Coverage across P1→P2→P3."""
    r = {
        p: _load_results(experiments_dir, p)
        for p in (1, 2, 3)
    }

    # Collect samples that appear in at least one pipeline
    all_samples = sorted(set().union(*[set(r[p].keys()) for p in (1, 2, 3)]))
    if not all_samples:
        return

    pipeline_labels = ["P1", "P2", "P3"]
    metrics_cfg = [
        ("der",     "DER %",      "#e74c3c"),
        ("f1",      "F1 %",       "#2ecc71"),
        ("purity",  "Purity %",   "#3498db"),
        ("coverage","Coverage %", "#f39c12"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    colors_samples = plt.cm.tab10(np.linspace(0, 1, max(len(all_samples), 1)))

    for ax_idx, (metric_key, metric_label, _) in enumerate(metrics_cfg):
        ax = axes[ax_idx]
        for si, sample in enumerate(all_samples):
            values = []
            xs = []
            for pi, p in enumerate((1, 2, 3)):
                if sample in r[p] and metric_key in r[p][sample]:
                    values.append(r[p][sample][metric_key])
                    xs.append(pi)
            if values:
                ax.plot(
                    xs, values,
                    marker="o", linewidth=2,
                    color=colors_samples[si],
                    label=sample,
                )
                # Annotate last point
                ax.annotate(
                    f"{values[-1]:.1f}",
                    xy=(xs[-1], values[-1]),
                    xytext=(4, 2), textcoords="offset points",
                    fontsize=7, color=colors_samples[si],
                )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(pipeline_labels)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Pipeline Comparison — Metrics across P1 → P2 → P3", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(experiments_dir, f"pipeline_comparison_line_{_date_tag()}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_results(experiments_dir: str, pipeline_num: int) -> dict:
    path = os.path.join(experiments_dir, f"pipeline{pipeline_num}_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_results(experiments_dir: str, pipeline_num: int, results: dict):
    os.makedirs(experiments_dir, exist_ok=True)
    path = os.path.join(experiments_dir, f"pipeline{pipeline_num}_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate diarization results and generate PNG charts"
    )
    parser.add_argument("--pipeline", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--hyp_path", required=True, help="Hypothesis diarization file")
    parser.add_argument("--ref_path", required=True, help="Reference diarization file")
    parser.add_argument("--sample_key", default="unknown", help="Sample name (used as chart label)")
    parser.add_argument("--experiments_dir", default="experiments", help="Directory for charts and JSON")

    args = parser.parse_args()

    metrics = evaluate_diarization_files(args.ref_path, args.hyp_path)

    results = _load_results(args.experiments_dir, args.pipeline)
    results[args.sample_key] = metrics
    _save_results(args.experiments_dir, args.pipeline, results)

    # Existing bar/stacked charts
    if args.pipeline == 1:
        plot_pipeline1_der_chart(results, args.experiments_dir)
    elif args.pipeline == 2:
        plot_pipeline2_prf_chart(results, args.experiments_dir)
    elif args.pipeline == 3:
        results_p2 = _load_results(args.experiments_dir, 2)
        plot_pipeline3_purity_chart(results, results_p2, args.experiments_dir)

    # New: per-sample table + cross-pipeline line chart
    plot_metrics_table(metrics, args.sample_key, args.pipeline, args.experiments_dir)
    plot_pipeline_comparison_line(args.experiments_dir)
