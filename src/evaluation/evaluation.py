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
    out_path = os.path.join(_dated_dir(experiments_dir), "pipeline1_der.png")
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
    out_path = os.path.join(_dated_dir(experiments_dir), "pipeline2_prf.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


def plot_pipeline3_purity_chart(results_p3: dict, results_p2: dict, experiments_dir: str, method: str = ""):
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
    method_label = f" [{method}]" if method else ""
    ax.set_title(f"Pipeline 3{method_label} — DER Before vs After Cleansing")
    ax.legend()
    fig.tight_layout()
    suffix = f"_{method}" if method else ""
    out_path = os.path.join(_dated_dir(experiments_dir), f"pipeline3{suffix}_purity.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


# ── New: metrics summary table ────────────────────────────────────────────────

def plot_metrics_table(metrics: dict, sample_key: str, pipeline_num: int, experiments_dir: str, method: str = ""):
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
    suffix = f"_{method}" if (pipeline_num == 3 and method) else ""
    out_path = os.path.join(
        _dated_dir(experiments_dir), f"pipeline{pipeline_num}{suffix}_table_{sample_key}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


# ── New: cross-pipeline line chart ───────────────────────────────────────────

def plot_pipeline_comparison_line(experiments_dir: str):
    """Line chart comparing all metrics across P1→P2→P3, one subplot per metric."""
    r = {
        p: _load_results(experiments_dir, p)
        for p in (1, 2)
    }
    # For pipeline 3, merge all method results (best DER wins per sample)
    r[3] = {}
    for m in ("ahc", "cdgcn", "vbx", "dover-lap", "nme-sc", ""):
        for sample, metrics in _load_results(experiments_dir, 3, m).items():
            if sample not in r[3] or metrics["der"] < r[3][sample]["der"]:
                r[3][sample] = metrics

    # Collect samples that appear in at least one pipeline
    all_samples = sorted(set().union(*[set(r[p].keys()) for p in (1, 2, 3)]))
    if not all_samples:
        return

    pipeline_labels = ["P1", "P2", "P3"]

    # All metrics matching columns in the all-samples summary table
    # Organised as two logical rows: error/confusion metrics, then quality/impurity metrics
    metrics_cfg = [
        # Row 1 — error & confusion components
        ("der",                 "DER %",          "#e74c3c"),
        ("false_alarm_pct",     "FA %",            "#c0392b"),
        ("missed_detection_pct","MD %",            "#e67e22"),
        ("confusion_pct",       "Conf %",          "#f39c12"),
        ("purity",              "Purity %",        "#3498db"),
        # Row 2 — quality & impurity metrics
        ("coverage",            "Coverage %",      "#2980b9"),
        ("f1",                  "F1 %",            "#2ecc71"),
        ("coverage_hyp",        "H/V %",           "#27ae60"),
        ("impurity_0",          "Imp₀ %",          "#9b59b6"),
        ("impurity_5",          "Imp₀.₅ %",        "#8e44ad"),
    ]

    n_metrics = len(metrics_cfg)   # 10
    n_cols = 5
    n_rows = (n_metrics + n_cols - 1) // n_cols  # 2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.2, n_rows * 4.0),
        sharey=False,
    )
    axes = axes.flatten()

    colors_samples = plt.cm.tab10(np.linspace(0, 1, max(len(all_samples), 1)))

    for ax_idx, (metric_key, metric_label, color) in enumerate(metrics_cfg):
        ax = axes[ax_idx]
        legend_added = False
        for si, sample in enumerate(all_samples):
            values = []
            xs = []
            for pi, p in enumerate((1, 2, 3)):
                if sample in r[p] and metric_key in r[p][sample]:
                    v = r[p][sample][metric_key]
                    if v == v:   # skip NaN
                        values.append(v)
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
                legend_added = True

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(pipeline_labels)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_title(metric_label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if legend_added and ax_idx == 0:
            ax.legend(fontsize=7, loc="best")

    # Hide any unused subplots (in case n_metrics < n_rows*n_cols)
    for ax_idx in range(n_metrics, len(axes)):
        axes[ax_idx].set_visible(False)

    # Shared legend below the figure (all sample colours in one row)
    handles = [
        plt.Line2D([0], [0], color=colors_samples[si], marker="o",
                   linewidth=2, label=sample)
        for si, sample in enumerate(all_samples)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(all_samples), 6),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )

    fig.suptitle(
        "Pipeline Comparison — All Metrics across P1 → P2 → P3",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(_dated_dir(experiments_dir), "pipeline_comparison_line.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


# ── New: all-samples summary table ───────────────────────────────────────────

def plot_all_samples_summary_table(results: dict, pipeline_num: int, experiments_dir: str, method: str = ""):
    """
    Single table image with one row per sample and all key metrics as columns.
    Generated (and overwritten) every time a new sample result is added.
    """
    if not results:
        return

    COLS = [
        ("Sample",    None),
        ("DER %",     "der"),
        ("FA %",      "false_alarm_pct"),
        ("MD %",      "missed_detection_pct"),
        ("Conf %",    "confusion_pct"),
        ("Purity %",  "purity"),
        ("Cov %",     "coverage"),
        ("F1 %",      "f1"),
        ("H/V %",     "coverage_hyp"),
        ("Imp₀ %",    "impurity_0"),
        ("Imp₀.₅ %",  "impurity_5"),
    ]

    samples = sorted(results.keys())
    n_rows = len(samples)
    n_cols = len(COLS)

    # Build cell data
    cell_text = []
    der_values = []
    for sample in samples:
        m = results[sample]
        row = [sample]
        for _, key in COLS[1:]:
            val = m.get(key, float("nan"))
            row.append(f"{val:.2f}" if not (val != val) else "—")
        cell_text.append(row)
        der_values.append(m.get("der", float("nan")))

    # Find best (lowest) and worst (highest) DER for highlighting
    valid_ders = [(v, i) for i, v in enumerate(der_values) if v == v]
    best_row  = min(valid_ders, key=lambda x: x[0])[1] if valid_ders else -1
    worst_row = max(valid_ders, key=lambda x: x[0])[1] if valid_ders else -1

    fig_w = max(14, n_cols * 1.35)
    fig_h = max(2.5, n_rows * 0.52 + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_labels = [c[0] for c in COLS]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Header row style
    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Data rows style
    for row_idx in range(1, n_rows + 1):
        sample_row = row_idx - 1
        for col in range(n_cols):
            cell = tbl[row_idx, col]
            if sample_row == best_row:
                cell.set_facecolor("#d5f5e3")   # light green — best DER
            elif sample_row == worst_row:
                cell.set_facecolor("#fdecea")   # light red   — worst DER
            else:
                cell.set_facecolor("#eaf0fb" if row_idx % 2 == 0 else "white")
            # Bold DER column
            if col == 1:
                cell.set_text_props(fontweight="bold")

    # Legend note
    fig.text(
        0.01, 0.01,
        "[green]=best DER  [red]=worst DER  |  FA=False Alarm  MD=Missed Detection  Conf=Speaker Confusion"
        "  H/V=Hypothesis/Video coverage  Imp=Impurity",
        fontsize=7, color="#555555", va="bottom",
    )

    method_label = f" [{method}]" if (pipeline_num == 3 and method) else ""
    ax.set_title(
        f"Pipeline {pipeline_num}{method_label} — All Samples Summary ({n_rows} samples)",
        fontsize=13, fontweight="bold", pad=14,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    suffix = f"_{method}" if (pipeline_num == 3 and method) else ""
    out_path = os.path.join(
        _dated_dir(experiments_dir), f"pipeline{pipeline_num}{suffix}_all_samples_table.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dated_dir(experiments_dir: str) -> str:
    """Return (and create if needed) experiments_dir/YYYYMMDD/ for today."""
    d = os.path.join(experiments_dir, datetime.now().strftime("%Y%m%d"))
    os.makedirs(d, exist_ok=True)
    return d


def _results_filename(pipeline_num: int, method: str = "") -> str:
    suffix = f"_{method}" if (pipeline_num == 3 and method) else ""
    return f"pipeline{pipeline_num}{suffix}_results.json"


def _load_results(experiments_dir: str, pipeline_num: int, method: str = "") -> dict:
    path = os.path.join(_dated_dir(experiments_dir), _results_filename(pipeline_num, method))
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_results(experiments_dir: str, pipeline_num: int, results: dict, method: str = ""):
    path = os.path.join(_dated_dir(experiments_dir), _results_filename(pipeline_num, method))
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def plot_pipeline3_methods_comparison(experiments_dir: str):
    """Grouped bar chart comparing DER across all 5 P3 methods, one group per sample."""
    all_methods = ["ahc", "cdgcn", "vbx", "dover-lap", "nme-sc"]
    method_results = {m: _load_results(experiments_dir, 3, m) for m in all_methods}

    # Only plot if at least 2 methods have data
    methods_with_data = [m for m in all_methods if method_results[m]]
    if len(methods_with_data) < 2:
        return

    all_samples = sorted(set().union(*[set(method_results[m].keys()) for m in methods_with_data]))
    if not all_samples:
        return

    n_methods = len(methods_with_data)
    n_samples = len(all_samples)
    x = np.arange(n_samples)
    width = 0.8 / n_methods

    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
    fig, ax = plt.subplots(figsize=(max(8, n_samples * 1.8), 5))

    for mi, m in enumerate(methods_with_data):
        ders = [method_results[m].get(s, {}).get("der", float("nan")) for s in all_samples]
        offset = (mi - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, ders, width=width, label=m, color=colors[mi % len(colors)])
        for bar, val in zip(bars, ders):
            if val == val:  # not NaN
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(all_samples, rotation=20, ha="right")
    ax.set_ylabel("DER %")
    ax.set_title("Pipeline 3 — DER Comparison across Methods")
    ax.legend(title="Method")
    fig.tight_layout()
    out_path = os.path.join(_dated_dir(experiments_dir), "pipeline3_methods_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Chart] Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate diarization results and generate PNG charts"
    )
    parser.add_argument("--pipeline", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--hyp_path", required=True, help="Hypothesis diarization file")
    parser.add_argument("--ref_path", required=True, help="Reference diarization file")
    parser.add_argument("--sample_key", default="unknown", help="Sample name (used as chart label)")
    parser.add_argument("--experiments_dir", default="experiment", help="Directory for charts and JSON")
    parser.add_argument("--method", default="", help="Cleansing method (pipeline 3 only)")

    args = parser.parse_args()

    metrics = evaluate_diarization_files(args.ref_path, args.hyp_path)

    results = _load_results(args.experiments_dir, args.pipeline, args.method)
    results[args.sample_key] = metrics
    _save_results(args.experiments_dir, args.pipeline, results, args.method)

    # Bar/stacked charts
    if args.pipeline == 1:
        plot_pipeline1_der_chart(results, args.experiments_dir)
    elif args.pipeline == 2:
        plot_pipeline2_prf_chart(results, args.experiments_dir)
    elif args.pipeline == 3:
        results_p2 = _load_results(args.experiments_dir, 2)
        plot_pipeline3_purity_chart(results, results_p2, args.experiments_dir, args.method)

    # Per-sample table + all-samples summary table + cross-pipeline line chart
    plot_metrics_table(metrics, args.sample_key, args.pipeline, args.experiments_dir, args.method)
    plot_all_samples_summary_table(results, args.pipeline, args.experiments_dir, args.method)
    plot_pipeline_comparison_line(args.experiments_dir)

    # P3 multi-method comparison (only generated if ≥ 2 methods have data)
    if args.pipeline == 3:
        plot_pipeline3_methods_comparison(args.experiments_dir)
