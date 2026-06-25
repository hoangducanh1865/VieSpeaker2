#!/usr/bin/env python3
"""Render a self-contained REPORT.md from a scenario results.json.

The report begins with a ready-to-use analysis PROMPT so you can copy-paste the
whole file into any LLM and immediately get a comparative analysis of the
scenarios. Can be run standalone:

    python experiment/scenarios/make_report.py experiment/<RUNTAG>/results.json
"""

import json
import os
import sys

_PROMPT = """\
<!-- ============================ ANALYSIS PROMPT ============================ -->
> **Bạn là chuyên gia speaker diarization.** Dưới đây là kết quả benchmark của hệ
> thống VieSpeaker2 (diarization tiếng Việt) trên nhiều *kịch bản* (scenario) khác
> nhau: các model Pipeline 1 (audio), Pipeline 2 (audio-visual, relabel-only),
> Pipeline 3 (cleansing: AHC / CDGCN / VBx / NME-SC) với nhiều speaker-embedding
> (ECAPA, WeSpeaker34/293, CAM++, ReDimNet), và chế độ **fusion** (DOVER-Lap).
> Mỗi ô là **DER %** (Diarization Error Rate, *thấp hơn = tốt hơn*) trên 6 mẫu thử.
>
> Hãy phân tích và trả lời:
> 1. **Xếp hạng tổng thể** các kịch bản theo Avg DER (collar=0 và collar=0.25). Kịch bản nào tốt nhất, kém nhất?
> 2. **Per-sample**: kịch bản tốt nhất cho từng mẫu? Mẫu nào khó nhất với mọi cấu hình và vì sao (dựa vào FA/MD/Confusion)?
> 3. **Pipeline 2 có giúp ích không?** So sánh `p1_*` vs `p2_*` vs `fusion_*`. Audio-visual và fusion có cải thiện so với chỉ audio?
> 4. **Embedding nào tốt nhất cho tiếng Việt?** So sánh các kịch bản P3 cùng method khác embedding (ví dụ AHC: ecapa vs wespeaker34 vs campplus).
> 5. **Thuật toán cleansing P3 nào hiệu quả nhất** (AHC vs CDGCN vs VBx vs NME-SC)?
> 6. Phân tích đánh đổi **FA (false alarm) / MD (missed detection) / Confusion** giữa các nhóm kịch bản.
> 7. **Khuyến nghị cấu hình tốt nhất** để triển khai, và 2-3 hướng cải thiện tiếp theo.
>
> *(English: You are a speaker-diarization expert. Each cell is DER % (lower is
> better). Rank scenarios overall and per-sample, judge whether audio-visual (P2)
> and fusion help over audio-only (P1), identify the best speaker embedding for
> Vietnamese and the best P3 cleansing algorithm, analyse the FA/MD/Confusion
> trade-offs, and recommend the best deployable configuration plus next steps.)*
<!-- ========================== END ANALYSIS PROMPT ========================== -->
"""


def _fmt(v, nd=2):
    if v is None:
        return "—"
    try:
        if v != v:  # NaN
            return "—"
        return f"{v:.{nd}f}"
    except Exception:
        return "—"


def _der(entry, sample):
    cell = entry["per_sample"].get(sample)
    if not cell:
        return None
    if "error" in cell:
        return None
    return cell.get("der")


def write_report(results, path):
    meta = results.get("meta", {})
    samples = meta.get("samples", [])
    scen = results.get("scenarios", {})

    # order scenarios by avg DER (successful first), failed last
    def sort_key(item):
        avg = item[1].get("avg", {})
        d = avg.get("der")
        return (d if d is not None else float("inf"))
    ordered = sorted(scen.items(), key=sort_key)

    lines = []
    lines.append("# VieSpeaker2 — Scenario Benchmark Report")
    lines.append("")
    lines.append(_PROMPT)
    lines.append("")
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- Run tag: `{meta.get('runtag','?')}`")
    lines.append(f"- Created: {meta.get('created','?')}")
    lines.append(f"- Git SHA: `{meta.get('git_sha','?')}`")
    lines.append(f"- Device: {meta.get('device','?')}")
    lines.append(f"- Base P1 for P2/P3/fusion: `{meta.get('base_p1','?')}`")
    lines.append(f"- Samples ({len(samples)}): {', '.join(samples)}")
    lines.append(f"- Scenarios: {len(scen)}")
    lines.append("- Metric: **DER %** (lower is better); DERc25 = DER at collar 0.25s.")
    lines.append("")

    # ---- Master table: per-sample DER + averages ----
    lines.append("## Master table — DER % per sample (collar=0)")
    lines.append("")
    header = ["Scenario"] + samples + ["**Avg DER**", "Avg DERc25", "Status"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for sid, e in ordered:
        row = [f"`{sid}`"]
        for s in samples:
            row.append(_fmt(_der(e, s)))
        avg = e.get("avg", {})
        row.append("**" + _fmt(avg.get("der")) + "**")
        row.append(_fmt(avg.get("der_collar025")))
        row.append(e.get("status", "?"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- Averages table: error decomposition ----
    lines.append("## Averages — error decomposition (mean over samples)")
    lines.append("")
    cols = [("Avg DER", "der"), ("Avg DERc25", "der_collar025"),
            ("FA %", "false_alarm_pct"), ("MD %", "missed_detection_pct"),
            ("Conf %", "confusion_pct"), ("Purity %", "purity"),
            ("Cov %", "coverage"), ("F1 %", "f1")]
    header = ["Scenario", "Description"] + [c[0] for c in cols]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for sid, e in ordered:
        avg = e.get("avg", {})
        row = [f"`{sid}`", e.get("desc", "")]
        for _, k in cols:
            row.append(_fmt(avg.get(k)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- Best per sample ----
    lines.append("## Best scenario per sample (lowest DER)")
    lines.append("")
    lines.append("| Sample | Best scenario | DER % |")
    lines.append("|---|---|---|")
    for s in samples:
        best_sid, best_der = None, None
        for sid, e in scen.items():
            d = _der(e, s)
            if d is not None and (best_der is None or d < best_der):
                best_der, best_sid = d, sid
        lines.append(f"| {s} | {'`'+best_sid+'`' if best_sid else '—'} | {_fmt(best_der)} |")
    lines.append("")

    # ---- Notes: failures / best-effort ----
    notes = []
    for sid, e in scen.items():
        if e.get("status") != "OK":
            errs = {s: c.get("error") for s, c in e["per_sample"].items() if "error" in c}
            be = " (best-effort)" if e.get("best_effort") else ""
            notes.append(f"- `{sid}` — **{e['status']}**{be}: " +
                         "; ".join(f"{s}: {msg}" for s, msg in list(errs.items())[:3]))
    if notes:
        lines.append("## Notes — partial / failed scenarios")
        lines.append("")
        lines.extend(notes)
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if len(sys.argv) < 2:
        print("usage: make_report.py <results.json> [out.md]")
        sys.exit(1)
    results_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(results_path), "REPORT.md")
    with open(results_path) as f:
        results = json.load(f)
    write_report(results, out)
    print("Wrote", out)


if __name__ == "__main__":
    main()
