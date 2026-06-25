#!/usr/bin/env python3
"""Fusion pipeline — combine N diarization hypotheses via DOVER-Lap.

Instead of a one-directional cascade (P1 -> P2 -> P3) that can only ever delete
or relabel, fusion takes several *independent* hypotheses and votes them together
with DOVER-Lap (greedy label mapping + weighted average voting). This keeps the
recall of the most complete hypothesis while letting the others correct labels —
errors are not amplified in a single direction.

Reuses the DOVER-Lap implementation bundled under clean_pipeline/dover_lap.

Usage:
    python fuse.py --inputs p1.txt p2.txt --output_dir data/fusion/<sample>
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Make `viespeaker` importable from a fresh checkout, then centralize sys.path.
_d = _HERE
while _d != os.path.dirname(_d):
    if os.path.isdir(os.path.join(_d, "src", "viespeaker")):
        sys.path.insert(0, os.path.join(_d, "src"))
        break
    _d = os.path.dirname(_d)
from viespeaker import bootstrap  # noqa: E402
bootstrap.setup()

from dover_lap.src.doverlap import DOVERLap  # noqa: E402
from dover_lap.libs.turn import Turn  # noqa: E402


def _txt_to_turns(path, file_id):
    turns = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                onset, offset = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if offset > onset:
                turns.append(Turn(onset=onset, offset=offset, speaker_id=parts[2], file_id=file_id))
    return turns


def _write(path, segments):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for s in segments:
            f.write(f"{s['start']:.3f} {s['end']:.3f} {s['speaker']}\n")


def run_fusion(input_paths, output_path, file_id=None):
    """Fuse the given hypothesis txt files into output_path. Returns output_path."""
    present = [p for p in input_paths if p and os.path.exists(p)]
    if not present:
        raise FileNotFoundError(f"No fusion inputs exist among: {input_paths}")

    if file_id is None:
        file_id = os.path.splitext(os.path.basename(present[0]))[0]

    turns_list = [_txt_to_turns(p, file_id) for p in present]
    turns_list = [t for t in turns_list if t]  # drop empty hypotheses

    if not turns_list:
        _write(output_path, [])
        return output_path

    if len(turns_list) == 1:
        combined = turns_list[0]
    else:
        combined = DOVERLap.combine_turns_list(
            turns_list,
            file_id=file_id,
            label_mapping="greedy",
            voting_method="average",
            weight_type="rank",
        )

    out_segs = sorted(
        [{"start": t.onset, "end": t.offset, "speaker": t.speaker_id} for t in combined],
        key=lambda x: x["start"],
    )
    _write(output_path, out_segs)
    print(f"[Fusion] {len(present)} inputs -> {len(out_segs)} segments. Output: {output_path}")
    return output_path


def main():
    ap = argparse.ArgumentParser(description="Fuse diarization hypotheses with DOVER-Lap")
    ap.add_argument("--inputs", nargs="+", required=True, help="Hypothesis txt files to fuse")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--file_id", default=None)
    args = ap.parse_args()
    out = os.path.join(args.output_dir, "fused_diarization.txt")
    run_fusion(args.inputs, out, file_id=args.file_id)


if __name__ == "__main__":
    main()
