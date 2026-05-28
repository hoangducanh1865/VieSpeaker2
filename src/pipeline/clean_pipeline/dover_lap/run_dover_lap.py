"""
DOVER-Lap: Diarization Output Voting and Error Reduction via Label Averaging.
Fuses multiple diarization hypotheses (P1 + P2) via greedy label mapping + weighted voting.
"""

import os
import sys

_DOVER_LAP_DIR = os.path.dirname(os.path.abspath(__file__))
_CLEAN_PIPELINE_DIR = os.path.dirname(_DOVER_LAP_DIR)

# Inject clean_pipeline dir so `dover_lap.libs.X` absolute imports resolve to local copy
if _CLEAN_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _CLEAN_PIPELINE_DIR)
if _CLEAN_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _CLEAN_PIPELINE_DIR)

from dover_lap.src.doverlap import DOVERLap
from dover_lap.libs.turn import Turn

for _p in [_CLEAN_PIPELINE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from clean import write_diarization


def _txt_to_turns(txt_path, file_id=None):
    """Parse plain-text diarization file into Turn objects."""
    if file_id is None:
        file_id = os.path.splitext(os.path.basename(txt_path))[0]
    turns = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                onset, offset = float(parts[0]), float(parts[1])
                if offset > onset:
                    turns.append(Turn(onset=onset, offset=offset,
                                      speaker_id=parts[2], file_id=file_id))
            except ValueError:
                continue
    return turns


def run_dover_lap(args) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "cleansed_diarization.txt")

    if not args.diarization_path2:
        raise ValueError("[DOVER-Lap] --diarization_path2 is required for dover-lap method")

    print("\n[DOVER-Lap] Step 1: Parsing diarization files")
    file_id = os.path.splitext(os.path.basename(args.diarization_path))[0]
    turns_p2 = _txt_to_turns(args.diarization_path, file_id)
    turns_p1 = _txt_to_turns(args.diarization_path2, file_id)
    print(f"  P2 turns: {len(turns_p2)}, P1 turns: {len(turns_p1)}")

    if not turns_p2 and not turns_p1:
        write_diarization(output_path, [])
        return output_path

    if not turns_p2:
        combined = turns_p1
    elif not turns_p1:
        combined = turns_p2
    else:
        print("\n[DOVER-Lap] Step 2: Fusing with greedy label mapping + average voting")
        combined = DOVERLap.combine_turns_list(
            [turns_p2, turns_p1],
            file_id=file_id,
            label_mapping="greedy",
            voting_method="average",
            weight_type="rank",
        )

    print(f"\n[DOVER-Lap] Step 3: Writing output ({len(combined)} turns)")
    out_segs = sorted(
        [{"start": t.onset, "end": t.offset, "speaker": t.speaker_id} for t in combined],
        key=lambda x: x["start"],
    )

    write_diarization(output_path, out_segs)
    print(f"[DOVER-Lap] Done. Output: {output_path}")
    return output_path
