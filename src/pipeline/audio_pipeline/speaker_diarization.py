"""
Speaker diarization module using pyannote.audio (Pipeline 1)
"""

import os
import sys
import argparse
import contextlib
import torch
from pathlib import Path
import warnings

# Make `viespeaker` importable from a fresh checkout (for the .env loader).
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d):
    if os.path.isdir(os.path.join(_d, "src", "viespeaker")):
        sys.path.insert(0, os.path.join(_d, "src"))
        break
    _d = os.path.dirname(_d)
from viespeaker.env import load as _load_dotenv  # noqa: E402  (configurable .env location)
from viespeaker.hf_compat import pretrained_auth_kwargs, pyannote_hf_hub_compat  # noqa: E402

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


@contextlib.contextmanager
def _torch_load_weights_only_false():
    """Temporarily force torch.load(weights_only=False) only while loading the
    pyannote pipeline (some checkpoints are not loadable under the new default).

    Scoped via context manager instead of a global monkeypatch so the rest of the
    process keeps PyTorch's safe default behaviour.
    """
    original = torch.load

    def _patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = _patched
    try:
        yield
    finally:
        torch.load = original


from pyannote.audio import Pipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
warnings.filterwarnings("ignore", message=r".*std\(\): degrees of freedom is <= 0.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")


class SpeakerDiarizer:
    """Perform speaker diarization on audio files using local GPU/CPU"""

    def __init__(
        self,
        token: str,
        model_name: str = "pyannote/speaker-diarization-precision-2",
        min_segment_duration: float = 0.5,
        max_gap_threshold: float = 0.5,
        overlap_policy: str = "keep",
    ):
        self.token = token
        self.model_name = model_name
        self.pipeline = None
        self.min_segment_duration = min_segment_duration
        self.max_gap_threshold = max_gap_threshold
        # overlap_policy:
        #   "keep"     -> keep pyannote's native overlapping segments (recommended;
        #                 dropping overlap is a guaranteed missed-detection floor)
        #   "drop"     -> keep only single-speaker spans (legacy behaviour)
        #   "dominant" -> on overlap, keep the longest-active speaker for that span
        self.overlap_policy = overlap_policy

    def load_pipeline(self):
        if self.pipeline is None:
            print("Loading speaker diarization pipeline...")
            auth = pretrained_auth_kwargs(Pipeline.from_pretrained, self.token)
            with _torch_load_weights_only_false(), pyannote_hf_hub_compat():
                self.pipeline = Pipeline.from_pretrained(self.model_name, **auth)

            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                print("Pipeline loaded on GPU (CUDA).")
            else:
                print("Pipeline loaded on CPU.")

    def diarize(self, audio_path: str, output_dir: str) -> str:
        self.load_pipeline()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Processing: {audio_path}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.pipeline(audio_path)
        except Exception as e:
            print(f"Error during inference: {e}")
            raise

        raw_segments = []
        if "precision" in self.model_name.lower():
            # pyannote/speaker-diarization-precision-* (pyannoteAI cloud)
            # yields (turn, speaker) — no track label
            for turn, speaker in result.speaker_diarization:
                raw_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        else:
            # Standard pyannote Annotation (e.g. speaker-diarization-3.1)
            # yields (turn, track, speaker)
            for turn, _, speaker in result.itertracks(yield_label=True):
                raw_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

        print(f"Raw segments found: {len(raw_segments)}")

        if os.getenv("VIESPEAKER_DEBUG"):
            debug_path = os.path.join(output_dir, "debug_output.txt")
            os.makedirs(output_dir, exist_ok=True)
            with open(debug_path, "w") as f_debug:
                f_debug.write(str(raw_segments))

        if self.overlap_policy == "drop":
            cleaned_segments = self._remove_overlaps(raw_segments)
        elif self.overlap_policy == "dominant":
            cleaned_segments = self._resolve_overlaps_dominant(raw_segments)
        else:  # "keep" — preserve pyannote's native (possibly overlapping) segments
            cleaned_segments = sorted(raw_segments, key=lambda s: (s["start"], s["end"]))
        print(f"Segments after overlap_policy='{self.overlap_policy}': {len(cleaned_segments)}")
        merged_segments = self._merge_segments(cleaned_segments)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(audio_path).stem}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            for seg in merged_segments:
                f.write(f"{seg['start']:.3f} {seg['end']:.3f} {seg['speaker']}\n")

        print(f"Diarization saved to: {output_path}")
        return output_path

    def _remove_overlaps(self, segments: list) -> list:
        """Keep only time regions with exactly 1 active speaker (sweep-line)."""
        if not segments:
            return []

        events = []
        for seg in segments:
            events.append((seg["start"], "start", seg["speaker"]))
            events.append((seg["end"], "end", seg["speaker"]))

        # Process 'end' before 'start' at same timestamp to avoid phantom overlaps
        events.sort(key=lambda x: (x[0], 0 if x[1] == "end" else 1))

        active_speakers = set()
        last_time = 0.0
        final_segments = []

        for current_time, event_type, speaker in events:
            if len(active_speakers) == 1 and current_time > last_time:
                final_segments.append({
                    "start": last_time,
                    "end": current_time,
                    "speaker": list(active_speakers)[0],
                })

            if event_type == "start":
                active_speakers.add(speaker)
            else:
                active_speakers.discard(speaker)

            last_time = current_time

        return [s for s in final_segments if (s["end"] - s["start"]) > 0.1]

    def _resolve_overlaps_dominant(self, segments: list) -> list:
        """On overlapping spans keep a single speaker (the one whose original
        segment is longest = a confidence proxy), instead of deleting the span.

        Unlike `_remove_overlaps`, this never drops speech — every covered span
        is emitted with exactly one label, so overlap no longer inflates MD.
        """
        if not segments:
            return []

        events = []
        for idx, seg in enumerate(segments):
            dur = seg["end"] - seg["start"]
            events.append((seg["start"], "start", idx, seg["speaker"], dur))
            events.append((seg["end"], "end", idx, seg["speaker"], dur))
        events.sort(key=lambda x: (x[0], 0 if x[1] == "end" else 1))

        active = {}  # idx -> (speaker, duration)
        last_time = 0.0
        spans = []
        for t, etype, idx, spk, dur in events:
            if t > last_time and active:
                # pick the active segment with the longest original duration
                best_idx = max(active, key=lambda i: active[i][1])
                spans.append({"start": last_time, "end": t, "speaker": active[best_idx][0]})
            if etype == "start":
                active[idx] = (spk, dur)
            else:
                active.pop(idx, None)
            last_time = t

        # stitch adjacent same-speaker spans produced by the sweep
        stitched = []
        for s in spans:
            if stitched and stitched[-1]["speaker"] == s["speaker"] and abs(stitched[-1]["end"] - s["start"]) < 1e-6:
                stitched[-1]["end"] = s["end"]
            else:
                stitched.append(dict(s))
        return [s for s in stitched if (s["end"] - s["start"]) > 0.1]

    def _merge_segments(self, segments: list) -> list:
        if not segments:
            return []
        merged = []
        for seg in segments:
            if (
                merged
                and merged[-1]["speaker"] == seg["speaker"]
                and (seg["start"] - merged[-1]["end"]) <= self.max_gap_threshold
            ):
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg.copy())
        return [s for s in merged if (s["end"] - s["start"]) >= self.min_segment_duration]


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Pipeline 1: Speaker Diarization (pyannote)")
    parser.add_argument("--audio_path", required=True, help="Path to input WAV file")
    parser.add_argument("--output_dir", default="data/diarization", help="Output directory")
    parser.add_argument("--model", default="pyannote/speaker-diarization-precision-2",
                        help="Model ID. Use 'pyannote/speaker-diarization-precision-2' (default, cloud)"
                             " or 'pyannote/speaker-diarization-3.1' (local GPU/CPU).")
    parser.add_argument("--min_segment_duration", type=float, default=0.5)
    parser.add_argument("--max_gap_threshold", type=float, default=0.5)
    parser.add_argument("--overlap_policy", choices=["keep", "drop", "dominant"], default="keep",
                        help="How to handle overlapped-speech regions. 'keep' (default) preserves "
                             "pyannote's native overlapping segments; 'drop' deletes overlap spans "
                             "(legacy, inflates missed detection); 'dominant' keeps one speaker per span.")
    args = parser.parse_args()

    # Token selection: precision-* uses PYANNOTEAI_API_KEY (pyannoteAI cloud key),
    # all other models use HUGGINGFACE_ACCESS_TOKEN.
    if "precision" in args.model.lower():
        api_key = os.getenv("PYANNOTEAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing PYANNOTEAI_API_KEY in environment or .env file. "
                "Create an API key at https://dashboard.pyannote.ai"
            )
        print(f"[P1] Using pyannoteAI cloud model: {args.model}")
    else:
        api_key = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
        if not api_key:
            raise RuntimeError("Missing HUGGINGFACE_ACCESS_TOKEN in environment or .env file")

    diarizer = SpeakerDiarizer(
        token=api_key,
        model_name=args.model,
        min_segment_duration=args.min_segment_duration,
        max_gap_threshold=args.max_gap_threshold,
        overlap_policy=args.overlap_policy,
    )
    try:
        output_path = diarizer.diarize(args.audio_path, args.output_dir)
        print(f"Output: {output_path}")
    except Exception as e:
        print(f"Failed: {e}")
        raise


if __name__ == "__main__":
    main()
