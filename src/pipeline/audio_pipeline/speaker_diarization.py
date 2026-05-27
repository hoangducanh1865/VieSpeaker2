"""
Speaker diarization module using pyannote.audio (Pipeline 1)
"""

import os
import argparse
import torch
from pathlib import Path
import warnings


def _load_dotenv():
    """Simple .env loader — avoids dependency on python-dotenv."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env")
    env_path = os.path.normpath(env_path)
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

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
    ):
        self.token = token
        self.model_name = model_name
        self.pipeline = None
        self.min_segment_duration = min_segment_duration
        self.max_gap_threshold = max_gap_threshold

    def _from_pretrained(self, **extra_kwargs):
        """Try token= first (new API), fall back to use_auth_token= (old API)."""
        try:
            return Pipeline.from_pretrained(self.model_name, token=self.token, **extra_kwargs)
        except TypeError:
            return Pipeline.from_pretrained(self.model_name, use_auth_token=self.token, **extra_kwargs)

    def load_pipeline(self):
        if self.pipeline is None:
            # Try offline cache first — skips the internet HEAD check entirely.
            # Falls back to online download only when cache is absent.
            try:
                print("Loading speaker diarization pipeline (offline cache)...")
                self.pipeline = self._from_pretrained(local_files_only=True)
            except Exception:
                print("Cache miss — downloading pipeline from HuggingFace...")
                self.pipeline = self._from_pretrained()

            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                print("Pipeline loaded on GPU (CUDA).")
            else:
                print("Pipeline loaded on CPU.")

    def diarize(self, audio_path: str, output_dir: str) -> str:
        # Result cache: skip cloud / local inference if output already exists.
        # Useful when re-running the orchestrator without wanting to redo P1.
        output_path = os.path.join(output_dir, f"{Path(audio_path).stem}.txt")
        if os.path.exists(output_path):
            print(f"[Cache] Output already exists, skipping inference: {output_path}")
            return output_path

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

        cleaned_segments = self._remove_overlaps(raw_segments)
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
    args = parser.parse_args()

    # Token selection: precision-* uses PYANNOTE_API_KEY (pyannoteAI cloud key),
    # all other models use HUGGINGFACE_ACCESS_TOKEN.
    if "precision" in args.model.lower():
        api_key = os.getenv("PYANNOTE_API_KEY") or os.getenv("HUGGINGFACE_ACCESS_TOKEN")
        if not api_key:
            raise RuntimeError(
                "Missing PYANNOTE_API_KEY in environment or .env file. "
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
    )
    try:
        output_path = diarizer.diarize(args.audio_path, args.output_dir)
        print(f"Output: {output_path}")
    except Exception as e:
        print(f"Failed: {e}")
        raise


if __name__ == "__main__":
    main()
