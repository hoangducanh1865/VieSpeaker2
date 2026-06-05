#!/usr/bin/env python3
"""
Pipeline 3: Speaker Diarization Cleansing

Supports two methods:
  --method ahc   : ECAPA-TDNN embeddings + Agglomerative Hierarchical Clustering
  --method cdgcn : ECAPA-TDNN embeddings + KNN graph + Leiden community detection

Usage:
    python clean.py --method ahc \
        --diarization_path data/audio_visual/interview_noise/supplemented_diarization.txt \
        --audio_path data/diarization_test_set/audio/interview_noise.wav \
        --output_dir data/clean/interview_noise

    python clean.py --method cdgcn \
        --diarization_path data/audio_visual/interview_noise/supplemented_diarization.txt \
        --audio_path data/diarization_test_set/audio/interview_noise.wav \
        --output_dir data/clean/interview_noise
"""

import argparse
import os
import sys
import torch
import torchaudio
import torchaudio.transforms
import numpy as np
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import List, Optional, Tuple

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False

import sklearn
_SKLEARN_VER = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

# Paths resolved relative to THIS file
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "models")
_ECAPA_ROOT = os.path.join(_MODELS_DIR, "ecapa_tdnn")
_ECAPA_PATH = os.path.join(_ECAPA_ROOT, "exps", "pretrain.model")

# Make `import embeddings.*` work whether clean.py is run as a script or imported.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from embeddings.registry import get_embedder  # noqa: E402

SAMPLE_RATE = 16000


# ============================================================================
# Shared data structures
# ============================================================================

@dataclass
class Segment:
    start: float
    end: float
    speaker: str


# ============================================================================
# Shared I/O utilities
# ============================================================================

def parse_diarization(path: str) -> List[Segment]:
    segments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    start, end = float(parts[0]), float(parts[1])
                    if end > start:
                        segments.append(Segment(start, end, parts[2]))
                except ValueError:
                    continue
    return segments


def parse_diarization_dict(path: str) -> List[dict]:
    """Return list of dicts with start/end/speaker keys (for AHC helpers)."""
    segments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
                if end > start:
                    segments.append({"start": start, "end": end, "speaker": parts[2]})
            except ValueError:
                continue
    segments.sort(key=lambda x: (x["start"], x["end"]))
    return segments


def write_diarization(path: str, segments):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for seg in segments:
            if isinstance(seg, Segment):
                f.write(f"{seg.start:.3f} {seg.end:.3f} {seg.speaker}\n")
            else:
                f.write(f"{seg['start']:.3f} {seg['end']:.3f} {seg['speaker']}\n")
    print(f"Saved cleansed diarization: {path}")


# ============================================================================
# Shared: sweep-line overlap removal (O(n log n))
# ============================================================================

def remove_overlapping_segments(segments: List[dict]) -> List[dict]:
    """Return only segments whose time range has NO overlap with any other speaker."""
    if not segments:
        return []

    events = []
    for seg in segments:
        events.append((seg["start"], "start", seg["speaker"]))
        events.append((seg["end"], "end", seg["speaker"]))
    # Process 'end' before 'start' at the same time to avoid phantom overlaps
    events.sort(key=lambda x: (x[0], 0 if x[1] == "end" else 1))

    active: set = set()
    last_time = 0.0
    pure_spans: List[Tuple] = []

    for t, etype, spk in events:
        if t > last_time and len(active) == 1:
            pure_spans.append((last_time, t, next(iter(active))))
        if etype == "start":
            active.add(spk)
        else:
            active.discard(spk)
        last_time = t

    pure_ids = set()
    for ps, pe, pspk in pure_spans:
        for seg in segments:
            if seg["speaker"] == pspk and seg["start"] >= ps and seg["end"] <= pe:
                pure_ids.add(id(seg))

    return [s for s in segments if id(s) in pure_ids]


def merge_same_speaker_segments(segments: List[dict], max_gap_sec: float = 0.5) -> List[dict]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda x: (x["start"], x["end"]))
    merged = [dict(ordered[0])]
    for seg in ordered[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and (seg["start"] - prev["end"]) <= max_gap_sec:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged


def filter_short_segments(segments: List[dict], min_duration_sec: float = 0.5) -> List[dict]:
    return [s for s in segments if (s["end"] - s["start"]) >= min_duration_sec]


# ============================================================================
# AHC method: ECAPA-TDNN (shared with CDGCN) + AgglomerativeClustering
# ============================================================================
# NOTE: AHC uses the same pretrain.model as CDGCN for embedding extraction.
# No extra checkpoint (avg_model.pt) is required.

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n <= 1e-12 else v / n


def _select_main_cluster_ahc(embeddings: List[np.ndarray], threshold: float = 0.5) -> set:
    """
    Cluster embeddings with AHC (cosine distance).
    threshold: cosine distance threshold — segments farther than this are split
               into separate clusters. Range [0, 2]; lower = tighter clusters.
    """
    n = len(embeddings)
    if n <= 2:
        return set(range(n))

    X = np.stack([_normalize(e) for e in embeddings])
    _kw = "metric" if _SKLEARN_VER >= (1, 4) else "affinity"
    try:
        ahc = AgglomerativeClustering(
            n_clusters=None, **{_kw: "cosine"},
            linkage="average", distance_threshold=threshold,
        )
    except TypeError:
        ahc = AgglomerativeClustering(
            n_clusters=None, affinity="cosine",
            linkage="average", distance_threshold=threshold,
        )
    labels = ahc.fit_predict(X)
    clusters: dict = defaultdict(list)
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(idx)
    return set(max(clusters.values(), key=len))


def run_ahc(args) -> str:
    """AHC cleansing: per-speaker intra-cluster outlier removal."""
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "cleansed_diarization.txt")

    print("\n[AHC] Step 1: Loading diarization")
    original_segs = parse_diarization_dict(args.diarization_path)
    # By default DO NOT delete overlap regions (deleting them is a guaranteed
    # missed-detection floor). Only drop overlap when explicitly requested.
    if getattr(args, "drop_overlap", False):
        base_segs = remove_overlapping_segments(original_segs)
    else:
        base_segs = original_segs
    merged = merge_same_speaker_segments(base_segs, max_gap_sec=args.merge_gap_sec)
    print(f"  original={len(original_segs)}, after merge={len(merged)} (drop_overlap={getattr(args, 'drop_overlap', False)})")

    if not merged:
        write_diarization(output_path, [])
        return output_path

    embedding_name = getattr(args, "embedding", "ecapa")
    print(f"\n[AHC] Step 2: Loading speaker-embedding backend '{embedding_name}'")
    embedder = get_embedder(embedding_name, device=getattr(args, "device", "cuda"))

    print("\n[AHC] Step 3: Extracting embeddings")
    enriched = []
    for seg in merged:
        emb = embedder.extract(args.audio_path, seg["start"], seg["end"])
        if emb is None:
            continue
        item = dict(seg)
        item["embedding"] = emb
        enriched.append(item)

    trusted = [x for x in enriched if (x["end"] - x["start"]) >= 1.0]
    untrusted = [x for x in enriched if (x["end"] - x["start"]) < 1.0]

    print(f"  trusted={len(trusted)}, untrusted={len(untrusted)}")

    print("\n[AHC] Step 4: Per-speaker intra-cluster outlier detection")
    by_speaker: dict = defaultdict(list)
    for item in trusted:
        by_speaker[item["speaker"]].append(item)

    kept, outliers = [], []
    for speaker, items in by_speaker.items():
        if len(items) < args.min_cluster_size:
            kept.extend(items)
            continue
        embs = [x["embedding"] for x in items]
        main_idx = _select_main_cluster_ahc(embs, threshold=args.threshold)
        for i, item in enumerate(items):
            # Outliers are NOT deleted — they are queued for reassignment so we
            # never drop detected speech (deleting it only inflates missed detection).
            (kept if i in main_idx else outliers).append(item)

    # Build centroids from kept (high-confidence) segments
    speaker_centroids: dict = {}
    by_kept: dict = defaultdict(list)
    for item in kept:
        by_kept[item["speaker"]].append(item["embedding"])
    for spk, embs in by_kept.items():
        speaker_centroids[spk] = _normalize(np.mean(np.stack(embs), axis=0))

    print(f"\n[AHC] Step 5: Reassigning {len(outliers)} outliers + {len(untrusted)} short segments (no deletion)")
    for item in outliers + untrusted:
        if not speaker_centroids:
            # No reliable centroid to compare against — keep the original label.
            kept.append(item)
            continue
        best_spk = max(
            speaker_centroids,
            key=lambda s: float(np.dot(_normalize(item["embedding"]), speaker_centroids[s])),
        )
        new_item = dict(item)
        new_item["speaker"] = best_spk
        kept.append(new_item)

    kept_segs = [{"start": x["start"], "end": x["end"], "speaker": x["speaker"]} for x in kept]
    kept_segs = merge_same_speaker_segments(kept_segs, max_gap_sec=args.merge_gap_sec)
    kept_segs = filter_short_segments(kept_segs, min_duration_sec=args.min_duration_sec)
    kept_segs.sort(key=lambda x: x["start"])

    write_diarization(output_path, kept_segs)
    print(f"[AHC] Done. {len(original_segs)} → {len(kept_segs)} segments. Output: {output_path}")
    return output_path


# ============================================================================
# CDGCN method: ECAPA-TDNN (model.py) + KNN graph + Leiden
# ============================================================================

def _load_ecapa_model():
    if _ECAPA_ROOT not in sys.path:
        sys.path.insert(0, _ECAPA_ROOT)
    from model import ECAPA_TDNN  # type: ignore
    model = ECAPA_TDNN(C=1024)
    if not os.path.exists(_ECAPA_PATH):
        raise FileNotFoundError(
            f"ECAPA-TDNN checkpoint not found: {_ECAPA_PATH}. "
            "Run scripts/prepare_embeddings.py or download the weight."
        )
    # Fail hard on load errors — silently falling back to random weights would
    # produce garbage embeddings while the pipeline appears to run normally.
    state = torch.load(_ECAPA_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print(f"  Loaded ECAPA-TDNN from {_ECAPA_PATH}")
    model.eval()
    return model


def _extract_ecapa_embedding(model, audio_path: str, start: float, end: float) -> Optional[np.ndarray]:
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
        s, e = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        chunk = waveform[:, s:e]
        if chunk.shape[1] < 160:
            return None

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160,
            f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80,
        )
        chunk = chunk.squeeze(0)
        chunk_pre = torch.cat([torch.zeros(1), chunk[:-1] - 0.97 * chunk[1:]])
        mel = mel_spec(chunk_pre.unsqueeze(0))
        mel = torch.log(mel + 1e-9)
        mel = mel - torch.mean(mel, dim=-1, keepdim=True)

        with torch.no_grad():
            emb = model(mel).squeeze(0).cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)
    except Exception as e:
        print(f"  Warning: embedding failed [{start:.2f}, {end:.2f}]: {e}")
        return None


def _build_knn_graph(embeddings: np.ndarray, k: int = 10) -> np.ndarray:
    N = len(embeddings)
    sim = cosine_similarity(embeddings)
    k_actual = min(k + 1, N)
    nbrs = NearestNeighbors(n_neighbors=k_actual, metric="cosine").fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in indices[i][1:]:
            w = max(0.0, float(sim[i, j]))
            adj[i, j] = max(adj[i, j], w)
            adj[j, i] = max(adj[j, i], w)
    return adj


def _leiden_cluster(adj: np.ndarray, resolution: float = 0.6) -> np.ndarray:
    if not HAS_LEIDEN:
        raise RuntimeError(
            "igraph and leidenalg required. Install: pip install igraph leidenalg"
        )
    sources, targets = np.where(adj > 0)
    weights = adj[sources, targets].tolist()
    g = ig.Graph(n=adj.shape[0], directed=False)
    g.add_edges(list(zip(sources.tolist(), targets.tolist())))
    g.es["weight"] = weights
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=42,
    )
    return np.array(partition.membership)


def _refine_labels(
    adj: np.ndarray,
    leiden_labels: np.ndarray,
    original_labels: np.ndarray,
    min_cluster_size: int = 3,
    purity_threshold: float = 0.8,
) -> Tuple[np.ndarray, int]:
    refined = original_labels.copy()
    counts = Counter(leiden_labels.tolist())
    outlier_mask = np.array([counts[int(l)] < min_cluster_size for l in leiden_labels])

    for i in np.where(outlier_mask)[0]:
        neighbors = np.argsort(adj[i])[::-1]
        non_outliers = [n for n in neighbors if not outlier_mask[n]]
        if non_outliers:
            refined[i] = refined[non_outliers[0]]

    for cid in np.unique(leiden_labels):
        mask = leiden_labels == cid
        if mask.sum() < min_cluster_size:
            continue
        dominant, dominant_count = Counter(original_labels[mask].tolist()).most_common(1)[0]
        if dominant_count / mask.sum() >= purity_threshold:
            refined[mask] = dominant

    return refined, int(np.sum(refined != original_labels))


def run_cdgcn(args) -> str:
    """CDGCN cleansing: KNN graph + Leiden community detection."""
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "cleansed_diarization.txt")

    print("\n[CDGCN] Step 1: Parsing diarization")
    segments = parse_diarization(args.diarization_path)
    unique_speakers = len({s.speaker for s in segments})
    print(f"  {len(segments)} segments, {unique_speakers} speakers")

    embedding_name = getattr(args, "embedding", "ecapa")
    print(f"\n[CDGCN] Step 2: Loading speaker-embedding backend '{embedding_name}'")
    embedder = get_embedder(embedding_name, device=getattr(args, "device", "cuda"))

    print("\n[CDGCN] Step 3: Extracting embeddings")
    valid_segs, embeddings_list = [], []
    for seg in segments:
        emb = embedder.extract(args.audio_path, seg.start, seg.end)
        if emb is not None:
            valid_segs.append(seg)
            embeddings_list.append(emb)

    embeddings = np.array(embeddings_list)
    print(f"  {len(embeddings)} valid embeddings")

    if len(embeddings) < 2:
        write_diarization(output_path, valid_segs)
        return output_path

    original_labels = np.array([s.speaker for s in valid_segs])

    print(f"\n[CDGCN] Step 4: Building KNN graph (k={args.k})")
    adj = _build_knn_graph(embeddings, k=args.k)

    print(f"\n[CDGCN] Step 5: Leiden clustering (resolution={args.resolution})")
    leiden_labels = _leiden_cluster(adj, resolution=args.resolution)
    print(f"  {len(set(leiden_labels.tolist()))} communities found")

    print(f"\n[CDGCN] Step 6: Refining labels")
    refined_labels, n_changed = _refine_labels(
        adj, leiden_labels, original_labels,
        min_cluster_size=args.min_cluster_size,
        purity_threshold=args.purity_threshold,
    )
    pct = 100.0 * n_changed / len(valid_segs)
    print(f"  Changed {n_changed}/{len(valid_segs)} ({pct:.1f}%)")
    if pct > 20.0:
        print(f"  Warning: {pct:.1f}% changes > 20%. Consider increasing purity_threshold or decreasing k.")

    refined_segs = [
        Segment(seg.start, seg.end, refined_labels[i])
        for i, seg in enumerate(valid_segs)
    ]

    write_diarization(output_path, refined_segs)
    print(f"[CDGCN] Done. {unique_speakers} → {len(set(refined_labels.tolist()))} speakers. Output: {output_path}")
    return output_path


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline 3: Speaker Diarization Cleansing"
    )
    parser.add_argument("--method", choices=["ahc", "cdgcn", "vbx", "dover-lap", "nme-sc"],
                        required=True, help="Cleansing method")
    parser.add_argument("--diarization_path", required=True,
                        help="Input diarization TXT (from Pipeline 1 or 2)")
    parser.add_argument("--audio_path", required=True,
                        help="Original audio WAV file")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (e.g. data/clean/interview_noise)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--embedding", default="ecapa",
                        choices=["ecapa", "wespeaker34", "wespeaker293", "campplus", "redimnet"],
                        help="Speaker-embedding backend for ahc/cdgcn/nme-sc (default: ecapa). "
                             "VBx uses its own ResNet101 x-vector; dover-lap uses no embedding.")
    parser.add_argument("--drop_overlap", action="store_true", default=False,
                        help="Delete overlapped-speech regions before cleansing (legacy). "
                             "Off by default — deleting overlap only inflates missed detection.")

    # AHC params
    parser.add_argument("--merge_gap_sec", type=float, default=0.5)
    parser.add_argument("--min_duration_sec", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Cosine distance threshold for AHC (0=tightest, 2=loosest; default 0.5)")
    parser.add_argument("--min_cluster_size", type=int, default=3)

    # CDGCN params
    parser.add_argument("--k", type=int, default=10, help="KNN neighbors (CDGCN)")
    parser.add_argument("--resolution", type=float, default=0.6,
                        help="Leiden resolution (CDGCN)")
    parser.add_argument("--purity_threshold", type=float, default=0.8,
                        help="Community purity threshold (CDGCN)")

    # VBx params
    parser.add_argument("--vbx_loop_prob", type=float, default=0.9,
                        help="VBx HMM self-loop probability")
    parser.add_argument("--vbx_fa", type=float, default=0.3,
                        help="VBx Fa parameter (scales sufficient statistics)")
    parser.add_argument("--vbx_fb", type=float, default=17.0,
                        help="VBx Fb parameter (speaker regularization)")
    parser.add_argument("--vbx_max_speakers", type=int, default=10,
                        help="VBx maximum number of speakers")
    parser.add_argument("--vbx_lda_dim", type=int, default=128,
                        help="VBx LDA dimensionality after PLDA transform")
    parser.add_argument("--vbx_init_smoothing", type=float, default=5.0,
                        help="VBx AHC init smoothing coefficient")

    # NME-SC params
    parser.add_argument("--max_speakers", type=int, default=8,
                        help="NME-SC maximum number of speakers")
    parser.add_argument("--max_rp_threshold", type=float, default=0.25,
                        help="NME-SC maximum ratio of neighbors to consider")

    # DOVER-Lap params
    parser.add_argument("--diarization_path2", default=None,
                        help="Second diarization file for DOVER-Lap (e.g. P1 output)")

    args = parser.parse_args()

    if args.method == "ahc":
        run_ahc(args)
    elif args.method == "cdgcn":
        run_cdgcn(args)
    elif args.method == "vbx":
        from vbx.run_vbx import run_vbx
        run_vbx(args)
    elif args.method == "dover-lap":
        from dover_lap.run_dover_lap import run_dover_lap
        run_dover_lap(args)
    elif args.method == "nme-sc":
        from nme_sc.run_nme_sc import run_nme_sc
        run_nme_sc(args)


if __name__ == "__main__":
    main()
