"""
NME-SC: Auto-tuning Spectral Clustering via Normalized Maximum Eigengap.
Uses ECAPA-TDNN embeddings (reused from AHC/CDGCN) + cosine similarity + NME analysis.
"""

import os
import sys
import shutil
import numpy as np

_NME_SC_DIR = os.path.dirname(os.path.abspath(__file__))
_CLEAN_PIPELINE_DIR = os.path.dirname(_NME_SC_DIR)

# Inject dirs before any local imports
for _p in [_NME_SC_DIR, _CLEAN_PIPELINE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from spectral_opt import GraphSpectralClusteringClass, spectral_clustering, get_X_conn_from_dist

from clean import (
    parse_diarization,
    write_diarization,
    merge_same_speaker_segments,
    _load_ecapa_model,
    _extract_ecapa_embedding,
)


def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix scaled to [0, 1]."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    X_norm = X / norms
    sim = X_norm @ X_norm.T
    return (sim + 1.0) / 2.0  # scale [-1, 1] → [0, 1]


def run_nme_sc(args) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "cleansed_diarization.txt")

    print("\n[NME-SC] Step 1: Loading diarization")
    segments = parse_diarization(args.diarization_path)
    if not segments:
        write_diarization(output_path, [])
        return output_path

    print("\n[NME-SC] Step 2: Loading ECAPA-TDNN model")
    model = _load_ecapa_model()

    print("\n[NME-SC] Step 3: Extracting embeddings")
    embeddings, valid_segs = [], []
    for seg in segments:
        emb = _extract_ecapa_embedding(model, args.audio_path, seg.start, seg.end)
        if emb is not None:
            embeddings.append(emb)
            valid_segs.append(seg)

    print(f"  Extracted {len(embeddings)} embeddings from {len(segments)} segments")

    if len(embeddings) < 2:
        shutil.copy(args.diarization_path, output_path)
        print("[NME-SC] Too few segments, copying input to output")
        return output_path

    X = np.stack(embeddings)  # (N, 192)
    sim = _cosine_sim_matrix(X)  # (N, N) in [0, 1]

    print("\n[NME-SC] Step 4: NME analysis (auto-tuning threshold and speaker count)")
    # NMEanalysis doesn't use self — call as unbound with None
    X_conn, rp_threshold, est_k, lambdas, rp_p_neighbors = GraphSpectralClusteringClass.NMEanalysis(
        None, sim,
        SPK_MAX=args.max_speakers,
        max_rp_threshold=args.max_rp_threshold,
    )
    print(f"  Estimated speakers: {est_k}, rp_threshold: {rp_threshold:.4f}")

    if est_k < 1:
        est_k = 1

    print("\n[NME-SC] Step 5: Spectral clustering")
    if est_k == 1:
        labels = np.zeros(len(valid_segs), dtype=int)
    else:
        labels = spectral_clustering(X_conn, n_clusters=est_k)

    print("\n[NME-SC] Step 6: Writing output")
    out_segs = [
        {"start": s.start, "end": s.end, "speaker": f"SPEAKER_{l:02d}"}
        for s, l in zip(valid_segs, labels)
    ]
    out_segs.sort(key=lambda x: x["start"])
    out_segs = merge_same_speaker_segments(out_segs)

    write_diarization(output_path, out_segs)
    print(f"[NME-SC] Done. {len(segments)} → {len(out_segs)} segments. Output: {output_path}")
    return output_path
