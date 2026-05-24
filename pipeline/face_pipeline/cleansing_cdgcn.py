#!/usr/bin/env python3
"""
Speaker Diarization Cleansing Pipeline

Refines speaker diarization output using:
1. ECAPA-TDNN embeddings
2. KNN graph construction
3. Leiden community detection
4. Outlier removal and speaker re-assignment
"""

import argparse
import os
import sys
import torch
import torchaudio
import torchaudio.transforms
import numpy as np
from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple

# Try to import optional dependencies
try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# ============================================================================
# Configuration and Paths
# ============================================================================


ECAPA_PATH = "reference/sv/ECAPA-TDNN/exps/pretrain.model"
ECAPA_ROOT = "reference/sv/ECAPA-TDNN"
SAMPLE_RATE = 16000

DEFAULT_DIARIZATION_INPUT = "pipeline/face_pipeline/result/result_audio_visual/interview_noise/supplemented_diarization.txt"
DEFAULT_AUDIO_PATH = "data/diarization_test_set/audio/interview_noise.wav"
DEFAULT_OUTPUT_DIR = "pipeline/face_pipeline/result/result_cleansing/interview_noise"
DEFAULT_OUTPUT_FILE = "diarization_cleansed.txt"


# ============================================================================
# Data Classes and Parsers
# ============================================================================

@dataclass
class Segment:
    """Represents a speaker diarization segment"""
    start: float
    end: float
    speaker: str


def parse_diarization(path: str) -> List[Segment]:
    """Parse diarization file into Segment objects
    
    Format: "<start> <end> <SPEAKER_ID>" per line
    """
    segments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                segments.append(Segment(float(parts[0]), float(parts[1]), parts[2]))
    return segments


def save_segments(segments: List[Segment], path: str):
    """Save refined segments to file
    
    Format: "<start> <end> <SPEAKER_ID>" per line
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for seg in segments:
            f.write(f"{seg.start:.3f} {seg.end:.3f} {seg.speaker}\n")
    print(f"Saved refined diarization → {path}")


# ============================================================================
# ECAPA-TDNN Model Loading and Embedding Extraction
# ============================================================================

def load_ecapa_model():
    """Load ECAPA-TDNN model from pretrained checkpoint
    
    Returns:
        ECAPA_TDNN model in eval mode
    """
    sys.path.insert(0, ECAPA_ROOT)
    from model import ECAPA_TDNN
    
    model = ECAPA_TDNN(C=1024)
    
    # Load pretrained weights
    try:
        state = torch.load(ECAPA_PATH, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"✓ Loaded ECAPA-TDNN from {ECAPA_PATH}")
    except Exception as e:
        print(f"✗ Failed to load ECAPA checkpoint: {e}")
        print("  Using random initialization for testing")
    
    model.eval()
    return model


def extract_embedding(model, audio_path: str, start: float, end: float) -> np.ndarray:
    """Extract ECAPA-TDNN embedding for an audio segment
    
    Args:
        model: ECAPA_TDNN model
        audio_path: Path to audio file
        start: Start time in seconds
        end: End time in seconds
        
    Returns:
        192-dimensional embedding vector (L2-normalized) or None if segment too short
    """
    try:
        # Load audio and resample to 16kHz
        waveform, sr = torchaudio.load(audio_path)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Take first channel if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
        
        # Extract segment
        start_sample = int(start * SAMPLE_RATE)
        end_sample = int(end * SAMPLE_RATE)
        chunk = waveform[:, start_sample:end_sample]
        
        # Skip segments too short (< 160 samples = 10ms)
        if chunk.shape[1] < 160:
            return None
        
        # Convert to mel-spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            window_fn=torch.hamming_window,
            n_mels=80
        )
        
        # Apply pre-emphasis
        chunk = chunk.squeeze(0)
        chunk_pre = torch.cat([torch.zeros(1), chunk[:-1] - 0.97 * chunk[1:]], dim=0)
        
        # Get mel-spectrogram
        mel_feat = mel_spec(chunk_pre.unsqueeze(0))
        mel_feat = torch.log(mel_feat + 1e-9)
        mel_feat = mel_feat - torch.mean(mel_feat, dim=-1, keepdim=True)
        
        # Extract embedding
        with torch.no_grad():
            embedding = model(mel_feat)  # Output shape: (1, 192)
        
        # L2 normalize
        embedding = embedding.squeeze(0).cpu().numpy()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
        
    except Exception as e:
        print(f"  Warning: Failed to extract embedding for segment [{start:.2f}, {end:.2f}]: {e}")
        return None


# ============================================================================
# KNN Graph Construction
# ============================================================================

def build_knn_graph(embeddings: np.ndarray, k: int = 10) -> np.ndarray:
    """Build KNN adjacency matrix from embeddings
    
    Args:
        embeddings: Shape (N, embedding_dim)
        k: Number of nearest neighbors
        
    Returns:
        Adjacency matrix (N, N) with cosine similarity weights
    """
    N = len(embeddings)
    
    # Compute cosine similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Find k-nearest neighbors (including self)
    k_actual = min(k + 1, N)
    nbrs = NearestNeighbors(n_neighbors=k_actual, metric='cosine').fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    
    # Build adjacency matrix
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in indices[i][1:]:  # Skip self-loop (first neighbor is self)
            w = max(0, sim_matrix[i, j])  # Use cosine similarity as weight
            adj[i, j] = max(adj[i, j], w)
            adj[j, i] = max(adj[j, i], w)
    
    return adj


# ============================================================================
# Leiden Community Detection
# ============================================================================

def leiden_cluster(adj: np.ndarray, resolution: float = 0.6) -> np.ndarray:
    """Apply Leiden community detection on adjacency matrix
    
    Args:
        adj: Adjacency matrix (N, N)
        resolution: Resolution parameter for Leiden algorithm
        
    Returns:
        Community assignments (N,) array where label[i] is community ID
    """
    if not HAS_LEIDEN:
        raise RuntimeError("igraph and leidenalg required for Leiden clustering. "
                          "Install: pip install igraph leidenalg")
    
    # Get edge list and weights
    sources, targets = np.where(adj > 0)
    weights = adj[sources, targets].tolist()
    
    # Create igraph
    g = ig.Graph(n=adj.shape[0], directed=False)
    g.add_edges(list(zip(sources.tolist(), targets.tolist())))
    g.es['weight'] = weights
    
    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=42
    )
    
    return np.array(partition.membership)


# ============================================================================
# Refinement Logic
# ============================================================================

def refine_labels(
    adj: np.ndarray,
    leiden_labels: np.ndarray,
    original_labels: np.ndarray,
    min_cluster_size: int = 3,
    purity_threshold: float = 0.8
) -> Tuple[np.ndarray, int]:
    """Refine speaker labels using Leiden communities
    
    Two-stage refinement:
    1. Outlier removal: Segments in tiny communities → reassigned to nearest neighbor
    2. Community remapping: High-purity communities → map all to dominant speaker
    
    Args:
        adj: Adjacency matrix (N, N)
        leiden_labels: Community assignments from Leiden
        original_labels: Original speaker labels
        min_cluster_size: Minimum community size threshold
        purity_threshold: Purity threshold for community remapping
        
    Returns:
        Refined labels array, number of changed segments
    """
    refined = original_labels.copy()
    N = len(original_labels)
    
    # Count community sizes
    counts = Counter(leiden_labels)
    
    # Stage 1: Outlier removal (communities smaller than min_cluster_size)
    outlier_mask = np.array([counts[l] < min_cluster_size for l in leiden_labels])
    
    for i in np.where(outlier_mask)[0]:
        # Find nearest non-outlier neighbor
        neighbors = np.argsort(adj[i])[::-1]
        non_outlier_neighbors = [n for n in neighbors if not outlier_mask[n]]
        
        if non_outlier_neighbors:
            # Reassign to nearest non-outlier
            nearest_neighbor = non_outlier_neighbors[0]
            refined[i] = refined[nearest_neighbor]
    
    # Stage 2: Community remapping for high-purity communities
    unique_communities = np.unique(leiden_labels)
    
    for cid in unique_communities:
        mask = leiden_labels == cid
        community_size = mask.sum()
        
        # Skip tiny communities (already handled in Stage 1)
        if community_size < min_cluster_size:
            continue
        
        # Get original speaker distribution in this community
        community_speakers = original_labels[mask]
        dominant_speaker, dominant_count = Counter(community_speakers).most_common(1)[0]
        purity = dominant_count / community_size
        
        # If community is pure enough, trust Leiden and remap
        if purity >= purity_threshold:
            refined[mask] = dominant_speaker
    
    # Count changes
    num_changed = np.sum(refined != original_labels)
    
    return refined, num_changed


# ============================================================================
# Main Pipeline
# ============================================================================

def run_cleansing(
    diarization_path: str = DEFAULT_DIARIZATION_INPUT,
    audio_path: str = DEFAULT_AUDIO_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_file: str = DEFAULT_OUTPUT_FILE,
    k: int = 10,
    resolution: float = 0.6,
    min_cluster_size: int = 3,
    purity_threshold: float = 0.8,
):
    """Run complete diarization cleansing pipeline
    
    Args:
        diarization_path: Input diarization file
        audio_path: Audio file path
        output_dir: Output directory
        output_file: Output filename
        k: KNN neighbors
        resolution: Leiden resolution parameter
        min_cluster_size: Minimum community size for non-outliers
        purity_threshold: Purity threshold for speaker remapping
    """
    
    print("\n" + "=" * 70)
    print("Speaker Diarization Cleansing Pipeline")
    print("=" * 70)
    
    # Step 0: Verify input files
    print(f"\n[Step 0] Checking input files...")
    if not os.path.exists(diarization_path):
        raise FileNotFoundError(f"Diarization file not found: {diarization_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    print(f"✓ Diarization: {diarization_path}")
    print(f"✓ Audio: {audio_path}")
    
    # Step 1: Parse input diarization
    print(f"\n[Step 1] Parsing diarization file...")
    segments = parse_diarization(diarization_path)
    unique_speakers = len(set(s.speaker for s in segments))
    print(f"✓ Loaded {len(segments)} segments from {unique_speakers} speakers")
    
    # Step 2: Extract ECAPA-TDNN embeddings
    print(f"\n[Step 2] Loading ECAPA-TDNN model...")
    ecapa_model = load_ecapa_model()
    
    print(f"\n[Step 3] Extracting embeddings for {len(segments)} segments...")
    valid_segments = []
    embeddings = []
    
    for idx, seg in enumerate(segments):
        if (idx + 1) % max(1, len(segments) // 10) == 0:
            print(f"  {idx + 1}/{len(segments)} segments processed")
        
        emb = extract_embedding(ecapa_model, audio_path, seg.start, seg.end)
        if emb is not None:
            valid_segments.append(seg)
            embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    print(f"✓ Successfully extracted {len(embeddings)} embeddings")
    
    if len(embeddings) < 2:
        print("✗ Not enough valid segments for clustering. Skipping refinement.")
        output_path = os.path.join(output_dir, output_file)
        save_segments(valid_segments, output_path)
        return
    
    original_labels = np.array([s.speaker for s in valid_segments])
    
    # Step 4: Build KNN graph
    print(f"\n[Step 4] Building KNN graph (k={k})...")
    adj = build_knn_graph(embeddings, k=k)
    print(f"✓ KNN graph built: {adj.shape[0]} nodes, {np.count_nonzero(adj) / 2:.0f} edges")
    
    # Step 5: Leiden community detection
    print(f"\n[Step 5] Leiden community detection (resolution={resolution})...")
    leiden_labels = leiden_cluster(adj, resolution=resolution)
    num_communities = len(set(leiden_labels))
    print(f"✓ Found {num_communities} communities")
    
    # Step 6: Refine labels
    print(f"\n[Step 6] Refinement (min_cluster_size={min_cluster_size}, purity={purity_threshold})...")
    refined_labels, num_changed = refine_labels(
        adj, leiden_labels, original_labels,
        min_cluster_size=min_cluster_size,
        purity_threshold=purity_threshold
    )
    
    pct_changed = 100.0 * num_changed / len(valid_segments)
    print(f"✓ Changed {num_changed}/{len(valid_segments)} segments ({pct_changed:.1f}%)")
    
    # Adaptive threshold adjustment if too many changes
    if pct_changed > 20.0:
        print(f"\n  ⚠ Warning: {pct_changed:.1f}% changes exceeds 20% threshold")
        print(f"    Consider increasing purity_threshold or decreasing k")
    
    # Step 7: Save output
    print(f"\n[Step 7] Saving refined diarization...")
    refined_segments = [
        Segment(seg.start, seg.end, refined_labels[i])
        for i, seg in enumerate(valid_segments)
    ]
    
    output_path = os.path.join(output_dir, output_file)
    save_segments(refined_segments, output_path)
    
    # Print summary
    new_speakers = len(set(refined_labels))
    print(f"\nSummary:")
    print(f"  Input speakers: {unique_speakers}")
    print(f"  Output speakers: {new_speakers}")
    print(f"  Segments changed: {pct_changed:.1f}%")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface with argparse"""
    parser = argparse.ArgumentParser(
        description="Speaker Diarization Cleansing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--diarization_path",
        type=str,
        default=DEFAULT_DIARIZATION_INPUT,
        help="Path to input diarization file"
    )
    
    parser.add_argument(
        "--audio_path",
        type=str,
        default=DEFAULT_AUDIO_PATH,
        help="Path to audio file (WAV format, 16kHz)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for refined diarization"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output filename"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of KNN neighbors"
    )
    
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.6,
        help="Leiden resolution parameter (higher = more communities)"
    )
    
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=3,
        help="Minimum community size threshold for outlier detection"
    )
    
    parser.add_argument(
        "--purity_threshold",
        type=float,
        default=0.8,
        help="Purity threshold for community speaker remapping [0.0-1.0]"
    )
    
    args = parser.parse_args()
    
    # Run cleansing pipeline
    run_cleansing(
        diarization_path=args.diarization_path,
        audio_path=args.audio_path,
        output_dir=args.output_dir,
        output_file=args.output_file,
        k=args.k,
        resolution=args.resolution,
        min_cluster_size=args.min_cluster_size,
        purity_threshold=args.purity_threshold,
    )


if __name__ == "__main__":
    main()
