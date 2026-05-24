# Bo 0.5, ngan thi (< 1) ko dua vao graph --> cluster tren graph nay, sau do 

import argparse
import os
import subprocess
import sys
import librosa
import torch
from collections import defaultdict
from asv_model.ecapa.get_model import compute_fbank, get_pretrained_model, apply_cmvn

import numpy as np
from sklearn.cluster import AgglomerativeClustering

def parse_diarization_file(filepath):
    """
    Parse diarization lines.
    Supports both formats:
    - <start> <end> <speaker>
    - <speaker> <start> <end>
    """
    segments = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            # Try format: start end speaker
            try:
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2]
                if end > start:
                    segments.append({"start": start, "end": end, "speaker": speaker})
                    continue
            except ValueError:
                pass

            # Try format: speaker start end
            try:
                speaker = parts[0]
                start = float(parts[1])
                end = float(parts[2])
                if end > start:
                    segments.append({"start": start, "end": end, "speaker": speaker})
            except ValueError:
                continue

    segments.sort(key=lambda x: (x["start"], x["end"], x["speaker"]))
    return segments


def write_diarization_file(filepath, segments):
    with open(filepath, "w") as f:
        for seg in segments:
            f.write(f"{seg['start']:.3f} {seg['end']:.3f} {seg['speaker']}\n")


def merge_same_speaker_segments(segments, max_gap_sec=0.5):
    """Merge neighboring segments if same speaker and silence gap <= threshold."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda x: (x["start"], x["end"], x["speaker"]))
    merged = [dict(ordered[0])]

    for seg in ordered[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if seg["speaker"] == prev["speaker"] and gap <= max_gap_sec:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged

def remove_overlapping_segments(segments):
    cleaned = []

    for i, seg in enumerate(segments):
        is_overlap = False
        for j, other in enumerate(segments):
            if i == j:
                continue

            # overlap nếu giao nhau theo thời gian và khác speaker
            if seg["speaker"] != other["speaker"]:
                overlap_start = max(seg["start"], other["start"])
                overlap_end = min(seg["end"], other["end"])

                if overlap_end > overlap_start:
                    is_overlap = True
                    break

        if not is_overlap:
            cleaned.append(seg)

    return cleaned

def filter_short_segments(segments, min_duration_sec=0.5):
    return [seg for seg in segments if (seg["end"] - seg["start"]) >= min_duration_sec]


def extract_audio_segment(input_audio_path, output_wav_path, start_sec, end_sec, sample_rate=16000):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_audio_path,
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        output_wav_path,
    ]
    ret = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ret == 0 and os.path.exists(output_wav_path)

def select_main_cluster_indices_ahc(embeddings, sim_threshold=0.6):
    """
    Use sklearn AgglomerativeClustering directly.
    Keep the largest AHC cluster as the main speaker cluster.
    Smaller clusters are treated as ASV outliers.
    """

    n = len(embeddings)
    if n <= 2:
        return set(range(n))

    X = np.stack([_normalize(e) for e in embeddings], axis=0)

    distance_threshold = 1.0 - sim_threshold

    try:
        ahc = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        ahc = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )

    labels = ahc.fit_predict(X)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    main_cluster = max(clusters.values(), key=len)

    return set(main_cluster)

def _normalize(v):
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v
    return v / n


def _build_similarity_graph(embeddings, sim_threshold):
    """Create adjacency list where cosine similarity >= threshold."""
    n = len(embeddings)
    X = np.stack([_normalize(e) for e in embeddings], axis=0)
    sim = X @ X.T

    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= sim_threshold:
                graph[i].append(j)
                graph[j].append(i)
    return graph, sim


def _connected_components(graph):
    n = len(graph)
    visited = [False] * n
    components = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(sorted(comp))
    return components


def select_main_cluster_indices(embeddings, sim_threshold=0.55):
    """
    Inner-speaker clustering.
    Keep the largest connected component in cosine-similarity graph.
    Tie-break by highest mean internal similarity.
    """
    n = len(embeddings)
    if n <= 2:
        return set(range(n))

    graph, sim = _build_similarity_graph(embeddings, sim_threshold)
    comps = _connected_components(graph)
    if len(comps) == 1:
        return set(comps[0])

    comps.sort(key=len, reverse=True)
    best = comps[0]
    best_size = len(best)

    ties = [c for c in comps if len(c) == best_size]
    if len(ties) > 1:
        best_score = -1e9
        for c in ties:
            if len(c) == 1:
                score = 1.0
            else:
                sub = sim[np.ix_(c, c)]
                denom = len(c) * len(c) - len(c)
                score = float((np.sum(sub) - len(c)) / max(denom, 1))
            if score > best_score:
                best_score = score
                best = c
    return set(best)


# def load_vietnamese_asv_model(initial_model_path=None, device="cuda"):
#     """
#     Load ASV model for embedding extraction.

#     Important: fusion_module1/main.py has heavy top-level side effects.
#     So we instantiate Trainer directly from fusion_module1/trainer.py.
#     """
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     fusion_root = os.path.join(current_dir, "fusion_module1")
#     if fusion_root not in sys.path:
#         sys.path.insert(0, fusion_root)

#     from trainer import Trainer

#     if initial_model_path is None:
#         initial_model_path = os.path.join(
#             fusion_root,
#             "exps",
#             "ecapa_vn",
#             "05_01_26",
#             "model_0100.model",
#         )

#     model = Trainer(
#         lr=0.001,
#         lr_decay=0.97,
#         C=512,
#         n_class=1211,
#         m=0.2,
#         s=30,
#         test_step=1,
#         device=device,
#         grad_accumulation_step=1,
#     )
#     model.load_parameters(initial_model_path)
#     model.eval()
#     return model

def load_vietnamese_asv_model():
    model = get_pretrained_model()
    model.eval()
    return model

def extract_embedding(asv_model, audio_path):
    waveform, _ = librosa.load(audio_path, sr=16000)
    waveform = torch.FloatTensor(waveform).unsqueeze(0)
    fbank = compute_fbank(waveform).unsqueeze(0)
    feat = apply_cmvn(fbank)
    _, emb = asv_model.forward(feat)

    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    emb = np.array(emb).flatten().astype(np.float32)
    return emb


def run_post_processing(
    diarization_path,
    audio_path,
    output_final_path,
    out_dir,
    merge_gap_sec=0.5,
    min_duration_sec=0.5,
    min_cluster_size=3,
    intra_sim_threshold=0.55,
    sample_rate=16000,
    asv_model_path=None,
    device="cuda",
):
    os.makedirs(out_dir, exist_ok=True)
    segments_dir = os.path.join(out_dir, "segment_wavs")
    os.makedirs(segments_dir, exist_ok=True)

    # 1) Initial diarization processing
    print("\n1. Initial diarization processing") 

    original_segments = parse_diarization_file(diarization_path)
    non_overlap_segments = remove_overlapping_segments(original_segments)

    merged_segments = merge_same_speaker_segments(original_segments, max_gap_sec=merge_gap_sec)
    # processed_segments = filter_short_segments(merged_segments, min_duration_sec=min_duration_sec)
    processed_segments = merged_segments
    processed_diar_path = os.path.join(out_dir, "processed_before_asv.txt")
    write_diarization_file(processed_diar_path, processed_segments)

    if not processed_segments:
        write_diarization_file(output_final_path, [])
        return {
            "original_count": len(original_segments),
            "processed_count": 0,
            "kept_after_asv": 0,
            "removed_asv": 0,
            "processed_diarization": processed_diar_path,
            "final_output": output_final_path,
            "report_path": None,
        }

    # 2) Load ASV model
    print("\n2. Loading ASV model")

    asv_model = load_vietnamese_asv_model()

    # 3) Extract per-segment audio + embeddings
    enriched = []
    for idx, seg in enumerate(processed_segments):
        seg_path = os.path.join(
            segments_dir,
            f"seg_{idx:05d}_{seg['speaker']}_{seg['start']:.3f}_{seg['end']:.3f}.wav",
        )
        ok = extract_audio_segment(
            input_audio_path=audio_path,
            output_wav_path=seg_path,
            start_sec=seg["start"],
            end_sec=seg["end"],
            sample_rate=sample_rate,
        )
        if not ok:
            continue


        emb = extract_embedding(asv_model, seg_path)

        item = dict(seg)
        item["segment_path"] = seg_path
        item["embedding"] = emb
        enriched.append(item)
    
    # Cleansing short segments that cannot join graph formation
    trusted = []
    untrusted = []

    for item in enriched:
        dur = item["end"] - item["start"]
        if dur >= 1:
            trusted.append(item)
        else:
            untrusted.append(item)
            
    # 4) Inner-speaker clustering and outlier filtering
    # by_speaker = defaultdict(list)
    # for item in enriched:
    #     by_speaker[item["speaker"]].append(item)

    # kept = []
    # removed = []

    # for speaker, items in by_speaker.items():
    #     if len(items) < min_cluster_size:
    #         kept.extend(items)
    #         continue

    #     embs = [x["embedding"] for x in items]
    #     main_idx = select_main_cluster_indices(embeddings=embs, sim_threshold=intra_sim_threshold)
    #     for i, item in enumerate(items):
    #         if i in main_idx:
    #             kept.append(item)
    #         else:
    #             removed.append(item)
    print("\n3. SPEAKER FILTERING START")

    by_speaker = defaultdict(list)
    for item in trusted:
        by_speaker[item["speaker"]].append(item)

    kept = []
    removed = []

    for speaker, items in by_speaker.items():

        # too few trusted segments -> keep directly
        if len(items) < min_cluster_size:
            kept.extend(items)
            continue

        embs = [x["embedding"] for x in items]

        main_idx = select_main_cluster_indices_ahc(
            embeddings=embs,
            sim_threshold=intra_sim_threshold
        )

        for i, item in enumerate(items):
            if i in main_idx:
                kept.append(item)
            else:
                removed.append(item)
    # Reassign short/untrusted segments
    
    speaker_centroids = {}
    by_kept_speaker = defaultdict(list)

    for item in kept:
        by_kept_speaker[item["speaker"]].append(item["embedding"])

    for speaker, embs in by_kept_speaker.items():
        centroid = np.mean(np.stack(embs), axis=0)
        speaker_centroids[speaker] = _normalize(centroid)


    def cosine(a, b):
        return float(np.dot(_normalize(a), _normalize(b)))


    reassigned_short = []
    print("\n4. SHORT SEGMENT REASSIGNMENT")

    for item in untrusted:

        if not speaker_centroids:
            kept.append(item)
            continue

        best_speaker = max(
            speaker_centroids.keys(),
            key=lambda spk: cosine(item["embedding"], speaker_centroids[spk])
        )

        new_item = dict(item)
        new_item["original_speaker"] = item["speaker"]
        new_item["speaker"] = best_speaker

        reassigned_short.append(new_item)
        kept.append(new_item)
        
    kept_segments = [{"start": x["start"], "end": x["end"], "speaker": x["speaker"]} for x in kept]
    kept_segments = merge_same_speaker_segments(kept_segments, max_gap_sec=merge_gap_sec)
    kept_segments = filter_short_segments(kept_segments, min_duration_sec=min_duration_sec)
    kept_segments.sort(key=lambda x: (x["start"], x["end"], x["speaker"]))

    write_diarization_file(output_final_path, kept_segments)

    # 5) Report
    report_path = os.path.join(out_dir, "post_processing_report.txt")
    with open(report_path, "w") as f:
        f.write("5. POST PROCESSING REPORT\n")
        f.write(f"Input diarization: {diarization_path}\n")
        f.write(f"Input audio: {audio_path}\n")
        f.write(f"Merge gap threshold: {merge_gap_sec:.2f}s\n")
        f.write(f"Min segment duration: {min_duration_sec:.2f}s\n")
        f.write(f"Inner-speaker sim threshold: {intra_sim_threshold:.2f}\n")
        f.write(f"Min cluster size for filtering: {min_cluster_size}\n\n")

        f.write(f"Original segments: {len(original_segments)}\n")
        f.write(f"After merge+duration filtering: {len(processed_segments)}\n")
        f.write(f"Segments with valid embeddings: {len(enriched)}\n")
        f.write(f"Removed as ASV outliers: {len(removed)}\n")
        f.write(f"Final kept segments: {len(kept_segments)}\n\n")

        f.write("[REMOVED OUTLIERS]\n")
        if not removed:
            f.write("None\n")
        else:
            for x in sorted(removed, key=lambda y: (y["start"], y["speaker"])):
                dur = x["end"] - x["start"]
                f.write(f"{x['start']:.3f} {x['end']:.3f} {x['speaker']} | dur={dur:.3f}s\n")

    return {
        "original_count": len(original_segments),
        "processed_count": len(processed_segments),
        "kept_after_asv": len(kept_segments),
        "removed_asv": len(removed),
        "processed_diarization": processed_diar_path,
        "final_output": output_final_path,
        "report_path": report_path,
    }


# def main():
#     SAMPLE_KEY = "singing"
#     parser = argparse.ArgumentParser(
#         description="Post-process diarization with Vietnamese ASV-based inner-speaker outlier removal."
#     )
#     parser.add_argument(
#         "--diarization_input",
#         default=f"pipeline/audio_pipeline/testing/test_diarization_output_precision-2/{SAMPLE_KEY}.txt",
#         help="Input diarization file",
#     )
#     parser.add_argument(
#         "--audio_path",
#         default=f"data/diarization_test_set/video/{SAMPLE_KEY}.mp4",
#         help="Original full audio file",
#     )
#     parser.add_argument("--out_dir", default="post_processing_results")
#     parser.add_argument("--output_final", default="final_post_processed_diarization.txt")

#     parser.add_argument("--merge_gap_sec", type=float, default=0.5)
#     parser.add_argument("--min_duration_sec", type=float, default=0.3)
#     parser.add_argument("--min_cluster_size", type=int, default=4)
#     parser.add_argument("--intra_sim_threshold", type=float, default=0.7)
#     parser.add_argument("--sample_rate", type=int, default=16000)

#     parser.add_argument("--asv_model_path", default=None)
#     parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

#     args = parser.parse_args()
#     args.out_dir = os.path.join(args.out_dir, SAMPLE_KEY)
#     os.makedirs(args.out_dir, exist_ok=True)
#     output_final = args.output_final
#     if not os.path.isabs(output_final):
#         output_final = os.path.join(args.out_dir, output_final)

#     stats = run_post_processing(
#         diarization_path=args.diarization_input,
#         audio_path=args.audio_path,
#         output_final_path=output_final,
#         out_dir=args.out_dir,
#         merge_gap_sec=args.merge_gap_sec,
#         min_duration_sec=args.min_duration_sec,
#         min_cluster_size=args.min_cluster_size,
#         intra_sim_threshold=args.intra_sim_threshold,
#         sample_rate=args.sample_rate,
#         asv_model_path=args.asv_model_path,
#         device=args.device,
#     )

#     print("Post-processing completed.")
#     print(f"Processed diarization: {stats['processed_diarization']}")
#     print(f"Final output: {stats['final_output']}")
#     print(f"Report: {stats['report_path']}")
#     print(
#         "Counts => "
#         f"original={stats['original_count']}, "
#         f"after_basic={stats['processed_count']}, "
#         f"removed_asv={stats['removed_asv']}, "
#         f"final={stats['kept_after_asv']}"
#     )

def main():

    diar_dir = "pipeline/audio_pipeline/testing/test_diarization_output_precision-2"
    audio_dir = "data/diarization_test_set/audio"

    output_root = "pipeline/face_pipeline/result/result_cleansing/input_cleansing_ahc_3"

    diar_files = sorted([
        f for f in os.listdir(diar_dir)
        if f.endswith(".txt")
    ])

    for diar_file in diar_files:

        sample_key = os.path.splitext(diar_file)[0]

        diarization_path = os.path.join(diar_dir, diar_file)

        audio_path = os.path.join(audio_dir, f"{sample_key}.wav")

        if not os.path.exists(audio_path):
            print(f"[SKIP] Missing audio: {audio_path}")
            continue

        out_dir = os.path.join(output_root, sample_key)
        os.makedirs(out_dir, exist_ok=True)

        output_final = os.path.join(
            out_dir,
            "final_post_processed_diarization.txt"
        )

        print(f"\n===== Processing {sample_key} =====")

        stats = run_post_processing(
            diarization_path=diarization_path,
            audio_path=audio_path,
            output_final_path=output_final,
            out_dir=out_dir,
            merge_gap_sec=0.5,
            min_duration_sec=0.3,
            min_cluster_size=4,
            intra_sim_threshold=0.6,
            sample_rate=16000,
            device="cuda",
        )

        print(
            f"[DONE] {sample_key} | "
            f"original={stats['original_count']} | "
            f"removed={stats['removed_asv']} | "
            f"final={stats['kept_after_asv']}"
        )
if __name__ == "__main__":
    main()