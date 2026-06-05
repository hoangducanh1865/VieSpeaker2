import argparse
import cv2
import os
import numpy as np
import subprocess
import re
import math
import torch
import python_speech_features
import pickle
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from scipy.io import wavfile

import sys
import sklearn
import onnxruntime as ort

# Detect ONNX Runtime CUDA support once at import time
_ORT_AVAILABLE_PROVIDERS = ort.get_available_providers()
_ORT_CUDA_AVAILABLE = 'CUDAExecutionProvider' in _ORT_AVAILABLE_PROVIDERS
_ORT_PROVIDERS = (
    ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if _ORT_CUDA_AVAILABLE else ['CPUExecutionProvider']
)
# insightface ctx_id: 0 = GPU device 0, -1 = CPU
_INSIGHTFACE_CTX_ID = 0 if _ORT_CUDA_AVAILABLE else -1

_AV_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [
    os.path.join(_AV_DIR, 'sort'),
    _AV_DIR,
    os.path.join(_AV_DIR, 'audio_visual_model', 'LR-ASD'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Tracking and Detection
import sort
from face_detection_model.SCRFD.nets.nn import FaceDetector

# InsightFace
from insightface.utils import face_align
from insightface.model_zoo import get_model

from ASD import ASD as LRASDModel

_SKLEARN_VER = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

ORIGINAL_FACE_SETUP = {
    'det_input_size': 640,
    'det_thresh': 0.5,
    'tracker_max_age': 30,
    'tracker_min_hits': 3,
    'tracker_iou_thresh': 0.3,
    'scene_cut_threshold': 0.7,
    'scene_cut_debounce': 1,
}

def calculate_iou(boxA, boxB):
    """Utility to match SORT trackers back to original detections to recover landmarks."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


class ASVDataPipeline:
    def __init__(
        self,
        detector_path,
        arcface_path,
        out_dir,
        det_input_size=640,
        det_thresh=0.5,
        tracker_max_age=30,
        tracker_min_hits=3,
        tracker_iou_thresh=0.3,
        scene_cut_threshold=0.7,
        scene_cut_debounce=1,
    ):
        # 1. Initialize Detector & Tracker
        self.detector = FaceDetector(onnx_file=detector_path, providers=_ORT_PROVIDERS)
        self.det_input_size = int(det_input_size)
        self.det_thresh = float(det_thresh)
        self.tracker_max_age = int(tracker_max_age)
        self.tracker_min_hits = int(tracker_min_hits)
        self.tracker_iou_thresh = float(tracker_iou_thresh)
        self.scene_cut_threshold = float(scene_cut_threshold)
        self.scene_cut_debounce = max(1, int(scene_cut_debounce))
        self.mot_tracker = sort.Sort(
            max_age=self.tracker_max_age,
            min_hits=self.tracker_min_hits,
            iou_threshold=self.tracker_iou_thresh,
        )
        
        # 2. Initialize ArcFace Recognizer
        # Note: Download a model like 'glintr100.onnx' or 'w600k_r50.onnx' from InsightFace
        print(f"Loading ArcFace model from {arcface_path}...")
        self.recognizer = get_model(arcface_path)
        self.recognizer.prepare(ctx_id=_INSIGHTFACE_CTX_ID)
        
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Dictionary to store metadata for Active Speaker Detection (ASD)
        # Format: {track_id: [list of aligned image paths]}
        self.track_history = defaultdict(list)
        # Keep frame-level bboxes for rendering annotations on the original video.
        # Format: {track_id: [(frame_idx, [x1, y1, x2, y2]), ...]}
        self.track_bboxes = defaultdict(list)
        self.track_embeddings = {}
        # Global offset so SORT track IDs remain unique after tracker reset on scene cut
        self._global_track_id_offset = 0

    def phase_0_preprocess_fps(self, video_path):
        """
        Ensures the input video is exactly 25 FPS for TalkNet compatibility.
        Saves the new video to the output directory to preserve the original.
        """
        print("\n--- Phase 0: Video Preprocessing (25 FPS Enforcement) ---")
        
        stream = cv2.VideoCapture(video_path)
        current_fps = stream.get(cv2.CAP_PROP_FPS)
        frame_count = stream.get(cv2.CAP_PROP_FRAME_COUNT)
        stream.release()

        duration_sec = 0.0
        if current_fps and current_fps > 0:
            duration_sec = frame_count / current_fps
        if duration_sec <= 30.0:
            raise ValueError("Input video must be longer than 30 seconds.")

        base_name = os.path.basename(video_path)
        name, ext = os.path.splitext(base_name)
        standardized_video_path = os.path.join(self.out_dir, f"{name}_25fps_hq.mp4")

        if os.path.exists(standardized_video_path):
            print(f"Standardized video already exists at: {standardized_video_path}")
            return standardized_video_path

        print("Converting to strict 25 FPS with high quality...")
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-r', '25',
            '-c:v', 'libx264', '-crf', '16', '-preset', 'slow',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k',
            standardized_video_path
        ]

        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(standardized_video_path):
            raise RuntimeError("FFmpeg failed to prepare high-quality 25 FPS video.")

        print(f"Conversion complete! Standardized video saved for pipeline use.")
        return standardized_video_path

    def phase_1_track_and_crop(self, video_path):
        """Processes video, tracks faces, aligns them, and saves to disk."""
        print("\n--- Phase 1: Detection, Tracking & Alignment ---")
        stream = cv2.VideoCapture(video_path)
        if not stream.isOpened():
            raise ValueError(f"Cannot open video stream at {video_path}")

        frame_count = 0
        saved_count = 0
        
        # --- NEW: Variable to hold the previous frame's color histogram ---
        prev_hist = None
        cut_streak = 0

        while True:
            success, frame = stream.read()
            if not success:
                break
                
            frame_count += 1

            # --- NEW: Scene Cut Detection Logic ---
            # Convert frame to HSV (better for lighting changes) and calculate color distribution
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            if prev_hist is not None:
                # Compare current frame to previous frame
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                # A correlation of 1.0 means identical. < 0.7 usually indicates a hard camera cut.
                if correlation < self.scene_cut_threshold:
                    cut_streak += 1
                    if cut_streak >= self.scene_cut_debounce:
                        print(f"[{frame_count}] Camera Cut Detected (score: {correlation:.2f}). Resetting Tracker.")
                        # Update global offset so new SORT IDs don't collide with old track IDs
                        if self.track_history:
                            self._global_track_id_offset = max(self.track_history.keys())
                        self.mot_tracker = sort.Sort(
                            max_age=self.tracker_max_age,
                            min_hits=self.tracker_min_hits,
                            iou_threshold=self.tracker_iou_thresh,
                        )
                        cut_streak = 0
                else:
                    cut_streak = 0

            prev_hist = hist
            # ---------------------------------------

            dets, kpss = self.detector.detect(
                frame,
                thresh=self.det_thresh,
                input_size=(self.det_input_size, self.det_input_size),
            )
            
            if dets is not None and len(dets) > 0:
                trackers = self.mot_tracker.update(dets)
            else:
                trackers = self.mot_tracker.update(np.empty((0, 5)))
                continue

            for d in trackers:
                trk_box = d[0:4]
                track_id = int(d[4]) + self._global_track_id_offset
                
                # Find the original detection to get the landmarks (kpss)
                best_iou = 0
                best_kps = None
                for i, det in enumerate(dets):
                    iou = calculate_iou(trk_box, det[0:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_kps = kpss[i]

                # If we have valid landmarks, align and save
                if best_kps is not None and best_iou > 0.3:
                    # norm_crop automatically crops and aligns face to 112x112 for ArcFace
                    aligned_face = face_align.norm_crop(frame, landmark=best_kps)
                    
                    track_dir = os.path.join(self.out_dir, f"tracks/track_{track_id}")
                    os.makedirs(track_dir, exist_ok=True)
                    
                    filename = f"frame_{frame_count:06d}.jpg"
                    filepath = os.path.join(track_dir, filename)
                    cv2.imwrite(filepath, aligned_face)
                    
                    self.track_history[track_id].append(filepath)
                    self.track_bboxes[track_id].append((frame_count, trk_box.tolist()))
                    saved_count += 1

            if frame_count % 500 == 0:
                print(f"Processed {frame_count} frames... Saved {saved_count} aligned faces.")

        stream.release()
        print(f"Phase 1 Complete. Saved {saved_count} aligned crops across {len(self.track_history)} tracks.")
        
        # Save pipeline state for resuming
        self._save_pipeline_state()

    def phase_2_extract_uniform_embeddings(self, num_samples=5):
        """Samples 5 uniform frames per track, extracts embeddings, and averages them."""
        print("\n--- Phase 2: Uniform Embedding Extraction ---")
        
        for track_id, file_paths in self.track_history.items():
            if len(file_paths) == 0:
                continue
                
            # Select uniform indices
            indices = np.linspace(0, len(file_paths) - 1, num_samples, dtype=int)
            sampled_paths = [file_paths[i] for i in indices]
            
            embeddings = []
            for path in sampled_paths:
                img = cv2.imread(path)
                if img is not None:
                    # Extract embedding. get_feat expects BGR 112x112 image
                    feat = self.recognizer.get_feat(img) 
                    embeddings.append(feat)
            
            if embeddings:
                # Average the embeddings for a robust representation
                avg_embedding = np.mean(embeddings, axis=0)
                # Normalize the averaged embedding (crucial for cosine distance)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                self.track_embeddings[track_id] = avg_embedding
                
        print(f"Phase 2 Complete. Extracted embeddings for {len(self.track_embeddings)} tracks.")

    def phase_3_cluster_identities(self, distance_threshold=0.6, merge_threshold=0.0):
        """Clusters track embeddings to group same speakers together.

        merge_threshold > 0 enables a second-pass "re-ID" merge: face clusters
        whose centroids are within `merge_threshold` cosine distance are merged.
        Scene cuts reset the tracker and mint fresh track IDs, so one person often
        ends up split across several clusters; this pass re-unifies them.
        """
        print("\n--- Phase 3: Agglomerative Hierarchical Clustering ---")
        if len(self.track_embeddings) == 0:
            print("No embeddings to cluster.")
            return {}

        track_ids = list(self.track_embeddings.keys())
        features = np.array(list(self.track_embeddings.values()))
        features = features.reshape(features.shape[0], -1)


        # If there's only 1 track, clustering will fail, so handle edge case
        if len(features) == 1:
            labels = [0]
        else:
            # Compute cosine distance matrix and use precomputed metric
            distance_matrix = pairwise_distances(features, metric='cosine')
            _metric_kw = "metric" if _SKLEARN_VER >= (1, 4) else "affinity"
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                **{_metric_kw: 'precomputed'},
                linkage='average',
                distance_threshold=distance_threshold,
            )
            labels = clusterer.fit_predict(distance_matrix)

        labels = list(labels)
        if merge_threshold and merge_threshold > 0 and len(set(labels)) > 1:
            labels = self._merge_close_clusters(features, labels, merge_threshold)

        # Group track IDs by their clustered identity
        clustered_identities = defaultdict(list)
        for track_id, label in zip(track_ids, labels):
            clustered_identities[label].append(track_id)

        print(f"Found {len(clustered_identities)} unique speaker identities.")
        for cluster_id, tracks in clustered_identities.items():
            print(f"  Speaker {cluster_id}: Track IDs {tracks}")

        return clustered_identities

    @staticmethod
    def _merge_close_clusters(features, labels, merge_threshold):
        """Greedily merge clusters whose (normalised) centroids are within
        `merge_threshold` cosine distance. Returns relabelled, compacted labels."""
        labels = list(labels)
        for _ in range(len(set(labels))):  # iterate until no merge happens
            uniq = sorted(set(labels))
            centroids = {}
            for c in uniq:
                idx = [i for i, l in enumerate(labels) if l == c]
                v = features[idx].mean(axis=0)
                n = np.linalg.norm(v)
                centroids[c] = v / n if n > 1e-8 else v
            merged = False
            for a in range(len(uniq)):
                for b in range(a + 1, len(uniq)):
                    ca, cb = uniq[a], uniq[b]
                    dist = 1.0 - float(np.dot(centroids[ca], centroids[cb]))
                    if dist < merge_threshold:
                        labels = [ca if l == cb else l for l in labels]
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break
        # compact label ids to 0..K-1
        remap = {c: i for i, c in enumerate(sorted(set(labels)))}
        return [remap[l] for l in labels]

    def _filter_overlapping_segments(self, segments, min_segment_sec=0.2):
        if len(segments) == 0:
            return []

        boundaries = sorted({point for start, end, _ in segments for point in (start, end)})
        if len(boundaries) < 2:
            return []

        pure_segments = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right <= left:
                continue

            midpoint = (left + right) / 2.0
            active_speakers = {
                speaker_label
                for start, end, speaker_label in segments
                if start <= midpoint < end
            }

            if len(active_speakers) == 1:
                pure_segments.append((left, right, next(iter(active_speakers))))

        merged_segments = []
        for start, end, speaker_label in pure_segments:
            if merged_segments and merged_segments[-1][2] == speaker_label and abs(merged_segments[-1][1] - start) < 1e-6:
                merged_segments[-1] = (merged_segments[-1][0], end, speaker_label)
            else:
                merged_segments.append((start, end, speaker_label))

        return [segment for segment in merged_segments if (segment[1] - segment[0]) >= min_segment_sec]

    def _merge_segments_per_speaker(self, segments, min_segment_sec=0.2):
        """Merge contiguous/overlapping segments for each speaker while preserving cross-speaker overlap."""
        if len(segments) == 0:
            return []

        segments_by_speaker = defaultdict(list)
        for start, end, speaker_label in segments:
            if end > start:
                segments_by_speaker[speaker_label].append((start, end))

        merged_segments = []
        for speaker_label, speaker_segments in segments_by_speaker.items():
            speaker_segments.sort(key=lambda x: x[0])
            cur_start, cur_end = speaker_segments[0]

            for start, end in speaker_segments[1:]:
                if start <= (cur_end + 1e-6):
                    cur_end = max(cur_end, end)
                else:
                    if (cur_end - cur_start) >= min_segment_sec:
                        merged_segments.append((cur_start, cur_end, speaker_label))
                    cur_start, cur_end = start, end

            if (cur_end - cur_start) >= min_segment_sec:
                merged_segments.append((cur_start, cur_end, speaker_label))

        merged_segments.sort(key=lambda x: x[0])
        return merged_segments
    
    def phase_4_active_speaker_detection(self, video_path, clustered_identities, lrasd_weights,
                                         output_txt_path="diarization_output.txt", asd_threshold=0.0,
                                         min_segment_sec=0.2, allow_overlap=True, annotated_mp4_path=None):
        """Runs LR-ASD inference and generates diarization segments per clustered speaker."""
        print("\n--- Phase 4: Active Speaker Detection (LR-ASD) ---")
        if len(clustered_identities) == 0:
            print("No clustered identities provided. Writing an empty diarization file.")
            with open(output_txt_path, 'w') as f:
                f.write("")
            return output_txt_path

        # LR-ASD expects 16kHz mono audio and visual frames on a 25fps timeline.
        audio_path = os.path.join(self.out_dir, "temp_audio_16k.wav")
        print("Extracting 16kHz mono audio track...")
        audio_cmd = ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000', '-vn', audio_path]
        ret = subprocess.call(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ret != 0 or not os.path.exists(audio_path):
            raise RuntimeError("FFmpeg failed to extract 16kHz mono audio for LR-ASD inference.")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            fps = 25.0
        print(f"Video FPS: {fps}")

        sample_rate, full_audio = wavfile.read(audio_path)
        if sample_rate != 16000:
            raise RuntimeError(f"Expected 16000Hz audio after extraction, but got {sample_rate}Hz")

        print(f"Loading LR-ASD weights from {lrasd_weights}...")
        use_cuda = torch.cuda.is_available()
        asd_model = LRASDModel(use_cuda=use_cuda)
        asd_model.loadParameters(lrasd_weights)
        asd_model.eval()
        device = asd_model.device
        print(f"[LR-ASD] Using device: {device}")

        duration_set = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
        final_diarization_segments = []
        track_prediction_summary = {}

        for speaker_id, track_ids in clustered_identities.items():
            speaker_label = f"SPEAKER_{speaker_id:02d}"
            print(f"Analyzing {speaker_label} (Tracks: {len(track_ids)})...")

            for track_id in track_ids:
                frame_paths = sorted(self.track_history[track_id])
                if len(frame_paths) < 10:
                    continue

                frame_numbers = []
                for p in frame_paths:
                    m = re.search(r'frame_(\d+)', os.path.basename(p))
                    if m:
                        frame_numbers.append(int(m.group(1)))
                if len(frame_numbers) == 0:
                    continue

                start_frame = frame_numbers[0]
                end_frame = frame_numbers[-1]
                start_time = start_frame / fps
                end_time = (end_frame + 1) / fps

                start_sample = max(0, int(round(start_time * sample_rate)))
                end_sample = min(len(full_audio), int(round(end_time * sample_rate)))
                if end_sample <= start_sample:
                    continue
                track_audio = full_audio[start_sample:end_sample]

                if track_audio.ndim > 1:
                    track_audio = np.mean(track_audio, axis=1)
                audio_feature = python_speech_features.mfcc(
                    track_audio, 16000, numcep=13, winlen=0.025, winstep=0.010
                )

                video_feature = []
                for p in frame_paths:
                    frame = cv2.imread(p)
                    if frame is None:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if gray.shape[0] != 112 or gray.shape[1] != 112:
                        gray = cv2.resize(gray, (112, 112))
                    video_feature.append(gray)
                if len(video_feature) == 0:
                    continue

                video_feature = np.array(video_feature)
                length = min((audio_feature.shape[0] - (audio_feature.shape[0] % 4)) / 100.0, video_feature.shape[0])
                if length <= 0:
                    continue

                audio_feature = audio_feature[:int(round(length * 100)), :]
                video_feature = video_feature[:int(round(length * 25)), :, :]
                if audio_feature.shape[0] == 0 or video_feature.shape[0] == 0:
                    continue

                per_duration_scores = []
                for duration in duration_set:
                    batch_size = int(math.ceil(length / duration))
                    scores = []
                    with torch.no_grad():
                        for i in range(batch_size):
                            a_start = i * duration * 100
                            a_end = (i + 1) * duration * 100
                            v_start = i * duration * 25
                            v_end = (i + 1) * duration * 25

                            input_a_np = audio_feature[a_start:a_end, :]
                            input_v_np = video_feature[v_start:v_end, :, :]
                            if input_a_np.shape[0] == 0 or input_v_np.shape[0] == 0:
                                continue

                            input_a = torch.FloatTensor(input_a_np).unsqueeze(0).to(device)
                            input_v = torch.FloatTensor(input_v_np).unsqueeze(0).to(device)
                            embed_a = asd_model.model.forward_audio_frontend(input_a)
                            embed_v = asd_model.model.forward_visual_frontend(input_v)
                            # LR-ASD audio/visual encoders can produce off-by-one temporal length.
                            min_t = min(embed_a.shape[1], embed_v.shape[1])
                            if min_t <= 0:
                                continue
                            embed_a = embed_a[:, :min_t, :]
                            embed_v = embed_v[:, :min_t, :]
                            out = asd_model.model.forward_audio_visual_backend(embed_a, embed_v)
                            score = asd_model.lossAV.forward(out, labels=None)
                            scores.extend(score)

                    if len(scores) > 0:
                        per_duration_scores.append(np.asarray(scores, dtype=float))

                if len(per_duration_scores) == 0:
                    continue

                min_len = min(len(s) for s in per_duration_scores)
                if min_len == 0:
                    continue
                score_arr = np.stack([s[:min_len] for s in per_duration_scores], axis=0)
                mean_scores = np.round(np.mean(score_arr, axis=0), 3).astype(float)
                if len(mean_scores) == 0:
                    continue

                binary_preds = (mean_scores > asd_threshold).astype(np.int32)
                track_prediction_summary[track_id] = {
                    'speaker_label': speaker_label,
                    'start_time': start_time,
                    'end_time': end_time,
                    'mean_scores': mean_scores,
                    'binary_preds': binary_preds
                }
                in_speech_segment = False
                seg_start_time = 0.0
                time_step = (end_time - start_time) / max(len(binary_preds), 1)

                for idx, is_speaking in enumerate(binary_preds):
                    current_global_time = start_time + (idx * time_step)
                    if is_speaking == 1 and not in_speech_segment:
                        seg_start_time = current_global_time
                        in_speech_segment = True
                    elif is_speaking == 0 and in_speech_segment:
                        seg_end_time = current_global_time
                        if (seg_end_time - seg_start_time) >= min_segment_sec:
                            final_diarization_segments.append((seg_start_time, seg_end_time, speaker_label))
                        in_speech_segment = False

                if in_speech_segment:
                    seg_end_time = end_time
                    if (seg_end_time - seg_start_time) >= min_segment_sec:
                        final_diarization_segments.append((seg_start_time, seg_end_time, speaker_label))

        if allow_overlap:
            final_diarization_segments = self._merge_segments_per_speaker(
                final_diarization_segments,
                min_segment_sec=min_segment_sec,
            )
        else:
            final_diarization_segments = self._filter_overlapping_segments(
                final_diarization_segments,
                min_segment_sec=min_segment_sec,
            )
        final_diarization_segments.sort(key=lambda x: x[0])
        print(f"\nWriting final diarization output to {output_txt_path}...")
        with open(output_txt_path, 'w') as f:
            for start, end, spk in final_diarization_segments:
                f.write(f"{start:.3f} {end:.3f} {spk}\n")

        if annotated_mp4_path is None:
            annotated_mp4_path = os.path.join(self.out_dir, "annotated_output.mp4")
        self._render_annotated_video(
            video_path=video_path,
            out_path=annotated_mp4_path,
            track_prediction_summary=track_prediction_summary,
            fps=fps,
            asd_threshold=asd_threshold
        )

        if os.path.exists(audio_path):
            os.remove(audio_path)

        print(f"Phase 4 Complete! LR-ASD diarization finished successfully. Annotated video: {annotated_mp4_path}")
        return output_txt_path

    def _render_annotated_video(self, video_path, out_path, track_prediction_summary, fps, asd_threshold):
        """Render per-frame ASD decisions as boxes + labels and save as MP4 with audio."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for annotation: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = fps if fps > 0 else 25.0

        temp_video_path = out_path + ".tmp.mp4"
        writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (width, height))

        frame_annotations = defaultdict(list)
        for track_id, boxes in self.track_bboxes.items():
            pred = track_prediction_summary.get(track_id)
            if pred is None:
                continue

            speaker_label = pred['speaker_label']
            mean_scores = pred['mean_scores']
            binary_preds = pred['binary_preds']
            start_time = pred['start_time']
            end_time = pred['end_time']
            duration = max(end_time - start_time, 1e-6)

            for frame_idx, bbox in boxes:
                frame_time = frame_idx / source_fps
                if frame_time < start_time or frame_time > end_time:
                    continue
                rel = (frame_time - start_time) / duration
                score_idx = int(min(max(round(rel * (len(mean_scores) - 1)), 0), len(mean_scores) - 1))
                score = float(mean_scores[score_idx])
                speaking = int(binary_preds[score_idx]) == 1
                frame_annotations[frame_idx].append({
                    'track_id': track_id,
                    'speaker_label': speaker_label,
                    'bbox': bbox,
                    'score': score,
                    'speaking': speaking
                })

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            for ann in frame_annotations.get(frame_idx, []):
                x1, y1, x2, y2 = [int(round(v)) for v in ann['bbox']]
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                color = (0, 255, 0) if ann['speaking'] else (0, 0, 255)
                state = 'SPEAK' if ann['speaking'] else 'SILENT'
                label = f"{ann['speaker_label']} ID:{ann['track_id']} {state} {ann['score']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            writer.write(frame)

        cap.release()
        writer.release()

        # Mux original audio into the annotated MP4.
        mux_cmd = [
            'ffmpeg', '-y', '-i', temp_video_path, '-i', video_path,
            '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'copy', '-c:a', 'aac',
            '-shortest', out_path
        ]
        mux_ret = subprocess.call(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if mux_ret != 0:
            # Fallback to silent video if audio mux fails.
            os.replace(temp_video_path, out_path)
        else:
            os.remove(temp_video_path)

        print(f"Annotated MP4 saved to: {out_path} (threshold={asd_threshold})")

    def _save_pipeline_state(self):
        """Save track history and bboxes to disk for resuming later."""
        state_path = os.path.join(self.out_dir, "pipeline_state.pkl")
        state = {
            'track_history': dict(self.track_history),
            'track_bboxes': dict(self.track_bboxes),
        }
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Pipeline state saved to: {state_path}")

    def _load_pipeline_state(self):
        """Load track history and bboxes from disk, rewriting paths if directory changed."""
        state_path = os.path.join(self.out_dir, "pipeline_state.pkl")
        if not os.path.exists(state_path):
            print(f"No saved pipeline state found at: {state_path}")
            return False
        
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            
            # Rebuild track_history with current directory paths
            # (in case output directory was moved)
            tracks_dir = os.path.join(self.out_dir, "tracks")
            rebuilt_history = defaultdict(list)
            
            for track_id in sorted(state['track_history'].keys()):
                old_paths = state['track_history'][track_id]
                if len(old_paths) == 0:
                    continue
                
                # Extract frame numbers from old paths
                track_folder = os.path.join(tracks_dir, f"track_{track_id}")
                if os.path.exists(track_folder):
                    # Scan and sort frame files in current location
                    frame_files = sorted([
                        os.path.join(track_folder, f)
                        for f in os.listdir(track_folder)
                        if f.endswith('.jpg')
                    ])
                    if frame_files:
                        rebuilt_history[track_id] = frame_files
                    else:
                        print(f"  [WARN] No frames found in {track_folder}")
                else:
                    print(f"  [WARN] Track folder not found: {track_folder}")
            
            self.track_history = rebuilt_history
            self.track_bboxes = defaultdict(list, state['track_bboxes'])
            print(f"Loaded pipeline state from: {state_path}")
            print(f"Restored {len(self.track_history)} tracks with updated paths.")
            return True
        except Exception as e:
            print(f"Failed to load pipeline state: {e}")
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='data/diarization_test_set/video/sample_0.mp4')
    parser.add_argument('--detector_model', default='face_detection_model/SCRFD/weights/model_3_kps.onnx')
    # Path to your ArcFace ONNX model (e.g., glintr100.onnx)
    parser.add_argument('--arcface_model', default='face_embedding_model/weights/glintr100.onnx') 
    parser.add_argument('--out_dir', default='results_with_overlap')
    parser.add_argument('--lrasd_model', default='audio_visual_model/LR-ASD/weight/finetuning_TalkSet.model')
    parser.add_argument('--det_input_size', type=int, default=ORIGINAL_FACE_SETUP['det_input_size'])
    parser.add_argument('--det_thresh', type=float, default=ORIGINAL_FACE_SETUP['det_thresh'])
    parser.add_argument('--tracker_max_age', type=int, default=ORIGINAL_FACE_SETUP['tracker_max_age'])
    parser.add_argument('--tracker_min_hits', type=int, default=ORIGINAL_FACE_SETUP['tracker_min_hits'])
    parser.add_argument('--tracker_iou_thresh', type=float, default=ORIGINAL_FACE_SETUP['tracker_iou_thresh'])
    parser.add_argument('--scene_cut_threshold', type=float, default=ORIGINAL_FACE_SETUP['scene_cut_threshold'])
    parser.add_argument('--scene_cut_debounce', type=int, default=ORIGINAL_FACE_SETUP['scene_cut_debounce'])
    parser.add_argument('--use_original_face_setup', action='store_true',
                        help='Use original face detection/tracking setup values')
    parser.add_argument('--asd_threshold', type=float, default=0.0)
    parser.add_argument('--cluster_threshold', type=float, default=0.7,
                        help='Agglomerative clustering distance threshold (higher is softer/merges more tracks)')
    parser.add_argument('--min_segment_sec', type=float, default=0.2)
    parser.add_argument('--allow_overlap', dest='allow_overlap', action='store_true', default=True,
                        help='Keep overlapping diarization segments (multi-speaker overlap).')
    parser.add_argument('--no_allow_overlap', dest='allow_overlap', action='store_false',
                        help='Drop overlap and keep only single-speaker regions.')
    parser.add_argument('--annotated_mp4', default=None, help='Path to save annotated ASD MP4. Default: <out_dir>/annotated_output.mp4')
    parser.add_argument('--skip-phase', default='', type=str,
                        help='Skip specific phases. Format: "0-1" (skip 0,1), "2" (skip only 2), "2-4" (skip 2,3,4). Phases: 0=fps, 1=track, 2=embed, 3=cluster, 4=asd')
    parser.add_argument('--skip-phase-0-1', action='store_true', dest='_deprecated_skip_0_1',
                        help='[DEPRECATED] Use --skip-phase 0-1 instead')
    args = parser.parse_args()

    # Handle deprecated flag
    if args._deprecated_skip_0_1:
        args.skip_phase = '0-1'
        print("[INFO] Using deprecated --skip-phase-0-1. Please use --skip-phase 0-1 instead.")

    # Parse skip_phase argument
    skip_phases = set()
    if args.skip_phase:
        try:
            parts = args.skip_phase.split('-')
            if len(parts) == 1:
                # Single phase: "2"
                skip_phases.add(int(parts[0]))
            elif len(parts) == 2:
                # Range: "0-1" or "2-4"
                start, end = int(parts[0]), int(parts[1])
                skip_phases.update(range(start, end + 1))
            else:
                raise ValueError("Invalid --skip-phase format")
            print(f"[INFO] Skipping phases: {sorted(skip_phases)}")
        except Exception as e:
            print(f"[ERROR] Invalid --skip-phase format: {e}")
            return

    if args.use_original_face_setup:
        args.det_input_size = ORIGINAL_FACE_SETUP['det_input_size']
        args.det_thresh = ORIGINAL_FACE_SETUP['det_thresh']
        args.tracker_max_age = ORIGINAL_FACE_SETUP['tracker_max_age']
        args.tracker_min_hits = ORIGINAL_FACE_SETUP['tracker_min_hits']
        args.tracker_iou_thresh = ORIGINAL_FACE_SETUP['tracker_iou_thresh']
        args.scene_cut_threshold = ORIGINAL_FACE_SETUP['scene_cut_threshold']
        args.scene_cut_debounce = ORIGINAL_FACE_SETUP['scene_cut_debounce']

    pipeline = ASVDataPipeline(
        detector_path=args.detector_model, 
        arcface_path=args.arcface_model, 
        out_dir=args.out_dir,
        det_input_size=args.det_input_size,
        det_thresh=args.det_thresh,
        tracker_max_age=args.tracker_max_age,
        tracker_min_hits=args.tracker_min_hits,
        tracker_iou_thresh=args.tracker_iou_thresh,
        scene_cut_threshold=args.scene_cut_threshold,
        scene_cut_debounce=args.scene_cut_debounce,
    )
    
    # --- Execute End-to-End Pipeline ---
    
    if 0 in skip_phases and 1 in skip_phases:
        # Load pre-computed phase 1 results
        if not pipeline._load_pipeline_state():
            print("ERROR: Cannot skip phases 0-1 without saved pipeline state.")
            print("Run without --skip-phase flag first to generate the state.")
            return
        # Construct the standardized video path
        base_name = os.path.basename(args.video_path)
        name, ext = os.path.splitext(base_name)
        standardized_video = os.path.join(args.out_dir, f"{name}_from30s_25fps_hq.mp4")
        if not os.path.exists(standardized_video):
            print(f"ERROR: Standardized video not found at {standardized_video}")
            return
    else:
        # 0. Preprocess Video (Returns the path to the 25fps version)
        if 0 not in skip_phases:
            standardized_video = pipeline.phase_0_preprocess_fps(args.video_path)
        else:
            base_name = os.path.basename(args.video_path)
            name, ext = os.path.splitext(base_name)
            standardized_video = os.path.join(args.out_dir, f"{name}_from30s_25fps_hq.mp4")
            if not os.path.exists(standardized_video):
                print(f"[SKIP Phase 0] Standardized video not found. Running Phase 0...")
                standardized_video = pipeline.phase_0_preprocess_fps(args.video_path)
            else:
                print(f"[SKIP Phase 0] Using existing standardized video: {standardized_video}")
        
        # 1. Track & Crop (Use the standardized video!)
        if 1 not in skip_phases:
            pipeline.phase_1_track_and_crop(standardized_video)
        else:
            if not pipeline._load_pipeline_state():
                print("[ERROR] Cannot skip phase 1 without saved pipeline state.")
                print("Run phase 1 first to generate the state.")
                return
            print("[SKIP Phase 1] Loaded saved pipeline state")
    
    # 2. Extract Embeddings
    if 2 not in skip_phases:
        pipeline.phase_2_extract_uniform_embeddings(num_samples=5)
    else:
        print("[SKIP Phase 2]")
    
    # 3. Cluster Identities
    if 3 not in skip_phases:
        final_clusters = pipeline.phase_3_cluster_identities(distance_threshold=args.cluster_threshold)
    else:
        print("[SKIP Phase 3]")
        # If skipping phase 3, we still need clusters for phase 4. Try to load from file or generate simple clusters.
        # For now, assume phase 3 wasn't skipped if we need to run phase 4.
        if 4 not in skip_phases:
            print("[WARNING] Phase 4 requires clustering. Cannot skip both phases 3 and 4.")
            final_clusters = pipeline.phase_3_cluster_identities(distance_threshold=args.cluster_threshold)
        else:
            final_clusters = {}
    
    # 4. Active Speaker Detection (Use the standardized video!)
    if 4 not in skip_phases:
        output_txt = os.path.join(args.out_dir, "diarization.txt")
        pipeline.phase_4_active_speaker_detection(
            video_path=standardized_video,
            clustered_identities=final_clusters,
            lrasd_weights=args.lrasd_model,
            output_txt_path=output_txt,
            asd_threshold=args.asd_threshold,
            min_segment_sec=args.min_segment_sec,
            allow_overlap=args.allow_overlap,
            annotated_mp4_path=args.annotated_mp4
        )
    else:
        print("[SKIP Phase 4]")

if __name__ == '__main__':
    main()