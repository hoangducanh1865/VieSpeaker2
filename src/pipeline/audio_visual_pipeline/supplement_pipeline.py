import argparse
import os
import sys
import json
import subprocess
import shutil
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Make `viespeaker` importable from a fresh checkout, then centralize sys.path.
_d = _THIS_DIR
while _d != os.path.dirname(_d):
    if os.path.isdir(os.path.join(_d, "src", "viespeaker")):
        sys.path.insert(0, os.path.join(_d, "src"))
        break
    _d = os.path.dirname(_d)
from viespeaker import bootstrap, paths  # noqa: E402
bootstrap.setup()

from face_pipeline import ASVDataPipeline, ORIGINAL_FACE_SETUP  # noqa: E402

_HERE = _THIS_DIR
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))


def parse_diarization_file(filepath):
	"""Read diarization format: <start> <end> <speaker>."""
	segments = []
	with open(filepath, "r") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) < 3:
				continue
			start, end = float(parts[0]), float(parts[1])
			spk = parts[2]
			if end > start:
				segments.append({"start": start, "end": end, "speaker": spk})
	return segments


def write_diarization_file(filepath, segments):
	with open(filepath, "w") as f:
		for seg in segments:
			f.write(f"{seg['start']:.3f} {seg['end']:.3f} {seg['speaker']}\n")


def build_overlap_matrix(audio_segs, video_segs, audio_speakers, video_speakers):
	"""Build overlap (seconds) matrix: rows=audio speakers, cols=video speakers."""
	cost = np.zeros((len(audio_speakers), len(video_speakers)), dtype=float)
	if cost.size == 0:
		return cost

	a_idx = {spk: i for i, spk in enumerate(audio_speakers)}
	v_idx = {spk: i for i, spk in enumerate(video_speakers)}

	for a in audio_segs:
		i = a_idx[a["speaker"]]
		for v in video_segs:
			j = v_idx[v["speaker"]]
			overlap = max(0.0, min(a["end"], v["end"]) - max(a["start"], v["start"]))
			if overlap > 0:
				cost[i, j] += overlap
	return cost


def _write_cost_matrix_table(file_obj, audio_speakers, video_speakers, cost_matrix):
	file_obj.write("[COST MATRIX TABLE - TOTAL OVERLAP (seconds)]\n")

	if len(audio_speakers) == 0 or len(video_speakers) == 0:
		file_obj.write("  Matrix is empty (missing audio speakers or video speakers).\n\n")
		return

	row_header = "Audio\\Video"
	row_header_width = max(len(row_header), max(len(spk) for spk in audio_speakers)) + 2
	col_widths = [max(len(v_spk), len("0.00")) + 2 for v_spk in video_speakers]

	def _separator_line():
		return "+" + "-" * row_header_width + "+" + "+".join("-" * w for w in col_widths) + "+\n"

	file_obj.write(_separator_line())
	header_cells = [f"{row_header:^{row_header_width}}"]
	header_cells.extend(f"{v_spk:^{w}}" for v_spk, w in zip(video_speakers, col_widths))
	file_obj.write("|" + "|".join(header_cells) + "|\n")
	file_obj.write(_separator_line())

	for r, a_spk in enumerate(audio_speakers):
		row_cells = [f"{a_spk:<{row_header_width}}"]
		for c, width in enumerate(col_widths):
			row_cells.append(f"{cost_matrix[r, c]:>{width}.2f}")
		file_obj.write("|" + "|".join(row_cells) + "|\n")

	file_obj.write(_separator_line())
	file_obj.write("\n")


def map_audio_to_video(audio_segs, video_segs, threshold, output_report, audio_name="N/A", video_name="N/A"):
	"""
	Map SD speaker labels (audio) to visual speaker labels using Hungarian assignment.
	A match is accepted only if total overlap >= threshold.
	"""
	audio_speakers = sorted({seg["speaker"] for seg in audio_segs})
	video_speakers = sorted({seg["speaker"] for seg in video_segs})

	cost = build_overlap_matrix(audio_segs, video_segs, audio_speakers, video_speakers)
	a_to_v = {}

	with open(output_report, "w") as f:
		f.write("=== SD <-> FACE SPEAKER MAPPING REPORT ===\n")
		f.write(f"Audio diarization: {audio_name}\n")
		f.write(f"Video diarization: {video_name}\n")
		f.write(f"Match threshold: {threshold:.2f}s\n")
		f.write("==========================================\n\n")
		_write_cost_matrix_table(f, audio_speakers, video_speakers, cost)

		if len(audio_speakers) == 0 or len(video_speakers) == 0:
			f.write("[WARN] Empty speaker set. No mapping possible.\n")
			return a_to_v, cost, audio_speakers, video_speakers

		row_ind, col_ind = linear_sum_assignment(-cost)
		matched_audio = set()
		matched_video = set()

		f.write("[MATCHES]\n")
		for r, c in zip(row_ind, col_ind):
			overlap = float(cost[r, c])
			a_spk = audio_speakers[r]
			v_spk = video_speakers[c]
			if overlap >= threshold:
				a_to_v[a_spk] = v_spk
				matched_audio.add(a_spk)
				matched_video.add(v_spk)
				f.write(f"  [+] {a_spk:<15} <-> {v_spk:<15} | overlap={overlap:.2f}s\n")
			else:
				f.write(f"  [REJECTED] {a_spk:<15} <-> {v_spk:<15} | overlap={overlap:.2f}s\n")

		f.write("\n[UNMAPPED AUDIO SPEAKERS]\n")
		unmapped_a = sorted(set(audio_speakers) - matched_audio)
		if not unmapped_a:
			f.write("  None\n")
		for a_spk in unmapped_a:
			idx = audio_speakers.index(a_spk)
			best_j = int(np.argmax(cost[idx]))
			best_overlap = float(cost[idx, best_j])
			best_v = video_speakers[best_j] if best_overlap > 0 else "N/A"
			f.write(f"  [-] {a_spk:<15} | best_overlap={best_overlap:.2f}s with {best_v}\n")

		f.write("\n[UNMAPPED VIDEO SPEAKERS]\n")
		unmapped_v = sorted(set(video_speakers) - matched_video)
		if not unmapped_v:
			f.write("  None\n")
		for v_spk in unmapped_v:
			idx = video_speakers.index(v_spk)
			best_i = int(np.argmax(cost[:, idx]))
			best_overlap = float(cost[best_i, idx])
			best_a = audio_speakers[best_i] if best_overlap > 0 else "N/A"
			f.write(f"  [-] {v_spk:<15} | best_overlap={best_overlap:.2f}s with {best_a}\n")

	return a_to_v, cost, audio_speakers, video_speakers


def map_video_to_audio_soft(audio_segs, video_segs, min_overlap_sec=0.5):
	"""Soft many-to-many mapping: assign each VIDEO (face) speaker to the AUDIO
	speaker it overlaps with the most.

	Face clustering often over-segments one person into several face clusters; a
	1-to-1 Hungarian assignment cannot represent that. Mapping video→audio by
	argmax overlap lets *several* face clusters collapse onto the same audio
	speaker, which is what we actually need for relabel decisions.

	Returns: dict {video_speaker: audio_speaker} (only confident links).
	"""
	audio_speakers = sorted({s["speaker"] for s in audio_segs})
	video_speakers = sorted({s["speaker"] for s in video_segs})
	cost = build_overlap_matrix(audio_segs, video_segs, audio_speakers, video_speakers)

	v_to_a = {}
	if cost.size == 0:
		return v_to_a
	for j, v_spk in enumerate(video_speakers):
		col = cost[:, j]
		best_i = int(np.argmax(col))
		if float(col[best_i]) >= min_overlap_sec:
			v_to_a[v_spk] = audio_speakers[best_i]
	return v_to_a


def compute_video_overlap_in_window(video_segs, start, end):
	"""Return per-video-speaker overlap seconds with [start, end]."""
	out = defaultdict(float)
	for seg in video_segs:
		overlap = max(0.0, min(end, seg["end"]) - max(start, seg["start"]))
		if overlap > 0:
			out[seg["speaker"]] += overlap
	return dict(out)


def apply_sd_primary_anomaly_supplement(
	audio_segs,
	video_segs,
	v_to_a,
	relabel_main_max_sec=0.25,
	relabel_other_min_sec=0.8,
	relabel_dom_ratio=0.65,
):
	"""SD-first supplement strategy — **label refinement only, never deletion**.

	The audio diarization (P1) is the authority on *where* speech is; the visual
	stream is only used to *correct the speaker label* of a segment, never to
	remove it. Every audio segment is emitted exactly once:
	  - RELABEL : strong, consistent visual evidence that a *different* audio
	              speaker (via the soft video→audio map) dominates this segment
	              while this speaker's own faces are essentially inactive.
	  - KEEP    : everything else (default).

	`v_to_a` maps each video (face) cluster to the audio speaker it overlaps most
	(see map_video_to_audio_soft), so several face clusters can back one audio
	speaker. Overlaps are aggregated per *audio* speaker, not per face cluster.
	"""
	final_segments = []
	decisions = []

	for seg in audio_segs:
		start = seg["start"]
		end = seg["end"]
		a_spk = seg["speaker"]
		duration = end - start
		if duration <= 0:
			continue

		overlap_map = compute_video_overlap_in_window(video_segs, start, end)

		# Aggregate face overlaps by the audio speaker each face maps to.
		own_overlap = 0.0
		other_by_audio = defaultdict(float)
		for v_spk, t in overlap_map.items():
			mapped = v_to_a.get(v_spk)
			if mapped is None:
				continue
			if mapped == a_spk:
				own_overlap += t
			else:
				other_by_audio[mapped] += t

		if not other_by_audio:
			final_segments.append(dict(seg))
			decisions.append({
				"start": start, "end": end, "sd_speaker": a_spk, "decision": "KEEP",
				"reason": "No competing mapped face speaker inside this SD segment.",
			})
			continue

		best_other_audio, best_other_overlap = max(other_by_audio.items(), key=lambda x: x[1])

		# Relabel only when the evidence is strong AND consistent:
		#  - this speaker's own faces are essentially inactive in the window,
		#  - another audio speaker's faces dominate both in absolute time and
		#    as a fraction of the segment.
		relabel_confident = (
			best_other_audio != a_spk
			and own_overlap <= relabel_main_max_sec
			and best_other_overlap >= relabel_other_min_sec
			and (best_other_overlap / duration) >= relabel_dom_ratio
		)

		if relabel_confident:
			final_segments.append({"start": start, "end": end, "speaker": best_other_audio})
			decisions.append({
				"start": start, "end": end, "sd_speaker": a_spk, "decision": "RELABEL",
				"new_speaker": best_other_audio,
				"reason": (
					f"Own faces inactive ({own_overlap:.2f}s) while {best_other_audio} faces "
					f"dominate ({best_other_overlap:.2f}s / {duration:.2f}s)."
				),
			})
		else:
			final_segments.append(dict(seg))
			decisions.append({
				"start": start, "end": end, "sd_speaker": a_spk, "decision": "KEEP",
				"reason": (
					f"Competing visual overlap ({best_other_overlap:.2f}s) not strong/consistent "
					f"enough vs own ({own_overlap:.2f}s) — keeping audio label (never deleting speech)."
				),
			})

	return final_segments, decisions


def merge_contiguous_segments(segments, gap_tolerance=0.1):
	if not segments:
		return []
	segs = sorted(segments, key=lambda x: (x["start"], x["speaker"]))
	merged = [dict(segs[0])]
	for seg in segs[1:]:
		prev = merged[-1]
		if seg["speaker"] == prev["speaker"] and seg["start"] <= prev["end"] + gap_tolerance:
			prev["end"] = max(prev["end"], seg["end"])
		else:
			merged.append(dict(seg))
	return merged


def write_anomaly_report(report_path, decisions):
	kept = sum(1 for d in decisions if d["decision"] == "KEEP")
	relabeled = sum(1 for d in decisions if d["decision"] == "RELABEL")

	with open(report_path, "w") as f:
		f.write("=== SD-FIRST SUPPLEMENT REPORT (relabel-only, no deletion) ===\n")
		f.write(f"Total SD segments evaluated: {len(decisions)}\n")
		f.write(f"KEEP: {kept}\n")
		f.write(f"RELABEL: {relabeled}\n")
		f.write("==========================================\n\n")

		for d in decisions:
			base = f"[{d['decision']}] {d['start']:.3f} {d['end']:.3f} {d['sd_speaker']}"
			if d["decision"] == "RELABEL":
				base += f" -> {d['new_speaker']}"
			f.write(base + "\n")
			f.write(f"  Reason: {d['reason']}\n")


def run_loconet_asd(video_path, clustered_identities, output_txt_path, loconet_root, 
                     max_frames=1000, inference_stride=5, min_segment_sec=0.2):
	"""
	Run LoCoNet for Active Speaker Detection and generate diarization file.
	
	Args:
		video_path: Path to input video
		clustered_identities: Dict from Phase 3 with speaker cluster assignments
		output_txt_path: Path to write diarization output
		loconet_root: Path to LoCoNet directory
		max_frames: Max frames to process
		inference_stride: Frame stride for inference
		min_segment_sec: Minimum segment duration
	"""
	is_dir = os.path.isdir(video_path)
	if is_dir:
		video_path_for_loconet = video_path
	else:
		video_path_for_loconet = video_path

	loconet_output_dir = os.path.join(
		os.path.dirname(output_txt_path),
		"loconet_output"
	)
	os.makedirs(loconet_output_dir, exist_ok=True)

	print(f"[LoCoNet] Running ASD with video: {video_path_for_loconet}")
	print(f"[LoCoNet] Output directory: {loconet_output_dir}")

	# Check if corresponding audio file exists, otherwise extract from video
	base_name = os.path.basename(video_path_for_loconet)
	name, _ = os.path.splitext(base_name)
	audio_base_path = os.path.join(
		os.path.dirname(os.path.dirname(video_path_for_loconet)),
		"audio",
		f"{name}.wav"
	)
	audio_path = os.path.join(loconet_output_dir, "audio.wav")

	if os.path.exists(audio_base_path):
		print(f"[LoCoNet] Found pre-existing audio: {audio_base_path}")
		shutil.copy(audio_base_path, audio_path)
		print(f"[LoCoNet] Audio copied to: {audio_path}")
	else:
		print(f"[LoCoNet] Extracting audio from video...")
		cmd_audio = [
			"ffmpeg", "-y", "-i", video_path_for_loconet,
			"-vn", "-acodec", "pcm_s16le",
			audio_path
		]
		result = subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		if result.returncode != 0:
			error_msg = result.stderr if result.stderr else "Unknown error"
			raise RuntimeError(f"Failed to extract audio from video.\nFFmpeg error: {error_msg}")
		print(f"[LoCoNet] Audio extracted to: {audio_path}")

	# Convert audio path to absolute path (LoCoNet runs from loconet_root directory)
	audio_path = os.path.abspath(audio_path)
	# Also convert video and output to absolute paths BEFORE changing directory
	video_path_abs = os.path.abspath(video_path_for_loconet)
	loconet_output_dir_abs = os.path.abspath(loconet_output_dir)

	cwd = os.getcwd()
	try:
		os.chdir(loconet_root)
		cmd = [
			"python",
			"process_single_video.py",
			"--video", video_path_abs,
			"--output", loconet_output_dir_abs,
			"--mode", "both",
			"--max-frames", str(max_frames),
			"--inference-stride", str(inference_stride),
			"--resume-path", str(paths.LOCONET_MODEL),
		]
		ret = subprocess.call(cmd)
		if ret != 0:
			raise RuntimeError(f"LoCoNet process_single_video.py failed with code {ret}")
	finally:
		os.chdir(cwd)

	# Parse LoCoNet output and generate diarization file
	# Assuming LoCoNet outputs speaker predictions in JSON or similar format
	speaker_predictions = parse_loconet_output(loconet_output_dir, clustered_identities)

	# Write diarization file
	with open(output_txt_path, "w") as f:
		for pred in speaker_predictions:
			start = pred["start"]
			end = pred["end"]
			speaker_id = pred["speaker_id"]
			duration = end - start
			if duration >= min_segment_sec:
				f.write(f"{start:.3f} {end:.3f} {speaker_id}\n")

	print(f"[LoCoNet] Diarization saved to {output_txt_path}")


def parse_loconet_output(loconet_output_dir, clustered_identities):
	"""
	Parse LoCoNet output and convert to diarization format.
	
	LoCoNet outputs scores.json with frame-level speaker predictions.
	This function extracts and segments by speaker ID.
	
	Returns: List of dicts with 'start', 'end', 'speaker_id' keys.
	"""

	# LoCoNet outputs scores.json or tracks.json
	possible_outputs = [
		os.path.join(loconet_output_dir, "scores.json"),
		os.path.join(loconet_output_dir, "tracks.json"),
		os.path.join(loconet_output_dir, "predictions.json"),
		os.path.join(loconet_output_dir, "output.json"),
		os.path.join(loconet_output_dir, "speakers.json"),
	]

	json_file = None
	for fpath in possible_outputs:
		if os.path.exists(fpath):
			json_file = fpath
			break

	if json_file is None:
		raise FileNotFoundError(
			f"Could not find LoCoNet output JSON in {loconet_output_dir}. "
			f"Checked: {possible_outputs}"
		)

	print(f"[LoCoNet] Loading predictions from {json_file}")
	with open(json_file, "r") as f:
		loconet_data = json.load(f)

	# Extract frames/predictions data based on format
	# scores.json format: list of speaker IDs per frame, or dict with "predictions" key
	# tracks.json format: dict with track information
	
	if isinstance(loconet_data, list):
		# Direct list of speaker IDs
		frames_data = loconet_data
		fps = 25.0
	elif isinstance(loconet_data, dict):
		# Try different key names
		if "predictions" in loconet_data:
			frames_data = loconet_data["predictions"]
			fps = loconet_data.get("fps", 25.0)
		elif "frames" in loconet_data:
			frames_data = loconet_data["frames"]
			fps = loconet_data.get("fps", 25.0)
		elif "scores" in loconet_data:
			frames_data = loconet_data["scores"]
			fps = loconet_data.get("fps", 25.0)
		else:
			# Assume whole dict is frames data
			frames_data = loconet_data
			fps = 25.0
	else:
		frames_data = loconet_data
		fps = 25.0

	# Map frame indices to speaker IDs
	speaker_segments = []
	if frames_data:
		current_speaker = None
		segment_start_frame = None

		for frame_idx, frame_data in enumerate(frames_data):
			if isinstance(frame_data, dict):
				speaker_id = frame_data.get("speaker_id") or frame_data.get("speaker") or frame_data.get("label")
			else:
				# Assume it's a numeric speaker ID directly
				speaker_id = frame_data

			# Skip None or -1 (no speaker/background)
			if speaker_id is None or speaker_id == -1:
				if current_speaker is not None and segment_start_frame is not None:
					start_time = segment_start_frame / fps
					end_time = frame_idx / fps
					speaker_label = _map_speaker_id_to_label(current_speaker, clustered_identities)
					speaker_segments.append({
						"start": start_time,
						"end": end_time,
						"speaker_id": speaker_label,
					})
					current_speaker = None
					segment_start_frame = None
				continue

			if speaker_id != current_speaker:
				if current_speaker is not None and segment_start_frame is not None:
					start_time = segment_start_frame / fps
					end_time = frame_idx / fps
					speaker_label = _map_speaker_id_to_label(current_speaker, clustered_identities)
					speaker_segments.append({
						"start": start_time,
						"end": end_time,
						"speaker_id": speaker_label,
					})
			current_speaker = speaker_id
			if segment_start_frame is None:
				segment_start_frame = frame_idx

		# Handle last segment
		if current_speaker is not None and segment_start_frame is not None:
			start_time = segment_start_frame / fps
			end_time = len(frames_data) / fps
			speaker_label = _map_speaker_id_to_label(current_speaker, clustered_identities)
			speaker_segments.append({
				"start": start_time,
				"end": end_time,
				"speaker_id": speaker_label,
			})

	return speaker_segments

def _map_speaker_id_to_label(speaker_id, clustered_identities):
	"""
	Map LoCoNet's numeric speaker ID to cluster label (SPEAKER_XX format).
	
	Args:
		speaker_id: Numeric ID from LoCoNet
		clustered_identities: Dict from Phase 3 clustering
	
	Returns: Speaker label string (e.g., 'SPEAKER_04')
	"""
	# If clustered_identities is available, use it to map speaker indices
	if clustered_identities and isinstance(clustered_identities, dict):
		# Convert speaker_id to int if it's not already
		if isinstance(speaker_id, str):
			try:
				speaker_idx = int(speaker_id)
			except (ValueError, TypeError):
				return str(speaker_id)
		else:
			speaker_idx = speaker_id

		# Check if this index exists in clusters
		if "clusters" in clustered_identities:
			clusters_list = clustered_identities["clusters"]
			if speaker_idx < len(clusters_list):
				# Format: SPEAKER_XX where XX is 2-digit padded index
				return f"SPEAKER_{speaker_idx:02d}"
			elif speaker_idx == -1 or speaker_id is None:
				return "SPEAKER_UNK"

	# Fallback: format as SPEAKER_XX
	try:
		spk_idx = int(speaker_id) if not isinstance(speaker_id, int) else speaker_id
		return f"SPEAKER_{spk_idx:02d}"
	except (ValueError, TypeError):
		return f"SPEAKER_{speaker_id}"

def run_face_pipeline_for_video_diarization(args, skip_phase_0_2=False, tracks_dir=None):
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

	if skip_phase_0_2:
		# Skip phases 0-2, assume tracks already exist
		print("\n[SKIPPED] Phases 0-2 (using pre-computed tracks)")
		# Reconstruct standardized video path
		base_name = os.path.basename(args.video_path)
		name, ext = os.path.splitext(base_name)
		standardized_video = os.path.join(args.out_dir, f"{name}_25fps_hq.mp4")
		if not os.path.exists(standardized_video):
			raise FileNotFoundError(f"Standardized video not found: {standardized_video}. Cannot skip phases 0-2.")
		print(f"Using pre-processed video: {standardized_video}")
		# Load pipeline state to restore track_history and embeddings
		pipeline._load_pipeline_state()
		# Extract embeddings from restored tracks
		pipeline.phase_2_extract_uniform_embeddings(num_samples=args.embedding_samples)
	else:
		standardized_video = pipeline.phase_0_preprocess_fps(args.video_path) # Phase 0: Chuẩn hóa FPS video
		pipeline.phase_1_track_and_crop(standardized_video) # Phase 1: Track + crop khuôn mặt
		pipeline.phase_2_extract_uniform_embeddings(num_samples=args.embedding_samples) # Phase 2: Trích xuất face embeddings
	
	clusters = pipeline.phase_3_cluster_identities(
		distance_threshold=args.cluster_threshold,
		merge_threshold=getattr(args, "cluster_merge_threshold", 0.0),
	) # Phase 3: Cluster identity (ai là ai) + optional re-ID merge across scene cuts

	video_diarization = args.video_diarization_output 
	if video_diarization is None:
		video_diarization = os.path.join(args.out_dir, "video_diarization.txt")

	# Phase 4: Active Speaker Detection → ra file diarization
	if args.audio_visual_model == "loconet":
		print("\n[Phase 4] Using LoCoNet for Active Speaker Detection")
		run_loconet_asd(
			video_path=standardized_video,
			clustered_identities=clusters,
			output_txt_path=video_diarization,
			loconet_root=args.loconet_root,
			max_frames=args.loconet_max_frames,
			inference_stride=args.loconet_inference_stride,
			min_segment_sec=args.min_segment_sec,
		)
	else:
		print("\n[Phase 4] Using LR-ASD for Active Speaker Detection")
		pipeline.phase_4_active_speaker_detection( 
			video_path=standardized_video,
			clustered_identities=clusters,
			lrasd_weights=args.lrasd_model,
			output_txt_path=video_diarization,
			asd_threshold=args.asd_threshold,
			min_segment_sec=args.min_segment_sec,
			allow_overlap=args.allow_overlap,
			annotated_mp4_path=args.annotated_mp4,
		)
	return video_diarization


def main():
	parser = argparse.ArgumentParser(
		description="Run face pipeline as SD supplement (SD-first anomaly detection)."
	)
	parser.add_argument("--video_path", required=True, help="Path to input video file")
	parser.add_argument("--sd_diarization", required=True,
						help="Path to Pipeline 1 diarization output TXT")

	# Face pipeline models
	parser.add_argument("--detector_model",
		default=str(paths.SCRFD_DEFAULT))
	parser.add_argument("--arcface_model",
		default=str(paths.ARCFACE_MODEL))
	parser.add_argument("--model", choices=["lr_asd", "loconet"], default="lr_asd",
					dest="audio_visual_model",
					help="Active Speaker Detection model: 'lr_asd' (LR-ASD) or 'loconet' (LoCoNet)")
	parser.add_argument("--lrasd_model",
		default=str(paths.LRASD_MODEL),
					help="Path to LR-ASD model")
	parser.add_argument("--loconet_root",
		default=os.path.join(_HERE, "audio_visual_model", "LoCoNet_ASD"),
					help="Path to LoCoNet root directory")

	# Face pipeline setup
	parser.add_argument("--out_dir", default=os.path.join(_ROOT, "data", "audio_visual"))
	parser.add_argument("--det_input_size", type=int, default=ORIGINAL_FACE_SETUP["det_input_size"])
	parser.add_argument("--det_thresh", type=float, default=ORIGINAL_FACE_SETUP["det_thresh"])
	parser.add_argument("--tracker_max_age", type=int, default=ORIGINAL_FACE_SETUP["tracker_max_age"])
	parser.add_argument("--tracker_min_hits", type=int, default=ORIGINAL_FACE_SETUP["tracker_min_hits"])
	parser.add_argument("--tracker_iou_thresh", type=float, default=ORIGINAL_FACE_SETUP["tracker_iou_thresh"])
	parser.add_argument("--scene_cut_threshold", type=float, default=ORIGINAL_FACE_SETUP["scene_cut_threshold"])
	parser.add_argument("--scene_cut_debounce", type=int, default=ORIGINAL_FACE_SETUP["scene_cut_debounce"])
	parser.add_argument("--cluster_threshold", type=float, default=0.7)
	parser.add_argument("--cluster_merge_threshold", type=float, default=0.0,
					help="If >0, merge face clusters whose centroids are within this cosine "
						 "distance (re-ID across scene cuts). 0 disables (default).")
	parser.add_argument("--asd_threshold", type=float, default=0.0,
					help="Confidence threshold for LR-ASD speaking decision (LR-ASD operating "
						 "point is 0.0; raise for higher visual precision).")
	parser.add_argument("--min_segment_sec", type=float, default=0.2,
					help="Minimum segment duration")
	parser.add_argument("--allow_overlap", dest="allow_overlap", action="store_true", default=True,
					help="Keep overlapping diarization segments from face ASD.")
	parser.add_argument("--no_allow_overlap", dest="allow_overlap", action="store_false",
					help="Drop overlap and keep only single-speaker regions from face ASD.")
	parser.add_argument("--embedding_samples", type=int, default=5)
	parser.add_argument("--annotated_mp4", default=None)
	# LoCoNet-specific parameters
	parser.add_argument("--loconet_max_frames", type=int, default=1000,
					help="Max frames to process in LoCoNet")
	parser.add_argument("--loconet_inference_stride", type=int, default=5,
					help="Inference stride for LoCoNet")
	parser.add_argument("--use_original_face_setup", action="store_true")
	parser.add_argument("--skip-phase", type=str, default=None,
					help="Skip phases (e.g., '0-3' to skip phases 0,1,2,3). Requires pre-computed results.")

	# Mapping + supplement behavior
	parser.add_argument("--mapping_threshold", type=float, default=1.5)
	parser.add_argument("--pollution_min_sec", type=float, default=0.3,
						help="Other-speaker involvement threshold in SD segment.")
	parser.add_argument("--anomaly_margin_sec", type=float, default=0.15,
						help="Min(overlap_other - overlap_mapped) to trust anomaly.")
	parser.add_argument("--relabel_main_max_sec", type=float, default=0.25)
	parser.add_argument("--relabel_other_min_sec", type=float, default=0.8)
	parser.add_argument("--relabel_dom_ratio", type=float, default=0.65)

	args = parser.parse_args()

	# Derive output paths from out_dir and SAMPLE_KEY (extracted from video_path basename)
	SAMPLE_KEY = os.path.splitext(os.path.basename(args.video_path))[0]
	os.makedirs(args.out_dir, exist_ok=True)
	os.makedirs(os.path.join(args.out_dir, SAMPLE_KEY), exist_ok=True)

	# Auto-compute output file paths
	args.video_diarization_output = os.path.join(args.out_dir, SAMPLE_KEY, "video_diarization.txt")
	args.output_fused = os.path.join(args.out_dir, SAMPLE_KEY, "supplemented_diarization.txt")
	args.mapping_report = os.path.join(args.out_dir, SAMPLE_KEY, "mapping_analysis.txt")
	args.anomaly_report = os.path.join(args.out_dir, SAMPLE_KEY, "supplement_anomaly_report.txt")

	if args.use_original_face_setup:
		args.det_input_size = ORIGINAL_FACE_SETUP["det_input_size"]
		args.det_thresh = ORIGINAL_FACE_SETUP["det_thresh"]
		args.tracker_max_age = ORIGINAL_FACE_SETUP["tracker_max_age"]
		args.tracker_min_hits = ORIGINAL_FACE_SETUP["tracker_min_hits"]
		args.tracker_iou_thresh = ORIGINAL_FACE_SETUP["tracker_iou_thresh"]
		args.scene_cut_threshold = ORIGINAL_FACE_SETUP["scene_cut_threshold"]
		args.scene_cut_debounce = ORIGINAL_FACE_SETUP["scene_cut_debounce"]

	# Parse skip-phase argument
	skip_phases = set()
	if args.skip_phase:
		parts = args.skip_phase.split('-')
		if len(parts) == 2:
			try:
				start = int(parts[0])
				end = int(parts[1])
				skip_phases = set(range(start, end + 1))
			except ValueError:
				print(f"[WARN] Invalid skip-phase format: {args.skip_phase}. Expected '0-3' or similar.")
		else:
			print(f"[WARN] Invalid skip-phase format: {args.skip_phase}. Expected '0-3' or similar.")
	
	print("\n=== STEP 1: Run face pipeline to get visual diarization ===")
	if 3 in skip_phases:
		# Skip phase 3 means skip entire face pipeline and use pre-computed video diarization
		if os.path.exists(args.video_diarization_output):
			video_diarization = args.video_diarization_output
			print(f"[SKIPPED] Phases 0-3 skipped. Loading video diarization from: {video_diarization}")
		else:
			print(f"[WARN] Video diarization file not found: {args.video_diarization_output}")
			print(f"[FALLBACK] Running phases 0-3 instead of skipping.")
			video_diarization = run_face_pipeline_for_video_diarization(args) # Chạy toàn bộ baseline nhận diện khuôn mặt qua 4 phase
																			  # Phase 0: Chuẩn hóa FPS video
																			  # Phase 1: Track + crop khuôn mặt
																			  # Phase 2: Trích xuất face embeddings
																			  # Phase 3: Cluster identity (ai là ai)
																			  # Phase 4: Active Speaker Detection → ra file diarization
	elif 0 in skip_phases or 1 in skip_phases or 2 in skip_phases:
		# Skip phase 0-2, check if results already exist (tracks folder)
		tracks_dir = os.path.join(args.out_dir, "tracks")
		if os.path.exists(tracks_dir) and len(os.listdir(tracks_dir)) > 0:
			print(f"[SKIPPED] Phases 0-2 detected as completed (found tracks in {tracks_dir})")
			print(f"[INFO] Running only phase 3 (clustering) and phase 4 (ASD)...")
			# Only run phase 3 and 4, skip 0-2
			video_diarization = run_face_pipeline_for_video_diarization(args, skip_phase_0_2=True, tracks_dir=tracks_dir)
		else:
			print(f"[INFO] First run detected - tracks folder not found.")
			print(f"[INFO] Running phase 0-2 to generate tracks (this will be reused on next run)...")
			print(f"[INFO] Next time you run with --skip-phase 0-2, phases 0-2 will be skipped.")
			video_diarization = run_face_pipeline_for_video_diarization(args)
	else:
		# If phase 3 is NOT skipped, run the entire face pipeline (phases 0-3 and phase 4)
		# Even if 0-2 are in skip_phases, we still run them because phase 3 needs their outputs
		video_diarization = run_face_pipeline_for_video_diarization(args) # Chạy toàn bộ baseline nhận diện khuôn mặt qua 4 phase
																		  # Phase 0: Chuẩn hóa FPS video
																		  # Phase 1: Track + crop khuôn mặt
																		  # Phase 2: Trích xuất face embeddings
																		  # Phase 3: Cluster identity (ai là ai)
																		  # Phase 4: Active Speaker Detection → ra file diarization
 
	print(f"Visual diarization saved at: {video_diarization}")

	print("\n=== STEP 2: Build SD<->Face mapping with overlap matrix ===")
	sd_segments = parse_diarization_file(args.sd_diarization)
	video_segments = parse_diarization_file(video_diarization)

	# Report uses the (legacy) Hungarian 1-1 view for human inspection...
	map_audio_to_video(
		audio_segs=sd_segments,
		video_segs=video_segments,
		threshold=args.mapping_threshold,
		output_report=args.mapping_report,
		audio_name=os.path.basename(args.sd_diarization),
		video_name=os.path.basename(video_diarization),
	)
	# ...but the supplement logic uses the soft many-to-many video→audio map.
	v_to_a = map_video_to_audio_soft(
		audio_segs=sd_segments,
		video_segs=video_segments,
		min_overlap_sec=args.mapping_threshold,
	)
	print(f"Mapping report written to: {args.mapping_report}")
	print(f"Soft video→audio map ({len(v_to_a)} links): {v_to_a}")

	print("\n=== STEP 3: SD-primary supplement (relabel-only, never deletes speech) ===")
	supplemented, decisions = apply_sd_primary_anomaly_supplement(
		audio_segs=sd_segments,
		video_segs=video_segments,
		v_to_a=v_to_a,
		relabel_main_max_sec=args.relabel_main_max_sec,
		relabel_other_min_sec=args.relabel_other_min_sec,
		relabel_dom_ratio=args.relabel_dom_ratio,
	)
	supplemented = merge_contiguous_segments(supplemented)

	write_diarization_file(args.output_fused, supplemented)
	write_anomaly_report(args.anomaly_report, decisions)

	print(f"Supplemented diarization written to: {args.output_fused}")
	print(f"Anomaly report written to: {args.anomaly_report}")
	print("Done.")


if __name__ == "__main__":
	main()
