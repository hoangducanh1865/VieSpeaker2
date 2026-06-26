#!/usr/bin/env python3

import argparse
import json
import subprocess
import time
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict
from scipy.io import wavfile

from loss_multi import lossAV
from model.loconet_encoder import locoencoder
from torchvggish import vggish_input


def to_easydict(value):
    if isinstance(value, dict):
        return EasyDict({key: to_easydict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_easydict(item) for item in value]
    return value


def load_cfg(cfg_path: Path, data_path_ava: Optional[Path] = None):
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = to_easydict(yaml.safe_load(handle))
    if data_path_ava is not None:
        cfg.DATA.dataPathAVA = str(data_path_ava)
    return cfg


def load_detector(proto_path: Path, weight_path: Path):
    return cv2.dnn.readNetFromCaffe(str(proto_path), str(weight_path))


def detect_faces(net, image, conf_th: float = 0.8):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(640, 640),
        mean=(104.0, 117.0, 123.0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    output = net.forward("detection_out")

    bboxes = np.empty((0, 5), dtype=np.float32)
    for det in output[0, 0]:
        score = float(det[2])
        if score < conf_th:
            continue
        x1 = det[3] * width
        y1 = det[4] * height
        x2 = det[5] * width
        y2 = det[6] * height
        bboxes = np.vstack((bboxes, (x1, y1, x2, y2, score)))

    return bboxes


def extract_audio(video_path: Path, audio_path: Path):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()

        duration_seconds = frame_count / fps if fps > 0 else 0.0
        sample_rate = 16000
        sample_count = max(1, int(round(duration_seconds * sample_rate)))
        silent_audio = np.zeros(sample_count, dtype=np.int16)
        wavfile.write(str(audio_path), sample_rate, silent_audio)
        print(f"[warn] ffmpeg not found; wrote silent audio to {audio_path}", flush=True)
        return

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def clamp_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def serialize_track_db(track_db):
    payload = {"tracks": {}, "frames": {}}
    for track_id, track in track_db["tracks"].items():
        payload["tracks"][str(track_id)] = {
            "last_box": track["last_box"],
            "last_seen": track["last_seen"],
            "frames": {str(frame_idx): item for frame_idx, item in track["frames"].items()},
        }
    for frame_idx, items in track_db["frames"].items():
        payload["frames"][str(frame_idx)] = items
    return payload


def load_track_db(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    track_db = {"tracks": {}, "frames": {}}
    for track_id, track in raw["tracks"].items():
        track_db["tracks"][int(track_id)] = {
            "last_box": track["last_box"],
            "last_seen": track["last_seen"],
            "frames": {int(frame_idx): item for frame_idx, item in track["frames"].items()},
        }
    for frame_idx, items in raw["frames"].items():
        track_db["frames"][int(frame_idx)] = items
    return track_db


def preprocess_video(video_path: Path, detector, output_dir: Path, args):
    face_root = ensure_dir(output_dir / "faces")
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)

    track_db = {"tracks": {}, "frames": {}}
    active_tracks = {}
    next_track_id = 0
    started_at = time.monotonic()
    frame_idx = 0

    print(f"[start] video={video_path} total_frames={total_frames} fps={fps:.2f}", flush=True)

    while True:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        ok, frame = capture.read()
        if not ok:
            break

        boxes = detect_faces(detector, frame, conf_th=args.conf_th)
        detections = []

        if len(boxes) > 0:
            order = np.argsort(-((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
            boxes = boxes[order]

            used_track_ids = set()
            for det in boxes:
                bbox = [float(det[0]), float(det[1]), float(det[2]), float(det[3])]
                best_track_id = None
                best_iou = 0.0
                for track_id, track_state in active_tracks.items():
                    if track_id in used_track_ids:
                        continue
                    score = iou(bbox, track_state["last_box"])
                    if score > best_iou:
                        best_iou = score
                        best_track_id = track_id

                if best_track_id is not None and best_iou >= args.track_iou_th:
                    track_id = best_track_id
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    track_db["tracks"][track_id] = {"last_box": bbox, "last_seen": frame_idx, "frames": {}}

                used_track_ids.add(track_id)
                x1, y1, x2, y2 = clamp_box(bbox, frame.shape[1], frame.shape[0])
                crop = frame[y1:y2, x1:x2]
                crop_path = None
                if crop.size > 0:
                    track_face_dir = ensure_dir(face_root / f"track_{track_id:04d}")
                    crop_path = track_face_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(crop_path), crop)

                det_item = {
                    "track_id": int(track_id),
                    "bbox": bbox,
                    "score": float(det[4]),
                    "crop": str(crop_path) if crop_path is not None else None,
                }
                detections.append(det_item)

                track_state = track_db["tracks"].setdefault(track_id, {"last_box": bbox, "last_seen": frame_idx, "frames": {}})
                track_state["last_box"] = bbox
                track_state["last_seen"] = frame_idx
                track_state["frames"][frame_idx] = {"bbox": bbox, "crop": det_item["crop"]}

            active_tracks = {
                track_id: track_state
                for track_id, track_state in track_db["tracks"].items()
                if frame_idx - track_state["last_seen"] <= args.max_track_gap
            }

        track_db["frames"][frame_idx] = detections

        if frame_idx % args.progress_every == 0 or frame_idx + 1 == total_frames:
            elapsed = time.monotonic() - started_at
            rate = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"[preprocess] {frame_idx + 1}/{total_frames or '?'} frames, tracks={len(track_db['tracks'])}, elapsed={elapsed:.1f}s, {rate:.2f} fps",
                flush=True,
            )

        frame_idx += 1

    capture.release()
    return track_db, fps, total_frames, frame_idx


def load_loconet(cfg, checkpoint_path: Path, device):
    encoder = locoencoder(cfg).to(device)
    speaker_head = lossAV().to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    encoder_state = {}
    speaker_state = {}

    for key, value in state_dict.items():
        if key.startswith("model.module.model."):
            encoder_state[key.replace("model.module.model.", "", 1)] = value
        elif key.startswith("model.module.lossAV."):
            speaker_state[key.replace("model.module.lossAV.", "", 1)] = value
        elif key.startswith("model.module.model"):
            encoder_state[key.replace("model.module.", "", 1)] = value
        elif key.startswith("model.module.lossAV"):
            speaker_state[key.replace("model.module.", "", 1)] = value
        elif key.startswith("model."):
            encoder_state[key.replace("model.", "", 1)] = value
        elif key.startswith("lossAV."):
            speaker_state[key.replace("lossAV.", "", 1)] = value

    encoder.load_state_dict(encoder_state, strict=False)
    speaker_head.load_state_dict(speaker_state, strict=False)
    encoder.eval()
    speaker_head.eval()
    return encoder, speaker_head


def build_audio_features(audio_waveform, audio_sr, fps, frame_indices):
    if len(frame_indices) == 0:
        return np.zeros((4, 64), dtype=np.float32)

    start_frame = frame_indices[0]
    end_frame = frame_indices[-1] + 1
    start_sample = max(0, int(round((start_frame / fps) * audio_sr)))
    end_sample = min(len(audio_waveform), int(round((end_frame / fps) * audio_sr)))
    segment = audio_waveform[start_sample:end_sample]
    if segment.size == 0:
        segment = audio_waveform[: max(1, int(round(audio_sr / fps)))]
    return vggish_input.waveform_to_examples(segment, audio_sr, len(frame_indices), fps, return_tensor=False)


def read_face_crop(crop_path: Optional[str]):
    if not crop_path:
        return np.zeros((112, 112), dtype=np.uint8)
    image = cv2.imread(crop_path)
    if image is None:
        return np.zeros((112, 112), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (112, 112))
    return image


def build_visual_features(track_db, frame_indices, candidate_track_ids):
    visual_features = []
    for track_id in candidate_track_ids:
        track = track_db["tracks"].get(track_id, {"frames": {}})
        frame_features = []
        for frame_idx in frame_indices:
            item = track["frames"].get(frame_idx)
            crop_path = item["crop"] if item else None
            frame_features.append(read_face_crop(crop_path))
        visual_features.append(np.asarray(frame_features))
    return np.asarray(visual_features)


def score_target_track(encoder, speaker_head, device, audio_waveform, audio_sr, fps, track_db, frame_idx, candidate_track_ids, window_frames):
    half_window = max(1, window_frames // 2)
    last_frame = max(track_db["frames"].keys(), default=frame_idx)
    start_frame = frame_idx - half_window
    frame_indices = []
    for offset in range(window_frames):
        current = start_frame + offset
        current = max(0, min(last_frame, current))
        frame_indices.append(current)

    audio_feature = build_audio_features(audio_waveform, audio_sr, fps, frame_indices)
    visual_feature = build_visual_features(track_db, frame_indices, candidate_track_ids)

    audio_tensor = torch.tensor(audio_feature, dtype=torch.float32, device=device).unsqueeze(0)
    visual_tensor = torch.tensor(visual_feature, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        b, s, t = visual_tensor.shape[:3]
        visual_flat = visual_tensor.view(b * s, *visual_tensor.shape[2:])
        audio_embed = encoder.forward_audio_frontend(audio_tensor)
        visual_embed = encoder.forward_visual_frontend(visual_flat)
        audio_embed = audio_embed.repeat(s, 1, 1)
        audio_embed, visual_embed = encoder.forward_cross_attention(audio_embed, visual_embed)
        outs_av = encoder.forward_audio_visual_backend(audio_embed, visual_embed, b, s)
        outs_av = outs_av.view(b, s, t, -1)[:, 0, :, :].contiguous().view(b * t, -1)
        logits = speaker_head.FC(outs_av)
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    center_index = min(len(probs) // 2, max(0, len(probs) - 1))
    return float(probs[center_index]) if len(probs) > 0 else 0.0


def render_overlay(args, cfg, video_path: Path, output_dir: Path, track_db, fps, total_frames):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available; running LoCoNet inference on CPU.", flush=True)

    encoder, speaker_head = load_loconet(cfg, Path(args.resume_path), device)

    audio_path = output_dir / "audio.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio waveform: {audio_path}")

    audio_sr, audio_waveform = wavfile.read(str(audio_path))
    if audio_waveform.ndim > 1:
        audio_waveform = np.mean(audio_waveform, axis=1).astype(audio_waveform.dtype)

    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    render_path = output_dir / args.render_name
    writer = cv2.VideoWriter(
        str(render_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    started_at = time.monotonic()
    frame_idx = 0
    last_scores = {}
    score_log = []

    while True:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        ok, frame = capture.read()
        if not ok:
            break

        frame_dets = track_db["frames"].get(frame_idx, [])
        visible_track_ids = [item["track_id"] for item in frame_dets]

        if visible_track_ids and frame_idx % args.inference_stride == 0:
            for target_track_id in visible_track_ids:
                others = [track_id for track_id in visible_track_ids if track_id != target_track_id]
                candidate_track_ids = [target_track_id] + others
                while len(candidate_track_ids) < cfg.MODEL.NUM_SPEAKERS:
                    candidate_track_ids.append(candidate_track_ids[-1])
                candidate_track_ids = candidate_track_ids[: cfg.MODEL.NUM_SPEAKERS]

                score = score_target_track(
                    encoder,
                    speaker_head,
                    device,
                    audio_waveform,
                    audio_sr,
                    fps,
                    track_db,
                    frame_idx,
                    candidate_track_ids,
                    window_frames=args.window_frames,
                )
                last_scores[target_track_id] = score
                score_log.append({"frame_idx": int(frame_idx), "track_id": int(target_track_id), "score": score})

        for item in frame_dets:
            track_id = item["track_id"]
            bbox = item["bbox"]
            score = last_scores.get(track_id)
            color = (0, 255, 0) if score is not None and score >= args.speaker_th else (0, 0, 255)
            x1, y1, x2, y2 = clamp_box(bbox, width, height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id}"
            if score is not None:
                label += f" {score:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if args.display:
            cv2.imshow("LoCoNet ASD", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        writer.write(frame)

        if frame_idx % args.progress_every == 0 or frame_idx + 1 == total_frames:
            elapsed = time.monotonic() - started_at
            rate = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"[infer] {frame_idx + 1}/{total_frames or '?'} frames, visible={len(visible_track_ids)}, elapsed={elapsed:.1f}s, {rate:.2f} fps",
                flush=True,
            )

        frame_idx += 1

    capture.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    (output_dir / "scores.json").write_text(json.dumps(score_log, indent=2), encoding="utf-8")
    return render_path, score_log


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Process one mp4 with detector, tracking, and LoCoNet inference.")
    parser.add_argument("--video", default=str(script_dir / "sample_0.mp4"))
    parser.add_argument("--proto", default=str(script_dir / "pretrained_model/models/VGGNet/WIDER_FACE/SFD_trained/deploy.prototxt"))
    parser.add_argument("--weights", default=str(script_dir / "pretrained_model/models/VGGNet/WIDER_FACE/SFD_trained/SFD.caffemodel"))
    parser.add_argument("--cfg", default=str(script_dir / "configs/multi.yaml"))
    parser.add_argument("--resume-path", default=str(script_dir / "pretrained_model/loconet_ava_best.model"))
    parser.add_argument("--data-path-ava", default=str(script_dir / "AVA_dataset"))
    parser.add_argument("--output", default="")
    parser.add_argument("--mode", choices=["preprocess", "infer", "both"], default="both")
    parser.add_argument("--conf-th", type=float, default=0.8)
    parser.add_argument("--speaker-th", type=float, default=0.5)
    parser.add_argument("--track-iou-th", type=float, default=0.3)
    parser.add_argument("--max-track-gap", type=int, default=2)
    parser.add_argument("--window-frames", type=int, default=200)
    parser.add_argument("--inference-stride", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=100, help="Print a progress line every N frames.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for quick testing.")
    parser.add_argument("--display", action="store_true", help="Show the annotated video in a window.")
    parser.add_argument("--render-name", default="overlay.mp4")
    args = parser.parse_args()

    video_path = Path(args.video)
    proto_path = Path(args.proto)
    weight_path = Path(args.weights)
    output_dir = Path(args.output) if args.output else video_path.with_suffix("").with_name(video_path.stem + "_processed")
    ensure_dir(output_dir)

    cfg = load_cfg(Path(args.cfg), Path(args.data_path_ava))

    audio_path = output_dir / "audio.wav"
    if args.mode in {"preprocess", "both"} or not audio_path.exists():
        extract_audio(video_path, audio_path)

    track_db = None
    total_frames = None
    fps = None

    if args.mode in {"preprocess", "both"}:
        detector = load_detector(proto_path, weight_path)
        track_db, fps, total_frames, _ = preprocess_video(video_path, detector, output_dir, args)
        (output_dir / "tracks.json").write_text(json.dumps(serialize_track_db(track_db), indent=2), encoding="utf-8")
    else:
        track_db = load_track_db(output_dir / "tracks.json")
        capture = cv2.VideoCapture(str(video_path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        capture.release()

    render_path = None
    score_log = []
    if args.mode in {"infer", "both"}:
        render_path, score_log = render_overlay(args, cfg, video_path, output_dir, track_db, fps, total_frames)

    summary = {
        "video": str(video_path),
        "weights": str(weight_path),
        "cfg": str(args.cfg),
        "resume_path": str(args.resume_path),
        "mode": args.mode,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "fps": fps,
        "total_frames": total_frames,
        "tracks_file": str(output_dir / "tracks.json") if (output_dir / "tracks.json").exists() else None,
        "scores_file": str(output_dir / "scores.json") if (output_dir / "scores.json").exists() else None,
        "render_path": str(render_path) if render_path is not None else None,
        "speaker_threshold": args.speaker_th,
        "detections": sum(len(items) for items in track_db["frames"].values()) if track_db else 0,
        "tracks": len(track_db["tracks"]) if track_db else 0,
        "score_events": len(score_log),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
