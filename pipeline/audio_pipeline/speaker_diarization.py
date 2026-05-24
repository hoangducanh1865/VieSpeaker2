"""
Speaker diarization module using pyannote.audio (Local/Offline Version)
FIXED: Handle library conflicts (DiarizeOutput vs Annotation)
"""

import os
import torch
from pathlib import Path
import warnings
import threading
import time
import sys
import torchaudio

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from pyannote.audio import Pipeline

# --- FIX CÁC CẢNH BÁO (Thêm đoạn này vào đầu file) ---

# 1. Bật tính toán TF32 (Khắc phục ReproducibilityWarning & Tăng tốc độ)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2. Ẩn cảnh báo std() vô hại (Khắc phục UserWarning pooling.py)
warnings.filterwarnings("ignore", message=".*std\(\): degrees of freedom is <= 0.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")

class SpeakerDiarizer:
    """Perform speaker diarization on audio files using local GPU/CPU"""
    
    def __init__(self, token: str, model_name: str = "pyannote/speaker-diarization-3.1", min_segment_duration: float = 1.0, max_gap_threshold: float = 2.0):
        self.token = token
        self.model_name = model_name
        self.pipeline = None
        self.min_segment_duration = min_segment_duration
        self.max_gap_threshold = max_gap_threshold
    
    def load_pipeline(self):
        if self.pipeline is None:
            print("⏳ Loading speaker diarization pipeline...")

            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.token
            )
            
            # chỉ dùng gpu nếu dùng model 3.1, model precision 2 dùng cloud
            if self.model_name == "pyannote/speaker-diarization-3.1":
                if torch.cuda.is_available():
                    self.pipeline.to(torch.device("cuda"))
                    print("🚀 Pipeline loaded on GPU (CUDA)!")
                else:
                    print("⚠️ Pipeline loaded on CPU.")
            
    
    def diarize(self, audio_path: str, output_dir: str) -> str:
        self.load_pipeline()
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"🎙️ Processing: {audio_path}")
        
        # --- CHẠY DIARIZATION ---
        try:
            # Bắt thêm các warning phát sinh trong quá trình chạy để log sạch hơn
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Estimate duration (seconds) to provide a simple progress estimate
                audio_duration = None
                try:
                    info = torchaudio.info(audio_path)
                    audio_duration = info.num_frames / float(info.sample_rate)
                except Exception:
                    audio_duration = None

                # Run pipeline in background thread so we can print progress
                result_container = {'result': None, 'error': None}

                def _run_pipeline():
                    try:
                        result_container['result'] = self.pipeline(audio_path)
                    except Exception as e:
                        result_container['error'] = e

                thread = threading.Thread(target=_run_pipeline, daemon=True)
                thread.start()

                start_time = time.time()
                # Heuristic speed factor: how many seconds of audio are processed per second of wallclock
                # If unknown, fall back to a conservative 8x (i.e., processing 1s audio takes ~8s)
                speed_factor = 8.0
                est_total = audio_duration * (1.0 if audio_duration is None else speed_factor) if audio_duration else 60.0

                # Print progress line and update until thread finishes
                while thread.is_alive():
                    elapsed = time.time() - start_time
                    pct = int(min(99, (elapsed / est_total) * 100)) if est_total > 0 else 0
                    sys.stdout.write(f"\r⏳ Processing (approx): {pct}% — {elapsed:.1f}s elapsed")
                    sys.stdout.flush()
                    time.sleep(1.0)

                # Join and check for errors
                thread.join()
                if result_container['error']:
                    raise result_container['error']

                result = result_container['result']
        except Exception as e:
            print(f"Error during inference: {e}")
            raise e
        
        raw_segments = []
         
        annotation = getattr(result, "speaker_diarization", result)

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            raw_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
                

        print(f"📊 Raw segments found: {len(raw_segments)}")

        with open("debug_output.txt", "w") as f_debug:
            f_debug.write(str(raw_segments))
        
        # Merge continuous segments
        cleaned_segments = self._remove_overlaps(raw_segments)
        merged_segments = self._merge_segments(cleaned_segments)
        
        # Create output
        os.makedirs(output_dir, exist_ok=True)
        audio_filename = Path(audio_path).stem
        output_filename = f"{audio_filename}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for seg in merged_segments:
                f.write(f"{seg['start']:.3f} {seg['end']:.3f} {seg['speaker']}\n")
        
        print(f"💾 Diarization saved to: {output_path}")
        return output_path

    def _remove_overlaps(self, segments: list) -> list:
        """
        Giữ lại các đoạn âm thanh tinh khiết (chỉ có đúng 1 người nói).
        Cắt bỏ hoàn toàn các phần thời gian có từ 2 người trở lên nói đè nhau.
        """
        if not segments:
            return []

        # Bước 1: Tạo các mốc sự kiện (Bắt đầu / Kết thúc)
        events = []
        for seg in segments:
            events.append((seg['start'], 'start', seg['speaker']))
            events.append((seg['end'], 'end', seg['speaker']))

        # Bước 2: Sắp xếp trục thời gian
        # RẤT QUAN TRỌNG: Nếu trùng thời gian, phải xử lý 'end' trước 'start' 
        # để tránh tạo ra tình trạng overlap ảo (người A vừa dứt, người B nói ngay)
        events.sort(key=lambda x: (x[0], 0 if x[1] == 'end' else 1))

        active_speakers = set()
        last_time = 0.0
        final_segments = []

        # Bước 3: Quét từ trái sang phải dọc theo trục thời gian
        for current_time, event_type, speaker in events:
            
            # Nếu trước thời điểm này chỉ có ĐÚNG 1 NGƯỜI ĐANG NÓI (Âm thanh tinh khiết)
            if len(active_speakers) == 1:
                pure_speaker = list(active_speakers)[0]
                # Lưu lại đoạn tinh khiết đó
                if current_time > last_time:
                    final_segments.append({
                        'start': last_time,
                        'end': current_time,
                        'speaker': pure_speaker
                    })

            if event_type == 'start':
                active_speakers.add(speaker)
            elif event_type == 'end':
                if speaker in active_speakers:
                    active_speakers.remove(speaker)

            last_time = current_time

        # Bước 4: Lọc bỏ các mảnh vỡ quá ngắn (ví dụ < 0.1s) do bị chặt chém
        cleaned_segments = [s for s in final_segments if (s['end'] - s['start']) > 0.1]
        
        return cleaned_segments
    
    def _merge_segments(self, segments: list) -> list:
        if not segments: return []
        merged = []
        for seg in segments:
            if merged and merged[-1]['speaker'] == seg['speaker'] and (seg['start'] - merged[-1]['end']) <= self.max_gap_threshold:
                merged[-1]['end'] = seg['end']
            else:
                merged.append(seg.copy())
        
        return [s for s in merged if (s['end'] - s['start']) >= self.min_segment_duration]

def main():
    # Token HF (Nhớ thay token thật của bạn vào đây nhé)
    # PYANNOTE_TOKEN = "sk_b4a8a1133ea145ebae021da2ae71f9db" 
    PYANNOTE_KEY = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    if not PYANNOTE_KEY:
        raise RuntimeError("Missing HUGGINGFACE_ACCESS_TOKEN in environment or .env file")
    AUDIO_PATH = "data/diarization_test_set/audio/interview_noise.wav"
    
    # Khai báo thêm thư mục lưu output
    OUTPUT_DIR = "pipeline/audio_pipeline/testing/test_diarization_output_precision-2" 
    
    try:
        diarizer = SpeakerDiarizer(token=PYANNOTE_KEY, model_name="pyannote/speaker-diarization-3.1")
        
        # Gọi hàm và hứng đường dẫn file kết quả trả về
        output_file_path = diarizer.diarize(AUDIO_PATH, OUTPUT_DIR)
        
        # # --- ĐỌC VÀ IN KẾT QUẢ RA MÀN HÌNH ---
        # print("\n" + "="*50)
        # print("🎯 KẾT QUẢ DIARIZATION (SEGMENTS & SPEAKERS)")
        # print("="*50)
        
        # with open(output_file_path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         # In từng dòng ra màn hình (VD: 34.490 40.615 SPEAKER_00)
        #         print(line.strip())
                
        # print("="*50)
        
    except Exception as e:
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    main()
