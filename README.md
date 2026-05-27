# VieSpeaker2

Speaker diarization system for Vietnamese speech combining audio-only pyannote diarization, audio-visual active speaker detection, and speaker embedding-based cleansing.

## Setup

```bash
conda activate VieSpeaker2
```

Place your HuggingFace token in a `.env` file at the project root:

```
HUGGINGFACE_ACCESS_TOKEN=hf_...
```

## Running

### Chạy toàn bộ (tham số mặc định)

```bash
python main.py --pipeline all
```

### Lệnh tổng quát

```bash
python main.py --pipeline <PIPELINE> \
               --sample <SAMPLE> \
               --method <METHOD> \
               --asd_model <ASD_MODEL> \
               --threshold <THRESHOLD> \
               --device <DEVICE>
```

### Bảng tham số

| Tham số | Giá trị | Mặc định | Mô tả |
|---------|---------|----------|-------|
| `--pipeline` | `1` / `2` / `3` / `all` | *(bắt buộc)* | Pipeline cần chạy. `all` chạy tuần tự P1→P2→P3 |
| `--sample` | `drama` / `interview_clean` / `interview_noise` / `movie` / `sample_0` / `singing` | *(tất cả)* | Sample cần xử lý. Bỏ qua để chạy toàn bộ 6 sample |
| `--method` | `ahc` / `cdgcn` | `ahc` | Thuật toán cleansing cho Pipeline 3 |
| `--asd_model` | `lr_asd` / `loconet` | `lr_asd` | Mô hình Active Speaker Detection cho Pipeline 2 |
| `--threshold` | `0.0` – `2.0` | `0.5` | Cosine distance threshold của AHC (thấp = chặt hơn) |
| `--device` | `cuda` / `cpu` | `cuda` | Thiết bị tính toán cho Pipeline 3 |
| `--p1_model` | HuggingFace model ID | `pyannote/speaker-diarization-3.1` | Mô hình diarization cho Pipeline 1 |
| `--min_segment_duration` | float (giây) | `0.5` | Loại bỏ segment ngắn hơn ngưỡng này (Pipeline 1) |
| `--max_gap_threshold` | float (giây) | `0.5` | Merge hai segment cùng speaker nếu khoảng cách ≤ ngưỡng (Pipeline 1) |
| `--merge_gap_sec` | float (giây) | `0.5` | Khoảng cách tối đa để merge segment (AHC) |
| `--min_duration_sec` | float (giây) | `0.5` | Loại bỏ segment ngắn hơn ngưỡng sau cleansing (AHC) |
| `--min_cluster_size` | int | `3` | Số segment tối thiểu để tạo cluster (AHC) |
| `--k` | int | `10` | Số láng giềng KNN (CDGCN) |
| `--resolution` | float | `0.6` | Resolution của Leiden clustering (CDGCN) |
| `--purity_threshold` | float | `0.8` | Ngưỡng purity để gộp community (CDGCN) |

## Outputs

| Path | Nội dung |
|------|----------|
| `data/diarization/<sample>.txt` | Kết quả Pipeline 1 |
| `data/audio_visual/<sample>/supplemented_diarization.txt` | Kết quả Pipeline 2 |
| `data/clean/<sample>/cleansed_diarization.txt` | Kết quả Pipeline 3 |
| `experiments/pipeline<N>_results.json` | Metrics tích lũy của pipeline N |
| `experiments/pipeline<N>_table_<sample>_<date>.png` | Bảng metrics đầy đủ theo sample |
| `experiments/pipeline<N>_*_<date>.png` | Biểu đồ bar/stacked theo pipeline |
| `experiments/pipeline_comparison_line_<date>.png` | Biểu đồ so sánh P1→P2→P3 |
