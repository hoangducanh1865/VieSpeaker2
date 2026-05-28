# VieSpeaker2

Hệ thống speaker diarization cho tiếng Việt, kết hợp 3 pipeline:

| Pipeline | Mô tả |
|----------|-------|
| **P1** — Audio-only | Diarization bằng pyannote (cloud hoặc local) |
| **P2** — Audio-visual | Bổ sung nhãn speaker từ Active Speaker Detection (face track) |
| **P3** — Cleansing | Lọc/cộng gộp segment bằng speaker embedding (AHC hoặc CDGCN) |

---

## Cài đặt

### Local

```bash
conda activate VieSpeaker2
```

Tạo file `.env` ở thư mục gốc:

```env
PYANNOTE_API_KEY=sk_...             # Pipeline 1 mặc định (pyannote/speaker-diarization-precision-2)
HUGGINGFACE_ACCESS_TOKEN=hf_...     # Chỉ cần nếu dùng pyannote/speaker-diarization-3.1
```

> `PYANNOTE_API_KEY` — tạo tại [dashboard.pyannote.ai](https://dashboard.pyannote.ai) (có free credits).

### Google Colab / Kaggle

1. Upload `notebook/VieSpeaker2.ipynb` và mở trên Colab.

2. Vào **Secrets** (🔑 ở thanh bên trái), thêm các secret:

   | Name | Value |
   |------|-------|
   | `PYANNOTE_API_KEY` | `sk_...` |
   | `HUGGINGFACE_ACCESS_TOKEN` | `hf_...` |

   Bật **Notebook access** cho từng secret.

3. Tải 2 file model lớn (vượt giới hạn GitHub) từ Google Drive:

   > 📁 [Download large model files](https://drive.google.com/drive/folders/1uETq0S36474-dRvErdTI3V2CUR-qS2aH?usp=sharing)

   Đặt đúng vị trí:

   | File | Đường dẫn |
   |------|-----------|
   | `glintr100.onnx` | `src/pipeline/audio_visual_pipeline/face_embedding_model/weights/glintr100.onnx` |
   | `loconet_ava_best.model` | `src/pipeline/audio_visual_pipeline/audio_visual_model/LoCoNet_ASD/pretrained_model/loconet_ava_best.model` |

4. Chạy các cell theo thứ tự. Cell cuối tương đương `python main.py --pipeline all`.

---

## Chạy

### Chạy toàn bộ (mặc định)

```bash
python main.py --pipeline all
```

### Chạy từng pipeline

```bash
# Chỉ Pipeline 1
python main.py --pipeline 1 --sample singing

# Chỉ Pipeline 2 (cần output P1 trước)
python main.py --pipeline 2 --sample singing

# Chỉ Pipeline 3 (cần output P2 trước)
python main.py --pipeline 3 --sample singing --method ahc
```

### Chạy full pipeline với 3 method mới

```bash
# VBx — Bayesian HMM + PLDA (ResNet101 x-vectors)
python main.py --pipeline all --method vbx

# DOVER-Lap — dung hợp P1 + P2 hypothesis
python main.py --pipeline all --method dover-lap

# NME-SC — Auto-tuning Spectral Clustering
python main.py --pipeline all --method nme-sc
```

### Lệnh đầy đủ

```bash
python main.py --pipeline <PIPELINE> \
               --sample <SAMPLE> \
               --p1_model <P1_MODEL> \
               --asd_model <ASD_MODEL> \
               --method <METHOD> \
               --threshold <THRESHOLD> \
               --device <DEVICE>
```

---

## Tham số

### Tổng quát

| Tham số | Giá trị | Mặc định | Mô tả |
|---------|---------|----------|-------|
| `--pipeline` | `1` / `2` / `3` / `all` | *(bắt buộc)* | Pipeline cần chạy. `all` chạy tuần tự P1→P2→P3 |
| `--sample` | `drama` / `interview_clean` / `interview_noise` / `movie` / `sample_0` / `singing` | *(tất cả)* | Sample cần xử lý. Bỏ qua để chạy toàn bộ 6 sample |

### Pipeline 1

| Tham số | Giá trị | Mặc định | Mô tả |
|---------|---------|----------|-------|
| `--p1_model` | model ID | `pyannote/speaker-diarization-precision-2` | Model diarization. `precision-2` chạy trên cloud pyannoteAI (cần `PYANNOTE_API_KEY`); `pyannote/speaker-diarization-3.1` chạy local (cần `HUGGINGFACE_ACCESS_TOKEN`) |
| `--min_segment_duration` | float (s) | `0.5` | Loại bỏ segment ngắn hơn ngưỡng này |
| `--max_gap_threshold` | float (s) | `0.5` | Merge 2 segment cùng speaker nếu khoảng cách ≤ ngưỡng |

> **Cache:** Lần đầu chạy sẽ tải config về `~/.cache/huggingface`. Lần sau load offline tự động. Nếu file output đã tồn tại, pipeline 1 sẽ bỏ qua inference.

### Pipeline 2

| Tham số | Giá trị | Mặc định | Mô tả |
|---------|---------|----------|-------|
| `--asd_model` | `lr_asd` / `loconet` | `lr_asd` | Mô hình Active Speaker Detection |

### Pipeline 3

| Tham số | Giá trị | Mặc định | Mô tả |
|---------|---------|----------|-------|
| `--method` | `ahc` / `cdgcn` / `vbx` / `dover-lap` / `nme-sc` | `ahc` | Thuật toán cleansing |
| `--device` | `cuda` / `cpu` | `cuda` | Thiết bị tính toán |
| `--threshold` | `0.0` – `2.0` | `0.5` | Cosine distance threshold (AHC) — thấp = cụm chặt hơn |
| `--merge_gap_sec` | float (s) | `0.5` | Khoảng cách tối đa để merge segment (AHC) |
| `--min_duration_sec` | float (s) | `0.5` | Loại bỏ segment ngắn hơn ngưỡng sau cleansing (AHC) |
| `--min_cluster_size` | int | `3` | Số segment tối thiểu để tạo cluster (AHC) |
| `--k` | int | `10` | Số láng giềng KNN (CDGCN) |
| `--resolution` | float | `0.6` | Resolution của Leiden clustering (CDGCN) |
| `--purity_threshold` | float | `0.8` | Ngưỡng purity để gộp community (CDGCN) |
| `--vbx_loop_prob` | float | `0.9` | Xác suất không đổi speaker giữa 2 frame (VBx) |
| `--vbx_fa` | float | `0.3` | Scale sufficient statistics — thấp = ít speaker hơn (VBx) |
| `--vbx_fb` | float | `17.0` | Speaker regularization — cao = ít speaker hơn (VBx) |
| `--vbx_max_speakers` | int | `10` | Số speaker tối đa (VBx) |
| `--vbx_lda_dim` | int | `128` | Số chiều LDA sau PLDA transform (VBx) |
| `--vbx_init_smoothing` | float | `5.0` | Độ mượt khởi tạo AHC → gamma (VBx) |
| `--max_speakers` | int | `8` | Số speaker tối đa (NME-SC) |
| `--max_rp_threshold` | float | `0.25` | Tỷ lệ láng giềng tối đa để scan (NME-SC) |

**Mô tả các method:**

| Method | Mô tả |
|--------|-------|
| `ahc` | ECAPA-TDNN embeddings + Agglomerative Hierarchical Clustering (loại outlier nội-cụm) |
| `cdgcn` | ECAPA-TDNN embeddings + KNN graph + Leiden community detection |
| `vbx` | Bayesian HMM Clustering + PLDA (ResNet101 x-vectors 256-dim, 16kHz) |
| `dover-lap` | Dung hợp hypothesis từ P1 + P2 qua greedy label mapping + weighted voting |
| `nme-sc` | Auto-tuning Spectral Clustering qua Normalized Maximum Eigengap (ECAPA embeddings) |

> **Lưu ý `dover-lap`:** Không cần audio — cần cả output P1 (`data/diarization/<sample>.txt`) và P2. Khi dùng `main.py`, P1 path được inject tự động.

---

## Outputs

### Diarization

| Path | Nội dung |
|------|----------|
| `data/diarization/<sample>.txt` | Kết quả Pipeline 1 |
| `data/audio_visual/<sample>/supplemented_diarization.txt` | Kết quả Pipeline 2 |
| `data/clean/<sample>/cleansed_diarization.txt` | Kết quả Pipeline 3 |

Định dạng mỗi file (mỗi dòng một segment):
```
<start_s> <end_s> <speaker_id>
```

### Evaluation

Kết quả được lưu vào `experiment/<YYYYMMDD>/` (tự động tạo folder theo ngày chạy):

| File | Nội dung |
|------|----------|
| `pipeline<N>_results.json` | Metrics tích lũy của pipeline N (tất cả sample) |
| `pipeline3_<method>_results.json` | Metrics của Pipeline 3, method cụ thể |
| `pipeline<N>_all_samples_table.png` | Bảng tổng hợp toàn bộ sample |
| `pipeline3_<method>_all_samples_table.png` | Bảng tổng hợp P3 theo method |
| `pipeline<N>_table_<sample>.png` | Bảng metrics chi tiết từng sample |
| `pipeline3_<method>_table_<sample>.png` | Bảng metrics chi tiết P3 theo method |
| `pipeline1_der.png` | Stacked bar: FA / MD / Confusion (P1) |
| `pipeline2_prf.png` | Grouped bar: Purity / Coverage / F1 (P2) |
| `pipeline3_<method>_purity.png` | Bar: DER before vs after cleansing (P3) |
| `pipeline3_methods_comparison.png` | Grouped bar: DER so sánh tất cả 5 method P3 (tự động khi có ≥ 2 methods) |
| `pipeline_comparison_line.png` | Line chart: toàn bộ metrics qua P1→P2→P3 |
