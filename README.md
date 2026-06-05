# VieSpeaker2

Hệ thống speaker diarization cho tiếng Việt, gồm 3 pipeline xếp tầng + 1 chế độ hợp nhất:

| Pipeline | Mô tả |
|----------|-------|
| **P1** — Audio-only | Diarization bằng pyannote (cloud `precision-2` hoặc local `3.1`) |
| **P2** — Audio-visual | **Chỉ tinh chỉnh nhãn** từ Active Speaker Detection (face track) — *không bao giờ xóa speech* |
| **P3** — Cleansing | Lọc/gộp segment bằng speaker embedding (AHC / CDGCN / VBx / NME-SC / DOVER-LAP) |
| **Fusion** | Hợp nhất nhiều hypothesis (P1 + P2 + …) bằng DOVER-Lap |

> **Embedding cho P3 có thể chọn:** `ecapa` (mặc định), `wespeaker34`, `wespeaker293`, `campplus` (zh-cn), `redimnet`.

---

## 1. Cài đặt trên server A100

Giả định project đặt tại `/home/user14/anhhd/sv/VieSpeaker2`.

```bash
cd /home/user14/anhhd/sv
git clone <repo-url> VieSpeaker2
cd VieSpeaker2

# Tạo môi trường conda (cài torch CUDA + toàn bộ deps qua pip)
conda env create -f environment.yml
conda activate VieSpeaker2
```

> File `requirements.txt` đã trỏ sẵn PyTorch CUDA index (`cu124`). Nếu driver CUDA khác, sửa tag `cu124` trong `requirements.txt` cho khớp (xem https://pytorch.org).

## 2. Tải model weights

Một số weight quá lớn nên **không** nằm trong git:

**a) Weight Pipeline 2 (>100MB)** — tải từ Google Drive và đặt đúng path:

| File | Đường dẫn |
|------|-----------|
| `glintr100.onnx` | `src/pipeline/audio_visual_pipeline/face_embedding_model/weights/glintr100.onnx` |
| `loconet_ava_best.model` | `src/pipeline/audio_visual_pipeline/audio_visual_model/LoCoNet_ASD/pretrained_model/loconet_ava_best.model` |

> 📁 [Download large model files](https://drive.google.com/drive/folders/1uETq0S36474-dRvErdTI3V2CUR-qS2aH?usp=sharing)

**b) Weight speaker-embedding cho P3** — đặt vào `src/pipeline/clean_pipeline/embeddings/weights/`:

```bash
# Nếu đã có folder stuff/ (chứa các model đã clone), copy tự động:
python scripts/prepare_embeddings.py
# Hoặc tải từ Google Drive và đặt vào các path mà script in ra.
```

Các weight sẵn trong git (không cần tải): ECAPA `pretrain.model`, VBx `ResNet101_16kHz`, SCRFD detector.

## 3. Tạo file `.env`

```env
PYANNOTEAI_API_KEY=sk_...           # P1 cloud (pyannote/speaker-diarization-precision-2)
HUGGINGFACE_ACCESS_TOKEN=hf_...     # P1 local (pyannote/speaker-diarization-3.1)
```

> `PYANNOTEAI_API_KEY` tạo tại [dashboard.pyannote.ai](https://dashboard.pyannote.ai).

## 4. Kiểm tra trước khi chạy

```bash
python scripts/selfcheck.py
```

Script báo cáo: import lõi, weight có đủ chưa, từng embedding backend extract được không, và key trong `.env`. Cảnh báo (WARN) ở model best-effort (loconet, redimnet) là chấp nhận được.

---

## 5. Chạy benchmark toàn bộ kịch bản (khuyến nghị)

Bộ kịch bản curated (P1 cloud/local, P3 × nhiều method × nhiều embedding, P2, fusion) nằm ở [experiment/scenarios/scenarios.py](experiment/scenarios/scenarios.py).

```bash
# Sanity nhanh (2 mẫu, 4 kịch bản nhẹ)
python experiment/scenarios/run_scenarios.py --smoke

# Chạy đầy đủ trên A100 qua SLURM
sbatch experiment/submit-job.sh
```

Hoặc chạy trực tiếp (không SLURM):

```bash
python experiment/scenarios/run_scenarios.py            # toàn bộ
python experiment/scenarios/run_scenarios.py --only p1_cloud p3_vbx fusion_p1_p2
python experiment/scenarios/run_scenarios.py --samples interview_noise movie
```

**Kết quả** lưu ở `experiment/<RUNTAG>/`:

| File | Nội dung |
|------|----------|
| `REPORT.md` | **Báo cáo tự chứa** — có sẵn *prompt phân tích* ở đầu file. Chỉ cần copy-paste cả file vào bất kỳ LLM nào để nhận phân tích so sánh các kịch bản. Tự cập nhật sau mỗi kịch bản. |
| `results.json` | Metrics thô của mọi kịch bản |
| `scenarios/<id>/<sample>/*.txt` | Diarization output từng kịch bản |
| `cache/` | P1/P2 trung gian dùng lại (gitignored) |

Runner **tái dùng** output: mỗi model P1 chạy 1 lần, mỗi P2 (face pipeline) chạy 1 lần / ASD, các kịch bản P3/fusion dùng lại cache → tiết kiệm credit cloud + thời gian. Mỗi kịch bản được bọc lỗi: một kịch bản fail không làm dừng cả lượt chạy.

---

## 6. Chạy thủ công từng pipeline (`main.py`)

```bash
# Toàn bộ cascade (P1→P2→P3)
python main.py --pipeline all --sample singing

# Từng pipeline
python main.py --pipeline 1 --sample singing --p1_model pyannote/speaker-diarization-3.1
python main.py --pipeline 2 --sample singing
python main.py --pipeline 3 --sample singing --method ahc --embedding wespeaker34

# Fusion (P1 + P2 qua DOVER-Lap)
python main.py --pipeline fusion --sample singing
```

### Tham số chính `main.py`

| Tham số | Giá trị | Mặc định | Mô tả |
|---------|---------|----------|-------|
| `--pipeline` | `1`/`2`/`3`/`all`/`fusion` | *(bắt buộc)* | Pipeline cần chạy |
| `--sample` | tên mẫu | *(tất cả 6)* | `drama`/`interview_clean`/`interview_noise`/`movie`/`sample_0`/`singing` |
| `--p1_model` | model id | `…/speaker-diarization-precision-2` | P1 cloud; hoặc `…/speaker-diarization-3.1` (local) |
| `--overlap_policy` | `keep`/`drop`/`dominant` | `keep` | **Mới.** `keep` giữ nguyên overlap (không tạo missed-detection giả); `drop` = hành vi cũ |
| `--asd_model` | `lr_asd`/`loconet` | `lr_asd` | Model ASD cho P2 |
| `--method` | `ahc`/`cdgcn`/`vbx`/`dover-lap`/`nme-sc` | `ahc` | Thuật toán P3 |
| `--embedding` | `ecapa`/`wespeaker34`/`wespeaker293`/`campplus`/`redimnet` | `ecapa` | **Mới.** Embedding cho `ahc`/`cdgcn`/`nme-sc` |
| `--collar` | float (s) | `0.0` | Collar chính cho DER. DER ở `collar=0.25` luôn được báo cáo kèm |
| `--device` | `cuda`/`cpu` | `cuda` | Thiết bị |

> Các tham số chi tiết của từng method P3 (AHC threshold, CDGCN k/resolution, VBx Fa/Fb, NME-SC max_speakers…) giữ nguyên như cũ — xem `python main.py -h` hoặc `python src/pipeline/clean_pipeline/clean.py -h`.

### Tham số Pipeline 2 (mới/đáng chú ý)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--cluster_merge_threshold` | `0.0` | Nếu >0, gộp các face-cluster có centroid gần nhau (re-ID xuyên scene-cut) để giảm over-segment |
| `--asd_threshold` | `0.0` | Ngưỡng "đang nói" của LR-ASD; tăng để precision hình ảnh cao hơn |

---

## 7. Định dạng output & đánh giá

Mỗi dòng output: `<start_s> <end_s> <speaker_id>`.

| Path | Nội dung |
|------|----------|
| `data/diarization/<sample>.txt` | P1 |
| `data/audio_visual/<sample>/supplemented_diarization.txt` | P2 |
| `data/clean/<sample>/cleansed_diarization.txt` | P3 |
| `data/fusion/<sample>/fused_diarization.txt` | Fusion |

Đánh giá (`src/evaluation/evaluation.py`) báo cáo **DER (collar=0)** và **DER (collar=0.25)**, cùng FA / MD / Confusion / Purity / Coverage / F1. Kết quả `main.py` lưu vào `experiment/<YYYYMMDD>/` (PNG + JSON).

---

## 8. Những thay đổi cốt lõi so với phiên bản trước

- **P2 không còn xóa speech** (bỏ `DISCARD`) — chỉ relabel khi bằng chứng hình ảnh mạnh & nhất quán; dùng mapping mềm many-to-many video→audio. Trước đây P2 xóa tới ~38% speech đúng ở mẫu nhiễu (`interview_noise` DER 57%→25%).
- **P3 AHC không vứt outlier** — gán lại về speaker gần nhất thay vì xóa. Mặc định không xóa vùng overlap.
- **P1 giữ overlap** mặc định (`--overlap_policy keep`).
- **Nhiều speaker-embedding** cắm rời cho P3 (so sánh ECAPA EN vs WeSpeaker vs CAM++ zh).
- **Fusion mode** (DOVER-Lap) thay cho cascade một chiều.
- **Eval collar 0.25** để so sánh được với literature.
- Bộ **scenario runner** + **REPORT.md** tự chứa để benchmark & nhờ LLM phân tích.
