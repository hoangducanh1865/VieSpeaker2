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

# Đăng ký package `viespeaker` (module dùng chung: paths/bootstrap/pipeline_api).
# --no-deps để KHÔNG đụng tới các dependency đã pin trong env.
pip install -e . --no-deps
```

> `requirements.txt` dùng PyTorch `cu118`, tương thích driver 470/CUDA 11.4 trên
> DGX của cụm. Dev/CPU (mac) dùng `requirements-cpu.txt`. Python ≥3.9
> (env hiện tại là 3.9).

## 2. Model weights + dữ liệu test (thư mục **assets** ngoài repo)

Toàn bộ **model weights** và **audio/video test KHÔNG nằm trong git** — chúng sống trong một thư mục **assets** đặt **cạnh** repo:

```
~/anhhd/sv/
├── VieSpeaker2/             # repo (chỉ code + label .txt)
└── VieSpeaker2_assets/      # weights + data  ← mặc định, KHÔNG cần set env
    ├── models/{ecapa_tdnn,vbx,embeddings,face_detection,face_embedding,asd}/…
    └── data/diarization_test_set/{audio,video,label_audio}/…
```

Code tự tìm assets ở `../VieSpeaker2_assets` (cạnh repo) → trên server tự là `~/anhhd/sv/VieSpeaker2_assets`, **không cần cấu hình**. Đặt nơi khác thì set `VIESPEAKER2_ASSETS=/đường/dẫn`. Ground-truth `label/*.txt` vẫn nằm trong repo.

**Lần đầu — đưa weights + data ra thư mục assets** (chạy từ gốc repo; chỉ relocate file đã có, **không cần Google Drive**):

```bash
python scripts/migrate_assets.py          # in ra các lệnh cp để xem trước
python scripts/migrate_assets.py --run     # hoặc copy luôn vào ../VieSpeaker2_assets
```

`redimnet` tải qua `torch.hub` (IDRnD/ReDimNet) lần đầu chạy (cần mạng) — không cần file cục bộ. Sau khi populate, `scripts/selfcheck.py` xác nhận đủ weight và in path đích cho file còn thiếu.

## 3. Tạo file `.env`

`.env` **không nằm trong repo**. Loader (`viespeaker.env`) tìm theo thứ tự: `$VIESPEAKER2_ENV_FILE` → `<repo>/.env` → `<repo_parent>/env/<repo_name>/.env`. Khuyến nghị đặt **ngoài repo** (server tự nhận):

```bash
mkdir -p ~/anhhd/sv/env/VieSpeaker2
cat > ~/anhhd/sv/env/VieSpeaker2/.env <<'EOF'
PYANNOTEAI_API_KEY=sk_...           # P1 cloud (pyannote/speaker-diarization-precision-2)
HUGGINGFACE_ACCESS_TOKEN=hf_...     # P1 local (pyannote/speaker-diarization-3.1)
EOF
chmod 600 ~/anhhd/sv/env/VieSpeaker2/.env
```

> `PYANNOTEAI_API_KEY` tạo tại [dashboard.pyannote.ai](https://dashboard.pyannote.ai). Đặt nơi khác thì set `VIESPEAKER2_ENV_FILE=/đường/dẫn/.env`.

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

# Chạy đầy đủ trên A100 qua SLURM (nộp từ thư mục gốc repo)
sbatch experiment/submit-job.sh
sbatch experiment/submit-job.sh --smoke                 # tham số chuyển thẳng cho run_scenarios.py
./experiment/run-interactive.sh --smoke                 # hoặc chạy interactive trên node (log realtime)
```

Chi tiết cách nộp/theo dõi job SLURM ở [mục 9](#9-chạy-trên-cluster-dgx-slurm--dgx01dgx02) bên dưới.

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
- **Weights + dữ liệu test ra khỏi git** → thư mục assets cạnh repo (xem mục 2); repo nhẹ, không còn tải Drive.
- **Đóng gói `viespeaker`** (paths/bootstrap/pipeline_api/logging) + test/CI/ruff; nạp audio 1 lần/file (cache) thay vì decode lại mỗi segment.

---

## 9. Chạy trên cluster DGX (SLURM — dgx01/dgx02)

Quy trình: code ở **local** → `git push` → **server** `git pull` → chạy qua SLURM.
Login: `ssh user14@172.16.3.8` (head node `bcm-headnode02`, **không có GPU**). Partition `defq` (dgx01 + dgx02, mỗi node 8× A100). Home NFS-share nên code/env/assets trong `/home/user14/...` thấy được ở mọi node. **Không SSH thẳng** vào compute node — chỉ qua SLURM.

**Đồng bộ + chuẩn bị (một lần):**
```bash
ssh user14@172.16.3.8
cd /home/user14/anhhd/sv/VieSpeaker2 && git pull
pip install -e . --no-deps                       # nếu chưa cài
python scripts/migrate_assets.py --run           # đưa weights+data ra ../VieSpeaker2_assets (một lần)
mkdir -p ~/anhhd/sv/env/VieSpeaker2              # .env ngoài repo (xem §3)
cat > ~/anhhd/sv/env/VieSpeaker2/.env <<'EOF'
PYANNOTEAI_API_KEY=sk_...
HUGGINGFACE_ACCESS_TOKEN=hf_...
EOF
chmod 600 ~/anhhd/sv/env/VieSpeaker2/.env
```

**Cách A — Batch job (khuyến nghị, chạy dài).** Nộp **từ gốc repo**:
```bash
sbatch experiment/submit-job.sh                  # full sweep
sbatch experiment/submit-job.sh --smoke          # tham số chuyển thẳng cho run_scenarios.py
sbatch experiment/submit-job.sh --only p1_local p3_ahc_ecapa
sbatch -w dgx02 experiment/submit-job.sh         # ép node (bỏ qua nếu node DRAIN)
```
Tài nguyên mặc định: 1× A100, 8 CPU, 64G (sửa trong [experiment/submit-job.sh](experiment/submit-job.sh) hoặc đè bằng `sbatch --mem=48G …`). **Đừng ép `-w`** nếu không cần (node DRAIN làm job kẹt `ReqNodeNotAvail`).

**Cách B — Interactive (debug/ngắn, log realtime):**
```bash
./experiment/run-interactive.sh --smoke          # srun xin 1 A100, chạy sweep, log trực tiếp
# hoặc shell thô trên node:
srun --partition=defq --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --pty bash -l
```

**Theo dõi job:**
```bash
squeue -u $USER                                  # PD=chờ, R=chạy
tail -f experiment/logs/viespeaker2_sweep_<jobid>.out
scontrol show job <jobid> | grep -iE "JobState|Reason|TRES"
sinfo -p defq -N -o "%N %t %C %G"; scancel <jobid>
```
`Reason=`: `Resources`=thiếu GPU/CPU/RAM; `Priority`=job khác ưu tiên; `QOSMax/AssocMax…PerUser`=vượt quota → đợi hoặc xin ít tài nguyên hơn.

Cả `submit-job.sh` và `run-interactive.sh` dùng chung [experiment/_job_body.sh](experiment/_job_body.sh); chỉnh qua env `VIESPEAKER_DIR` / `CONDA_BASE` / `CONDA_ENV` / `SKIP_SELFCHECK`.

---

## 10. Dataset & tái lập

**Test set: 6 mẫu** (`drama`, `interview_clean`, `interview_noise`, `movie`, `sample_0`, `singing`) — phủ hội thoại sạch/nhiễu, phim, hát. Số mẫu nhỏ ⇒ xem kết quả là chỉ báo định hướng, không phải kết luận tổng quát. Audio/video ở thư mục assets; ground-truth `data/diarization_test_set/label/*.txt` trong repo.

**BASE_P1 mặc định = `cloud`** (`precision-2`, phi xác định + tốn credit). Để **tái lập offline** dùng bản local:
```bash
python experiment/scenarios/run_scenarios.py --only p1_local        # hoặc đổi BASE_P1="local" trong scenarios.py
```

Kết quả chốt: chạy `run_scenarios.py` rồi dán `experiment/<RUNTAG>/REPORT.md` vào bảng dưới (chưa cập nhật số ở đây để tránh số liệu lỗi thời).

## 11. Dev (local / CPU / test)

```bash
pip install -r requirements-cpu.txt      # torch CPU + onnxruntime (mac/linux không GPU)
pip install -e ".[dev]"                  # ruff + black + pytest
ruff check . && pytest -q                # lint + test (test nặng tự skip nếu thiếu deps)
```
Đường dẫn weight/data tập trung ở [src/viespeaker/paths.py](src/viespeaker/paths.py); manifest assets ở [src/viespeaker/assets_manifest.py](src/viespeaker/assets_manifest.py). Mức log chỉnh bằng `VIESPEAKER_LOG_LEVEL=DEBUG`.
