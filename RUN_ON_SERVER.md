# Chạy VieSpeaker2 trên cluster DGX (SLURM — dgx01/dgx02)

Quy trình: code ở **local (MacBook)** → `git push` → **server** `git pull` → chạy trên dgx01/dgx02 qua SLURM.

- Login node: `ssh user14@172.16.3.8` (head node `bcm-headnode02`, **không có GPU** — chỉ để nộp/quản lý job).
- Repo trên server: `/home/user14/anhhd/sv/VieSpeaker2` (home được NFS share → mọi node đều thấy).
- Conda env: `VieSpeaker2` tại `/home/user14/miniconda3/envs/VieSpeaker2`.
- Partition: `defq` (gồm dgx01 + dgx02, mỗi node 8× A100).

---

## 1. Đồng bộ code (mỗi lần sửa ở local)

```bash
# Local (MacBook)
git add -A && git commit -m "..." && git push

# Server (head node)
ssh user14@172.16.3.8
cd /home/user14/anhhd/sv/VieSpeaker2
git pull
```

## 2. Cài một lần (nếu env chưa có)

Env `VieSpeaker2` đã tồn tại sẵn thì bỏ qua. Nếu chưa:

```bash
cd /home/user14/anhhd/sv/VieSpeaker2
conda env create -f environment.yml      # hoặc: conda activate ... && pip install -r requirements.txt
```

Tải các weight >100MB (không có trong git) theo bảng trong [README.md](README.md) mục 2.

## 3. Tạo file `.env` trên server (một lần)

`.env` bị gitignore nên **không được pull về** — phải tạo trực tiếp trên server. Home dùng chung NFS nên chỉ cần tạo 1 lần là mọi node đọc được. Dán key thật của bạn vào:

```bash
cat > /home/user14/anhhd/sv/VieSpeaker2/.env <<'EOF'
PYANNOTEAI_API_KEY=sk_xxx          # P1 cloud (precision-2) — dashboard.pyannote.ai
HUGGINGFACE_ACCESS_TOKEN=hf_xxx    # P1 local (3.1) — huggingface.co/settings/tokens
EOF
chmod 600 /home/user14/anhhd/sv/VieSpeaker2/.env
```

(Mẫu các biến: xem [.env.example](.env.example).)

---

## 4. Cách A — Batch job (khuyến nghị cho chạy dài)

Nộp **từ thư mục gốc repo** để đường dẫn log (`experiment/logs/`) khớp:

```bash
cd /home/user14/anhhd/sv/VieSpeaker2
sbatch experiment/submit-job.sh                      # full sweep
sbatch experiment/submit-job.sh --smoke              # sanity nhanh (2 mẫu)
sbatch experiment/submit-job.sh --only p1_local p3_ahc_ecapa
sbatch experiment/submit-job.sh --samples interview_noise movie
```

- Mọi tham số sau tên script được **chuyển thẳng** cho `run_scenarios.py`.
- Tài nguyên mặc định (sửa trực tiếp trong [experiment/submit-job.sh](experiment/submit-job.sh) hoặc đè bằng CLI): 1× A100, 8 CPU, 64G RAM.
  ```bash
  sbatch --cpus-per-task=4 --mem=48G experiment/submit-job.sh   # đè ngay trên CLI
  ```
- **Đừng ép `-w`** nếu không cần (node DRAIN sẽ làm job kẹt `ReqNodeNotAvail`); để SLURM tự chọn node trống.

## 5. Cách B — Interactive trên node (debug / chạy ngắn, log realtime)

```bash
cd /home/user14/anhhd/sv/VieSpeaker2
./experiment/run-interactive.sh --smoke        # xin 1 A100, chạy sweep, log trực tiếp
GPUS=1 CPUS=8 MEM=64G ./experiment/run-interactive.sh --only p1_local
```

Hoặc xin một shell thô trên node rồi tự chạy:

```bash
srun --partition=defq --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --pty bash -l
# (đang ở trên node) →
source /home/user14/miniconda3/bin/activate VieSpeaker2
cd /home/user14/anhhd/sv/VieSpeaker2
python experiment/scenarios/run_scenarios.py --smoke
# hoặc chạy thủ công 1 pipeline:
python main.py --pipeline 1 --sample interview_noise --p1_model pyannote/speaker-diarization-3.1
```

---

## 6. Theo dõi & quản lý job

```bash
squeue -u $USER                                   # job của tôi (PD=chờ, R=chạy)
tail -f experiment/logs/viespeaker2_sweep_<jobid>.out
scontrol show job <jobid> | grep -iE "JobState|Reason|TRES"
sinfo -p defq -N -o "%N %t %C %G"                 # trạng thái + GPU mỗi node
scancel <jobid>                                   # hủy job
```

Lý do pending (`Reason=`): `Resources` = thiếu GPU/CPU/RAM thật; `Priority` = job khác ưu tiên hơn; `QOSMax/AssocMax...PerUser` = vượt quota → đợi job cũ xong hoặc xin ít tài nguyên hơn.

---

## Biến môi trường tinh chỉnh script (tùy chọn)

Cả hai script đều dùng chung [experiment/_job_body.sh](experiment/_job_body.sh); có thể đè bằng env var:

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `VIESPEAKER_DIR` | submit dir / path cố định | Thư mục gốc repo |
| `CONDA_BASE` | `/home/user14/miniconda3` | Prefix miniconda |
| `CONDA_ENV` | `VieSpeaker2` | Tên conda env |
| `SKIP_SELFCHECK` | `0` | Đặt `1` để bỏ qua `scripts/selfcheck.py` |

Ví dụ: `SKIP_SELFCHECK=1 CONDA_ENV=VieSpeaker2 sbatch experiment/submit-job.sh --smoke`.
