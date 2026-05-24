#!/bin/bash -l

#SBATCH --job-name=supplement_loconet_a100
#SBATCH --comment="supplement_pipeline_with_loconet"
#SBATCH --partition=defq  # Available partition with A100 GPUs
#SBATCH --gres=gpu:a100:1  # Yêu cầu 1 GPU A100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100g

# 1. Chuyển đến đúng thư mục
cd "$(dirname "$0")"

# 2. Khởi tạo environment
source /home/user14/miniconda3/bin/activate face_detection_anhhd

# 3. Tạo logs folder
mkdir -p logs

# 4. Chạy lệnh
echo "Starting supplement_pipeline with LoCoNet on A100..."
python supplement_pipeline.py \
  --video_path 'data/diarization_test_set/video/interview_noise.mp4' \
  --audio_visual_model loconet \
  --loconet_root audio_visual_model/LoCoNet_ASD \
  --skip-phase 0-2 \
  --out_dir result/result_audio_visual

echo "Job completed!"