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

### Individual pipelines

```bash
# Pipeline 1 — audio diarization
python src/pipeline/audio_pipeline/speaker_diarization.py \
    --audio_path data/diarization_test_set/audio/interview_noise.wav \
    --output_dir data/diarization

# Pipeline 2 — audio-visual ASD supplement
python src/pipeline/audio_visual_pipeline/supplement_pipeline.py \
    --video_path data/diarization_test_set/video/interview_noise.mp4 \
    --sd_diarization data/diarization/interview_noise.txt \
    --model lr_asd

# Pipeline 3 — cleansing
python src/pipeline/clean_pipeline/clean.py \
    --method ahc \
    --diarization_path data/audio_visual/interview_noise/supplemented_diarization.txt \
    --audio_path data/diarization_test_set/audio/interview_noise.wav \
    --output_dir data/clean/interview_noise
```

### Orchestrated runs

```bash
# Single sample through all pipelines
python main.py --pipeline all --sample interview_noise

# All samples through all pipelines with CDGCN cleansing
python main.py --pipeline all --method cdgcn

# Single pipeline for all samples
python main.py --pipeline 1
```

## Outputs

| Path | Contents |
|------|----------|
| `data/diarization/<sample>.txt` | Pipeline 1 diarization |
| `data/audio_visual/<sample>/supplemented_diarization.txt` | Pipeline 2 output |
| `data/clean/<sample>/cleansed_diarization.txt` | Pipeline 3 output |
| `experiments/pipeline{N}_results.json` | Per-sample metrics |
| `experiments/pipeline{N}_*.png` | Evaluation charts |
