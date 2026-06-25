# Vendored third-party code

These sub-trees are **copies** of upstream projects, kept in-repo and adapted
lightly (path/import glue only). They retain their own licenses where present.
Exact upstream commit SHAs were not recorded when the code was first vendored —
fill them in when re-syncing, and prefer a git submodule or a pinned download for
future updates.

| Path (under `src/`) | Upstream | Purpose | License |
|---|---|---|---|
| `pipeline/audio_visual_pipeline/audio_visual_model/LR-ASD/` | [Junhua-Liao/LR-ASD](https://github.com/Junhua-Liao/LR-ASD) (Light-ASD) | P2 Active Speaker Detection | see `LR-ASD/LICENSE` |
| `pipeline/audio_visual_pipeline/audio_visual_model/LoCoNet_ASD/` | [SJTUwxz/LoCoNet_ASD](https://github.com/SJTUwxz/LoCoNet_ASD) | P2 ASD (alt) | upstream repo; bundles `dlhammer/` (see its LICENSE) |
| `pipeline/audio_visual_pipeline/face_detection_model/SCRFD/` | [deepinsight/insightface](https://github.com/deepinsight/insightface) (SCRFD) | P2 face detection | see `SCRFD/README.md` / insightface terms |
| `pipeline/audio_visual_pipeline/face_embedding_model/` (ArcFace `glintr100`, insightface packs) | [deepinsight/insightface](https://github.com/deepinsight/insightface) | P2 face embedding | insightface (non-commercial research) |
| `pipeline/audio_visual_pipeline/sort/` | [abewley/sort](https://github.com/abewley/sort) | P2 tracking | GPL-3.0 |
| `pipeline/clean_pipeline/models/ecapa_tdnn/` | [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) | P3 default embedding | MIT |
| `pipeline/clean_pipeline/vbx/` | [BUTSpeechFIT/VBx](https://github.com/BUTSpeechFIT/VBx) | P3 VBx clustering | Apache-2.0 |
| `pipeline/clean_pipeline/nme_sc/` | [tango4j/Auto-Tuning-Spectral-Clustering](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering) | P3 NME-SC | MIT |
| `pipeline/clean_pipeline/dover_lap/` | [desh2608/dover-lap](https://github.com/desh2608/dover-lap) | P3/fusion DOVER-Lap | Apache-2.0 |
| `pipeline/clean_pipeline/embeddings/backends/arch/campplus.py` | [modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker) (CAM++) | P3 embedding arch | Apache-2.0 |

The corresponding **weights** are not in git; they live in the assets dir
(see [README §2](README.md) and `src/viespeaker/assets_manifest.py`).
