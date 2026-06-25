# VieSpeaker2 — implementation notes (for maintainers)

Deep notes behind the infra refactor (assets-out-of-repo + packaging + cleanup).
User-facing instructions live in [../../README.md](../../README.md); this file is
the "why/how it's wired" for whoever edits the code. `stuff/` is gitignored, so
this file is **force-added** to git (`git add -f stuff/claude_guidance/...`).

## Architecture of the glue

`src/viespeaker/` is an installable package (`pip install -e . --no-deps`) holding
cross-cutting infra so pipeline code stops hard-coding paths and sys.path:

- `paths.py` — **single source of truth** for every weight/data path. `ASSETS_ROOT`
  defaults to `REPO_ROOT.parent / "VieSpeaker2_assets"` (sibling of the repo →
  `~/anhhd/sv/VieSpeaker2_assets` on the server, no env needed). Override with
  `VIESPEAKER2_ASSETS`; data alone with `VIESPEAKER2_DATA`. Labels stay in-repo.
- `assets_manifest.py` — declarative `Asset(dest, old, severity, note)` list. `dest`
  is the assets-dir target; `old` is the repo-relative source (drives the relocate
  cp commands + selfcheck). Keep in sync when adding a weight.
- `bootstrap.py` — `setup()` prepends the **first-party** vendored dirs (unique
  top-level names: `clean`, `face_pipeline`, `sort`, `evaluation`, `fuse`, …) to
  sys.path **once**. Collision-prone deep trees (`model.py` in ecapa_tdnn,
  `features.py`/`VBx.py` in vbx, `ASD.py`+`model/` in LR-ASD) are NOT added here —
  their consumers insert them at `sys.path[0]` just-in-time so the right module
  wins. Removing those local inserts WILL cause `from model import ECAPA_TDNN` to
  resolve the wrong `model`.
- `audio.py` — `load_mono(path, sr)` `lru_cache`d full-file load+resample+mono.
  Backends slice this instead of re-decoding the whole WAV per segment.
- `pipeline_api.py` — `run_p1/p2/p3/fusion/evaluate` wrap the subprocess scripts;
  shared by `main.py` and `experiment/scenarios/run_scenarios.py` (removed the
  duplicated `_run`/`_sh` + script-path constants). Stages stay in separate
  processes (dependency isolation preserved on purpose).
- `logging_setup.py` — `get_logger`; level via `VIESPEAKER_LOG_LEVEL`. Adopted in
  the orchestration layer (pipeline_api, main). Per-pipeline algorithm scripts keep
  their progress `print()`s by design (low value / high churn / tab-indented file).

Entry scripts (`main.py`, `clean.py`, `supplement_pipeline.py`, `fuse.py`,
`selfcheck.py`, `run_scenarios.py`) carry an identical 6-line **walk-up preamble**
that finds `src/viespeaker` upward and adds `src` to sys.path, so they run from a
fresh checkout even before `pip install -e .`.

## Assets relocation + CP sequencing (critical ordering)

The relocate must happen on the **server while it still has the old clone** (the
committed weights + the Drive-downloaded >100MB files are all physically present
there). Order:

1. **Local**: all code commits (done) — weights/media still tracked until the purge.
2. **CP1 (server, OLD commit)**: `python scripts/migrate_assets.py --run` copies
   everything into `../VieSpeaker2_assets`. Use **cp, not mv** — the repo's
   tracked copies are cleaned later by the reset; the `>100MB` Drive files exist
   only here so this is the one chance to relocate them. No Drive needed anymore.
3. **Local Phase 8**: `git filter-repo --invert-paths` strips weight/media/artifact
   paths from ALL history; `git push --force`.
4. **CP2 (server)**: `git fetch && git reset --hard origin/<branch>` → repo loses
   the old in-tree weight copies (now only in assets); history is light.
5. **CP3 (server)**: `pip install -e . --no-deps`; `python scripts/selfcheck.py`.

`backup/pre-refactor` tag (local) points at the pre-refactor HEAD with the full
big-file history — recovery point before the force-push. Do NOT push it (keeps
GitHub light).

## filter-repo path list (Phase 8)

Purge these path prefixes (weights, media, generated artifacts) from history:
- `src/pipeline/clean_pipeline/models/ecapa_tdnn/exps/pretrain.model`
- `src/pipeline/clean_pipeline/vbx/models/`
- `src/pipeline/clean_pipeline/embeddings/weights/`
- `src/pipeline/audio_visual_pipeline/face_detection_model/SCRFD/weights/`
- `src/pipeline/audio_visual_pipeline/face_embedding_model/weights/`
- `src/pipeline/audio_visual_pipeline/audio_visual_model/LR-ASD/weight/`
- `src/pipeline/audio_visual_pipeline/audio_visual_model/LoCoNet_ASD/pretrained_model/loconet_ava_best.model`
- `data/diarization_test_set/audio/`, `…/video/`, `…/label_audio/`
- `debug_output.txt`, `experiment/20260527/`, `experiment/20260528/`, `experiment/20260605/`
- `src/pipeline/audio_visual_pipeline/audio_visual_model/LoCoNet_ASD/sample_0_output/`

Keep: `data/diarization_test_set/label/*.txt`, tiny `s3fd.pth`, small config/text.

## Deliberate deviations (don't "fix" these without thinking)

- **Python stays ≥3.9** (not bumped to 3.10): the server conda env is 3.9; forcing
  3.10 would require rebuilding it. Tooling targets py39.
- **ECAPA now honours `--device`** (was always CPU). On `--device cuda` the numbers
  may drift by <0.1% vs CPU (float). The audio-cache refactor itself is numerically
  identical on CPU → run the regression check with `--device cpu`.
- **BASE_P1 stays `cloud`** (user has the key). Local 3.1 is the offline-reproducible
  alternative; documented, not made default.
- **`s3fd.pth` (256KB) stays in-repo** — LoCoNet loads it via hardcoded vendored
  paths; not worth editing third-party code to relocate a tiny file. LoCoNet's big
  `loconet_ava_best.model` is passed via `--resume-path` from supplement_pipeline.
- **sys.path not 100% removed** — impossible without rewriting vendored bare imports.
  Scattered duplicates were centralized into `bootstrap.setup()`; collision-prone
  just-in-time inserts remain on purpose.

## Verification checklist

- `python scripts/selfcheck.py` → core OK, all weights found under assets.
- Regression invariance (CPU): P3 AHC + VBx on one sample before/after the audio
  cache → identical DER.
- `ruff check . && pytest -q` green (integration tests skip without heavy deps).
- `python experiment/scenarios/run_scenarios.py --smoke` → REPORT.md produced.
- `du -sh .git` shrinks dramatically after filter-repo; `git ls-files` has no weights/media.
