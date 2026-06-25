#!/usr/bin/env python3
"""Copy speaker-embedding weights out of the (local-only) stuff/ folder into the
canonical, code-referenced location under src/.../embeddings/weights/.

This is a LOCAL convenience (stuff/ only exists on the dev machine). On the
server you do NOT need it for most weights:
  * wespeaker34 + campplus (+ all config.yaml) are committed in git -> arrive via
    `git pull`.
  * ECAPA (pretrain.model) and VBx (ResNet101 ONNX) are also committed.
  * wespeaker293 ONNX (>100MB) is NOT in git -> download from Drive (see README).
  * redimnet uses torch.hub (no local weight needed).

Usage:
    python scripts/prepare_embeddings.py            # copy whatever is present
    python scripts/prepare_embeddings.py --list     # just report status
"""

import argparse
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from viespeaker import paths  # noqa: E402

_SRC_STUFF = os.path.join(_ROOT, "stuff", "reference", "speaker_embedding_model")
# Embedding weights now live in the external assets dir (see viespeaker.paths).
# (For a full relocate of ALL weights+media, prefer scripts/migrate_assets.py.)
_DST = str(paths.EMBEDDINGS_WEIGHTS)

# (destination subdir, [ (src_relpath, dst_filename), ... ])
COPY_PLAN = {
    "wespeaker34": [
        ("pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin", "pytorch_model.bin"),
        ("pyannote/wespeaker-voxceleb-resnet34-LM/config.yaml", "config.yaml"),
    ],
    "wespeaker293": [
        ("wespeaker-voxceleb-resnet293-LM/speaker-embedding.onnx", "speaker-embedding.onnx"),
        ("wespeaker-voxceleb-resnet293-LM/config.yaml", "config.yaml"),
    ],
    "campplus": [
        ("campplus/campplus_cn_common.bin", "campplus_cn_common.bin"),
        ("campplus/config.yaml", "config.yaml"),
    ],
    # NOTE: redimnet is intentionally omitted — the ReDimNet backend loads the
    # model via torch.hub (IDRnD/ReDimNet), not from a local checkpoint.
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="Only report status, copy nothing")
    args = ap.parse_args()

    print(f"Source : {_SRC_STUFF}")
    print(f"Dest   : {_DST}\n")
    if not os.path.isdir(_SRC_STUFF) and not args.list:
        print("[WARN] stuff/ source folder not found. If weights are already in place "
              "(or downloaded from Drive), backends will still work.")

    copied = skipped = missing = 0
    for subdir, files in COPY_PLAN.items():
        dst_dir = os.path.join(_DST, subdir)
        os.makedirs(dst_dir, exist_ok=True)
        for src_rel, dst_name in files:
            src = os.path.join(_SRC_STUFF, src_rel)
            dst = os.path.join(dst_dir, dst_name)
            if os.path.exists(dst):
                print(f"  [have] {subdir}/{dst_name}")
                skipped += 1
                continue
            if not os.path.exists(src):
                print(f"  [MISS] {subdir}/{dst_name}  (source: {src_rel})")
                missing += 1
                continue
            if args.list:
                print(f"  [todo] {subdir}/{dst_name}")
                continue
            shutil.copy2(src, dst)
            print(f"  [copy] {subdir}/{dst_name}")
            copied += 1

    print(f"\nDone. copied={copied} already_present={skipped} missing={missing}")
    if missing:
        print("Missing files: download from the project's Google Drive and place them "
              "under the Dest paths above, or ensure stuff/ is present.")


if __name__ == "__main__":
    main()
