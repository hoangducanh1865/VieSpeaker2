#!/usr/bin/env python3
"""Generate (or run) the one-time commands that relocate weights + test media
out of the repo into the sibling assets dir ``../VieSpeaker2_assets``.

The assets dir defaults to a sibling of the repo (see viespeaker.paths), so the
emitted commands use a *relative* ``../VieSpeaker2_assets`` base and therefore
work unchanged on any machine when run from the repo root.

Usage:
    python scripts/migrate_assets.py            # print the bash to copy-paste (server CP1)
    python scripts/migrate_assets.py --run      # actually copy on THIS machine
    python scripts/migrate_assets.py --move      # use mv instead of cp (frees repo space)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from viespeaker import paths  # noqa: E402
from viespeaker import assets_manifest as M  # noqa: E402


def _rel_dest(asset) -> str:
    return str(asset.dest.relative_to(paths.ASSETS_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Execute the copy on this machine")
    ap.add_argument("--move", action="store_true", help="Move (mv) instead of copy (cp)")
    args = ap.parse_args()

    repo = paths.REPO_ROOT
    op = "mv" if args.move else "cp"
    present, missing = [], []
    for a in M.ALL:
        src = repo / a.old
        (present if src.exists() else missing).append(a)

    if not args.run:
        print('# Run from the repo root on the server. Relocates weights + test media')
        print('# into the sibling assets dir (default location, no env var needed).')
        print('set -e')
        print('ASSETS=../VieSpeaker2_assets')
        seen_dirs = set()
        for a in present:
            dest_rel = _rel_dest(a)
            dest_dir = os.path.dirname(dest_rel)
            if dest_dir and dest_dir not in seen_dirs:
                print(f'mkdir -p "$ASSETS/{dest_dir}"')
                seen_dirs.add(dest_dir)
            print(f'{op} "{a.old}" "$ASSETS/{dest_rel}"')
        if missing:
            print("\n# NOTE: these manifest entries were not found in the repo working tree")
            print("# (expected for >100MB Drive files on a fresh machine — relocate them")
            print("#  manually if present elsewhere):")
            for a in missing:
                print(f'#   [{a.severity}] {a.old}  ->  $ASSETS/{_rel_dest(a)}   ({a.note})')
        return

    # --run
    copied = skipped = 0
    for a in present:
        a.dest.parent.mkdir(parents=True, exist_ok=True)
        if a.dest.exists():
            skipped += 1
            continue
        (shutil.move if args.move else shutil.copy2)(str(repo / a.old), str(a.dest))
        copied += 1
        print(f"  [{op}] {a.old} -> {a.dest}")
    print(f"\nDone. {op}={copied} already_present={skipped} missing_in_repo={len(missing)}")
    for a in missing:
        print(f"  [missing] {a.old}  ({a.note})")


if __name__ == "__main__":
    main()
