#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.pipeline import reset_face_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset persistent known/stranger face database")
    parser.add_argument("--face-db-root", default="data/face_db")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive reset")
    args = parser.parse_args()

    if not bool(args.yes):
        raise SystemExit("Refusing to reset without --yes")

    face_db_root = Path(args.face_db_root)
    if not face_db_root.is_absolute():
        face_db_root = (PROJECT_ROOT / face_db_root).resolve()

    layout = reset_face_db(face_db_root)

    print("[face-db] reset complete")
    print(f"[face-db] root={layout['root']}")
    print(f"[face-db] known_identities={layout['known_identities']}")
    print(f"[face-db] stranger_sessions={layout['stranger_sessions']}")


if __name__ == "__main__":
    main()
