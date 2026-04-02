#!/usr/bin/env python3
"""Download zipped mesa assets and move them to mesa/sim/assets."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import gdown

ASSETS_ROOT = Path(__file__).resolve().parents[2] / "mesa" / "sim" / "assets"

ASSETS_URL = "https://drive.google.com/file/d/1I0jZn7_WII3Kywq3JeTupf9y5GErPUlO/view?usp=drive_link"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a zip archive from ASSETS_URL, unzip it, and move the "
            f"extracted assets folder to {ASSETS_ROOT}."
        )
    )
    parser.add_argument(
        "--assets-url",
        default=ASSETS_URL,
        help="Google Drive URL for the assets zip file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Overwrite {ASSETS_ROOT} if it already exists.",
    )
    return parser.parse_args()


def _find_extracted_assets(extract_root: Path) -> Path:
    direct_assets_path = extract_root / "assets"
    if direct_assets_path.is_dir():
        return direct_assets_path

    nested_assets_path = extract_root / "mesa" / "sim" / "assets"
    if nested_assets_path.is_dir():
        return nested_assets_path

    candidates = sorted(
        (path for path in extract_root.rglob("assets") if path.is_dir()),
        key=lambda path: len(path.parts),
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find an extracted 'assets' directory in archive contents."
        )
    if len(candidates) > 1:
        candidate_str = "\n".join(str(path) for path in candidates)
        raise RuntimeError(
            "Found multiple 'assets' directories in archive contents:\n"
            f"{candidate_str}"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()

    if ASSETS_ROOT.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {ASSETS_ROOT}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(ASSETS_ROOT)

    with tempfile.TemporaryDirectory(prefix="mesa_assets_download_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_path = tmp_path / "assets.zip"
        extract_root = tmp_path / "extracted"

        download_path = gdown.download(
            url=args.assets_url,
            output=str(zip_path),
            quiet=False,
            fuzzy=True,
        )
        if download_path is None or not zip_path.is_file():
            raise RuntimeError("Download failed: expected zip file was not created.")

        with ZipFile(zip_path) as archive:
            archive.extractall(extract_root)

        extracted_assets = _find_extracted_assets(extract_root)
        shutil.move(str(extracted_assets), str(ASSETS_ROOT))

    print(f"Moved extracted assets to: {ASSETS_ROOT}")


if __name__ == "__main__":
    main()