#!/usr/bin/env python3
"""Extract init_states.zip and place init_states under mesa/task_suites."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

from mesa import MESA_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unzip init_states.zip and move the extracted 'init_states' folder "
            "to mesa/task_suites."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing mesa/task_suites/init_states/mesa if present.",
    )
    return parser.parse_args()


def _find_extracted_init_states(extract_root: Path) -> Path:
    direct_path = extract_root / "mesa"
    if direct_path.is_dir():
        return direct_path

    candidates = sorted(
        (path for path in extract_root.rglob("mesa") if path.is_dir()),
        key=lambda p: len(p.parts),
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find an extracted 'mesa' directory in archive contents."
        )
    if len(candidates) > 1:
        candidate_str = "\n".join(str(path) for path in candidates)
        raise RuntimeError(
            "Found multiple 'mesa' directories in archive contents:\n"
            f"{candidate_str}"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    zip_path = Path(__file__).parent / "init_states.zip"
    destination_dir = Path(MESA_ROOT) / "task_suites" / "init_states"
    if destination_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file does not exist: {zip_path}")
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip path is not a file: {zip_path}")


    with tempfile.TemporaryDirectory(prefix="init_states_extract_") as tmp_dir:
        extract_root = Path(tmp_dir)
        with ZipFile(zip_path) as archive:
            archive.extractall(extract_root)

        extracted_init_states = _find_extracted_init_states(extract_root)
        shutil.move(str(extracted_init_states), str(destination_dir))

    print(f"Moved extracted init_states to: {destination_dir}")


if __name__ == "__main__":
    main()
