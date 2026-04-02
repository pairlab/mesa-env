#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="${SCRIPT_DIR}/setup"

shopt -s nullglob
python_scripts=("${SETUP_DIR}"/*.py)
shopt -u nullglob

if [[ ${#python_scripts[@]} -eq 0 ]]; then
  echo "No Python scripts found in ${SETUP_DIR}"
  exit 0
fi

for script in "${python_scripts[@]}"; do
  echo "Running ${script}..."
  uv run python "${script}"
done

echo "Setup complete."
