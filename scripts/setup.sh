#!/usr/bin/env bash
# One-time setup after `uv sync`:
#   1. point robosuite's image convention at OpenCV (required by `import mesa`),
#   2. download the simulation assets from Hugging Face (albertwilcox/mesa-assets),
#   3. download the fixed evaluation initial states (albertwilcox/mesa-init-states).
# Extra arguments are forwarded to download_assets.py (e.g. `--no-bimesa` to skip the BiMESA-only objects).

set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup"

echo "Configuring robosuite macros..."
uv run python "${SETUP_DIR}/setup_robosuite_macros.py"

echo "Downloading simulation assets..."
uv run python "${SETUP_DIR}/download_assets.py" "$@"

echo "Downloading evaluation initial states..."
uv run python "${SETUP_DIR}/download_init_states.py"

echo "Setup complete."
