# Installation

This project uses `uv` for dependency and environment management. The installation instructions have been tested on Ubuntu 24.04.

## 0) Install uv

Follow the official `uv` installation instructions [found here](https://docs.astral.sh/uv/getting-started/installation/).

Verify your installation:

```bash
uv --version
```

## 1) Clone MESA

First clone the repository

```bash
git clone https://github.com/pairlab/mesa-env.git mesa
```

and navigate into it

```bash
cd mesa
```

## 2) Install project dependencies

From the repository root, sync the environment:

```bash
uv sync
```

If you need to export data in LeRobot format, install the optional LeRobot dependencies:

```bash
uv sync --extra lerobot
```

## 3) Run setup scripts

Run the setup helper, which executes all Python setup scripts in `scripts/setup/`:

```bash
./scripts/setup.sh
```

Alternatively, you can individually run the scripts found in `scripts/setup`.