#!/usr/bin/env bash
# Lightweight environment/download preparation on an HPC login node.
# Training, tests and dataset featurization belong in LSF jobs, not here.
set -euo pipefail
module load python3/3.11.9
TASK_ROOT="${LATENTMOL_WORK_ROOT:-/work3/$USER/latentmol}"
mkdir -p "$TASK_ROOT"/{env,raw,runs,logs,releases,cache}
export UV_CACHE_DIR="$TASK_ROOT/cache/uv"
if [ ! -x "$TASK_ROOT/env/ae/bin/python" ]; then
    uv venv --python "$(command -v python3)" "$TASK_ROOT/env/ae"
fi
uv pip install --python "$TASK_ROOT/env/ae/bin/python" torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$TASK_ROOT/env/ae/bin/python" \
    numpy==2.1.3 rdkit==2024.9.6 PyYAML==6.0.2 pytest==8.3.5 setuptools==75.8.0
uv pip freeze --python "$TASK_ROOT/env/ae/bin/python" > "$TASK_ROOT/env/ae/resolved-requirements.txt"

TASK_DATA_URL=https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/main/models/zinc/250k_rndm_zinc_drugs_clean_3.csv
TASK_DATA_SHA=35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae
if [ ! -f "$TASK_ROOT/raw/zinc250k.csv" ]; then
    curl --fail --location --retry 2 "$TASK_DATA_URL" -o "$TASK_ROOT/raw/zinc250k.csv.part"
    printf '%s  %s\n' "$TASK_DATA_SHA" "$TASK_ROOT/raw/zinc250k.csv.part" | sha256sum -c -
    mv "$TASK_ROOT/raw/zinc250k.csv.part" "$TASK_ROOT/raw/zinc250k.csv"
fi
printf '%s  %s\n' "$TASK_DATA_SHA" "$TASK_ROOT/raw/zinc250k.csv" | sha256sum -c -
printf '%s\n' "$TASK_DATA_URL" > "$TASK_ROOT/raw/source-url.txt"

