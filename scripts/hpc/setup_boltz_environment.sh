#!/usr/bin/env bash
# Pinned optional environment/assets; all installation runs on DTU HPC.
set -euo pipefail
: "${LSB_JOBID:?Run this bootstrap through LSF}"
TASK_ROOT="/work3/$USER/latentmol"
TASK_ENV="$TASK_ROOT/env/boltz311"
TASK_CACHE="$TASK_ROOT/cache/boltz2"
export UV_CACHE_DIR="$TASK_ROOT/cache/uv"
export TMPDIR="$TASK_ROOT/cache/tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
module load python3/3.11.9
unset PYTHONHOME PYTHONPATH
mkdir -p "$TASK_CACHE"
if [ ! -e "$TASK_ENV" ]; then
    "$HOME/.local/bin/uv" venv --python /appl9/python/3.11.9/bin/python3 "$TASK_ENV"
fi
"$HOME/.local/bin/uv" pip install --python "$TASK_ENV/bin/python" \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$HOME/.local/bin/uv" pip install --python "$TASK_ENV/bin/python" \
    'boltz @ git+https://github.com/jwohlwend/boltz.git@b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc' \
    torch==2.5.1 numpy==1.26.4 rdkit==2024.9.6 pytest==8.3.5
"$HOME/.local/bin/uv" pip freeze --python "$TASK_ENV/bin/python" > "$TASK_CACHE/environment-freeze.txt"
fetch_asset() {
    local task_name="$1" task_hash="$2"
    local task_target="$TASK_CACHE/$task_name"
    if [ -f "$task_target" ]; then
        printf '%s  %s\n' "$task_hash" "$task_target" | sha256sum -c -
        return
    fi
    curl --fail --location --retry 3 --output "$task_target.partial" \
        "https://huggingface.co/boltz-community/boltz-2/resolve/6fdef46d763fee7fbb83ca5501ccceff43b85607/$task_name"
    printf '%s  %s\n' "$task_hash" "$task_target.partial" | sha256sum -c -
    mv "$task_target.partial" "$task_target"
}
fetch_asset boltz2_aff.ckpt dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e
fetch_asset boltz2_conf.ckpt 090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1
fetch_asset mols.tar 39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7
if [ ! -d "$TASK_CACHE/mols" ]; then
    tar -xf "$TASK_CACHE/mols.tar" -C "$TASK_CACHE"
fi
printf 'Bootstrap completed for LSF job %s\n' "$LSB_JOBID"
