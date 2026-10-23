#!/usr/bin/env bash
# Separate official DeFoG environment and tiny native interface smoke.
# Invoke through run_job.sh from an immutable release inside an LSF allocation.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
case "${1:-}" in
    setup|smoke) task_mode="$1" ;;
    *) printf 'Usage: defog_interface_smoke.sh setup|smoke\n' >&2; exit 2 ;;
esac
task_root="${LATENTMOL_WORK_ROOT:-/work3/$USER/latentmol}"
task_env="$task_root/env/defog311"
task_asset="$task_root/cache/defog-official-guacamol"
if [ "$task_mode" = setup ]; then
    mkdir -p "$task_asset"
    if [ ! -x "$task_env/bin/python" ]; then
        "$HOME/.local/bin/uv" venv --python /appl9/python/3.11.9/bin/python3 "$task_env"
    fi
    "$HOME/.local/bin/uv" pip install --python "$task_env/bin/python" \
        torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    "$HOME/.local/bin/uv" pip install --python "$task_env/bin/python" \
        numpy==1.26.4 scipy==1.13.1 rdkit==2024.9.6 \
        pytorch-lightning==2.5.0 torchmetrics==1.9.0 torch_geometric==2.6.1 \
        omegaconf==2.3.0 pandas==2.2.3 matplotlib==3.10.1 \
        wandb==0.18.7 tqdm==4.65.0 pytest==8.3.5 \
        PyYAML==6.0.2 networkx==2.8.7 imageio==2.31.1 \
        hydra-core==1.3.2 overrides==7.3.1 PyGSP==0.5.1 seaborn==0.13.2
    "$HOME/.local/bin/uv" pip freeze --python "$task_env/bin/python" > \
        "$task_asset/resolved-requirements.txt"
    "$task_env/bin/python" -c 'import torch, torch_geometric, pytorch_lightning, rdkit; print("defog-env-ready", torch.__version__, torch_geometric.__version__, pytorch_lightning.__version__, rdkit.__version__)'
    export PYTHONPATH="$PWD/src:$task_asset/DeFoG/src:$task_asset/DeFoG${PYTHONPATH:+:$PYTHONPATH}"
    "$task_env/bin/python" - "$task_asset" <<'PY'
from pathlib import Path
import sys
import torch
from latentmol.defog.interface_smoke import _load_native_model, _verify_assets
asset = Path(sys.argv[1])
_verify_assets(asset / "DeFoG", asset / "guacamol.ckpt")
model = _load_native_model(asset / "DeFoG", asset / "guacamol.ckpt", torch.device("cpu"))
print("official-defog-checkpoint-loaded", dict(model.output_dims), dict(model.input_dims))
PY
    exit 0
fi
test -x "$task_env/bin/python"
source "$task_env/bin/activate"
export PYTHONPATH="$PWD/src:$task_asset/DeFoG/src:$task_asset/DeFoG${PYTHONPATH:+:$PYTHONPATH}"
python -m latentmol.experiments --run-root "$task_root/runs" \
    --name tests-stage3-native-defog-contract -- \
    python -m pytest -q tests/test_defog_contract.py
python -m latentmol.defog.interface_smoke \
    --defog-source "$task_asset/DeFoG" \
    --checkpoint "$task_asset/guacamol.ckpt" \
    --data "$task_root/data/zinc215k-seed17-v1" \
    --run-root "$task_root/runs"
