#!/bin/bash
set -euo pipefail
: "${LSB_JOBID:?Allocated GPU required}"
source_root="$1"
output_root="$2"
module load python3/3.12.11 cuda/12.9.1 cudnn/v9.13.0.50-prod-cuda-12.X
study="$source_root/experiments/learned-prior/protenix-defog-composition-30000"
export PYTHONPATH="$source_root/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
unset PYTHONHOME PYTHONOPTIMIZE

coordinator=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
latentmol=/work3/s234818/latentmol
cd "$source_root"
"$coordinator" scripts/hpc/storage_accounting.py --output-root "$output_root" \
    --cap-mib 128 --file-cap 4200 > "$output_root/storage.json"
exec "$coordinator" "$study/scripts/stage2_pair.py" \
    --config "$study/configs/stage2-pair.yaml" --out "$output_root/pair" \
    --boltz-context /zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared \
    --boltz-python "$latentmol/env/boltz311/bin/python" \
    --boltz-checkpoint "$latentmol/cache/boltz2/boltz2_aff.ckpt" \
    --teacher-python "$latentmol/env/defog311/bin/python" \
    --teacher-source "$latentmol/cache/defog-official-guacamol/DeFoG" \
    --teacher-checkpoint "$latentmol/cache/defog-official-guacamol/guacamol.ckpt"
