#!/usr/bin/env bash
# One isolated allocation; arguments are the immutable source archive, its
# SHA256, and an output root that must not yet exist.
set -euo pipefail
: "${LSB_JOBID:?An allocated GPU job is required}"
[[ $# == 3 ]] || { echo 'Usage: ipsae-hpc.sh ARCHIVE SHA256 NEW_OUTPUT' >&2; exit 2; }
archive=$1
sha=$2
out=$3
[[ "$sha" =~ ^[0-9a-f]{64}$ ]] || { echo 'Invalid archive SHA256' >&2; exit 2; }
[[ "$out" = /work3/s234818/latentmol/runs/ipsae-* ]] || { echo 'Invalid output root' >&2; exit 2; }
[[ "$(sha256sum "$archive" | cut -d' ' -f1)" == "$sha" ]] || { echo 'Archive hash mismatch' >&2; exit 1; }
mkdir "$out"
local_source=$(mktemp -d "${TMPDIR:-/tmp}/ipsae-${LSB_JOBID}-XXXXXXXX")
tar -xzf "$archive" -C "$local_source"
cd "$local_source"
module load python3/3.11.9
unset PYTHONHOME
export PYTHONPATH="$local_source/src:$local_source/tests"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 LATENTMOL_SOURCE_SHA256="$sha"
export IPSAE_OUTPUT_ROOT="$out"
ulimit -c 0
ownership=/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-release-26/source/scripts/cuda_ownership.py
[[ "$(sha256sum "$ownership" | cut -d' ' -f1)" == 578d2b6c9453ad14ce6778d306aa6c1d926acb442dcf8ae9f9a75bccc6b2e3cb ]] || { echo 'CUDA ownership helper changed' >&2; exit 1; }
export IPSAE_CUDA_HELPER="$ownership"
# Resolve LSF task GPU mappings before either PyTorch or CUDA initialization.
CUDA_VISIBLE_DEVICES=$(/usr/bin/python3 - <<'PY'
import importlib.util, os
spec = importlib.util.spec_from_file_location('cuda_ownership', os.environ['IPSAE_CUDA_HELPER'])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.resolve_visibility(os.environ)['selected_cuda_visible_devices'])
PY
)
export CUDA_VISIBLE_DEVICES
python=/work3/s234818/latentmol/env/boltz311/bin/python
checkpoint=/work3/s234818/latentmol/runs/source156-vae-epoch45-structural-attempt002/epoch0045/epoch_0045.pt
checkpoint_sha=5c4946c15515e70bd01281eea7b74693c18598c4b47f6724fcc9898deb1302be
experiment=experiments/latent-design/protenix-vae
"$python" - <<'PY'
import json, os
from pathlib import Path
source = Path('experiments/latent-design/protenix-vae')
out = Path(os.environ['IPSAE_OUTPUT_ROOT'])
for arm in ('baseline', 'ipsae'):
    value = json.loads((source / f'ipsae-{arm}-native.json').read_text())
    value['source_sha256'] = os.environ['LATENTMOL_SOURCE_SHA256']
    with (out / f'{arm}-native.json').open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
PY
for arm in ipsae baseline; do
  "$python" -u -m latentmol.design.latent_boltz \
    --config "$experiment/ipsae-lr3-500.json" \
    --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
    --native-config "$out/$arm-native.json" --output "$out/$arm" \
    --max-updates 500 --threads 4
done
"$python" -u "$experiment/ipsae_gradient_probe.py" \
  --run "$out/baseline" --run "$out/ipsae" \
  --native-config "$out/ipsae-native.json" --checkpoint "$checkpoint" \
  --checkpoint-sha256 "$checkpoint_sha" --ipsae-weight 0.3 \
  --steps 0,250,500 --output "$out/gradient-comparison.json"
