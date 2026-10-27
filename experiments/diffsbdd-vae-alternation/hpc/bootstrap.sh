#!/usr/bin/env bash
# Allocated runtime admission only. Provisioning is separate and must first
# satisfy the experiment's storage gate and authenticate both new environments.
set -euo pipefail
test -n "${LSB_JOBID:-}" || { echo 'An LSF allocation is required' >&2; exit 2; }
source_root="$1"
shift
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$source_root/src"
runtime=/work3/s234818/latentmol/env/diffsbdd-v1-attempt001/bin/python
reference_runtime=/work3/s234818/latentmol/env/diffsbdd-native-v1-attempt001/bin/python
test -x "$runtime" || { echo 'New combined runtime has not passed provisioning; no numerical stage started' >&2; exit 2; }
test -x "$reference_runtime" || { echo 'New original-compatible runtime has not passed provisioning; no numerical stage started' >&2; exit 2; }
"$runtime" -c "import importlib.metadata as m; assert m.version('torch')=='2.5.1+cu121'; assert m.version('torch-scatter').startswith('2.1.2'); assert m.version('openbabel-wheel')=='3.1.1.22'; assert m.version('boltz')=='2.2.1'"
"$reference_runtime" -c "import importlib.metadata as m; assert m.version('torch')=='2.0.1+cu118'; assert m.version('pytorch-lightning')=='1.8.4'"
exec "$@"
