#!/usr/bin/env bash
# protenix_latent_run.sh already bound this allocation's GPU; it is only a device router.
set -eo pipefail
exp="$PWD/experiments/latent-design/direct-jump"
source "$exp/protenix_env.sh"
export PYTHONPATH="$PWD/src:$DIRECT_JUMP_MOSAIC_SOURCE/src/mosaic/small_molecule/Stage2:$PYTHONPATH"
export DIRECT_REPAIR_PYTHON=/work3/s234818/latentmol/env/boltz311/bin/python
exec "$DIRECT_JUMP_PYTHON" -u "$exp/repair_search.py" "$@"
