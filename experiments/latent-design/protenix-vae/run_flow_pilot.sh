#!/usr/bin/env bash
set -euo pipefail
: "${LSB_JOBID:?Run the pilot inside an allocation}"

mode=$1
output=$2
plan=$3
plan_sha=$4
shift 4
root=/work3/s234818/latentmol
torch_python=$root/env/boltz311/bin/python
mosaic_python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
defog_python=$root/env/defog311/bin/python
checkpoint=$root/runs/source156-vae-epoch45-structural-attempt002/epoch0045/epoch_0045.pt
experiment=experiments/latent-design/protenix-vae
wrapper=scripts/hpc/protenix_latent_run.sh
assets=$root/protenix_occupancy
study=flow-pilot.json
fresh_config=flow-fresh-dtu.json
if [[ $# -ge 2 ]]; then
    study=$1
    fresh_config=$2
    shift 2
fi

stage() {
    local phase=$1
    shift
    bash "$wrapper" "$torch_python" "$experiment/flow_pilot.py" "$phase" \
        --checkpoint "$checkpoint" --config "$experiment/$study" --output "$output" \
        --mosaic-config "$experiment/native-dtu.json" --mosaic-python "$mosaic_python" \
        --defog-python "$defog_python" --plan "$plan" --plan-sha256 "$plan_sha" "$@"
}

fresh() {
    bash "$wrapper" /usr/bin/env \
        "PYTHONPATH=$PWD/src:$assets/environment-attempt003/deps" \
        "PROTENIX_ROOT_DIR=$assets/data" LAYERNORM_TYPE=torch \
        "$torch_python" -m latentmol.design.temperature_fresh --config "$experiment/$fresh_config" \
        --run "$1" --plan "$plan" --plan-sha256 "$plan_sha"
}

case "$mode" in
    rescore)
        /usr/bin/python3 - "$output" "$@" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

output, parent, tuning_sha, candidate_sha = map(str, sys.argv[1:])
parent = Path(parent)
payloads = {}
for name, expected in (("tuning.json", tuning_sha), ("fresh-candidates.json", candidate_sha)):
    raw = (parent / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Saved tuning input changed: " + name)
    payloads[name] = raw
panel = json.loads(payloads["tuning.json"])
panel["optimization_job_id"] = panel["job_id"]
panel["job_id"] = os.environ["LSB_JOBID"]
panel["rescoring_source"] = dict(path=str(parent), tuning_sha256=tuning_sha,
                                candidate_sha256=candidate_sha)
output = Path(output)
output.mkdir(exist_ok=False)
with (output / "tuning.json").open("x", encoding="utf-8") as stream:
    json.dump(panel, stream, indent=2, allow_nan=False)
with (output / "fresh-candidates.json").open("xb") as stream:
    stream.write(payloads["fresh-candidates.json"])
PY
        fresh "$output"
        ;;
    tune)
        stage tune "$@"
        fresh "$output"
        ;;
    select)
        mkdir -- "$output"
        exec /usr/bin/python3 "$experiment/flow_lr_selection.py" --config "$experiment/$study" \
            --output "$output/selection.json" "$@"
        ;;
    report)
        mkdir -- "$output"
        unset PYTHONHOME
        exec "$torch_python" "$experiment/flow_followup.py" --config "$experiment/$study" \
            --output "$output/comparison.json" "$@"
        ;;
    latent)
        stage teacher "$@"
        fresh "$output"
        stage students "$@"
        fresh "$output/students"
        ;;
    repair)
        mkdir -- "$output"
        stage repairs "$@"
        fresh "$output/repair"
        ;;
    *) echo "Unknown pilot mode $mode" >&2; exit 2 ;;
esac
