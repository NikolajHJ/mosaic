set -euo pipefail
root=/zhome/95/e/206121/mosaic-runs
base=/work3/s234818/mosaic/releases/20261016-brd4-noisy-uniform-5000-01
pilot="$root/20261016-brd4-pilot-01/payload/study"
readiness="$root/20261016-brd4-noisy-uniform-readiness-01"
assessment="$readiness/payload/assessment/assessment.json"
[[ "$(bjobs -noheader -o stat 29736756 | tr -d '[:space:]')" == DONE ]]
echo "4e61063a54e64f671a9c4ce97b009b0fa5f22e659401235c3d4eb4d14a441185  $assessment" | sha256sum -c -
echo "7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4  $pilot/summary.json" | sha256sum -c -
bjobs -l 29736756 > "$readiness/job-final.txt"
run="$root/20261016-brd4-noisy-uniform-5000-01"
mkdir "$run"
response=$(bsub -J mosaic-brd4-noisy-uniform-5000 -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -gpu 'num=1:mode=exclusive_process' -W 16:00 -w 'done(29736756)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_contact.lsf" "$base/source" "$run" ea4ceab78cb7605fdd715cb79bebacefc03ec8c8a5947834fa58540d5028c02e examples/configs/brd4_noisy_uniform_5000.json "$base/source/tests/data/brd4_3mxf/fixture.json" "$pilot" 7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4 "$assessment" 4e61063a54e64f671a9c4ce97b009b0fa5f22e659401235c3d4eb4d14a441185 1)
echo "$response"
pattern='Job <([0-9]+)>'
[[ $response =~ $pattern ]]
native=${BASH_REMATCH[1]}
report="$root/20261016-brd4-noisy-uniform-5000-report-01"
mkdir "$report"
bsub -J mosaic-brd4-noisy-uniform-report -q hpc -n 1 -R 'rusage[mem=4GB]' -W 01:30 -w "ended($native)" -oo "$report/stdout.%J.log" -eo "$report/stderr.%J.log" <<'JOB'
#!/bin/bash
set -euo pipefail
module load python3/3.12.11
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
release=/work3/s234818/mosaic/releases/20261016-brd4-noisy-uniform-5000-01/source
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-noisy-uniform-5000-report-01"
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
mkdir "$run/payload"
export MPLCONFIGDIR="$run/payload/mplcache"
"$python" "$release/scripts/verify_stage0_release.py" "$release" ea4ceab78cb7605fdd715cb79bebacefc03ec8c8a5947834fa58540d5028c02e
"$python" "$release/scripts/hpc/zhome_quota_preflight.py" --run-dir "$run/payload" --minimum-free-gib 5
"$python" "$release/scripts/render_protenix_trajectory.py" --study "$root/20261016-brd4-noisy-uniform-5000-01/payload/study" --output "$run/payload/report" > "$run/payload/render.log" 2>&1
JOB
