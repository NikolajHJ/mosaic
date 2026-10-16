set -euo pipefail
root=/zhome/95/e/206121/mosaic-runs
base=/work3/s234818/mosaic/releases/20261016-brd4-uniform-5000-01
pilot="$root/20261016-brd4-pilot-01/payload/study"
readiness="$root/20261016-brd4-uniform-readiness-01"
assessment="$readiness/payload/assessment/assessment.json"
echo "58aa2f68a88a13fa5d21d6d504808850a51e864651f4bff2e82cea793fca2fa1  $assessment" | sha256sum -c -
echo "7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4  $pilot/summary.json" | sha256sum -c -
bjobs -l 29736713 > "$readiness/job-final.txt"
run="$root/20261016-brd4-uniform-5000-01"
mkdir "$run"
response=$(bsub -J mosaic-brd4-uniform-5000 -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -gpu 'num=1:mode=exclusive_process' -W 16:00 -w 'done(29736713)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_contact.lsf" "$base/source" "$run" cefff19ac9b68d4b546427565b1036fc625c0cde42f22a5157116c930d8252fd examples/configs/brd4_uniform_5000.json "$base/source/tests/data/brd4_3mxf/fixture.json" "$pilot" 7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4 "$assessment" 58aa2f68a88a13fa5d21d6d504808850a51e864651f4bff2e82cea793fca2fa1 1)
echo "$response"
pattern='Job <([0-9]+)>'
[[ $response =~ $pattern ]]
native=${BASH_REMATCH[1]}
report="$root/20261016-brd4-uniform-5000-report-01"
mkdir "$report"
bsub -J mosaic-brd4-uniform-report -q hpc -n 1 -R 'rusage[mem=4GB]' -W 01:30 -w "ended($native)" -oo "$report/stdout.%J.log" -eo "$report/stderr.%J.log" <<'JOB'
#!/bin/bash
set -euo pipefail
module load python3/3.12.11
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
release=/work3/s234818/mosaic/releases/20261016-brd4-uniform-5000-01/source
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-uniform-5000-report-01"
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
mkdir "$run/payload"
export MPLCONFIGDIR="$run/payload/mplcache"
"$python" "$release/scripts/verify_stage0_release.py" "$release" cefff19ac9b68d4b546427565b1036fc625c0cde42f22a5157116c930d8252fd
"$python" "$release/scripts/hpc/zhome_quota_preflight.py" --run-dir "$run/payload" --minimum-free-gib 5
"$python" "$release/scripts/render_protenix_trajectory.py" --study "$root/20261016-brd4-uniform-5000-01/payload/study" --output "$run/payload/report" > "$run/payload/render.log" 2>&1
JOB
