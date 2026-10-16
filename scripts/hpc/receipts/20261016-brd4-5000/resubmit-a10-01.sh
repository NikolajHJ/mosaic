set -euo pipefail
root=/zhome/95/e/206121/mosaic-runs
base=/work3/s234818/mosaic/releases/20261016-brd4-5000-02
old="$root/20261016-brd4-5000-02"
old_report="$root/20261016-brd4-5000-report-01"
pilot="$root/20261016-brd4-pilot-01"
assessment="$root/20261016-brd4-assessment-01/payload/assessment/assessment.json"
# Queue switching is disabled by eswitch. These are ordinary owner-authorized
# cancellation and submission operations to a queue already open to this user.
[[ $(bjobs -noheader -o stat 29736600 | xargs) == PEND ]]
[[ $(bjobs -noheader -o stat 29736612 | xargs) == PEND ]]
bjobs -l 29736600 > "$old/pre-cancellation.txt"
bjobs -l 29736612 > "$old_report/pre-cancellation.txt"
bread -i 0 -w 29736576 > "$old/pilot-power-notice-full.txt"
bkill 29736612 > "$old_report/cancellation.txt"
[[ $(bjobs -noheader -o stat 29736600 | xargs) == PEND ]]
bkill 29736600 > "$old/cancellation.txt"
[[ $(bjobs -noheader -o stat 29736600 | xargs) == EXIT ]]
[[ $(bjobs -noheader -o stat 29736612 | xargs) == EXIT ]]
bjobs -l 29736600 > "$old/cancellation-final.txt"
bjobs -l 29736612 > "$old_report/cancellation-final.txt"
echo "cfc44f1e6619a0d0e97bdb761c006be76b4bbad0547dd3cd74656a504ac93cbd  $assessment" | sha256sum -c -
echo "7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4  $pilot/payload/study/summary.json" | sha256sum -c -
run="$root/20261016-brd4-5000-03"
mkdir "$run"
response=$(bsub -J mosaic-brd4-5000-lr003 -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -gpu 'num=1:mode=exclusive_process' -W 16:00 -w 'done(29736579)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_contact.lsf" "$base/source" "$run" 80bb2413904b955cc4ebc25a0c01e1940fd185b3bddfb6cbd3afed76ad75a4f2 examples/configs/brd4_5000.json "$base/source/tests/data/brd4_3mxf/fixture.json" "$pilot/payload/study" 7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4 "$assessment" cfc44f1e6619a0d0e97bdb761c006be76b4bbad0547dd3cd74656a504ac93cbd)
echo "$response"
pattern='Job <([0-9]+)>'
[[ $response =~ $pattern ]]
native=${BASH_REMATCH[1]}
report="$root/20261016-brd4-5000-report-02"
mkdir "$report"
bsub -J mosaic-brd4-5000-report -q hpc -n 1 -R 'rusage[mem=4GB]' -W 01:30 -w "ended($native)" -oo "$report/stdout.%J.log" -eo "$report/stderr.%J.log" <<'JOB'
#!/bin/bash
set -euo pipefail
module load python3/3.12.11
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
release=/work3/s234818/mosaic/releases/20261016-brd4-5000-03/source
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-5000-report-02"
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
mkdir "$run/payload"
export MPLCONFIGDIR="$run/payload/mplcache"
"$python" "$release/scripts/verify_stage0_release.py" "$release" d6d4bbc17545df536e8dcea70f50d863f966b67ec25fb0a3b3f04f86739a9f9a
"$python" "$release/scripts/hpc/zhome_quota_preflight.py" --run-dir "$run/payload" --minimum-free-gib 5
"$python" "$release/scripts/render_protenix_trajectory.py" --study "$root/20261016-brd4-5000-03/payload/study" --output "$run/payload/report" > "$run/payload/render.log" 2>&1
JOB
