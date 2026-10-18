#!/bin/bash
# Invoke only after authenticating actual CPU DONE and its assessment hash.
set -euo pipefail
base=$1
manifest=$2
tag=$3
ready_job=$4
assessment_sha=$5
[[ "$base" == /work3/s234818/mosaic/releases/* && "$tag" =~ ^[a-z0-9-]+$ && "$ready_job" =~ ^[0-9]+$ ]]
[[ "$manifest" =~ ^[0-9a-f]{64}$ && "$assessment_sha" =~ ^[0-9a-f]{64}$ ]]
root=/zhome/95/e/206121/mosaic-runs
ready="$root/$tag-readiness"
assessment="$ready/payload/assessment.json"
[[ "$(bjobs -noheader -o stat "$ready_job" | tr -d '[:space:]')" == DONE ]]
echo "$assessment_sha  $assessment" | sha256sum -c -
bjobs -l "$ready_job" > "$ready/job-final.txt"
native="$root/$tag-native"
report="$root/$tag-report"
mkdir "$native" "$report"
response=$(bsub -J mosaic-brd4-chem50 -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' \
  -gpu 'num=1:mode=exclusive_process' -W 23:00 -w "done($ready_job)" \
  -oo "$native/stdout.%J.log" -eo "$native/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_chemistry.lsf" "$base/source" "$native" "$manifest" "$assessment" "$assessment_sha")
echo "$response" | tee "$native/submission.txt"
pattern='Job <([0-9]+)>'
[[ $response =~ $pattern ]]
native_job=${BASH_REMATCH[1]}
bsub -J mosaic-brd4-chem-report -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 01:00 \
  -w "ended($native_job)" -oo "$report/stdout.%J.log" -eo "$report/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_chemistry_report.lsf" "$base/source" "$report" "$manifest" "$native" "$ready" \
  | tee "$report/submission.txt"
