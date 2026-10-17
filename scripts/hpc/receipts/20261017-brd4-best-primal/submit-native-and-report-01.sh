#!/bin/bash
set -euo pipefail
plan_sha=$1
base=/work3/s234818/mosaic/releases/20261017-brd4-best-primal-03
root=/zhome/95/e/206121/mosaic-runs
readiness="$root/20261017-brd4-best-primal-readiness-02"
plan="$readiness/payload/plan/plan.json"
manifest=cb3f850acd5f0810bf6eb318869dc33f5657eb6c22f3f0c37e9c95327c424903
[[ "$plan_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$(bjobs -noheader -o stat 29745459 | tr -d '[:space:]')" == DONE ]]
echo "$plan_sha  $plan" | sha256sum -c -
bjobs -l 29745459 > "$readiness/job-final.txt"
jobs=()
for study in chain uniform noisy; do
  run="$root/20261017-brd4-$study-best-primal-01"
  mkdir "$run"
  response=$(bsub -J "mosaic-brd4-$study-best-primal" -q gpua10 -n 4 \
    -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -gpu 'num=1:mode=exclusive_process' \
    -W 16:00 -w 'done(29745459)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" \
    bash -l "$base/source/scripts/hpc/brd4_continue.lsf" "$base/source" "$run" \
    "$manifest" "$plan" "$plan_sha" "$study")
  echo "$response" | tee "$run/submission.txt"
  pattern='Job <([0-9]+)>'
  [[ $response =~ $pattern ]]
  jobs+=("${BASH_REMATCH[1]}")
done
report="$root/20261017-brd4-best-primal-report-01"
mkdir "$report"
bsub -J mosaic-brd4-best-primal-report -q hpc -n 2 -R 'span[hosts=1]' \
  -R 'rusage[mem=4GB]' -W 03:00 \
  -w "done(29745459) && ended(${jobs[0]}) && ended(${jobs[1]}) && ended(${jobs[2]})" \
  -oo "$report/stdout.%J.log" -eo "$report/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_continuation_report.lsf" "$base/source" "$report" "$manifest" \
  | tee "$report/submission.txt"
bjobs -w "${jobs[@]}"
