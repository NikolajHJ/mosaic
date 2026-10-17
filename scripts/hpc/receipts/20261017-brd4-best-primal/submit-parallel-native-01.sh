#!/bin/bash
set -euo pipefail
assessment_sha=$1
base=/work3/s234818/mosaic/releases/20261017-brd4-parallel-01
root=/zhome/95/e/206121/mosaic-runs
ready="$root/20261017-brd4-parallel-readiness-01"
manifest=1cf635c222bf750c298c7b4bcac30ec36eeb47e40f74480466d813cd64041504
plan="$root/20261017-brd4-best-primal-readiness-03/payload/plan/plan.json"
plan_sha=4837faf4a35e49927c8fe828b65dfb91ee871da9028ade9db27e6bfb9db1ade1
[[ "$assessment_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$(bjobs -noheader -o stat 29745498 | tr -d '[:space:]')" == DONE ]]
echo "$assessment_sha  $ready/payload/assessment.json" | sha256sum -c -
echo "$plan_sha  $plan" | sha256sum -c -
bjobs -l 29745498 > "$ready/job-final.txt"
jobs=()
tags=(15 30 50)
for index in 0 1 2; do
  tag=${tags[$index]}
  run="$root/20261017-brd4-chain$tag-best-primal-01"
  mkdir "$run"
  wall=16:00
  [[ "$index" == 0 ]] && wall=01:00
  response=$(bsub -J "mosaic-brd4-chain$tag-best-primal" -q gpua10 -n 4 \
    -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -gpu 'num=1:mode=exclusive_process' \
    -W "$wall" -w 'done(29745498)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" \
    bash -l "$base/source/scripts/hpc/brd4_continue.lsf" "$base/source" "$run" \
    "$manifest" "$plan" "$plan_sha" chain "$index")
  echo "$response" | tee "$run/submission.txt"
  pattern='Job <([0-9]+)>'
  [[ $response =~ $pattern ]]
  jobs+=("${BASH_REMATCH[1]}")
done
report="$root/20261017-brd4-best-primal-report-02"
mkdir "$report"
bsub -J mosaic-brd4-parallel-report -q hpc -n 2 -R 'span[hosts=1]' \
  -R 'rusage[mem=4GB]' -W 03:00 \
  -w "done(29745498) && ended(${jobs[0]}) && ended(${jobs[1]}) && ended(${jobs[2]}) && ended(29745473) && ended(29745474)" \
  -oo "$report/stdout.%J.log" -eo "$report/stderr.%J.log" \
  bash -l "$base/source/scripts/hpc/brd4_parallel_report.lsf" "$base/source" "$report" "$manifest" \
  | tee "$report/submission.txt"
bjobs -w "${jobs[@]}" 29745473 29745474
