#!/bin/bash
set -euo pipefail
root=/zhome/95/e/206121/mosaic-runs
[[ "$(bjobs -noheader -o stat 29745472 | tr -d '[:space:]')" == PEND ]]
[[ "$(bjobs -noheader -o stat 29745475 | tr -d '[:space:]')" == PEND ]]
bjobs -l 29745472 > "$root/20261017-brd4-chain-best-primal-01/job-before-parallel-split.txt"
bjobs -l 29745475 > "$root/20261017-brd4-best-primal-report-01/job-before-parallel-split.txt"
bkill 29745472 29745475
