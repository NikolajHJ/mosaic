set -euo pipefail
root=/zhome/95/e/206121/mosaic-runs
base=/work3/s234818/mosaic/releases/20261016-brd4-combined-report-01
readiness="$root/20261016-brd4-comparison-readiness-01"
[[ "$(bjobs -noheader -o stat 29736903 | tr -d '[:space:]')" == DONE ]]
bjobs -l 29736903 > "$readiness/job-final.txt"
run="$root/20261016-brd4-combined-report-01"
mkdir "$run"
bsub -J mosaic-brd4-combined-report -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 02:30 -w 'done(29736903) && ended(29736629) && ended(29736735) && ended(29736761)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_combined_report.lsf" "$base/source" "$run" 40f9864f21f9c03c811492a15e5a7e2b3dee9e2f6b597ec3909f343dbeaf2075
