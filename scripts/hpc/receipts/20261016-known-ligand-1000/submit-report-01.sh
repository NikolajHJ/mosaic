set -euo pipefail
project=/work3/s234818/mosaic
base="$project/releases/20261016-known-ligand-1000-05"
run="$project/runs/20261016-known-ligand-1000-report-01"
mkdir "$run"
bsub -J mosaic-known-1000-report -q hpc -n 1 -R 'rusage[mem=4GB]' -W 00:15 -w 'ended(29736126)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/render_protenix_trajectory.lsf" "$base/source" "$run" "$project/runs/20261016-known-ligand-1000-01/payload/study" 6 544be74174b9568bb712e866483f4cd054400d80fab3ca880561267d3b23373b
