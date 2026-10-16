set -euo pipefail
project=/work3/s234818/mosaic
base="$project/releases/20261016-known-ligand-1000-05"
run="$project/runs/20261016-known-ligand-1000-evidence-01"
mkdir "$run"
bsub -J mosaic-known-1000-evidence -q hpc -n 1 -R 'rusage[mem=2GB]' -W 00:15 -w 'ended(29736412)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/known_ligand_collect.lsf" "$base/source" "$run" 544be74174b9568bb712e866483f4cd054400d80fab3ca880561267d3b23373b "$project/runs/20261016-known-ligand-1000-evidence-01.tar.gz" "$project/runs/20261016-known-ligand-readiness-01" "$project/runs/20261016-known-ligand-pilot-01" "$project/runs/20261016-known-ligand-assessment-01" "$project/runs/20261016-known-ligand-assessment-02" "$project/runs/20261016-known-ligand-1000-01" "$project/runs/20261016-known-ligand-1000-report-01" "$project/runs/20261016-known-ligand-collection-smoke-01"
