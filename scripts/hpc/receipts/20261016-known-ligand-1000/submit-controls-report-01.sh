set -euo pipefail
project=/work3/s234818/mosaic
base="$project/releases/20261016-known-ligand-1000-07"
run="$project/runs/20261016-known-ligand-controls-report-01"
mkdir "$run"
bsub -J mosaic-known-controls-report -q hpc -n 1 -R 'rusage[mem=2GB]' -W 00:10 -w 'done(29736511) && done(29736412)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/augment_protenix_controls.lsf" "$base/source" "$run" 05cd383db826477b59389c38d99870ae55f3a6acd46667bc8c0ee7b9301050c0 "$project/runs/20261016-known-ligand-1000-report-01/payload/report" "$project/runs/20261016-known-ligand-hard-controls-01/payload/study" 7dc78247be4be56442ea4569c944d942dab1a8e31daf3b02e7cb36bb085335b9
