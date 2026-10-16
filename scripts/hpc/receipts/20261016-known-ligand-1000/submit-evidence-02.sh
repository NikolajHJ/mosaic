set -euo pipefail
project=/work3/s234818/mosaic
base="$project/releases/20261016-known-ligand-1000-07"
run="$project/runs/20261016-known-ligand-1000-evidence-02"
mkdir "$run"
bsub -J mosaic-known-1000-evidence-v2 -q hpc -n 1 -R 'rusage[mem=2GB]' -W 00:15 -w 'ended(29736521)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/known_ligand_collect.lsf" "$base/source" "$run" 05cd383db826477b59389c38d99870ae55f3a6acd46667bc8c0ee7b9301050c0 "$project/runs/20261016-known-ligand-1000-evidence-02.tar.gz" "$project/runs/20261016-known-ligand-readiness-01" "$project/runs/20261016-known-ligand-pilot-01" "$project/runs/20261016-known-ligand-assessment-01" "$project/runs/20261016-known-ligand-assessment-02" "$project/runs/20261016-known-ligand-1000-01" "$project/runs/20261016-known-ligand-1000-report-01" "$project/runs/20261016-known-ligand-collection-smoke-01" "$project/runs/20261016-known-ligand-1000-evidence-01" "$project/runs/20261016-known-ligand-hard-controls-01" "$project/runs/20261016-known-ligand-controls-report-01"
