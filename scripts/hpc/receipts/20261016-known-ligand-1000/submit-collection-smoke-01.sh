set -euo pipefail
project=/work3/s234818/mosaic
echo 'eae3eb9109ba89e42bde1c262340abe536495f987f156bc73dd34150aa4fdecb  /work3/s234818/mosaic/known-ligand-1000-20261016-05.tar.gz' | sha256sum -c -
base="$project/releases/20261016-known-ligand-1000-05"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$project/known-ligand-1000-20261016-05.tar.gz" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/known_ligand_collect.lsf"
run="$project/runs/20261016-known-ligand-collection-smoke-01"
mkdir "$run"
bsub -J mosaic-known-collection-smoke -q hpc -n 1 -R 'rusage[mem=2GB]' -W 00:10 -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/known_ligand_collect.lsf" "$base/source" "$run" 544be74174b9568bb712e866483f4cd054400d80fab3ca880561267d3b23373b "$project/runs/20261016-known-ligand-collection-smoke-01.tar.gz" "$project/runs/20261016-known-ligand-assessment-01"
