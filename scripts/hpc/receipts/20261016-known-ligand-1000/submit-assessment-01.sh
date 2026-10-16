set -euo pipefail
project=/work3/s234818/mosaic
echo '6979ab616a55432888bc67237c490f475a245e7a3bc5c4fd3fb31110cf310b68  /work3/s234818/mosaic/known-ligand-1000-20261016-03.tar.gz' | sha256sum -c -
base="$project/releases/20261016-known-ligand-1000-03"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$project/known-ligand-1000-20261016-03.tar.gz" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/known_ligand_assess.lsf"
run="$project/runs/20261016-known-ligand-assessment-01"
mkdir "$run"
bsub -J mosaic-known-assess -q hpc -n 1 -R 'rusage[mem=4GB]' -W 00:15 -w 'done(29736017)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/known_ligand_assess.lsf" "$base/source" "$run" 83f62cdb6d0265b09b033663f9b30712b8d319c6410a27dae7fe003256ab9507 "$project/runs/20261016-known-ligand-pilot-01/payload/study"
