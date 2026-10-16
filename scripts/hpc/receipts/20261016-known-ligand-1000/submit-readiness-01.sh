set -euo pipefail
project=/work3/s234818/mosaic
archive="$project/known-ligand-1000-20261016-02.tar.gz"
echo 'ff8dd32ace4d5b75b283dc58a65f9749b5e1b541388eb91b215a5fae463a8d35  /work3/s234818/mosaic/known-ligand-1000-20261016-02.tar.gz' | sha256sum -c -
base="$project/releases/20261016-known-ligand-1000-02"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$archive" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/known_ligand_readiness.lsf"
bash -n "$base/source/scripts/hpc/known_ligand_1000.lsf"
run="$project/runs/20261016-known-ligand-readiness-01"
mkdir "$run"
bsub -J mosaic-known-readiness -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/known_ligand_readiness.lsf" "$base/source" "$run" 4060ecaae546339cab71991ba6f42813c996451540eacdc9218b36e071c79cc1
