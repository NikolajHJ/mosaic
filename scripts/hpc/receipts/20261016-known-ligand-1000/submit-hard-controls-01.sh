set -euo pipefail
project=/work3/s234818/mosaic
echo '5937111ebd0f63cd3035e916237103af00b21941bed67c599b37cfa543edcc38  /work3/s234818/mosaic/known-ligand-1000-20261016-06.tar.gz' | sha256sum -c -
base="$project/releases/20261016-known-ligand-1000-06"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$project/known-ligand-1000-20261016-06.tar.gz" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/known_ligand_controls.lsf"
bash -n "$base/source/scripts/hpc/augment_protenix_controls.lsf"
run="$project/runs/20261016-known-ligand-hard-controls-01"
mkdir "$run"
bsub -J mosaic-known-hard-controls -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=8GB]' -gpu 'num=1:mode=exclusive_process' -W 00:15 -w 'done(29736126)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/known_ligand_controls.lsf" "$base/source" "$run" 1a9ae1fd0f4437571b2d2ac4fbb37fef0fce424330267f4a1fd6a24621269ef0 "$project/runs/20261016-known-ligand-1000-01/payload/study" 2a159de33e26f050a8dcb4e8bee8d0059af2a141aa1f3da787cc4219e3ce2fa1
