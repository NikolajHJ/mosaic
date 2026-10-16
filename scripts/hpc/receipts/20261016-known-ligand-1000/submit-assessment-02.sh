set -euo pipefail
project=/work3/s234818/mosaic
echo '9b7aaf4253f9380b694eae353e6ba7fd91aaccf26aea68245a11ecdf96c7c04a  /work3/s234818/mosaic/known-ligand-1000-20261016-04.tar.gz' | sha256sum -c -
base="$project/releases/20261016-known-ligand-1000-04"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$project/known-ligand-1000-20261016-04.tar.gz" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/known_ligand_assess.lsf"
run="$project/runs/20261016-known-ligand-assessment-02"
mkdir "$run"
bsub -J mosaic-known-recover-assess -q hpc -n 1 -R 'rusage[mem=4GB]' -W 00:15 -w 'done(29736017)' -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/known_ligand_assess.lsf" "$base/source" "$run" 8df25aaa80a129dd56ef9fdbf8e12a2f7c4f37bd8668d72f172e544d205fb990 "$project/runs/20261016-known-ligand-pilot-01/payload/study" dc1e1d948c99d487f5c06897964037cd2130a983f003ad4fc1643d996bad14ba
