set -euo pipefail
project=/work3/s234818/mosaic
archive="$project/brd4-combined-report-20261016-01.tar.gz"
echo 'e5b047d437da7d1bfb9c23d55d5b20be6c12f147289188132aca62daa9b6b45c  /work3/s234818/mosaic/brd4-combined-report-20261016-01.tar.gz' | sha256sum -c -
base="$project/releases/20261016-brd4-combined-report-01"
mkdir "$base"
mkdir "$base/source"
tar -xzf "$archive" -C "$base/source"
chmod -R a-w "$base/source"
bash -n "$base/source/scripts/hpc/brd4_comparison_readiness.lsf"
bash -n "$base/source/scripts/hpc/brd4_combined_report.lsf"
root=/zhome/95/e/206121/mosaic-runs
run="$root/20261016-brd4-comparison-readiness-01"
mkdir "$run"
bsub -J mosaic-brd4-comparison-ready -q hpc -n 2 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -W 00:15 -oo "$run/stdout.%J.log" -eo "$run/stderr.%J.log" bash -l "$base/source/scripts/hpc/brd4_comparison_readiness.lsf" "$base/source" "$run" 40f9864f21f9c03c811492a15e5a7e2b3dee9e2f6b597ec3909f343dbeaf2075
