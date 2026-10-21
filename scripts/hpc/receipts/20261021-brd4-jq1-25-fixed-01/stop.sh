set -uo pipefail
out=/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-01-stop
mkdir "$out"
date -Is > "$out/requested-at.txt"
bjobs -l 29770093 29770124 > "$out/before.txt" 2>&1
bkill 29770093 29770124 > "$out/bkill-reply.txt" 2>&1
cat "$out/bkill-reply.txt"
bjobs -l 29770093 29770124 > "$out/after-initial.txt" 2>&1
old=/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-protenix-lr03-01
for p in "$old/cpu/stderr.29776049.log" "$old/cpu/payload/hpc-contract.log" "$old/after-cpu/stderr.29776050.log"; do if test -f "$p"; then echo "$p"; tail -n 70 "$p"; fi; done
