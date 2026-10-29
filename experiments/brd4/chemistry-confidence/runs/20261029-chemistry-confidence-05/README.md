# Confidence margin attempt005

Completed29October2026: actual CPU29848105 DONE,35cases passed/no failures,
errors/skips and12matched chemistry-only pairs completed. See [results](results.md),
[independent audit](independent-verification.json), [source identity](source-comparison.json)
and [actual same-job scheduler/LSF](scheduler.json).

Prepared after attempt004 passed35CPU fixtures and all12matched
pairs. Scientific library/test/comparison bytes remain exact004. Audit of the
harness found that child TMPDIR/TEMP/TMP were still inherited, although source,
outputs and caches were explicitly node-local. No saved-run write was observed
or authorized. This attempt explicitly resets all child temp variables to its
own compute-node `/tmp` subtree and independently verifies Python's actual temp
directory before testing. It also refuses any prior execution marker even if
the local transfer folder is missing. All original guards remain enabled.

New immutable source/full35fixtures+24chemistry-only trajectories completed with
the stronger locality proof. Earlier outcomes remain preserved;004's
numerical pass is not relabelled as failure or as this new source's pass.
Same4CPU hpcint allocation/840-second process deadline/15-minute site limit.
No native binding model or VAE training runs; composition stays off, saved runs
and current scientific settings/jobs stay unchanged. See [protocol](../../README.md).

Submitted11:50:49UTC, start11:50:51/end11:53:18 on n-62-29-6. Tests21.14seconds;
full assessment141.149seconds. Exact scientific/test/control/compare bytes from004
were unchanged; all stored scalar update records and sampled-validity counts
matched the engineering repeat. This is not independent-seed replication.

Source146files, manifestdf24a831267b9303748c228fb2aa9d9e31bb471bc0b615946de72a4d2a0716b7;
archive405849bytes SHA564d994e2cd336cf4ef20d502794e00a1b70f4cea856335edb2ef6c0f87ae58c.
Returned archive339473bytes SHA2b971b0a7e67c36e586b35e5e1cae477ca5cc450a01a09b95814d3813e6884b8;
all82members/72sample archives/source/raw35caseXML/LSF/scheduler/schema passed
independent stdlib audit. Runtime and tempdir-proof authenticate actual child
temporary paths beneath `/tmp/chem-confidence-squ_4cuh/tmp`.
Large arrays and exact raw evidence remain laptop results/hpc/TAG/allocated01.
