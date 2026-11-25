# Does ChEMBL broaden the VAE's training coverage?

The audit uses the structure-only ChEMBL 37 export from EMBL-EBI. It streams
the published SMILES through the current MOSES graph schema and exact
encode/decode round-trip. The canonical identity is stereo-free, as in the
existing VAE. Molecules must be connected and supported by the checkpoint's
50-heavy-atom vocabulary. Counts distinguish rejected source rows, duplicate
ChEMBL identities, exact overlap with each MOSES split, and novel eligible
identities. Ring-containing Bemis–Murcko scaffolds are compared with both the
MOSES training split and all MOSES splits. Acyclic molecules are counted
separately.

The release-pinned structure file is
`chembl_37_chemreps.txt.gz`, 292,540,141 bytes, with published SHA-256
`ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7`.
The local acquisition used the EBI HTTP endpoint and matched the release
checksum. [ChEMBL 37 release](https://chembl.blogspot.com/2026/05/chembl-37-is-here-bigger-cleaner-and.html).

The [matched comparison](epoch45-chembl-comparison.json) starts both arms
from epoch 45 with canonical input ordering. Each arm receives 400 updates
and 12,800 examples. The mixture arm replaces 1,920 of those example slots
with ChEMBL identities absent from every MOSES split. Both arms use the same
optimizer settings, reconstruction and KL objectives, and frozen prior
anchor. Interpolation validity is evaluated on held-out pairs to measure the
effect of changing the data source.

Evaluation uses the same MOSES holdout, prior draws and interpolation paths
in both arms. A separate ChEMBL panel tests reconstruction on scaffolds
excluded from the mixture's training draw and absent from all MOSES splits.
Keep the existing MOSES validation and final test assignments intact.

## Audit run

DTU job **29891386** started on `milan` on 4 November 2026 at 15:27 CET.
The allocation has 12 CPU slots, 11 chemistry workers and a 90-minute
wall limit. The worker hashes both input sources before reading the
complete ChEMBL release and all three MOSES splits.
[Submission and command](../../outputs/hpc-submissions/chembl37-moses-audit-20261104a/submission.json).

The audit writes `summary.json` and `eligible.csv.gz` under
`/work3/s234818/latentmol/runs/chembl37-moses-audit-20261104a`.

**Audit completed** in 482 seconds. All 2,897,819 source rows were read.
The filters rejected 615,710 rows and 107,525 passing rows were duplicate
canonical graphs. Of 2,174,584 distinct passing graphs, 92,425 were already
in MOSES (90,300 training, 1,047 validation and 1,078 test).

The remaining **2,082,159 novel eligible molecules** include **1,584,232**
with a ring scaffold absent from all MOSES splits, spanning **612,327**
distinct new Bemis–Murcko scaffolds. The novel set also contains 17,599
acyclic molecules. The largest rejection categories were graph round-trip
(349,270), more than 50 heavy atoms (129,141) and disconnected structures
(120,712). The eligible table is about 50 MiB compressed.
[Verified audit summary](../../outputs/hpc-results/chembl37-moses-audit-20261104a/summary.json).

## Comparison entry point

The comparison requires a complete audit. It verifies the summary digest,
the eligible CSV digest and the original MOSES manifest before training.
The six local audit, comparison and interpolation tests passed, including
an end-to-end fixture with both arms and all evaluation panels. The
[staged source](../../outputs/source-releases/chembl37-matched-comparison-20261104a.tar.build.json)
is ready for an allocated job.

The deterministic draw from the completed audit produced 1,920 training
molecules covering 1,746 ring scaffolds and 1,024 held-out molecules
covering 983 ring scaffolds. Every selected graph re-encoded exactly.
[Real-data panel check](../../outputs/hpc-results/chembl37-moses-audit-20261104a/comparison-panel-check.json).

In an allocated GPU job using the existing checkpoint-comparison shell:

```sh
LATENTMOL_EVALUATION_MODULE=latentmol.training.chembl_comparison \
bash experiments/vae/run_checkpoint_comparison.sh \
  --config experiments/vae/epoch45-chembl-comparison.json \
  --audit /work3/s234818/latentmol/runs/chembl37-moses-audit-20261104a \
  --audit-sha256 ba70ac2018135ce6719694c60cec9434c8b9f7189652d295c3163632e3ab0759 \
  --output NEW_OUTPUT_DIRECTORY --device cuda
```

## Closest-neighbor coverage probe

Local CPU, 4 November 2026. Drew 250 MOSES training and 250 novel ChEMBL
queries with seed 1749104, distinct from a mixed 100,000-molecule reference
sample. The 25,000-reference panel is a prefix of the same shuffled pool.
Used exact top-one radius-2 Morgan fingerprint Tanimoto within each sample,
with no atom-count, scaffold or mapping prefilter and no fallback neighbor.
Equal-size pairs use the existing one-second atom matcher and common-core
threshold of 0.8. The stricter column additionally applies the pilot's
similarity, atom-size and edit limits.

| Reference size | Equal atom count | Atom map passes | Also passes similarity and edit limits |
| ---: | ---: | ---: | ---: |
| 25,000 | 61/500 | 14/500 | 6/500 |
| 100,000 | 66/500 | 24/500 | 13/500 |

In the larger panel, 434 pairs differed in atom count, 35 had too little
common core and seven hit the mapping timeout. The estimate covers sampled
reference pools. Full-corpus coverage requires a search over the final
training split. The check took 46 seconds and used no model weights.
[Summary and per-query observations](../../outputs/nearest-pair-coverage/20261104a/summary.json).

### One terminal atom

The optional-atom matcher keeps the entire shared core identical in element,
formal charge and bond type. A single extra terminal atom aligns to an absent
slot in the smaller endpoint. The isolated interpolation loss weights its
conditional charge and bond targets by target occupancy at 25%, 50% and 75%.
The decoder then applies its normal occupancy gate to obtain marginal bonds.
Either canonical endpoint frame can be scored.

The local replay admitted **9 of 136** one-atom-difference candidates without
trying any alternate neighbors. Eight also passed the 0.65 similarity limit.
Combining the nine with the original 24 equal-size mapped pairs raises
sampled coverage from **24/500 (4.8%) to 33/500 (6.6%)** before the separate
similarity and edit limits. The replay took 34 seconds.
[Replay](../../outputs/nearest-pair-coverage/20261104a/optional-atom-replay.json).

Twenty-one focused fixtures and regressions passed. The isolated loss gave
finite gradients through the frozen epoch-45 decoder for a fluorine and an
iodine addition. A latent step of length 0.01 lowered the loss for both.
These checks cover the mapping and objective. Training quality and full-corpus
coverage remain to be measured.
[Latent smoke](../../outputs/nearest-pair-coverage/20261104a/optional-atom-latent-smoke.json).
