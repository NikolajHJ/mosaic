# Do epochs 43–45 improve generation and local robustness?

## Completed — 3 November 2026 11:44:41 CET

Native **29877255 DONE**; scientific summary completed in **863.58 s (14.39 min)**.
Frozen models/checkpoints unchanged. [Downloaded complete summary](../../outputs/hpc-results/vae-epoch43-45-comparison-20261103-a02/completed-summary.json).

| Epoch | Fresh prior argmax validity | Categorical validity | Exact reconstruction |
| --- | ---: | ---: | ---: |
| 43 | 75.49% | 64.35% | 99.84% |
| 44 | 74.88% | 64.26% | 99.88% |
| 45 | 75.41% | 63.30% | 99.87% |

Each prior/reconstruction panel has 10,000 observations. These point estimates
do not show a clear epoch-45 prior-validity gain; paired intervals and neighborhood
results remain in the complete summary. Training stays paused pending review
and the separately authorized [interpolation test](interpolation.md).

User authorized this evaluation on 3 November 2026. VAE training stays paused
at the completed epoch-45 checkpoint. This script has no training command,
optimizer, continuation callback or automatic checkpoint promotion.

## Matched protocol

Use [the config](epoch43-45-comparison.json) and the reusable
`latentmol.evaluation.checkpoint_comparison` runner. Evaluate the three frozen
checkpoints sequentially in **one job**; same graph schema, precision/device,
decoder batch 64 and temperature 1. No repair, rejection or valid-only filtering.

- **10,000 fresh N(0,I) latents per checkpoint**, five new seeds × 2,000. Reuse
  the identical vectors across checkpoints. Evaluate argmax and one categorical
  draw per vector using a common seeded RNG stream. Report these separately.
- **10,000 held-out validation molecules**, selected once without replacement.
  Exclude the declared historical 256-row health panel and 4,096-row panels
  (seeds 904017 and 17). This exclusion does not prove they were never examined
  in any other historical analysis. The final **test split stays untouched**.
  Encode to each checkpoint's deterministic posterior mean, then decode.
- **Latent perturbations:** the first 1,000 validation molecules and first
  1,000 prior vectors; four shared unit-RMS Gaussian directions at scales
  0.01, 0.05, 0.1 and 0.2. Around real molecules, multiply each coordinate by
  that checkpoint's posterior standard deviation. Around prior vectors use
  standard-normal units. Save actual L2 displacement: these two scales are
  different measurements, not directly equivalent neighborhoods.

This is 62,000 hard-graph attempts per checkpoint, 186,000 total. All raw hard
graphs, including invalid/duplicate outputs, are saved as gzip JSONL. Shared
prior vectors/directions and encoded posterior anchors are saved in NumPy
files. Molecule generation is assessed without external binding models.

## Readouts and interpretation

Report chemical validity, strict representation-roundtrip validity, connectivity,
empty graphs, component/atom-count histograms, failure reasons, unique-valid
yield, sulfur atom/molecule fractions, relaxed expected composition/atom entropy
and bounded fingerprint/scaffold diversity. Reconstruction includes exact
canonical identity, exact labels and reconstruction cross-entropies.
Neighborhoods include validity, identity changes, invalid-to-valid rescues and
fingerprint similarity to their own unperturbed decodes.

The chemistry classifier is the existing `classify_repair_graph`; there is no
new valence policy or sulfur restriction here. Chemical and representation
failures have separate counts. Validity is not synthesis or binding evidence.

Report Wilson intervals for independent prior/molecule panels and paired
validity/exact-reconstruction differences with gained/lost counts and an
approximate paired 95% interval. Neighborhood uncertainty clusters the four
directions by anchor; do not count them as four independent molecules. All
intervals are pointwise, approximate and unadjusted for multiple comparisons.
Checkpoints share one training lineage; this is not a training-seed replication.

## Run and budget

```sh
python -m latentmol.evaluation.checkpoint_comparison \
  --config experiments/vae/epoch43-45-comparison.json \
  --output NEW_OUTPUT --device cuda
```

Planned allocation: **one exclusive V100, four CPU slots × 4 GB, four hours**.
Use the existing node-local source launcher to isolate execution from scratch
mount stalls. It verifies a frozen source archive and preserves the scientific
worker's return code. Substantive models run only in the allocation. Frozen
model/checkpoint hashes, data/schema identities, finite inputs/outputs and
output collisions are checked. Quality declines are recorded as outcomes.

Small CPU fixtures exercise complete evaluation, matched panels, raw invalid
output retention, paired counts, checkpoint freezing and collision detection.
All **five tests passed**, including against the exact staged source snapshot;
the actual CLI help and shell syntax checks passed. These are small CPU fixtures,
not native GPU quality results.

Outputs: `summary.json`, `summary.md`, `comparison.csv`, `panels.json`, per-arm
summaries/raw graphs, and `failure.json` on failure. The main result is the
fresh-panel comparison, not a decision made automatically by the script.

## Submission — 3 November 2026 11:27 CET

**Launch record: 29877255, RUN since 11:30:06 CET on `gpuv100` at this snapshot.**
It reuses the identical source/config/checkpoints after the launch-only fix below.
Native LSF confirms
four CPU slots, 4 GB per slot resource reservation, four-hour wall limit and
`num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=16G`. This is one
evaluation job, no dependent training callback. The existing unrelated Protenix
job remains unchanged. Native GPU evaluation has no result yet.
At 11:30:56 the corrected wrapper has loaded the Python module and resolved
the allocated GPU without guessing; the launcher is still in input/model
startup (no ready marker or checkpoint quality result yet).
[Startup evidence](../../outputs/hpc-results/vae-epoch43-45-comparison-20261103-a02/status-1793701856843442900.json).

The exact three checkpoint files, dataset manifest, GPU routing helper and
uploaded archive were SHA-checked. Sealed source archive SHA:
`3daab857d33c997b6d1a1556a27e6c1d0562054bceb11dad92c1e8d6b80b31b3`.
The existing standard-library node-local launcher is copied to this job's own
home control directory; no existing release, callback or checkpoint was edited.

- [Build and allocation](../../outputs/hpc-submissions/vae-epoch43-45-comparison-20261103-a01/build.json)
- [Read-only asset verification](../../outputs/hpc-submissions/vae-epoch43-45-comparison-20261103-a01/preflight.json)
- [Current submission and exact scheduler command](../../outputs/hpc-submissions/vae-epoch43-45-comparison-20261103-a02/submission.json)
- [Local fixture results](../../outputs/implementation-checks/vae-epoch43-45-comparison-a01.xml)

Remote output:
`/work3/s234818/latentmol/runs/vae-epoch43-45-comparison-20261103-a02`.
Home logs and launch receipt:
`/zhome/95/e/206121/dbmol-control/vae-epoch43-45-comparison-20261103-a02`.
VAE training stays paused pending review; this job cannot resume it.

The first job 29877241 started at 11:27:27 and EXIT 127 at 11:27:30 before
importing the models: the nested `bash -lc` command could not find the relative
runner script. Archive inspection confirms that script exists. Metadata-only
replay after extraction finds it successfully, so the exact source-context
loss in the batch shell was not reproduced. The replacement removes that
nested shell/string expansion entirely: direct argv invokes `bash` from the
existing launcher's node-local working directory, as existing VAE launches do.
The failed receipt/logs and source remain intact; no scientific setting changed.
[First submission](../../outputs/hpc-submissions/vae-epoch43-45-comparison-20261103-a01/submission.json),
[terminal check and metadata replay](../../outputs/hpc-submissions/vae-epoch43-45-comparison-20261103-a02/prepare.json).
