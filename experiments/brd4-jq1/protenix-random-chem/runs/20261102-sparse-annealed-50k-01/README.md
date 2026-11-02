# Sparse structural repair with chemistry, cooling and noise schedules

Question: can a sharper relaxed graph, a sustained strong chemistry phase and
annealed exploration produce a valid molecule with an improved native contact score?

User authorized one fresh 50,000-update run and a conditional additional 20,000
updates on 2 November 2026. Existing runs, snapshots, sources and results stay unchanged.

Actual GPU **29873389 RUN / gpuv100**, coherent 442 applied updates verified at
11:09:06 UTC. Allocated endpoint checker **29873393 PEND / hpc**, dependent on
`done(29873389)`. No tail GPU has been submitted yet.

| Global updates | Chemistry coefficient | Molecular temperature | Noise ratio |
|---|---|---|---|
| 0-20,000 | 1 | Linear 0.7 to 0.58 | Linear 0.4 to 0.1333 |
| 20,000-30,000 | Geometric 1 to 30 | Linear 0.58 to 0.52 | Linear 0.1333 to 0 |
| 30,000-50,000 | 30 | Linear 0.52 to 0.4 | 0 |
| Optional 50,000-70,000 | 100 | 0.4 | 0 |

Temperature is globally `0.7 - 0.3*min(update/50000,1)`, represented FP32.
Noise is globally `0.4*max(0,1-update/30000)` times each atom/bond block's
binding-only logit-gradient update norm, including LR. Its row-centered Gaussian
directions use independent PCG64 seed 230017; the complete stream is checkpointed.
Zero binding-gradient blocks receive zero noise. There are no draws after 30k.
Model key 17 remains separate and unchanged.

Initialization is the existing sparse seed17 recipe: Dirichlet atom identities,
a connected Pruefer tree plus 2% extra candidate edges, soft present/absent bond
probabilities. N25 occupied neutral slots, LR0.3, T-dependent graph probabilities,
original frozen Protenix/JQ1 context/objective and one finite candidate are retained.
Composition and confidence margin remain off. Structural proposals precede the
unchanged bounded beam/progress fallbacks and categorical repair loss. Targets
are detached and fixed during each optimizer trial; atom identities remain free
to change during optimization. No clipping, projection, backtracking or gradient
normalization is introduced.

The prescribed temperature is a dynamic stopped-gradient scalar, avoiding a
new JIT cache key for each update. The source and trial share the same teacher,
coefficient and temperature. Cooling the next source changes the model input,
so its native value/gradient and teacher are refreshed. This adds one native
backward evaluation during cooling, not an extra optimizer update. Once T is
clamped, the ordinary native-result reuse is restored. Source/trial/next
temperatures and comparison scope are saved explicitly.

## Conditional continuation

The allocated CPU checker authenticates successful full 50k completion,
50,001 contiguous saved states, 50,000 updates, the endpoint snapshot/key,
original frozen identities/source and actual job/LSF receipts. It checks the
actual saved endpoint argmax with chemical sanitization and connectivity.

- Valid connected endpoint: record `skipped_endpoint_already_valid`; no GPU submitted.
- Invalid endpoint: write a new immutable child plan and submit exactly one
  20k GPU continuation from the exact saved 50k logits, key, teacher and RNG state.
  Chemistry changes immediately from 30 to 100; temperature stays 0.4, noise
  stays zero, and binding/LR/model/context remain unchanged. New child output
  only; the 50k parent is never edited. The child has no further continuation rule.

The predicate concerns the optimized argmax, not the repair teacher or sampled
validity. A valid graph midway through the tail does not truncate its requested
20k budget. Unknown submission outcomes prevent blind retries.

## Resources and evidence

Both producer phases use one exclusive 32GB V100, `mps=no`, 8 CPU slots and
4GiB/slot. Main wall18h15: actual compatible V100 upper warm sample0.508s/update,
2x allowance for fresh native evaluation while cooling, 50k updates,25% margin
and setup/cleanup. Tail wall4h15 covers20k at constantT with ordinary native reuse.
Current V100 use was1/4 account GPUs; other families already had queued arms.
No healthy job was moved or cancelled. Group cap3GiB+256MiB/11,500 files,
including2GiB main,1GiB tail and controls, with live remaining-output/reserve checks.
No files were deleted.

Allocated CPU **29873381 DONE**, all79 cases passed with zero failures/errors/skips,
plus the actual parent CLI startup. Cases cover original behavior, the global
schedules, cold pullback gradients, dynamic-T JIT reuse, fixed trial targets/T,
refresh after cooling, uninterrupted/resumed RNG and immediate coefficient100,
and valid/invalid/failed/corrupt conditional outcomes. No model qualification is
inferred from these synthetic controls.

Canonical report adapter CPU **29873411 DONE**,44 focused cases passed. The
existing adapter accepts explicitly declared scheduled-T states and NPZs,
checks source/trial/next T, and retains fixed-temperature transition checks.
Trial contact at sourceT and next-state contact at cooledT remain separate
observations. Changing T/chemistry means the relaxed curve and surrogate totals
are not one fixed objective. The renderer is unchanged; the current published
report was not rebuilt for ordinary training progress. Two conditional completion
candidates cover the 50k parent and, if it exists, full0-70k history.

[Configuration](configuration.json), [actual campaign/jobs](campaign.json),
[byte/XML/receipt audit](verification.json). Raw receipts and reusable observer:
`results/hpc/20261102-sparse-annealed-50k-01/observe.py` (use fresh output labels).
Immutable remote source:
`/work3/s234818/mosaic/runs/20261102-sparse-annealed-50k-01/source01/source`.

Preparation for a NEW namespace only:

```text
python experiments/brd4-jq1/protenix-random-chem/scripts/prepare_chemistry_screen.py --repo . --tag NEW-TAG --version source01 --comparison annealed50k
```

Current preparation/staging/controls/dispatch already ran. Never replay them.
Read actual `run/jobs.json`, submission/held/release receipts and conditional
assessment before any recovery. Completion is exploratory, not derivative or
experimental binding certification. Keep all losses; matrices every20 plus
endpoints/latest/best,128 unfiltered categorical samples every1000 and endpoint.
