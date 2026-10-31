# Chemistry strategies from random starts

User-authorized comparison of ideas1,2,3,4,5,7 and combinations,31October2026.
The user permits a less dense starting graph. These are fresh exploratory
experiments; existing trajectories, jobs, frozen scorers and saved results are
not modified. Current actual jobs and outcomes are in [campaign.json](campaign.json).

## Question and matrix

Can a random, initially imperfect molecule become valid while retaining useful
Protenix contact optimization? Ten strategies each use paired initialization
seeds17 and23,20,000 updates each. Two additional seed17 dense controls use the
exact old random step-zero logits: delayed baseline and combined strategy.
Total:22 trajectories,440,000 requested updates.

| Variant | Change from the shared delayed control |
| --- | --- |
| delayed | Original0.001 through5k, geometric to100 at50k |
| constant1 | Chemistry coefficient1 from the first step |
| constant10 | Chemistry coefficient10 from the first step |
| chemfirst | Coefficient10; binding off until valid connected argmax or5k cap, then2k-step ramp to full binding |
| severity | Atom-preserving severity-aware partial repair after original repair fails |
| adaptive | Base repair gradient influence target1, coefficient bounded0.1–100, EMA0.9 |
| neighbor | Existing frozen-neighbor probability guidance, additional weight0.1 |
| temperature2 | Constant temperature2, including scorer/preparation mapping and chain rule |
| earlyseverity | Constant1 plus severity repair |
| combined | Chemistry-first, early minimum1/adaptive1–100, severity, neighbor guidance and temperature2 |

The20k control intentionally retains the50k schedule horizon: it assesses the
early regime, not completion of that ramp. The combined arm's coefficient is
adaptive, not strictly constant. Adaptive balancing uses the **base repair**
gradient; the fixed-weight neighbor term is additional. Coefficients, targets
and source neighbor context remain detached/fixed within each trial.

Main starts use atom Dirichlet(1), a random Prüfer tree plus independent extra
edges with probability0.02. Present edges choose single/double80%/20%; selected
bond probability0.8, alternatives0.05. Nonedges use NONE0.98, alternatives0.005.
All rows have positive support. This produces connected heterogeneous random
graphs, not guaranteed valid molecules or a predefined carbon chain. Variants
share logits within seed; temperature2 intentionally changes soft probabilities.

## Shared settings and observations

Frozen original Protenix tiny/context/JQ1 pocket;25 occupied neutral slots,
LR0.3, model key17 independent of initialization seeds. Native and original
matrix repair computations remain FP32 with the existing deterministic runtime
profile. The existing neighbor term alone uses an explicit scoped FP64
calculation, restoring native FP32 afterward. Composition and confidence margin
are off; no additional noise. Exactly one finite fixed candidate per update,
without line search, clipping or gradient normalization.

The new experimental runner mixes unit component gradients after AD. This is
the mathematical weighted sum; it does not claim bitwise identity to historical
weighted-inside-AD arithmetic. Temperature pullbacks have independent synthetic
AD controls. Original native loader/hook/features/loss/evaluator/pocket/I/O ASTs
were checked against the authenticated scientific643 snapshot and match.

Save all scalar losses and fixed-target before/after values, argmax validity,
identity changes, per-block gradient norms, preparation route and timings.
Matrices/teacher receipts every20 states plus endpoints/latest/best/best-valid.
At0/every1000/end, retain all128 independent categorical samples, including
invalid draws, and report sanitized/connected validity and unique valid SMILES.
No sample repair, rejection or native rescoring is hidden in these diagnostics.
Contact scores are relaxed-model measurements, not binding/activity evidence.

## Execution and recovery

Canonical files are in this experiment's `scripts/` and `tests/` directories:
`protenix_chemistry_screen.py`, `prepare_chemistry_screen.py`,
`chemistry_screen_ops.py`, `chemistry_screen_control.py`, and
`chemistry_screen_stage.sh`. Shared optional severity repair is in the library.

Use32GBV100,8CPU/4GiB per slot. One allocated focused CPU check and four25-update
native implementation/timing probes precede automatic long-run dispatch. These
are finite-path checks, not a native/FD qualification campaign. Whole20k
reservations use the measured warm upper sample×1.25 plus30min setup/cleanup;
arms exceeding the queue's24h limit are recorded separately, without blocking
fitting arms or silently shortening experiments. Probe states are never adopted.

Own aggregate output cap12GiB (22×512MiB plus1GiB shared controls),65,000 files;
the common exploratory storage helper counts current bytes once. No cleanup.
Entry/periodic runtime/capacity checks and frozen/input identities remain.

Read actual `jobs.json`, each submission/release receipt and source snapshot
before recovery. Never replay an accepted or unknown submission, alter a queued
snapshot, or write into old output directories. Source01CPU29862450 retained40
passes/two fixture reference errors; its unstarted dependent jobs29862451/52 were
authenticated and retired. Source02 fixes only those reference tests plus
metadata timing/independent oversized-arm dispatch; production math is unchanged.

When results arrive, use the canonical saved-result reporting command/cache.
Extend its input adapter if necessary; never rerun an optimizer to fit a report.
Compare validity and valid-contact performance jointly, not minimum chemistry
loss alone. Two seeds provide an exploratory signal, not a robust efficacy claim.
