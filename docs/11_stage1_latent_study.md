# Stage 1: training-code priors and paired latent exploration

Recorded 28 September 2026 from source016. **Both jobs completed; both density
promotion gates failed.** AE job **29505503** and VAE job **29505504** each
passed 21 selected tests before the study (15.33 and 14.37 seconds). Each requested
one gpua10 GPU, four CPU cores, 4 GB per core and six hours. Results below are
followed by the unchanged executed method. Submission history and evidence
identities are retained in [the ledger](05_experiment_ledger.md).

The question is whether a cheap empirical latent prior preserves useful hard
molecular changes better than size control or a local trust region. This study
uses no Boltz, synthesis or off-target scorer. It does not retrain the graph
models and does not read the original test split.

## Observed results and decision

AE run `20260928T063833Z_ae-latent-stage1_580abd75` and VAE run
`20260928T064117Z_vae-latent-stage1_bd6a6a7a` both have completed manifests.
Each study took approximately 130 seconds from its recorded start to finish,
excluding the preceding regression run and scheduler wait. No per-arm runtime
comparison is available in this release.

Each arm contains **2,304 attempts from the same 256 pilot molecules**, with
three radii and three repeats. All attempts, including invalid outputs, stay in
the denominator. A size-compliant change is a valid output different from its
valid unperturbed baseline and within two heavy atoms of its input molecule.

| Model / arm | Valid / 2,304 | Valid fraction | Changed and size-compliant / 2,304 |
|---|---:|---:|---:|
| AE / perturbed | 1,488 | 64.58% | 335 |
| AE / size | 1,500 | 65.10% | 345 |
| AE / trust | 2,018 | 87.59% | 283 |
| AE / density | 1,459 | 63.32% | 379 |
| VAE / perturbed | 1,191 | 51.69% | 310 |
| VAE / size | 1,191 | 51.69% | 311 |
| VAE / trust | 1,629 | 70.70% | 290 |
| VAE / density | 1,144 | 49.65% | 298 |

The following differences are **density minus control**, in percentage points,
with seed-clustered 95% bootstrap intervals. They preserve dependence among the
nine attempts from each molecule.

| Model / control | Changed-size-compliant gain [95% interval] | Validity gain [95% interval] |
|---|---:|---:|
| AE / trust | +4.17 [+2.56, +5.60] | -24.26 [-25.78, -22.66] |
| AE / size | +1.48 [+0.48, +2.47] | -1.78 [-2.78, -0.87] |
| VAE / trust | +0.35 [-1.04, +1.69] | -21.05 [-22.92, -19.18] |
| VAE / size | -0.56 [-1.22, 0.00] | -2.04 [-3.13, -1.04] |

**The predeclared density-promotion flag is false for both models.** AE density
increased changed-candidate yield but failed the validity requirement, including
the lower-bound requirement against size. VAE density did not establish a
positive changed-candidate gain and failed the validity requirement. Hard size
compliance among density-arm changed-valid outputs was 100% for both; identity
coverage was 1.8125 for AE and 1.7375 for VAE. Those checks do not offset the
failed conditions. The 200-molecule promotion reserve was **not used**.

Calibration chose bandwidth **1.4 / 1.0** for AE/VAE and trust radius **3.5**
for both. The best calibration density-validity AUC was **0.4808 / 0.6436**;
these same-cohort calibration values do not establish a reliable validity
classifier. PCA retained **63.54% / 53.31%** of normalized training-code variance
in 32 components. The residual guards and limited feature projection are part
of the tested method, not evidence that every relevant chemical direction was
captured.

| Model / interpolation | Valid intermediates / 384 | Valid and distinct from both endpoints / 384 | All five valid, with some distinct intermediate / 128 paths |
|---|---:|---:|---:|
| AE / near | 87 | 61 | 1 |
| AE / far | 141 | 91 | 1 |
| VAE / near | 130 | 48 | 1 |
| VAE / far | 174 | 105 | 4 |

VAE had more valid sampled intermediate points, while AE had stronger endpoint
reconstruction in the earlier audit. Most complete interpolated paths failed
the stronger all-valid criterion. The nearest input partners here had mean
Tanimoto only 0.2709 (far partners 0.0266): “near” is relative to this cohort.
Neither these counts nor linear interpolation establish semantic continuity or
a general latent-quality winner. This result argues against promoting the
tested density restraint; it does not rule out every empirical prior or other
calibration method.

Evidence: [AE summary](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage1/20260928T063833Z_ae-latent-stage1_580abd75/summary.json>) and
[VAE summary](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage1/20260928T064117Z_vae-latent-stage1_bd6a6a7a/summary.json>).
The verified results archive is `stage1-source016-results-20260928T0645Z.tar.gz`,
SHA256 `8828a2a6ad08344984d604cf9d538ec44ff14e61a10fd30523438a06e69fbe87`.

## Fixed model and data identities

### Source017 paired interpolation follow-up

CPU job **29505583 completed**, after one test passed, as audit run
`20260928T073544Z_ae-vae-interpolation-comparison_4e92495d`. It compares the
already recorded source016 paths; it makes no new model calls. Source017 SHA256
is `06ef6636810f195b10ff0eea98c4458a297bf3c19685ded3532f96fa578b9686`.

The audit verifies matching source cohort identities, graph schema, dataset and
split, pair identities and interpolation grids. It separates all paths from
the subset whose two endpoints decode validly in both models. Bootstrap
resampling groups the near/far paths owned by the same input anchor.

| Paired cohort | Intermediate attempts per model | VAE minus AE validity | Anchor-clustered 95% interval |
|---|---:|---:|---:|
| 223 paths with common valid endpoints | 669 | +11.510 percentage points | [+7.692, +15.408] |
| All 256 paths | 768 | +9.896 percentage points | [+6.380, +13.672] |

These are conditional intervals for this fixed partner pool and single model
training seed. Reuse of right endpoints creates dependence beyond the anchor
clustering; the intervals are not unconditional uncertainty over chemical
space. Only **three paths** additionally had valid midpoint decodes in both
models. Their mean minimum midpoint-to-decoded-endpoint Tanimoto was **0.05673
for AE / 0.05182 for VAE**. That tiny selected subset cannot establish semantic
continuity. The follow-up supports better sampled intermediate validity for
this VAE on these paths, while leaving the failed density-promotion decisions
and absence of a general optimization-quality winner unchanged.

### Frozen identities

The paired-audit [summary](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage1/paired-audit-source017/summary.json>),
[run manifest](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage1/paired-audit-source017/run.json>) and
[common-midpoint records](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage1/paired-audit-source017/common_midpoint_similarity.json>) are now downloaded.
Their local SHA256 values match the verified HPC transfer receipt in
[the ledger](05_experiment_ledger.md). This verifies artifact identity, not
additional model predictions or broader scientific claims.

The selected AE epoch **108** and VAE epoch **94** checkpoints are the sealed
best-validation-validity choices from [the full-training audit](10_full_training_audit.md).
Their training budgets differ, so this compares the selected usable models,
not identical compute budgets. Both have one model-training seed.

- AE checkpoint SHA256:
  `1b1d2a32aedfadb376d5a5e2d8802d574ed5c1d72851ed79517acaef418e9b95`.
- VAE checkpoint SHA256:
  `80b22b9c5a5e9010457b49246f830f0425c5578458eab28ddf2a10f5f632cee9`.
- Submitted archive `latentmol-source-016-stage1.tar.gz`, SHA256:
  `39e4232444ba5c8fc89d5ff3543a7a2efce8325ac7e1302b582f528d1bb044c9`.
- Remote release: `/work3/s234818/latentmol/releases/source-016-stage1`.
- Data: `/work3/s234818/latentmol/data/zinc215k-seed17-v1`, with
  **215,000 training / 7,347 validation / 7,347 test** molecules. Dataset/split
  hashes are recorded in the submission ledger and run configuration.

The launch script verifies the expected checkpoint hash, then runs
`test_latent_priors.py`, `test_models.py` and `test_evaluation.py`. The study
starts only if those selected regressions pass. Model loading checks graph
schema, and the study checks dataset identity and declared AE/VAE mode. Both
encoder and decoder stay frozen in evaluation mode. All encoding uses the
deterministic mean, including for the VAE; its empirical reference is therefore
neither standard-normal draws nor sampled aggregate posterior codes.

## Code responsibilities

| Module | Responsibility |
|---|---|
| `latent_priors.py` | Training-only scaling, PCA geometry, Gaussian KDE and residual guard; serializable frozen reference |
| `latent_optimization.py` | Batched Adam updates of normalized latent coordinates, with a movement bound and finite-value checks |
| `latent_study.py` | Cohorts, calibration, objectives, paired arms, chemistry evaluation, interpolation, metrics and artifacts |
| `scripts/hpc/latent_stage1.sh` | Checkpoint identity check, regression gate and study invocation |

Chemistry conversion and checking occur after hard decoding. They are not in
the differentiable objective. Frozen decoder probabilities provide expected
atom count for the size term; density and distance act directly on latents.

## Cohorts and matched starts

A validation permutation with seed **20260928** fixes three disjoint groups
before calibration: **256 calibration**, **256 pilot**, and **200 reserved
promotion** molecules. The submitted job records the reserve identities but
does not evaluate or optimize them. These are development cohorts: validation
already contributed to model selection and is not an untouched test set.

AE and VAE use the same molecule identities and numerical direction seeds.
Directions are unit vectors from a Gaussian draw with seeds **9107, 9108 and
9109**. Starting displacements have normalized L2 radii **2, 4 and 6**, with
three repeats at each radius. The four arms share every start within a model.
The corresponding direction vectors in independently learned latent spaces do
not imply the same molecular edit.

Unperturbed decoded means supply each molecule's baseline. A changed outcome
must differ from its own decoded baseline, not merely from the input molecule.
Invalid baselines are retained, and their rescues are counted separately.

## Training-only geometry and calibration

Encode all **215,000 training means**. Per dimension, subtract the training mean
and divide by the population standard deviation, floored at the larger of
**0.1 times the median training standard deviation** and **0.001**. Fit the
normalized covariance on all training codes and retain **32 principal
components**. Whiten these components using eigenvalues floored at **0.0001**.
Select **20,000** training references using seed **17** for the Gaussian KDE.

The density is computed in these 32 whitened coordinates. It is not a normalized
density over the full 128-dimensional space and is not molecular likelihood.
A separate guard penalizes the norm of the discarded-coordinate residual above
its **95th percentile among training codes**. Record PCA variance retained so
the dimensionality reduction can be assessed.

Calibrate independently for AE and VAE:

1. Perturb the calibration means at radius 4 using direction repeat 0. Among
   bandwidths **0.7, 1.0 and 1.4**, choose the highest AUC for distinguishing
   valid from invalid hard decodes by density. AUC requires at least ten examples
   of each class; if none is evaluable, bandwidth 1.0 remains the default. Ties
   retain the earlier candidate.
2. Set the density floor to the **10th percentile of densities at valid,
   unperturbed calibration baselines**. No valid baseline is a failure, not a
   reason to substitute pilot molecules.
3. Compare trust radii **1.5, 2.5 and 3.5** on that same calibration perturbation.
   Choose the largest changed-valid count, then total validity, then the smaller
   radius. This calibration comparator is distinct from the final
   hard-size-compliant promotion metric.

The calibrated bandwidth, threshold and trust radius are fixed before pilot
scoring. Raw density values are not comparable between the two model spaces.

## Four arms and the latent update

Write normalized position as `u`, its unperturbed anchor as `u0`, and the input
heavy-atom count as `n0`. Define the shared soft size penalty:

```text
S(u) = 0.2 * max(0, |expected_atom_count(decode(u)) - n0| - 2)^2
```

| Arm | Objective or operation |
|---|---|
| `perturbed` | Decode the shared displaced start without optimization |
| `size` | Minimize `S(u)` |
| `trust` | Minimize `S(u) + max(0, ||u-u0|| - calibrated_radius)^2` |
| `density` | Minimize `S(u) + max(0, threshold-log_KDE(u))^2 + max(0, PCA_residual(u)-training_limit)^2` |

Each optimized arm uses **ten Adam steps**, learning rate **0.15**, in batches
of **16** candidates. After every step, each candidate is projected into an L2
ball of radius **2** around its *perturbed starting position*. This shared
movement cap is separate from the calibrated trust arm's distance to `u0`.
Objectives and latent gradients must be finite. Gradient norms and movement are
logged; no gradient-norm clipping policy is implied by the position projection.

There are **2,304 attempts per arm per model**: 256 pilot molecules times three
radii times three repeats. Repeated attempts remain correlated within their
seed molecule. The soft size term does not guarantee hard size compliance;
evaluation separately requires a hard count within two atoms of the input.

## Interpolation diagnostic

For the first **128 pilot molecules**, choose the nearest and farthest other
pilot molecule by Morgan-fingerprint Tanimoto similarity (radius 2, 2,048 bits).
AE and VAE share these input-chemistry pairs. Linearly interpolate their encoded
means at **0, 0.25, 0.5, 0.75 and 1**.

Each near/far group has 128 paths and 384 intermediate attempts. Record valid
intermediates, valid intermediates distinct from both decoded endpoints, and
paths where all five points are valid with at least one distinct intermediate.
Per-point similarities are measured against decoded endpoints. This measures
sampled paths only, not continuous manifold validity or chemical feasibility
between sampled points.

## Outcomes and the promotion decision

Hard decoding uses the project's conditional-bond hardening and RDKit checks
without repair. Every attempt retains baseline, perturbed start, endpoint,
SMILES or failure reason, expected/discrete size, densities and PCA residual.
Summaries report validity, valid-to-different-valid changes, hard-size-compliant
changes, invalid-to-valid rescues, unchanged valid outputs, unique identities,
successful seed counts and novelty relative to the full training split.

Density is compared with trust, size and unoptimized perturbations. Paired
bootstrap intervals use **2,000 resamples**, seed **20260928**, resampling whole
seed molecules while retaining their nine correlated attempts. The implemented
density-promotion flag requires all of the following:

- Against both trust and size, the 95% interval's lower bound for the gain in
  valid, changed, hard-size-compliant fraction is **greater than zero**.
- Against both controls, the validity-gain lower bound is **greater than -0.02**.
- The ratio of distinct successful identities to seed molecules with success
  is **at least 0.8**. This is an identity-coverage ratio, not a bounded diversity
  fraction; several distinct successes per seed can make it exceed one.
- **At least 90%** of density-arm changed-valid outputs satisfy the hard size
  range.

The flag records a decision signal; it does not automatically run the 200-molecule
reserve. Failed or inconclusive gates must remain visible. One training seed,
selected checkpoint budgets, a development cohort and finite sampling limit the
conclusion. A density increase alone never establishes validity or useful change.

Exact reconstruction and size-bin statistics can be derived from retained
attempt rows but are not separate aggregate fields in this release. Run records
provide overall timestamps; per-arm wall-time measurements are not implemented.
These limitations must be retained when comparing quality and cost.

## Durable artifacts and evidence limits

Expected artifacts include `run.json`, `metrics.jsonl`, `training_means.pt`,
`latent_reference.pt`, `cohorts.json`, `calibration.json`, `interpolations.json`,
`attempts.jsonl` and `summary.json`. Reference metadata carries checkpoint,
dataset, split, graph-schema and mode identities, training count and the saved
mean-code hash. The run records reference/mean-code hashes; loading the saved
reference checks exact metadata equality.

Downloaded evidence now confirms the regression counts, completed run IDs,
calibration choices, PCA retention, paired outcomes and intervals, interpolation
results and failed promotion flags above. Those outcomes do not establish
binding, synthesis or off-target performance. Later studies remain governed by
[the staged experiment plan](09_next_experiment_plan.md).
