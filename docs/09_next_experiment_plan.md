# Proposed next experiments: audit, prior, binding, synthesis and selectivity

**Supersession, 28 September 2026:** the user now requires frozen external
pretrained priors/scorers with interchangeable input adapters. New in-house
chemistry-prior/scorer training is excluded, except small supervised model-input
bridges/representation decoders that preserve one shared canonical molecule and
are frozen before optimization. Stage 3's custom denoiser fallback, Stage 4's
new synthesis surrogate and Stage 5's in-house activity-model training are no
longer active paths. Historical plans/results, including source022/024 failures,
are retained below. The current contract/gates are in
[docs23](23_frozen_model_plugin_contract.md).

Prepared 28 September 2026. The user subsequently **authorized continuing
experiments through this staged plan**. The proposed budgets and experiment
gates below remain subject to the supersession above; concrete configurations and run outcomes must still
be recorded separately. No new run is claimed by this status update.
Technical background and primary references are in
[the gradient research note](08_multiobjective_gradient_research.md).

## Current evidence

Completion was verified at **28 September 2026 06:20:41 UTC**. Both source015
workflows completed, with scheduler output explicitly reporting successful
completion. Each completed eight final evaluations: validation and test for the
common epoch-60, best-validation-reconstruction, best-validation-validity and last
checkpoint roles. The original test split has now been evaluated after each
model's continuation and selection decisions were frozen.

| Record | AE | VAE |
|---|---|---|
| Scheduler job | 29503898 | 29503899 |
| Workflow | `20260927T182827Z_ae-full-workflow_04509a97` | `20260927T182856Z_vae-full-workflow_ade6e67e` |
| Final epoch | 120 | 100 |
| Workflow finished (UTC, 28 September) | 01:41:13 | 00:30:56 |
| Stopping decision | `budget_reached_without_recent_gain` | `stop_after_validation_plateau` |
| Selected epochs: common / best reconstruction / best validity / last | 60 / 61 / 108 / 120 | 60 / 62 / 94 / 100 |
| Matched epoch-60 full-validation valid / exact, out of 7,347 | 6,578 / 6,236 | 6,028 / 5,571 |

The final training budgets differ; use the matched epoch-60 evaluation for the
initial equal-budget comparison. Both ran on separate A10 allocations. VAE prior sampling at
epoch 60 produced only 13/1,024 valid molecules. At neighborhood scale 0.2, each
model produced 14 valid-to-different-valid changes and six invalid-to-valid
rescues among 256 attempts. Local optimization has evidence of usable changes;
unconditional generation was weak at epoch 60. AE reconstructed more molecules
correctly at that matched checkpoint, but the best optimization representation
remains an open question.

The existing validation rules governed the completed extensions. Both final
trend records have `overfitting_warning: true`; completion is not proof of global
convergence or evidence that more training necessarily helps.
Source015's GPU gate passed 83 tests with one optional Boltz check skipped. The
verified split is 215,000/7,347/7,347 with preserved prior memberships.
The authoritative completion evidence is
[full-training-completion-20260928.json](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/full-training-completion-20260928.json>).
The unchanged source015 archive SHA-256 is
`375c7f6a5a67f229d5c95d307413047b26e244de82a1c71ff09a0471bc192f12`.
Earlier running snapshots remain in the experiment ledger.

## Common comparison rules

Freeze each selected encoder, decoder and scorer during latent optimization.
Match molecule identities across AE/VAE arms, and distinguish three optimization
random seeds from three independently trained models: the current training
campaign has one model-training seed. Record initial reconstruction differences;
do not silently restrict comparison to seeds that one decoder handles well.

Use training molecules only for permitted references, feature normalization and
supervised model-input bridge fitting. New denoiser/scorer training is excluded.
Before prior development, reserve
separate portions of validation for calibration/tuning and pilot evaluation.
Keep subsequent promotion seeds disjoint from the pilot. Those validation
molecules have already participated in model selection; call them development
evaluation, not a new untouched test. The original campaign test has been
evaluated under its sealed, predeclared evaluation plan. Do not reuse its outcomes to tune
these methods or choose another checkpoint.

Proposed cheap prior study: **256 shared validation molecules**, with a separate
calibration subset, matched normalized perturbations and three random repeats.
The first Boltz engineering check instead uses **one existing prepared context,
eight latent starts, ten update steps and two random repeats per arm**. These
are correlated starts around one chemical context, not independent target tests.
Only after useful hard changes, fresh-pose validation and runtime profiling
should the study expand to **32 seed molecules, 20 steps and three optimization
RNG seeds per arm**, with new native contexts prepared as necessary. Stratify by
size and retain selected identities and failures in outcome denominators. A focused
structural pilot would cover **two or three protein contexts total**, comprising
an intended target and one or two justified countertargets once a target context
is chosen. No proteins are selected here.

Declare both expensive-oracle-call and wall-time caps, including gradient passes,
preparation, failures and independent final rescoring. Compare at matched oracle
budgets and show wall-time efficiency separately; equal steps alone are insufficient.
All sample counts below are exploratory and should be adjusted using measured
variance and cost before freezing a protocol.

Compare every optimized hard molecule with its own unperturbed decoded baseline;
an encoder reconstruction error must not count as an optimization improvement.
Report valid-to-valid changes separately from rescues of invalid baselines.
Record each objective's value, latent gradient norm and pairwise gradient cosine,
including domination, conflict and clipping. Calibrate objective weights on a
separate cohort and freeze them for comparison. A lower weighted sum alone does
not establish improvement in the intended individual properties.

## Stage 0: audit the full-data training campaign

Retrieve scheduler completion records, run manifests, epoch histories, phase
decisions, saved checkpoints, evaluation summaries and errors. Verify source,
config, dataset/split and transferred-checkpoint hashes. For each model, establish
the actual last epoch, update count and stop reason. A scheduler interruption is
not convergence; reaching the cap while still improving is budget limitation.

Compare common epoch-60 and epoch-80 checkpoints when both exist, then report
the declared best-reconstruction, best-validity and last roles. Deduplicate
identical checkpoint files while preserving their roles. Any unmatched later
endpoints need their different training budgets stated. Confirm each model's
selection and continuation were sealed before its prescribed test evaluation.

Inspect full-validation curves and deterministic train/validation probes, with
the last 20 epochs shown within the learning-rate phases. Check validity,
exactness, atom counts, failure categories, gradient clipping and throughput.
For latent quality, report changed valid neighborhood outputs, not validity alone;
for VAE also inspect posterior variance/KL, active mean dimensions and actual prior
sample validity. The nominal normal prior is not evidence of a useful generative
distribution. Keep AE posterior statistics marked inapplicable.

**Gate:** complete provenance and trustworthy reconstruction/neighborhood evidence.
If validity remains weak or local moves mostly fail, investigate representation
or decoding first; do not proceed simply because training loss is low. Freeze
checkpoint choices for subsequent pilots using validation evidence only.

## Stage 1: compare inexpensive priors before involving Boltz

For each frozen model, compare size constraints alone, a normalized local latent
trust region, and an empirical density restraint using KDE or a regularized
Gaussian mixture. Fit on training codes; tune scaling, bandwidth/covariance,
density threshold and trust-region radius on the calibration subset. Use a
thresholded penalty rather than unbounded density maximization. Keep AE/VAE
calibration separate; their raw density scales are not comparable.

Perturb held-out development seeds by matched normalized displacements, then test
whether the restraint restores or preserves valid decodes while permitting
different molecules. Separate deterministic encoded means from VAE aggregate
posterior samples. If PCA is used, retain a residual check for discarded
directions. Density references from an external model are a later ablation:
they require an actual differentiable, compatible feature path.

**Outcomes:** valid fraction over all attempts, valid-and-changed fraction, exact
reconstruction, unchanged outputs, novelty, diversity, size-bin performance and
runtime. An apparent gain obtained by mapping every seed to the same molecule
fails the purpose of the prior.

**Gate:** the chosen prior improves a predeclared validity/change tradeoff over
size plus local trust-region controls on disjoint pilot seeds. Report paired
confidence intervals clustered by seed molecule, preserving correlated RNG
repeats. A gain whose uncertainty spans material harm is inconclusive, not a
promotion claim. Only after evidence of improvement should a frozen protocol
replicate on **200 new development seeds**; confidence intervals and all failures
remain mandatory.

## Stage 2: add one on-target Boltz context

Start with the small engineering budget above. Compare target-only latent
optimization, prior-only optimization, target-plus-prior optimization, and
target plus a simple latent trust region. Include unperturbed and matched random
displacement controls. On promotion, add a direct relaxed-graph optimization
baseline to test the embedding's value. Use the same seed identities,
occupancy policy, size limits and oracle budgets. Report starting hard graphs
and initial scores because AE/VAE reconstructions may differ. Include known
positive/negative controls separately from the generic molecule-seed cohort.

Start with the existing fixed-occupancy, fixed-pose adapter and its verified
gradient tests. Use the binary output for binder discrimination; the continuous
IC50-like head is only justified within a validated active-molecule setting.
Save every latent, soft graph, hard graph and score. Rebuild ligand features and
generate fresh poses for a predeclared candidate subset from every arm and its
baseline, using equal inference budgets and repeated pose seeds. Deduplicate
identical molecules while retaining attempted-trajectory denominators. Measure
score variation and failures. This native hard-molecule evaluation is a separate
gate from the fast fixed-context objective; it is still predicted binding.

**Gate:** more valid, changed hard candidates with independently improved target
scores at matched cost, without unacceptable size growth or diversity loss.
Lower continuous loss with identical hard chemistry does not pass. Separate
atom-count optimization is blocked by the current Boltz adapter: it needs a new
occupancy/geometry design and its own acceptance tests, not an added loss weight.

A later variable-size baseline can sample decoder `NO_ATOM` choices, rebuild and
score each discrete molecule with native Boltz, and estimate gradients to `z`
using the actual sampling probabilities. Keep explicit size limits and account
for the additional oracle cost. A fully soft occupancy-aware Boltz adapter is an
alternative requiring new hard-input parity and derivative checks. Neither
branch is implemented or necessary for the initial fixed-count combination test.

## Stage 3: test a chemical denoiser only if it adds value

**Current scope:** frozen external pretrained models only. The custom-schema
denoiser fallback in the original paragraph below is historical and superseded;
failed source022/024 runs do not authorize another such training attempt.

Retain the density/trust-region baseline. First perform a small direct DeFoG
interface and gradient check at fixed occupancy where representation parity is
established. Compare two possible objectives: density/distance in its internal
features, and a denoising-consistency term. The feature branch needs real-molecule
references with matching time/noise/conditioning and appropriate invariant
pooling. A small denoiser trained on our element/charge/bond/`NO_ATOM` schema is
the fallback if pretrained compatibility is inadequate. Do not silently drop
charge or reinterpret an absorbing noise token as atom absence. The proposed
denoising-consistency term is not exact graph likelihood.

Prefer direct tensor derivatives wherever the full path is meaningful. DiGress
has similar categorical-interface questions; GDSS offers continuous score fields;
FragFM would require larger architectural changes. They remain alternatives,
not three additional training commitments.

A tensor-only straight-through pilot might use **8–16 samples**; a black-box
score-function pilot might use **16–32 samples**. These are alternative branches,
not requirements to implement every estimator. Use the actual sampling law,
sample each unordered bond once, avoid double occupancy gating, and penalize
invalid or failed attempts rather than discard them. REINFORCE can score discrete
outputs without differentiating the oracle, but does not implement variable
occupancy in Boltz automatically.

**Gate:** additional hard-candidate improvement over Stage 2 at measured cost.
Validate ordinary soft derivatives by finite differences; do not expect an ST
soft backward pass to equal finite differences of its hard forward function.
Check expected-reward estimators by tiny enumeration and repeated Monte Carlo.

## Stage 4: establish a synthesis oracle before trusting its surrogate

**Current scope:** frozen external synthesis models, with a supervised input
bridge if required. The separately trained synthesis-surrogate arm described
below is historical and superseded; direct frozen-model objectives and ordinary
route verification remain candidate paths.

First run a small, predeclared subset of pilot candidates through SynFormer or
PrexSyn with fixed catalog/templates, search budget and stopping rules. Define
separate outcomes for exact candidate recovered, similar analog recovered,
invalid route, timeout, and no route found within budget. Execute returned routes
and compare canonical product identity; confidence in a route or a similar
product is not synthesis evidence for the candidate itself.

If oracle coverage is useful, compare a soft-graph synthesis surrogate against a
learned condition adapter into the frozen synthesis model. For direct SynFormer
research, use a declared teacher-forced route loss, including fingerprint-denoising
training terms where appropriate. Preserve adapter, route and product errors
separately. Conditional route confidence is not generic synthesizability, and
summing probability over all possible routes is not an informative reward.

Include a small direct SynFormer gradient feasibility check explicitly. A
surrogate gradient is a separate experimental arm and must not be described as
backpropagation through SynFormer. PrexSyn's descriptor-conditioned route losses
provide another adapter option; discrete fingerprint construction still requires
a differentiable bridge. Do not train both alternatives at scale before testing
native route-recovery coverage and adapter fidelity.

**Gate:** improved exact-product route recovery among valid optimized candidates,
with denominators including failures and the same planner budget. Keep a
held-out planner audit independent of surrogate training. A recovered computational
route remains distinct from experimental synthesis success.

## Stage 5: introduce off-target objectives and combine successful branches

**Current scope:** select/calibrate existing pretrained activity predictors and
frozen structural scorers. Training an in-house multitask activity model is
superseded. A small input bridge must align features rather than replace the
pretrained activity predictor.

Build an inexpensive calibrated multitask activity panel with documented endpoint
definitions, measured positive/negative support and chemical-domain coverage.
Missing activity is unknown, not inactive. Select the smaller Boltz countertarget
panel only after the intended target and relevant controls are known. Every
protein needs its own prepared context and calibration; raw cross-target scores
are not automatically comparable selectivity measurements.

Retain an on-target activity floor and explicit countertarget ceilings. A smooth
maximum or individual constraints avoids hiding one strong liability in an
average. Compare target-plus-prior with the addition of off-target guidance,
then combine only branches that independently passed their earlier gates.
Avoid a full factorial expansion before useful components have been established.

**Gate:** a higher fraction of all attempts jointly valid, changed, on-target,
below the declared countertarget thresholds and, when required, independently
route-recovered. Report each component separately as well. Freeze the combined
protocol before confirmation on new scaffolds and separately held-out target
contexts. That confirmation requires a new evaluation plan; the already allocated
campaign test cannot be adaptively reused as fresh evidence.

## Durable evidence and decisions

Each proposed stage becomes executable only after its chosen configuration,
cohort, acceptance metric and resource budget are explicitly adopted. Use
immutable source releases and record model/checkpoint, reference-code, adapter,
data/split/cohort, protein-context and synthesis-catalog hashes. Preserve RNG
seeds, all submitted jobs, logs, failed attempts and actual resource use in the
[experiment ledger](05_experiment_ledger.md).

Report unrepaired hard outcomes and independent rescoring, expected and discrete
size, score/projection gaps, novelty/diversity, per-target results, route identities
and elapsed time. Record a stop decision when a component adds cost without
measurable hard-candidate benefit. The immediate next decision is which completed,
validation-selected checkpoints to freeze for Stage 1. Current evidence supports
testing local neighborhoods of both models on HPC. Final training decisions and
the prescribed test evaluations are complete; new methods must retain the
separate development and confirmation protocol above.
