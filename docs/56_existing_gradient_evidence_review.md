# Review of existing Boltz–DeFoG failure evidence

Reviewed 15 October 2026 using already downloaded, hash-verified HPC artifacts.
This review ran no model, numerical test, HPC command or submission. Local
Python 3.12.14 standard-library JSON reads and arithmetic on saved scalar
results establish the comparisons below. Source identities, commands, runtime
environments and capsule hashes remain in the [experiment ledger](05_experiment_ledger.md).

The evidence supports several distinct causes. Early failures included real
adapter and process-launch bugs, which later native runs demonstrably passed.
The remaining failures are derivative-validation failures of the Boltz
objective at particular latent states. They are not evidence that DeFoG cannot
sample repair targets or that its discrete sampling must be differentiated.
Forward-mode and reverse-mode automatic derivatives agree at the latest failed
states; finite-step secants sometimes disagree or lack the required adjacent
passing scales. This does not uniquely identify the responsible upstream
operation or certify long-run stability.

## What the completed runs establish

The table separates a completed diagnostic from a passed scientific gate.
Native numerical results cited here were computed on allocated DTU HPC;
fixture counts are engineering evidence only.

| Actual experiment / evidence | Failure class or scope | Established finding and remaining limit |
| --- | --- | --- |
| Source069 startup; source076 controls **29634064/29634065**, native **29634091**; [negative and positive probes](../outputs/hpc-results/source076-defog-mps-recovery/extracted-evidence001/) | GPU process admission | Without MPS, the DeFoG child failed CUDA allocation while the Boltz parent retained CUDA. Managed MPS admitted both configured interpreters to the same GPU; the subsequent native allocation also passed startup. This fixes the demonstrated admission problem, not molecular compatibility or efficacy. |
| Source076 → source079; [native-contract diagnosis](48_defog_comprehensive_diagnosis.md#failure-history) and [source079 evidence](../outputs/hpc-results/source079-defog-contract-recovery/) | Adapter implementation | The confidence validator rejected the valid native `[B,1,T,3]` frame; the transition validator wrongly required a zero edge diagonal instead of native one-hot `NONE`. Source079 corrected these assumptions and passed its actual six-update native repair/gradient gate. No unsupported charge projection was silently introduced. |
| Source079 ten-seed long cohort; [full campaign evidence](../outputs/hpc-results/source079-long-run-investigation-002/) | Runtime budget **and** numerical validation | Six cases failed numerical guards: contact at prior 0/20, 3/50, 7/50; combined loss at 5/20, 8/60, 9/20. Four hit the 6,000-second child cap: priors 1/2/4 inside gate 60 after 60 updates; prior 6 in line search after 56. For 1/2/4, Boltz validation cost 3,440–3,907 seconds versus DeFoG 653–803 seconds. No complete 100-update endpoint. |
| Source080 three-state replay; [actual case receipts](../outputs/hpc-results/source080-attribution-002/runs/source-080-failed-gate-attribution-attempt001-replay/20261009T141517Z_source080-failed-gate-attribution_4b1c235b/cases.json) | Numerical attribution | All three observers completed; repeats and grad/no-grad endpoint graph/output comparisons matched, scalar assembly was exact and component-gradient assembly differed only by small floating-point amounts. Original failures reproduced; no old guard was certified. This excludes simple stochastic drift or gross assembly error for these states. |
| Source081 **29670234**; [actual case receipts](../outputs/hpc-results/source081-attention-causal-006/runs/source-081-attention-causal-attempt001-replay/20261009T223433Z_source081-native-attention-causal_3248aad0/cases.json) | Negative mechanism test | All three states had zero wholly masked triangle rows and identical native/stopped-logit/widened-mask controls. The proposed masked-attention mechanism is excluded for these states. Fixed-cotangent decoder readouts passed for prior 0 contact and prior 5 binding, but prior 3 exposed a different decoder-local discrepancy. |
| Source082 **29670381**; [prior-3 attribution](../outputs/hpc-results/source082-decoder-causal-004/runs/source-082-decoder-causal-attempt002-replay/20261009T232136Z_source082-decoder-causal_a0618b2a/prior-3/20261009T232249Z_prior-3_c3260a08/attribution.json) | Finite-step decoder branch crossings | Original AD **0.01717366** versus FD **0.01559777 / 0.01546945** at steps 0.001/0.003; holding baseline absolute-value signs keeps the exact baseline/gradient and gives **0.01720160 / 0.01718873**, which pass. Nonzero-cotangent branch crossings were observed. Strong state-specific evidence for secants crossing decoder kinks, not an incorrect central derivative or a production replacement. Stable occupancy did not remove the larger-step mismatch. |
| Source083 **29670416**; [case receipts](../outputs/hpc-results/source083-head-causal-004/runs/source-083-head-causal-attempt001-replay/20261010T002406Z_source083-head-causal_779c4eab/cases.json); source086 **29672980**, [case receipts](../outputs/hpc-results/source086-precision-head-003/runs/source-086-precision-head-attempt001-replay/20261010T075304Z_source086-precision-head_c4737535/cases.json) | Contact arithmetic / head attribution | Cached wider contact assembly improved prior 0, and source086's actual opt-in contact/assembly arithmetic passed that state's contact and combined checks. Prior 5 retained an affinity-regression discrepancy; its final scalar half-ULP bound was insufficient to explain it. Partial new-arithmetic validation does not certify the old gate or every trajectory. |
| Source087 **29674150**; [prior-5 attribution](../outputs/hpc-results/source087-affinity-causal-attempt002-005/runs/source-087-affinity-causal-attempt002-replay/20261010T102212Z_source087-affinity-causal-attempt002_474638f7/prior-5/20261010T102334Z_prior-5_f2a301b1/attribution.json) | Upstream sensitivity / cancellation | Regression AD **−0.03723935** versus FD **−0.04455447** at 0.001 persists through wider small MLPs and held small-head ReLUs, with no such crossings at that scale. Opposing member contributions separately pass, while their mean fails. Component passes cannot certify a cancellation-sensitive sum. |
| Source088 **29674301**; [regression decoder attribution](../outputs/hpc-results/source088-affinity-decoder-attempt001-008/runs/source-088-affinity-decoder-attempt001-replay/20261010T123411Z_source088-affinity-decoder_240f0eec/prior-5/20261010T123534Z_prior-5_ff568782/attribution.json) | Localization limit | Original and held-absolute-value regression decoder readouts both have neighboring passing scales in all four directions. The prior-5 residual was not reproduced in that decoder cut; together with source087, this leaves the shared-graph-to-pooled-affinity path, without uniquely locating an operation. |
| Source089 **29674537**; [completed six-update smoke](../outputs/hpc-results/source089-defog-minimal-smoke-attempt001-003/runs/source-089-defog-minimal-smoke-attempt001-smoke/controller/20261010T123626Z_defog-minimal-integration-smoke_3bfec22e/smoke.json) | Successful bounded integration | Six accepted guided updates, eight eligible native outputs, two selected targets and passed guards at 0/5/6. Binding and repair gradients reached the same `z`. Raw candidates **0/14 valid**, argmax states **0/7 valid**; no fresh-pose scoring or chemical advantage. A functioning short gradient path is established. |
| Source091 **29679843**; [control gate 15](../outputs/hpc-results/source091-defog-matched-pilot-attempt001-002/runs/source-091-defog-matched-pilot-attempt001-pilot/boltz_only/optimization/20261010T202405Z_matched-pilot-boltz_only_491449c5/fixed_initial_mask/gate-0015.json), [guided gate 15](../outputs/hpc-results/source091-defog-matched-pilot-attempt001-002/runs/source-091-defog-matched-pilot-attempt001-pilot/boltz_defog/optimization/20261010T210520Z_matched-pilot-boltz_defog_b6f3af95/fixed_initial_mask/gate-0015.json) | Numerical gate | Fifteen accepted updates per arm, then control combined FD and guided affinity FD failed before update 16. Guided refresh 15 generated four eligible targets and held-target repair checks passed. Neither trajectory completed; this was not a DeFoG sampler crash. |
| Source093 **29680551**; [exact-state cases](../outputs/hpc-results/source093-first-suite-002/runs/source-093-first-suite-attempt001-replay/20261011T001225Z_source093-defog-pilot-failed-gate-replay_6bc2685a/cases.json) | Repeatability / endpoint attribution | Both observers completed. Repeated values/gradients and grad-enabled/no-grad baselines agree exactly; paired endpoints have identical graph hashes and contact selections. Combined original gradient-direction FD still fails at every stored scale. No simple mode, random-forward or bookkeeping explanation was demonstrated. |
| Source097 **29686651**; [complete boundary cases](../outputs/hpc-results/source097-first-suite-002/runs/source-097-operation-boundary-recovery-attempt001-replay/20261011T100922Z_source097-operation-boundary_6351579e/cases.json) | Affinity localization | All 14 cut checks completed with actual heterogeneous ensemble depths eight/four. Guided affinity first diverges in the pooled-to-shared-head finite-step readout, correlated with readout ReLU crossings. This motivated the causal branch intervention below; a cut-location correlation alone was not proof. |
| Source101 **29699446**; [guided attribution](../outputs/hpc-results/source101-first-suite-003/runs/source-101-affinity-head-metadata-recovery-attempt001-replay/20261013T042738Z_source101-affinity-head-metadata-recovery_314be622/boltz_defog/20261013T043120Z_boltz_defog_6d5a4740/attribution.json) | Finite-step native readout crossings | Guided AD **4.07505941**, native FD **4.54388559**, cached native FP64 FD **4.54406987** at 0.0001; holding baseline ReLU branches gives **4.07414864**. Branch-switch contribution **0.46992122** dominates measured FP32 contribution **−0.00018428**. This explains this affinity interval; it is not a blanket explanation of the later control trajectory. |
| Source102 **29699981**; [structural cases](../outputs/hpc-results/source102-first-suite-002/runs/source-102-structural-loss-diagnostic-attempt001-replay/20261013T053906Z_source102-structural-loss-diagnostic_deb4c7c6/cases.json); source103 **29706116**, [contact cases](../outputs/hpc-results/source103-first-suite-004/runs/source-103-contact-boundary-diagnostic-attempt001-replay/20261014T085919Z_source103-contact-boundary-diagnostic_01c76680/cases.json) | Complete derivative boundaries / contact arithmetic | Source102 complete-frontier VJPs match direct VJPs exactly for 11 objectives per case; source103 all ten contact chained VJPs match exactly. Holding final contact winners/confidence reductions did not explain the residual; smallest input-ReLU probes showed no crossings. Widening the final contact affine head brought the guided small-step slopes within the existing tolerance, with pocket/anchor cancellation amplification about five. No unique upstream operation was identified. |
| Source105 readiness **29715519**, promotion **29715520**; [actual readiness summary](../outputs/hpc-results/source105a002-first-suite-003/runs/source-105-contact-precision-branch-validation-attempt002-validation/20261014T164118Z_source105a002-contact-precision-readiness_c5685678/validation.json) | Passed new saved-state contract | Both original source091 failed states passed the explicitly new head64/`branch_local_v2` contract, including native natural contact FD, independent forward AD and exact baseline parity. No optimizer update or new DeFoG call occurred. Old natural combined FD still failed in both cases, and guided natural affinity FD failed; old `local_v1` was not certified. |
| Source105 pilot **29716584**; [control gate 15](../outputs/hpc-results/source105a002-first-suite-004/runs/source-105-contact-precision-branch-validation-attempt002-pilot/boltz_only/optimization/20261014T205327Z_matched-pilot-boltz_only_f632abf8/fixed_initial_mask/gate-0015.json), [guided gate 15](../outputs/hpc-results/source105a002-first-suite-004/runs/source-105-contact-precision-branch-validation-attempt002-pilot/boltz_defog/optimization/20261014T213903Z_matched-pilot-boltz_defog_65ba68e5/fixed_initial_mask/gate-0015.json) | New-state numerical failure | Both arms accepted 15 updates and passed audits 0/5/10, then failed audit 15 before update 16. Control failed combined branch FD; guided failed contact FD. Exact baseline parity and all independent JVP checks passed. All 16 native DeFoG workers returned 0; 15 eligible targets and four target selections were recorded. No endpoint/fresh-pose efficacy evaluation. |

Source097's job identity is confirmed by its [actual native receipt](../outputs/hpc-results/source097-first-suite-002/submissions/source097-operation-boundary-recovery-attempt001/receipts/native-receipt.json).
Case artifacts establish completion even where finished scheduler records
have aged out. A controller's exit zero or scheduler `DONE` never overrides
failed child scientific outcomes.

## Existing data explain the latest failures more precisely

The independent [saved-result arithmetic report](../outputs/diagnostics/contact-result-existing-evidence-review-20261015.json),
SHA256 `63305fb4aea0d4776f461c7552d7739c39e761022c234799702eb598a23f3045`,
records source artifact hashes, matched direction hashes, component/combined
secants and normalized errors for the comparisons below. It reads existing
JSON; it is not a new numerical model experiment.

### The control's combined loss is sensitive to cancellation

The source105 control step-15 file stores the **same direction hash**
`7c5aea9af819b04d29995614fdfd3e5854bb08330df271ec329cd74063904b56`
for combined and all three component `random_2` rows. At the same epsilon
`0.00003`, each component's branch-held FD passes its own 5% rule, but the
combined FD fails:

| Component | Analytic directional slope | Saved FD slope | Objective weight |
| --- | ---: | ---: | ---: |
| Affinity | −3.273288678 | −3.176182508 | 1 |
| Confidence | 1.266226241 | 1.204311848 | 2.5 |
| Contact | 0.291425489 | 0.281539811 | 2.5 |
| Combined objective | 0.620846711 | 0.538446638 | assembled above |

The weighted saved component secants reproduce the combined secant to about
`2e-12`. This excludes a scalar assembly bug at these particular endpoints.
Opposing weighted slopes leave a small net derivative; the ratio of summed
absolute weighted analytic contributions to the absolute analytic component
sum is about **11.54**. The net error is about **13.27%** even though each
component passes its separate relative test. The analytic component sum and
native combined projection differ by about `6.1e-6`, consistent with their
separately stored floating-point reductions; they are not claimed bitwise
equal. This cancellation example is a concrete explanation of this failed
random-direction predicate, not an explanation of every control direction.

The reverse effect is also stored: at **control step 5**, the same-direction
`random_2` rows at `1e-4` fail separately for affinity, contact and confidence
(error/tolerance approximately **1.2133 / 1.2462 / 1.9152**), while their
combined error/tolerance is **0.5868**, a pass. Weighted component secants
recompose the net to within `4.5e-12`. Thus net cancellation can both amplify
and hide component discrepancies. A justified component-based check must use
the **same actual direction and endpoints** and independently certify its
required components; merely adding component tolerances would not establish
that the net derivative passed. No source108 native certificate is implied.

### The control's own-gradient secants show rapid endpoint variation

At the same control state, native reverse AD is **6.523304606** and independent
forward AD is **6.523338433** along the combined own-gradient direction. The
saved central branch-held secants are **7.012759269**, **8.019930706** and
**26.545086058** at `1e-5`, `3e-5` and `1e-4`, respectively. At `1e-4`, the
saved linear prediction on the *actual rounded endpoint chord* is
**6.523339646**, whereas the plus/minus one-sided slopes are **81.146082633**
and **−28.055910517**. Endpoint rounding alone therefore does not explain this
large discrepancy. Holding the twelve affinity readout ReLU branches did not
remove it; the model input uses the enforced fixed 22-slot mask, and measured
contact winners did not change on these small probes. Natural `NO_ATOM`
selection was not independently audited by these endpoint records. These
facts establish strong finite-interval nonlinearity or
unobserved internal branch sensitivity; the saved state does not identify
one particular upstream operation as its unique cause.

### The guided contact failure is the adjacent-scale rule, with small residuals

For guided contact `random_2`, reverse AD is **−0.007872803281**, independent
forward AD **−0.007872757154**: absolute difference about `4.6e-8`.
The stored natural contact secants are:

| Epsilon | Central secant | Existing pointwise tolerance result |
| --- | ---: | --- |
| 0.0001 | −0.008097118788 | Pass |
| 0.0003 | −0.007316493762 | Fail |
| 0.001 | −0.008044541183 | Pass |

There is no two-adjacent-scale passing window, so the declared gate correctly
reports failure. The model input retains the enforced fixed 22-slot mask;
measured contact winners remain unchanged through 0.001. This is not an audit
of the decoder's unconstrained natural `NO_ATOM` selections. Contact's
own-gradient probes pass at `3e-5`, `1e-4`, `3e-4`; the
failure is direction-specific, not a zero or universally broken contact
gradient. Readout holding is insufficient, while output `float64` alone does
not imply every upstream feature was evaluated at that precision. Existing
coverage cannot uniquely separate upstream rounding, rapid curvature and
other unobserved branch crossings in this contact direction.

### Smaller steps also expose arithmetic limits

The source101 guided cached-head tangent control gives another independent
mechanism. At step **`9.23155e-8`**, with no observed readout branch crossing,
native FP32 head AD is **4.215132557** versus FD **3.631850873**; the widened
head's own AD is **4.215132382175** versus FD **4.215132382601**. This is a
different, cached **head-input tangent experiment**, not the latent-direction
probe whose AD is 4.075059414. It demonstrates arithmetic sensitivity inside
that controlled head path. Its reported `1.47705e-6` open-cell radius is
explicitly head-only, not a guaranteed smooth radius in the full latent
space. Decreasing epsilon can avoid some crossings while worsening FP32
subtraction/feature-rounding effects; simply choosing ever smaller probes is
not a universal remedy.

## Instrumentation failures that must not be counted as scientific evidence

| Failed attempt | Actual cause | Verified recovery / scope |
| --- | --- | --- |
| Source066/067 preparation | Fixture duplicated an immutable context; interpreter symlink resolution selected the wrong environment | Later fixture and lexical interpreter-path recovery; preparation only. |
| Source082 attempt001 CPU | Invalid copied-decoder baseline fixture | Corrected attempt002 reached the actual native branch-control result above. |
| Source087 attempt001 | Cached diagnostic/autograd phase lifetime exhausted GPU memory | Attempt002 **29674150** completed all 36 FD rows without timeout/OOM. This fixes observer execution, not the affinity mismatch. |
| Source090 CPU **29679550** | Tuple/list and integer/string JSON provenance comparison false alarm after fixture passes | Source091 canonicalized verified metadata and reached the real pilot. [Original failed CPU log](../outputs/hpc-results/source090-defog-matched-pilot-attempt001-002/runs/20261010T200439Z_source-090-defog-matched-pilot-attempt001-cpu_1fe67190/command.log). |
| Source096 native **29683997** | Observer assumed equal ensemble depths; controller masked failed children with exit zero | Both cases failed before first cut. Source097 supported actual depths eight/four and propagated failures. [Original cases](../outputs/hpc-results/source096-first-suite-005/runs/source-096-operation-boundary-attempt001-replay/20261011T085012Z_source096-operation-boundary_f6501f27/cases.json). |
| Source099 native **29693807** | Observer read `graph_sha256` from the analytic record instead of saved no-grad metadata | Both children failed baseline capture, not DeFoG or gradient computation. Source101 corrected the reader and completed both actual probes. [Original cases](../outputs/hpc-results/source099-first-suite-002/runs/source-099-affinity-head-isolation-attempt001-replay/20261011T202836Z_source099-affinity-head-isolation_d3b0f66c/cases.json). |
| Source105 attempt001 / source108 attempt001 local preparation | Runtime-manifest/collector packaging issue; later an incorrectly transcribed 65-character reference hash | Rejected before upload/submission; original artifacts retained. These were not numerical experiments. Source108 remains a candidate whose native readiness has not passed. |
| Source104 chemistry benchmark **29712276** | Parent identity rejected before updates; source106 allocated metadata inspection isolated JSON float `1e-5` parsed as YAML string `"1e-05"` | Source107 strict JSON reader has CPU regression coverage, but its native checkpoint/optimizer/RNG continuation remains queued in the last verified snapshot. This separate training problem does not explain Boltz FD failures. [Actual metadata](../outputs/hpc-results/source106-parent-metadata-002/parent-metadata.json). |

Source107's **169** CPU checks and source108's **181 + 135** CPU checks are
not native gradient, checkpoint-resume or chemical-advantage certificates.
Queued source106/source108 observers supply no new numerical findings for
this review. No additional runs were submitted for this evidence audit.

## Interpretation for the current method

DeFoG uses a candidate-start discrete denoising/repair interface, with explicit
vocabulary and charge limits. The current surrogate holds an accepted repair
target fixed and differentiates its decoder categorical NLL into the shared
latent vector; it does not differentiate sampling or claim a DeFoG likelihood.
Successful native targets, repair checks and accepted guided updates establish
that this plumbing works within the tested representation. Source079 charged
seeds 8/9 were explicitly rejected before any native DeFoG transition and later
failed Boltz-only guards, further excluding DeFoG sampling as the universal
cause of these failures. Neutral target projection remains lossy and is not
charge preservation.

The existing measurements justify distinguishing **local automatic derivative
consistency**, **finite-interval secant accuracy**, and **optimization success**.
They do not justify discarding failures, declaring all gradients wrong,
identifying an inherent DeFoG incompatibility, or claiming a completed useful
molecule. Cancellation, decoder/readout branch crossings and limited arithmetic
precision each have concrete state-specific evidence. The exact operation
behind the latest control own-gradient and guided contact residuals remains
unidentified from the stored primitive coverage. Any later changed acceptance
contract must be named and evaluated separately while retaining the original
failed predicates; a new certificate cannot retroactively pass them.

See [the matched-pilot history](51_defog_matched_pilot.md),
[the sealed head64/branch-local contract](54_contact_precision_and_branch_validation.md)
and [the queued recovery description](55_trajectory_and_parent_identity_recovery.md)
for immutable identities and implementation scope.
