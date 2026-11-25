# Why sulfur can accumulate under chemical guidance

Reviewed **29 October 2026**. This is a cross-experiment mechanism audit, not a
change to any running loss, checkpoint, optimizer, target policy or validation
gate. It combines source inspection, established saved results and the separate
allocated Mosaic sulfur-gradient attribution. Eight new allocated CPU component
checks also passed as described below. No training or optimization ran for this
documentation update or the component audit.

**The recurrence has more than one cause.** The old live maximum-valence loss
contains a direct element-substitution shortcut. Atom-preserving/frozen-target
losses remove that immediate shortcut but can still change element distributions
through shared parameters, refreshed targets and witness selection. The current
Protenix–DeFoG trajectory shows a different measured sequence: an early native
contact-score push toward sulfur followed by reinforcement from the conditional
DeFoG teacher. These mechanisms should not be reported as one universal defect
of every validity loss.

## 1. The old maximum-valence proxy gives sulfur a structural advantage

The [versioned rule table](../src/latentmol/chemistry_regularization.py#L46)
follows the [RDKit allowed-valence rules](https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences).
For the neutral elements used here, carbon has allowed valence 4, nitrogen 3,
oxygen 2, phosphorus 3/5 and sulfur 2/4/6. The sulfur rules allow legitimate
expanded-valence chemistry; replacing them with a universal sulfur cap of 2 would
change the chemistry incorrectly. Charge can change the applicable rule, so
the comparison below explicitly holds neutral charge fixed.

The old [live probability-weighted loss](../src/latentmol/chemistry_regularization.py#L260)
uses the largest permitted value, not a complete molecular-validity test:

\[
L_i=\sum_{a,q}p_i(a)p_i(q)\,\mathbb E[(V_i-c(a,q))_+],
\qquad c(a,q)=\max\operatorname{AllowedValences}(a,q).
\]

For unchanged bonds, occupancy and neutral charge,

\[
\mathbb E[(V-6)_+]\leq\mathbb E[(V-4)_+].
\]

Moving atom probability from carbon to sulfur therefore never increases this
component. It strictly lowers it when there is bond-order mass above 4. At a
bond-order sum of 5, the unnormalized carbon excess is 1 and the sulfur excess 0.
The optimizer can lower the penalty by changing the atom label while leaving
the excessive bonding in place. The implementation's shared positive scale and
occupancy normalization do not remove this comparison when those quantities
are held fixed. Squaring the positive excess also preserves the ordering.

This is an **analytic component-level incentive**, not a measured fraction of
the historical training drift. The bond-order distribution is handled more
carefully than a hinge on mean valence, but exact averaging of an incomplete
rule still leaves the element-label escape route. Zero undercoordination cost
accounts for omitted implicit hydrogens; forcing every heavy atom to attain
its maximum valence would introduce another incorrect incentive. Aromatic
order 1 in this proxy is an optimistic lower bound and does not certify rings,
Kekulization, radicals or the full molecule.

## 2. Why this shortcut can be attractive repeatedly

Dense or initially random graphs contain many excessive incident bond orders.
A higher cap can make an unchanged local graph cheaper than several coordinated
bond changes. When a repair search explicitly permits atom edits, one C-to-S
substitution can also compete with multiple bond edits under the declared edit
cost. That is a property of the objective and allowed search operations, not
evidence of a physical binding advantage for sulfur. Phosphorus or permitted
charge changes can offer related escape routes; the neutral sulfur cap is the
largest in this project's finite element vocabulary. Even a complete validity
test can prefer an allowed C-to-S substitution when atom identity is free and
the objective rewards validity alone. The incomplete maximum-cap proxy makes
that route broader; full validation alone does not supply a composition prior.

Passing a narrow local valence test is easier than matching the training
distribution, forming a useful drug-like molecule, or improving binding.
RDKit sanitization and schema roundtrip are stronger than the local proxy,
but their success still does not establish energetics, synthesis, stability or
binding. A validity improvement accompanied by a wholesale element shift must
retain both facts in the result.

## 3. Atom-preserving and frozen-cap losses have different semantics

[Atom-preserving repair](../src/latentmol/atom_preserving_chemistry.py#L34)
fixes atom labels, charges, occupied slots and alignment in its search target.
The target verification rejects atom/charge substitutions. This prevents the
search from repairing an overbonded carbon by relabelling it sulfur.

The later [failure-guidance preparation](../src/latentmol/repair_failure_guidance.py#L161)
detaches the source graph and freezes its caps, while
[local admissibility](../src/latentmol/repair_failure_guidance.py#L330) reads
current bond probabilities against those saved caps. Within that fixed
preparation, changing current atom probabilities does not increase the cap.
The immediate live-mixture C-to-S gradient from section 1 is absent in that
component. Optional atom-edited witnesses are a separately declared search
policy, not an implicit property of the bond-only path.

This does **not** make every element output invariant:

- Accepted full repair targets include cross-entropy for unchanged atom and
  charge labels. A current sulfur label can therefore be reinforced even when
  only target bonds were edited. Already-valid skipping and self-guidance must
  be distinguished; the original atom-preserving policy skips already-valid
  candidates without an identity self-target.
- During decoder training, bond losses and real reconstruction update shared
  parameters. During latent optimization, changes in shared latent coordinates
  can move several output heads. Zero direct atom-logit gradient in one term
  does not guarantee zero indirect element change through these shared paths.
- A new preparation can select new hard labels/caps after an update. Holding
  one target fixed inside a derivative check does not make the entire sequence
  a single fixed objective.
- Search success, failure and target coverage can depend on composition. A
  restricted accepted subset can teach a different distribution from all
  attempted molecules, even without an explicit sulfur reward.

These limits are already explicit in the
[atom-preserving continuation protocol](53_atom_preserving_triangle_continuation.md#loss-weighting-and-the-composition-prior).
Its weak population composition penalty and retention gates discourage drift
without mathematically guaranteeing its absence. The
[epoch36 collapse diagnostic](70_epoch36_collapse_mechanism.md) independently
demonstrated shared-decoder objective interference: a total-loss improvement
can worsen prior repair and raw validity. That diagnostic was not a
sulfur-specific causal experiment.

## 4. The current Protenix–DeFoG sulfur feedback is measured separately

The companion Mosaic experiment uses a native contact score plus a refreshed,
stopped DeFoG KL target. It does not use the old maximum-valence penalty.
Allocated CPU **29848106**, completed 29 October 2026 at 12:51:17 CET, decomposed
the stored gradients at 20 authenticated saved states. No model forward,
optimizer intervention or fresh chemistry evaluation occurred. Source and
frame hashes, teacher element mapping, and the stored gradient equations passed.

| Saved state | Current sulfur argmax count | DeFoG sulfur argmax count | Native local push on expected S | Weighted DeFoG local push on expected S |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4 | 0 | +0.005165546 | -0.000944273 |
| 16,579 | 17 | 23 | -0.003876141 | +0.004225332 |

The pushes are derivatives of the sum of sulfur probabilities under gradient
descent, per unit learning rate at the saved point. Positive increases sulfur
probability. They are local components, not complete counterfactual trajectories.
At update 16,579 the net is still positive, about +0.000349191. The current soft expected
sulfur count is 11.76427, while the teacher's is 20.84360; these soft counts must
not be confused with the 17/23 argmax counts.

Initially the contact objective pushes toward sulfur while the teacher opposes
it. Around the sampled state 9,000, the teacher component also starts increasing
sulfur. Later the native component opposes sulfur growth, but the teacher push
is larger. This is conditional teacher reinforcement of an already pathological
trajectory, not evidence that an unconditional molecular prior intrinsically
prefers sulfur-rich molecules. Increasing teacher time, graph changes and hard
auxiliary features changed together; their individual causal roles in the
teacher flip have not been isolated.

Global gradient matching does not solve this. At initialization, total native
and weighted-prior norms were equal, but atom-head norms were 0.02361847 and
0.00525442, respectively: native atom guidance was about 4.5 times larger. The
prior's norm was concentrated in 300 unordered bond decisions versus 25 atoms.
Later even similarly sized vectors can oppose each other in the sulfur
direction. Equal norms establish neither agreement nor correct teacher targets.

Companion evidence requires the sibling Mosaic checkout: its run record is
`experiments/learned-prior/protenix-defog-balanced-30000/runs/20261029-sulfur-attribution-01/README.md`;
the verified saved JSON is
`results/hpc/20261029-defog-investigation-01/sulfur/summary.json`, SHA256
`b3b44619afdd6fddcefac2e8690221d581d35c6b002df12510e26ac597645abd`.
These artifacts are not silently reproduced as DBMol numerical results.
The [companion Mosaic mechanism report](../../mosaic/docs/sulfur_bias_in_chemical_guidance.md)
provides the corresponding source and diagnostic overview; this cross-repository
link also assumes sibling checkouts.

## 5. What the observed trend does and does not establish

| Evidence | Observation | Limit |
| --- | --- | --- |
| [Historical dense100-epoch continuation](45_dense_chemistry_100_epoch_continuation.md) | Across the same unfiltered 4,096 prior states, expected carbon/sulfur changed 16.4789/0.39095 to 3.6838/17.5771. Expected size stayed near 22.09 to 22.07 while raw validity increased. | Real drift beyond a valid-only selection effect; the loss, additional fine-tuning and frozen-encoder regime were not causally separated. |
| Current Mosaic saved-gradient decomposition | Early native sulfur increase, later conditional teacher reinforcement, with valid element mapping. | Local directional attribution; the cause inside the teacher is unisolated and the run was still in progress at the cited snapshot. |
| [Original 36 to completed 40 frozen-reference continuation](75_frozen_reference_epoch40_continuation.md) | Fixed-panel sulfur frequency 1.5070% to 1.4394%, while raw validity 641/1,024 to 753/1,024. | A counterexample to inevitable sulfur inflation, not a matched isolation of which safeguard prevented it or an independent-seed guarantee. |

Reporting composition only among valid molecules can additionally exaggerate
or hide a trend through selection. Report expected and hard element counts over
**all attempts**, plus the valid subset separately, with occupancy/size and
target-coverage denominators. A stable total atom count is not a stable element
distribution. The historical dense result above already uses all prior states,
so valid-only selection cannot explain that particular drift.

## 6. Practical interpretation and prospective checks

Preserve legitimate sulfur chemistry. The supported response is to qualify the
guidance mechanism and its target distribution, rather than banning sulfur or
pretending the allowed-valence table is a drug-likeness distribution.

- Separate direct element, charge, occupancy and bond derivatives. Demonstrate
  a fixed-bond C-to-S contrast for a live-cap loss and the absence of that direct
  gradient for a genuinely frozen-cap component.
- Report atom/bond gradient norms, element-specific directional effects and
  gradient conflicts. One scalar coefficient cannot generally equalize each
  output head and every chemical direction.
- Audit the actual teacher probabilities and proposed witnesses. Include valid
  reference molecules with normal sulfur chemistry, invalid dense graphs,
  hard-feature boundaries, controlled fixed-time queries and mapping checks.
- Retain atom/charge-preserving targets where that is the intended experiment.
  Any allowed identity edit needs explicit cost, coverage and distribution
  accounting. Full-target NLL, bond-only NLL and valid self-guidance are distinct
  objectives and must not be conflated.
- When a composition trust region is used, derive its reference from an
  authenticated empirical/frozen population and include failed attempts. It is
  a distribution prior, not a validity oracle. Continue raw validation, size,
  diversity and retained-reference checks independently.

On 30 October 2026, the user authorized a separate prospective Mosaic
Protenix–DeFoG correction using fixed empirical composition JS. The
[companion correction section](../../mosaic/docs/sulfur_bias_in_chemical_guidance.md#prospective-fixed-composition-correction--30-october-2026)
records the six-element reference, fixed coefficient, allocated calibration,
native-gradient diagnosis and mandatory fresh qualification/pilot gates. This
cross-repository link assumes sibling checkouts. CPU29852742 passed its
31-prespecified-saved-state controls; the coefficient remains a heuristic based
on old native gradients, rather than evidence of a successful new trajectory.
GPU29852743 reproduced a coarse native atom AD/FD disagreement. The subsequent
frozen-model/context full-FP64 CPU29853002 reference matched AD to central and
both one-sided FD at fine scales: at `1e-6`, AD/central FD were
`0.00509808668019`/`0.00509808667504`. A nearby sharp bend explains the
coarse-scale discrepancy; its particular internal operation has not been
isolated. This is no native-gradient or precision-runtime fix, and the old
combined gate remains failed. CPU29852864 passed attempt02's 98 cases with zero
failures/errors/skips, but that does not qualify prospective draft03. Draft03
explicitly replaces the old foreign-state coarse prerequisite with the bound
fine diagnostic, retains all fresh actual native/component/combined gates,
and adds measured strict fixed-teacher descent with saved backtracking trials.
Its planned 100-case CPU qualification, fresh 31-state panel and matched
64+64-update pilot remain required. It keeps the original FP32/default runtime;
at this documentation checkpoint source03 was unsealed and no GPU optimizer
was submitted. Raw JS can penalize legitimate sulfur compositions; neither
exact motif preservation nor a hard sulfur ban is claimed. The frozen-neighbor term
in [docs85](85_neighbor_aware_chemistry.md) has zero direct atom-logit derivative
at fixed preparation and cannot itself oppose this direct-logit atom drift.
No DBMol training objective, run or checkpoint changes through this proposal,
and this documentation update ran no experiment.

**Submission update, 30 October 2026, 10:53 CET:** final Mosaic attempt04
CPU29853132 passed all **100 cases** (26 own/74 inherited), zero
failures/errors/skips. Its resource-only output cap is 3.25 GiB; ten
scientific/test/proof files remain exact attempt03 bytes. The
[verified final collection](../../mosaic/results/hpc/20261030-protenix-defog-composition-30000-04/collection-001/verification.json)
records GPU29853158 submitted at **10:52:38 CET**, then **PEND** for physical
GPU availability. No GPU preflight, pilot or new production update existed
at that snapshot. The fixed-composition/strict-descent method still requires
the fresh actual native/component/combined checks, 31-state panel, matched
64+64 pilot and runtime admission before production. This is submission
evidence, not a sulfur-reduction or valid-molecule result.

## 7. Controlled component audit completed

Allocated CPU **29848155** ran on `n-62-31-4` on **29 October 2026,
13:02:39–13:02:47 CET**, scheduler DONE. All **8 cases passed**, with zero
failures/errors/skips, in 5.7812 seconds of recorded diagnostic runtime. The
runtime was Torch2.5.1+cu121, RDKit2024.09.6 and FP64 CPU; CUDA remained
uninitialized. The allocation used one CPU slot, 4GiB host memory and a
15-minute limit. No learned models, training, optimization or WORK3 writes
were involved.

Five original source files were snapshotted and hash-bound. The original
function AST nodes for heavyweight modules were extracted and compiled, with
their hashes recorded, instead of replacing the components with new surrogate
formulas. The frozen-cap case invokes the exact local risk component with
supplied synthetic identity/cap rows; it does not test end-to-end
`prepare_failure_guidance` or witness-search behavior. The local collection
verified the size/SHA of all three result/log
files. The [companion attempt record](../../mosaic/experiments/validation/sulfur-guidance-audit/runs/20261029-component-audit-01/README.md),
[source provenance](../../mosaic/experiments/validation/sulfur-guidance-audit/inputs/source-provenance.json),
[raw summary](../../mosaic/results/hpc/20261029-sulfur-guidance-audit-01/output/summary.json)
and [collection receipt](../../mosaic/results/hpc/20261029-sulfur-guidance-audit-01/collection-001.json)
require the sibling Mosaic checkout.

| Actual component/control | Measured result | Interpretation |
| --- | --- | --- |
| Legacy live probability-weighted excess, neutral near-hard five-neighbor star | Carbon loss 0.0554624652; sulfur approximately 1.65e-34. Sulfur-logit AD −0.01386561630 agrees with FD −0.01386561629. | Gradient descent directly increases sulfur probability at fixed bonds. The same sign was observed for all five fixed supported C/S charge classes, −2 through +2. |
| Atom-detached version of the legacy component | Atom gradient unused; bond gradient remains present. | Removing the live element path removes this immediate substitution derivative, while narrowing the loss's guidance scope. |
| Exact frozen-cap risk with supplied synthetic rows | Atom gradient unused; cap 4 risk 1.06526137. Explicit external replacement with cap 6 gives risk 0. | Frozen caps block the immediate AD shortcut. Changed supplied caps change the objective; that external change is not an atom derivative of the fixed prepared loss. |
| Actual fixed sulfur-containing target graph NLL | Sulfur-logit derivative −0.01428571429. | A fixed valid target can reinforce sulfur already present in its labels. This does not establish why a real search or teacher selected that target. |
| Actual population JS composition component | Matching-distribution loss 0; enriched example loss 0.59510803. The C-to-S count-transfer derivative is +0.01945910. | The tested composition prior opposes sulfur enrichment, rather than sharing the legacy excess term's direct incentive. It is an aggregate distribution control. |
| RDKit controls | Thioether `CSC`, sulfoxide `CS(=O)C`, sulfone `CS(=O)(=O)C` and SF6 `FS(F)(F)(F)(F)F` sanitized; the corresponding CF6 string was rejected. | Legitimate sulfur chemistry must remain representable. SF6 is a valence control, not a drug-likeness example. |

A separate toy shared-parameter control also showed that a bond-risk descent
direction can increase sulfur probability through a shared parameter despite
an unused direct atom-logit gradient. It is an existence counterexample only;
it does not attribute any trained decoder's drift. The complete eight-case
summary retains fixed-bond degrees 2/4/5/6/7 and all five charge-class AD/FD
results, rather than presenting only the favorable five-neighbor example.

Historical account-bound allocated command, preserved for this exact attempt:

```text
/work3/s234818/latentmol/env/boltz311/bin/python \
  /zhome/95/e/206121/mosaic-runs/sulfur-guidance-audit-20261029-01/source/scripts/audit.py \
  --root /zhome/95/e/206121/mosaic-runs/sulfur-guidance-audit-20261029-01/source \
  --manifest a1d8c4353f5743cb02344194f7648a1f98c93b6d96a04d9e513b5facef1a5f86 \
  --output /zhome/95/e/206121/mosaic-runs/sulfur-guidance-audit-20261029-01/output
```

Summary SHA256:
`60e1cf2a428dc8ba6db91cf949ead819f0e076009336315a71e23f30d2d752d3`.
The source archive SHA256 is
`04ba03ef5a50d5ab17a8b7ea0535c3f46d5af54192a36d0d62f7a4f2e486b5b6`.
These results establish the tested component mechanisms and controls. They do
not determine the fraction of the historical training collapse caused by each
mechanism, validate a replacement loss, establish binding efficacy, or isolate
the internal reason for the DeFoG teacher's conditional sulfur preference.

## Evidence and source identities

The original and epoch100 JSON metric files were inspected as saved metadata,
without importing scientific modules or recomputing molecular measurements:

- [Original metrics](../outputs/hpc-results/status-20261010-check002/original/metrics.json),
  SHA256 `e4d85f78ec467fe21cfd6dd0336df4a1b8d13385b87b549e103b883a38495ceb`.
- [Epoch100 prior metrics](../outputs/hpc-results/status-20261010-check002/decoder/e100/prior/metrics.json),
  SHA256 `681d2c35f64615e4a827e6f2dd62b87dde3fed426578c42864aa5c58f9d8cef2`.
- [Historical capsule verification](../outputs/hpc-results/status-20261010-check002.tar.gz.verification.json)
  retains the96-member/95-file checked export and its original source context.
- [Rule and live-mixture implementation](../src/latentmol/chemistry_regularization.py#L260),
  [fixed-label search and target checks](../src/latentmol/atom_preserving_chemistry.py#L223),
  [detached failure preparation](../src/latentmol/repair_failure_guidance.py#L127),
  and [frozen-cap local bond risk](../src/latentmol/repair_failure_guidance.py#L330).
  Line references describe the working source reviewed 29 October 2026; immutable
  experiment source identities remain in their original receipts.

The [global experiment ledger](05_experiment_ledger.md) is the authoritative
chronology. No running source, scientific setting, checkpoint or failed result
is replaced by this explanation.
