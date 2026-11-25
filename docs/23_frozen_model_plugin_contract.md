# Frozen external models: shared-molecule adapter contract

Active user direction, 28 September 2026. **Do not train new in-house chemistry
priors, denoisers, energy critics, synthesis surrogates or activity scorers.**
Use externally pretrained model checkpoints, frozen and replaceable through
explicit interfaces. Small supervised input bridges or representation decoders
are allowed when necessary, then frozen before molecular optimization. This
does not authorize retraining a replacement scorer under the name of an adapter.

The already-trained molecular AE/VAE and historical experiments remain evidence.
The unexecuted exact-schema critic proposal in docs20 is rejected under the
current preference; source022/024 denoiser failures remain in the ledger. This
document records the source034 interface implementation and its failed HPC
verification, followed by **source035's 44 passing selected HPC regressions**.
No actual new external checkpoint, bridge training or chemistry-improvement
experiment was run.

The implemented additive source034 scope is deliberately
small: `latentmol.graph_policy` for shared graph construction,
`latentmol.scoring` for `BranchManifest`, `InputSupport`, `GraphScoreBranch`,
`evaluate_branches` and a future-only hard-graph protocol, plus
`BoltzGraphBranch` as a thin facade over the existing adapter/model. The old
Boltz adapter import remains a compatibility re-export. A mock two-branch
gradient test is included in the selected HPC suite. Source034's two facade
failures remain recorded; the corrected **source035 suite passed all 44 tests**.
There is no broad registry or new prior/scorer training in this scope.

## Source035 correction: selected HPC regressions passed

Local archive:
`outputs/source-releases/latentmol-source-035-frozen-score-contract-fix.tar.gz`.
HPC archive:
`/work3/s234818/latentmol/releases/source-035-frozen-score-contract-fix.tar.gz`;
release directory:
`/work3/s234818/latentmol/releases/source-035-frozen-score-contract-fix`.
Matching SHA256:
`aee578cc7b848b65623ff62ea7714a0bd996ae2978149b8c24f587d4750ed313`.

The Boltz eligibility correction is the masked bond expression
`graph.bonds[0][masked][:, 1:].any()`. A focused diagonal-rejection test was
added and the wrapper's run naming changed to 035; no model/scoring objective
or molecular policy changed. CPU **29508977 is DONE**, `hpc`, two CPU slots,
4 GB memory request, 30 minutes. It runs the same selected suite plus the new
regression through `boltz_run.sh` / `frozen_score_contract_tests.sh`.
Logs: `/work3/s234818/latentmol/logs/tests-frozen-score-contract-035.%J.{out,err}`;
XML: `/work3/s234818/latentmol/logs/tests-frozen-score-contract-035-29508977.xml`.
**44 tests passed in 13.16 seconds**; LSF reports successful completion.
Run `20260928T142648Z_tests-frozen-score-contract-035_526671fe` records completed
status, return code 0, job ID 29508977 and the exact source035 archive hash above.
The selected suite verifies interface/policy/facade behavior, including mock
composition gradients; it does not validate a new external checkpoint or prove
useful chemistry. No bridge training occurred. Source034's failed immutable
release remains preserved rather than being relabeled successful.

The combined evidence bundle contains both immutable source archives, failed
source034 and completed source035 run directories, scheduler stdout/stderr and
JUnit XMLs. HPC export:
`/work3/s234818/latentmol/exports/frozen-score-contract-source034-035-results-20260928.tar.gz`.
Local: `outputs/hpc-results/stage3/frozen-score-contract-source034-035-results-20260928.tar.gz`.
Remote/local SHA256 matches:
`409dd782c393961a4de58d2ca8be4ae3c3121f5a942971b47bbbd7856b60f5ec`.
The local archive hash was independently rechecked.

## Source034 release and failed CPU verification

Local archive:
`outputs/source-releases/latentmol-source-034-frozen-score-contract.tar.gz`.
HPC archive:
`/work3/s234818/latentmol/releases/source-034-frozen-score-contract.tar.gz`;
release directory:
`/work3/s234818/latentmol/releases/source-034-frozen-score-contract`.
Matching SHA256:
`853357b72ef96c23dbd6bf3339b6a4120b9f05ff069edd556ca227ca30c50694`.

CPU job **29508965 exited with code 1**, `hpc`, two CPU slots, `rusage mem=4GB`,
30-minute wall limit. It uses `scripts/hpc/boltz_run.sh` and
`frozen_score_contract_tests.sh` for `test_scoring.py`, `test_graph_policy.py`,
`test_boltz_scoring.py`, `test_boltz_adapter.py` and `test_boltz_latent_pilot.py`.
Logs: `/work3/s234818/latentmol/logs/tests-frozen-score-contract-034.%J.{out,err}`.
Failed run: `20260928T142349Z_tests-frozen-score-contract-034_b834b5c1`, under
`/work3/s234818/latentmol/runs`, started approximately 14:23:49 UTC.
**41 tests passed and two failed in 12.97 seconds.** Both new Boltz-facade
failures occur at `src/latentmol/boltz/scoring.py:112`: boolean indexing
`graph.bonds[0, masked, 1:]` mismatches the `[2,2]` mask with the tensor dimensions.
Existing selected regressions passed, but **source034 as a whole is not verified**.
This is an implementation/test failure, not a scheduler/authentication issue.

Preserve the failed run manifest, immutable release, scheduler files
`tests-frozen-score-contract-034.29508965.{out,err}` and corresponding
`-29508965.xml` artifact under the recorded log root. Source035 correction is
completed its selected HPC tests as source035, recorded above.
These are interface/policy regressions, not a new trained model or molecular
optimization experiment.

## Implemented API and declared limits

`graph_from_decoder(decoded, spec, *, temperature=1., fixed_active=None)` creates
the shared graph under one declared policy. It validates schema/batch dimensions
and boolean fixed masks. Soft occupancy supports batches; the fixed-occupancy
policy remains batch-one like the existing Boltz path. The original
`fixed_occupancy_graph` import remains available through a compatibility re-export.

`evaluate_branches(graph, branches, *, spec, allow_lossy=False)` passes the same
graph object to every relaxed branch and returns nested named tensors. It checks
schema and eligibility before scoring and requires registered frozen-eval
`nn.Module` implementations. Branches treat graph tensors as immutable.
`BranchManifest`, `BranchCapabilities` and `InputSupport` declare provenance,
capabilities and exact/lossy/unsupported eligibility; hard-only branches remain
a separate future protocol, without an implemented sampling-gradient algorithm.
Objective sign, normalization and weights remain outside the branches.

`BoltzGraphBranch(scorer, prepared, manifest, *, allow_lossy_charge=False)` wraps
the existing adapter/scorer. Its manifest is constructed with
`boltz_branch_manifest(prepared, *, adapter_source_sha256, recycling_steps,
validation_report_sha256=None)`. It declares **partial gradients**, fixed
occupancy and the existing fixed-context restrictions. A soft categorical
charge distribution maps lossily to expected scalar charge: both the facade's
`allow_lossy_charge=True` and composition's `allow_lossy=True` are required.
Exact one-hot charges pass exact eligibility within the declared context;
that label does not prove full soft-native parity or chemical validity.

Call `BoltzGraphBranch.verify_context_unchanged()` at run boundaries, before and
after execution. It checks the current prepared tensor context and schema
against construction identity; it is not repeated on every forward. Manifests
are provenance **assertions**, not authentication of arbitrary injected models
or mocks. Production must use verified `load_affinity_model` /
`FrozenBoltzAffinity` and audited checkpoint bytes. Mock composition tests
cannot establish a real new checkpoint's fidelity or useful chemical gradients.

## One molecular state, several model views

For each optimized latent `z`, produce **one canonical shared `RelaxedGraph`**
from the selected frozen molecular decoder. Use one declared policy for slot
ordering, temperature, occupancy, atom elements, formal charges and symmetric
bonds. Every external branch receives that same graph. A branch may reindex or
translate its features, but must not independently decode a different molecule
from `z` or silently neutralize, discard or substitute unsupported chemistry.

```text
z -> frozen molecular decoder -> shared RelaxedGraph
                              -> model A input adapter -> frozen model A -> objective A
                              -> model B input adapter -> frozen model B -> objective B
                              -> model C input adapter -> frozen model C -> objective C
```

The existing canonical graph is topology-only, with 50 slots, nine real elements
plus NO_ATOM, categorical charges and aromatic bonds. Keeping this as the common
state does not imply every external checkpoint supports it. Lossy mappings such
as expected scalar charge must be named as partial representations, with lost
information and unsupported cases reported. If a shared restriction is required
for a study, declare it before scoring and apply it to every relevant branch.
Current Boltz support fixes 13 occupied slots and a pose; modular interfaces do
not add variable-size gradients or fresh geometry.

Model-specific native standardization is an identity audit, not permission to
quietly score a different molecule. Compare canonical input and effective native
identities with explicit atom maps/permutation policy. If they differ, report
the divergence and either abstain or use a separately declared common identity
policy. The source018 charge-collapse results illustrate this requirement.

## Design boundaries beyond the implemented facade

The user-directed [alternating DeFoG reference proposal](37_defog_repair_guidance.md)
adds a separate kind of guidance, implemented but numerically unvalidated: a frozen hard repair
oracle produces an explicitly different **constant target** for decoder CE.
It is not a model-specific alternative molecular input silently scored by
another branch. All actual activity/scoring branches still receive the same
current canonical relaxed graph; reference identity, slot alignment, edits
and target-selection policy are separate artifacts. No gradient through
categorical draws, DeFoG or target refresh is claimed. Native charge/vocabulary
exclusions remain explicit, with no neutralization or hidden substitution.
This does not authorize a trained replacement scorer, a validity guarantee
or retroactive promotion of the historical failed DeFoG derivative interface.
The opt-in fixed-mask runner holds `FrozenRepairTarget` constant across native
score, FD and line-search probes; the provider audits exact request/native
labels/slot/sampler/frozen-state evidence. Empty sampling overrides preserve
the checkpoint's settings. A completed Boltz-only fallback is not a repair
guidance pass: the new study requires actual accepted native references,
nonzero reference gradients, reference-only FD and accepted guided updates.
Authoring these boundaries does not pass HPC fixtures or native model checks.

| Component | Proposed responsibility | Required output/metadata |
|---|---|---|
| `Adapter.prepare(hard_seed, native_assets)` | Native discrete preprocessing, pose/reference construction and fixed context setup | Immutable context, identities/maps, source/checkpoint/asset hashes and fixed-feature policy |
| `Adapter.adapt(graph, context)` | Transform the shared graph into model input tensors without replacing molecular identity | Native inputs, supported/unsupported channels and any declared approximation; differentiable when pathwise mode is claimed |
| Frozen model forward | Execute the pinned external checkpoint in its declared inference configuration while retaining input gradients when supported | Named native outputs, diagnostic values and execution status |
| `Objective(native_outputs, context)` | Define direction, scalar loss and normalization from selected native outputs | Individual raw/normalized loss terms, applicability and declared calibration; no hidden trainable replacement scorer |
| Native hard evaluator | Rebuild ordinary inputs for the same hardened shared molecule and run independent evaluation | Effective identity, new pose/preprocessing when required, scores, failures and cost |

Each branch declares separate capability axes: **feature fidelity** (exact,
lossy or unsupported by channel), **fixed versus variable occupancy**,
**gradient scope** (pathwise, partial pathwise, straight-through or none; any
black-box estimator is separate), and **native recomputation at hard evaluation**
(which features/poses are rebuilt versus fixed). Record hard-versus-relaxed
input support, schema/size range and model/adapter/context/config
identities; and validation status. Unsupported capability should produce an
explicit abstention or error, not a zero gradient silently interpreted as support.
Swappability means an implementation can meet the same declared contract; it
does not mean arbitrary checkpoints are interchangeable without new audits.

The small Boltz facade failed source034 verification and passed the corrected
source035 selected regression suite;
it does not implement a new external prior branch. DeFoG's tested public checkpoint
remains an experimental partial interface with failed gradient gates, not a
validated chemical prior. RDKit/SMILES conversions and categorical sampling
remain outside a claimed ordinary pathwise derivative.

## Permitted bridge training

Use a deterministic tensor mapping first. If one is insufficient, a small bridge
may learn **only supervised reproduction of the frozen model's native input or
early embedding on paired hard molecules and declared context**. Fit on training
molecules, verify hard identity and native-feature fidelity on held-out
development molecules, record bridge size/loss/data
and code hashes, and freeze both bridge and external checkpoint before optimizing
`z`. Include decoder-like soft inputs only under a declared, tested alignment
policy; hard-input fidelity alone does not establish soft-input behavior.
Only after held-out hard parity should the soft path receive finite-difference
checks and hard-candidate outcome tests.

Do not train this bridge on desired binding/validity/synthesis scores as a new
surrogate, tune it jointly to improve the optimization reward, or let it invent
a model-specific chemical identity. Prefer `B_m(G, context)` when sufficient.
The user's small decoder from latent to model input is allowed as
`B_m(z, G, context)` when native features cannot be constructed from `G` alone,
but requires hard-input/native-feature parity and cycle/identity consistency
with that shared graph. Otherwise branches could score different implicit
molecules. Feature MSE alone is not a global identity or soft-gradient guarantee.
For pose/topology-dependent features, a small bridge may be insufficient: declare
unsupported behavior or a fixed-feature approximation. Missing chemical semantics
in the external checkpoint cannot be assumed recoverable through a bridge.

## Gates before combining gradients

1. **Provenance and compatibility:** freeze exact external source, weights,
   configuration and bridge/context artifacts. Audit supported features, size
   range, identity transformations and absence semantics. Reject silent loss.
2. **Hard-input parity/alignment:** compare native preprocessing and adapter
   inputs for the same hard graphs. Require exact parity for deterministic
   equivalent mappings; predeclare tolerances and report errors for a learned
   approximation. Verify native-forward agreement under the same fixed context.
3. **Derivative verification:** check finite, nonvanishing supported chemical
   and latent gradients, multi-scale finite differences for the declared smooth
   objective, and unchanged frozen weights/context. ST and black-box estimators
   require their own estimator tests, not false pathwise parity claims.
4. **Useful hard behavior:** use matched starts/budgets and separate relaxed
   gains from valid changed hard molecules. Retain unsupported/invalid attempts,
   size/charge/entropy shifts, identity collapse, diversity and actual cost.
5. **Native confirmation and combination:** rebuild the same hard molecules'
   native inputs and compare with original/random controls under equal native
   budgets. Audit each objective separately, then gradient norms/conflicts and
   trade-offs jointly. Multiple frozen models do not guarantee valid, selective
   or synthesizable molecules.

New adapters remain research candidates until these gates pass. Objective
weights may be calibrated on a declared development cohort, but external model
weights and validated bridges remain frozen during the molecular search. No
part of this design makes new in-house scorer training an active fallback.

## Checkpoint choice and a hard-sample fallback

No audited pretrained prior is plug-compatible with the complete 50-slot,
charged/aromatic/NO_ATOM soft schema. **JODO ZINC 2D is the leading restricted
pretrained comparator in this interface assessment**, not a proven best prior:
fixed count at most 38, scalar charge, hard masks and no explicit aromatic
channel. Its published checkpoint remains untested here. GraphEBM is excluded
from the active branch where it would require new scorer weights; GDSS/MoFlow
have additional schema gaps, and CoCoGraph is a hard-molecule comparator.
The public DeFoG checkpoint is at most a neutral-compatible hard-projection
candidate requiring its own untested continuation wrapper; its full soft-gradient
gate failed and most source025 draws were charged/unsupported.
[Official JODO configuration](https://raw.githubusercontent.com/GRAPH-0/JODO/main/configs/vpsde_zinc_2d_jodo.py),
[JODO parser](https://raw.githubusercontent.com/GRAPH-0/JODO/main/datasets/zinc_dataset.py),
[GraphEBM implementation](https://diveintographs.readthedocs.io/en/latest/_modules/dig/ggraph/method/GraphEBM/graphebm.html),
[DeFoG experiment evidence](18_stochastic_defog_projection_protocol.md).

A common fallback interface is to **sample hard graphs from the actual decoder
categorical law and evaluate/filter them with a compatible frozen model**.
This permits native discrete preprocessing without inventing a differentiable
bridge. It does not make every sampled graph valid or every external model
compatible. Retain unsupported/invalid/failed attempts, native identity changes
and actual oracle cost; do not select a model-specific alternative molecule.

Optionally, a separately validated score-function estimator can optimize
`E[(R(G) - b) * gradient_z log q_z(G)]` without differentiating the external
model. Use the actual sampling law, including occupancy and symmetric-bond
dependencies; count each unordered bond once. The baseline must be independent
of the current sample or use an appropriate leave-one-out construction. Retain
failure rewards rather than condition silently on successful samples. This
route may be costly and high variance and requires estimator checks; a straight-
through tensor trick does not automatically cross RDKit or native sampling.
[Primary discussion of discrete-variable estimators](https://arxiv.org/abs/1711.00123).
It is a proposed fallback, not an implemented variable-size Boltz path or a
validated prior. Fitting a learned reward predictor to replace these calls would
violate the current no-new-scorer preference.
