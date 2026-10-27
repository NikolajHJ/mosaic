# DiffSBDD, the DBMol VAE and alternating molecular hallucination

Experiment ID: `learned-prior/diffsbdd-vae-alternation`.
Prepared 27 October 2026, Europe/Paris. **Status: source review and prospective
plan only. No model was loaded, mapping trained, numerical fixture run, or HPC
job submitted for this experiment.** The user requested a concrete plan;
implementation and execution are subsequent work. Existing campaigns remain
independent. Actual future attempts belong in this home and in the
[Mosaic experiment ledger](../../../docs/experiment_ledger.md).

## 1. Question and proposed decision

Can a small supervised adapter connect a frozen DBMol VAE to frozen DiffSBDD,
so alternating pocket-conditioned denoising and latent optimization produces
better valid molecules than generation followed by optimization, at comparable
scorer cost?

Implement the first adapter at **DiffSBDD's early ligand atom encoder**, not at
an arbitrary hidden EGNN layer. Retain native coordinates, timestep, noise and
pocket processing. Use the ordinary VAE encoder at explicit, detached graph
handoffs rather than immediately adding a learned inverse adapter. Train no
affinity surrogate and update neither pretrained model.

Three separate deliverables must pass: feature mapping, chemistry-preserving
return to the native sampler, and useful alternating optimization. Feature fit
alone establishes only the first. The return mechanism below is an experimental
sampling modification, not a claim to sample an exact conditional posterior.

## 2. Source actually inspected

DiffSBDD upstream commit:
`5d0d38d16c8932a0339fd2ce3f67ade98bbdff27`.
[Source audit](source-audit.json) records the inspected file hashes and public
checkpoint metadata. Only source and metadata were downloaded; not weights.
Hashes are over raw Git blob bytes, independently of Windows checkout newlines.

| Inspected boundary | Finding and consequence |
| --- | --- |
| [`dynamics.py`, `EGNNDynamics.forward`](https://github.com/arneschneuing/DiffSBDD/blob/5d0d38d16c8932a0339fd2ce3f67ade98bbdff27/equivariant_diffusion/dynamics.py#L87) | Splits coordinates and atom features, applies `atom_encoder`, then combines ligand/pocket embeddings and time. This is the proposed chemical boundary. Coordinates are separate inputs. |
| [`crossdock_fullatom_cond.yml`](https://github.com/arneschneuing/DiffSBDD/blob/5d0d38d16c8932a0339fd2ce3f67ade98bbdff27/configs/crossdock_fullatom_cond.yml) | Declares joint width **128**, six EGNN layers, 500 diffusion steps and normalization factors `[1,4]`. The constructor's default joint width 16 is not this model's configuration. Loaded checkpoint metadata must confirm the actual values. |
| [`conditional_model.py`, `sample_p_zs_given_zt`](https://github.com/arneschneuing/DiffSBDD/blob/5d0d38d16c8932a0339fd2ce3f67ade98bbdff27/equivariant_diffusion/conditional_model.py#L432) | The recurrent state is noisy coordinates and atom features; hidden embeddings are recomputed each step. A hidden-feature regression alone does not define sampler resumption. |
| Same file, `xh_given_zt_and_epsilon` and `sample_given_pocket` | Provides a clean-state estimate and a 500-step sampling loop. The high-level sampler is under `no_grad`; a new wrapper must expose saved state and exact stepwise noise. Do not differentiate through the entire sampler. |
| [`analysis/molecule_builder.py`](https://github.com/arneschneuing/DiffSBDD/blob/5d0d38d16c8932a0339fd2ce3f67ade98bbdff27/analysis/molecule_builder.py) | Bonds are inferred from atom types and geometry, normally through OpenBabel; they are not a diffused categorical state. Its native conversion also reconstructs atoms to remove radicals. Keep raw point clouds and distinguish this builder's output from an untouched chemical graph. |
| `lightning_modules.py`, `generate_ligands` | Supports an explicit residue-list pocket; restores the original pocket coordinate frame after sampling. High-level output may sanitize, retain a largest fragment or relax structures. These operations must not silently filter our denominators. |
| `ConditionalDDPM.remove_mean_batch` | Translates both ligand and pocket to make the ligand mean zero. Track the translation and restore the receptor frame before scoring. Do not independently recenter the ligand. |

Use the **conditional full-atom CrossDocked checkpoint**,
`crossdocked_fullatom_cond.ckpt`, from [Zenodo record 8183747](https://zenodo.org/records/8183747).
The API reported 17,861,341 bytes and MD5
`166b0c056b31ffdf31d489a63e91e05b`. Download and SHA256-authenticate it during
future allocated asset preparation; a runtime/checkpoint SHA is not yet known.
The upstream source has an MIT license; retain its notice and the checkpoint
record's own terms in the asset receipt.

## 3. Frozen VAE and existing evidence

Primary planned VAE: the latest **completed checkpoint in the supplied status**,
epoch 39, SHA256
`e5d7f99b9eb0fe04cd8b6b800e4bd90b6f1c24a65368216ce384a95061a7f3b0`.
Do not read a changing epoch-40 training file or select a checkpoint based on
this experiment's results. Replacing epoch 39 requires a new asset binding and
repeating decoder/adapter validation.

Confirm strict loading, the 50-slot graph schema, latent width 256, atom hidden
width 96, actual bond-head inputs and feature-capture parity before fitting.
These widths follow the inspected VAE lineage, not a new checkpoint load here.
The epoch-39 fixed monitoring count, 724/1,024 raw/strict-valid, is neither a
full-test result nor a guarantee of valid local neighborhoods.

DBMol's source137 bridge is bound to **original epoch 36 and Boltz**, not this
VAE/DiffSBDD pair. Its training recipe is a reference; do not reuse its weights
or certificate. The matched epoch-36 comparison stopped before optimization on
a case003 **relaxed-path** derivative failure; the later diagnostic has no
saved native result in the supplied status. None of that is a passed gate for
this plan. See DBMol `docs/76_epoch36_boltz_embedding_bridge.md` and
`docs/78_embedding_latent_comparison.md`.

## 4. Scope of the first experiment

- Target: Mosaic's existing BRD4 **3MXF/JQ1** input and biological pocket.
  Bind the actual protein, MSA and residue IDs/insertions from the existing
  pocket artifact. Feed that explicit residue list to DiffSBDD; do not let its
  reference-ligand distance rule silently choose a different pocket.
- First optimization cohort: **25 heavy atoms, neutral, connected**, with
  elements C/N/O/S/F/Cl and the VAE's five bond states. Occupancy is fixed inside
  every inner optimization and sampler trajectory. No atom birth/death or
  charged/stereochemical optimization is claimed.
- DiffSBDD's CrossDocked vocabulary is C/N/O/S/B/Br/Cl/P/I/F. Match by symbol,
  never index position. DBMol lacks B; Mosaic's current graph boundary is the
  narrower six-element subset. Unsupported native samples remain recorded and
  count as ineligible for guidance, not resampled replacements.
- Do not suppress unsupported native atom categories in the ordinary diffusion
  baseline. Restriction introduced by VAE feedback is explicit and is controlled
  by the exact-codec and no-hallucination handoff arms below.
- Start with native de novo diffusion from prespecified seeds. Any later
  ligand-seeded partial-noise study is a separate cohort and cannot replace
  unsuccessful de novo starts.

## 5. The forward mapping to train

Let `z` be the molecular VAE latent, `u_i(z)` the final atom hidden state,
`q_i` a distribution over the six pilot elements, and `H` the **frozen native
DiffSBDD atom encoder**. For occupied atoms, train:

```text
q_theta(z) = softmax(MLP_theta(u(z)))
c_theta(z) = native_normalize(scatter_by_element(q_theta(z)))
e_theta(z,t,eps) = H(alpha(t) * c_theta(z) + sigma(t) * eps)
```

`scatter_by_element` embeds six categories into the actual native vocabulary
with zeros in other clean categories. Native Gaussian corruption still acts
on every native feature channel. Native normalization includes the loaded
bias and scale, not an assumed divide-by-four hardcode.

This is a **constrained learned mapping into an internal representation**.
Its output has a known native atom-feature preimage, so returning to the
sampler is defined. Do not train an unconstrained 128-dimensional head and
assume its output can be inverted by `atom_decoder`: that decoder predicts
noise-related outputs after EGNN processing, not a general inverse.

Proposed trainable head: LayerNorm(96), Linear(96,128), SiLU,
Linear(128,128), SiLU, Linear(128,6). Mask inactive VAE slots; keep an explicit
permutation between VAE slots, diffusion atoms and native scorer atoms.
Coordinates and pocket are supplied separately, not inferred from `z`.

**Mandatory zero-training control:** take the ordinary VAE conditional element
probabilities, apply the exact vocabulary/normalization conversion and `H`.
This is already an available mapping. Compare it with the learned head; the
experiment must establish whether supervised smoothing improves anything over
this exact composition. Early-feature matching is mostly an atom-identity
problem and does not align the full molecular graph or solve binding.

## 6. Paired training data and fitting

Proposed bounded corpus: first **20,000 distinct eligible source identities** in
authenticated DBMol training-manifest order, with at most 100,000 source attempts.
If insufficient, report the deficit before fitting; do not silently relax the
schema. Preserve original VAE validation/test roles. Build bridge-only groups
from VAE-training identities, joining source identity, decoded identity and
Bemis-Murcko scaffold; hash groups with seed 17 into approximately 80/10/10
train/development/audit. Record actual resulting counts. No identity/scaffold
collision across bridge splits, and no reserved VAE-test identity is admitted.

Five views per source: posterior mean, two posterior draws and two local latent
perturbations of RMS 0.02/0.05 per dimension. Their RNG seeds and grouping are
saved. Targets describe the **ordinary hard graph actually decoded at each
latent**, not the source molecule if reconstruction differs. Save invalid,
charged, unsupported and changed-count outcomes with reasons. Do not fit
invented targets for those views or replace them with favorable samples.

Native clean one-hot atom features of each decoded graph provide the targets.
Use the same Gaussian noise for target and predicted features, at model times
0, 0.04, 0.10 and 0.20; time 0 uses the checkpoint's actual small nonzero noise
schedule if present. Targets are computed by unmodified `H`. No docking,
affinity, confidence or desired property is a label. Coordinates and pocket
are unnecessary for this early chemical target; they enter downstream tests.

Loss: atom cross-entropy plus embedding MSE standardized using training-only
per-channel statistics with variance floor 1e-4, equal coefficient 1.
Average over occupied atoms before averaging examples. Preserve both raw and
standardized errors. Train only the small head: AdamW LR 1e-3, weight decay 1e-4,
batch 128, seed 17, at most 10,000 updates/3h50 internal. Evaluate every 250 updates;
select the lowest development loss, never the reserved audit result. Treat the
numbers as prospective settings, not measured runtime or selected results.

Generate views/noise online from saved seeds; cache compact latents, hard graphs
and receipts rather than all 50x50 pair states. Initial own-artifact cap 2GiB.

## 7. Feature and denoiser acceptance

First inject exact native early features and reproduce the ordinary denoiser
on identical tensors/noise. Then substitute adapter features and compare native
noise predictions and one-step means, separately for coordinates and chemistry.
Use 200 held-out identities, two conformers each and the four time levels,
with no model/scorer tuning on that audit. This downstream test is necessary
even though the fitting target itself is pocket-independent.

Proposed engineering thresholds to seal before execution:

- Exact boundary and no-guidance sampler replay: same-runtime tensor equality
  when operations/RNG are identical; cross-runtime parity at atol 2e-5/rtol 2e-5.
  Preserve failures rather than assuming this tolerance is attainable.
- Held-out learned atom agreement >=99.5%; embedding NRMSE <=0.05 at **each**
  audited time. Report per-element counts and missing category coverage.
- Denoiser coordinate/chemical NRMSE <=0.05 separately, and 95th-percentile
  per-example error <=0.10. Record absolute errors when the denominator is tiny;
  a near-zero target cannot be hidden by a favorable normalization.
- Every individual handoff additionally requires exact ordinary-decoder versus
  learned-head hard element identity. The population average never overrides it.
- Repeated local latent perturbations, including hard-category boundaries,
  compare learned/exact mappings and their denoiser outputs. Low MSE is not a
  gradient-accuracy or locality proof.
- Codec tests must preserve native atom IDs under canonical permutations,
  including symmetric graphs. Apply rigid rotations/translations jointly to
  pocket and ligand and require matching chemical outputs and transformed
  coordinates within the native replay tolerance. A topology-only VAE cannot
  supply missing stereochemistry or identify a unique conformation.

If the learned head fails but exact conversion works, report that outcome and
continue only an explicitly identified exact-conversion branch. Do not label
it a successful trained adapter. If both work equally, prefer the exact mapping.

## 8. Scorer integration is a real implementation task

Use frozen native PyTorch Boltz-2 affinity and structure/confidence trunks and
the existing DBMol composite reductions: affinity value minus binding logit,
plus 2.5 contact loss and 2.5 confidence loss, with the current declared component
weights. No learned VAE-to-Boltz bridge is needed for this study: the primary
gradient path is the ordinary VAE decoder -> relaxed graph -> native scorer.

Current Mosaic `src/mosaic/small_molecule/boltz2/dbmol.py` is a fixed JQ1-pose,
25-slot profile. Its context validator, protein coordinates, reference conformer,
frames, names and masks cannot be repurposed by replacing a coordinate tensor.
`boltz2/prepare.py` snapshots native predicted contexts; it does not presently
build arbitrary DiffSBDD-pose contexts. Therefore implement an experiment-owned
**new-pose preparation/evaluation wrapper**, rather than weakening the old profile.

For each admitted clean candidate, construct ordinary native input features from
that hard graph, the authenticated receptor/MSA, the supplied pose and exact
atom map. Recompute all chemistry-dependent auxiliaries and frame metadata.
Independently compare its one-hot adapted forward with the native evaluator at
that same supplied pose. Require the original biological pocket residues and
actual coordinate-frame consistency. Stop if native input construction cannot
represent that pose; do not silently redock or substitute a carbon/JQ1 context.

Hold every context field fixed during the inner latent updates. Changing graph
identity or pose afterward requires a new native preparation and endpoint check.
Save fixed-context proxy and rebuilt-hard scores separately. Gradient checks
cover components and total, atoms and bonds, the real encoder/decoder chain,
endpoint-simplex feasibility, independent primal/AD scalar execution and frozen
weights. Include wrong-gradient/null controls. Existing unresolved numerical
failures are not waived or diagnosed by this proposal.

## 9. Return from the VAE to a resumable diffusion state

Maintain native state `Y_t=(r_t,a_t)`, the actual shifted pocket, atom map,
timestep and a separately seeded Gaussian stream. At a guidance opportunity:

1. Compute the native clean estimate from `Y_t` and predicted noise. Save its
   raw coordinates/features. Restore its receptor frame for molecular/scorer work.
2. Run the pinned native molecule builder without largest-fragment selection,
   UFF relaxation or atom removal beyond the builder's documented native policy.
   Record that policy and all failures. Canonicalize only through a verified
   atom permutation. Require a neutral supported connected 25-atom graph.
3. Encode its graph with the VAE posterior mean. Require exact ordinary hard
   encode/decode identity, including bonds/charges, and save natural as well as
   fixed-occupancy decoding. Otherwise skip this opportunity without replacement.
4. Use this `z_base` and the fixed candidate context for K=1 or 5 latent updates.
   Weights remain frozen and only `z` is optimized. See the step policy below.
5. Decode the proposed graph and verify ordinary/adapter element agreement.
   Convert it and its pose to a clean native input `F(z)`. Geometry/topology
   transport must pass the policy below; a feature fit cannot bypass it.
6. Apply the correction in **native state space**, in its current coordinate
   frame and normalization:

   `Y_t_corrected = Y_t + alpha(t) * (F(z_new) - F(z_base))`.

7. Reapply only native joint centering to ligand/pocket, record the global
   translation, and perform the next native reverse step with its prescribed
   Gaussian noise. Do not invert a hidden tensor, reset the diffusion timestep,
   renoise from scratch, or consume an extra random draw on a rejected update.

This correction preserves the residual defined relative to `F(z_base)` and is
an exact no-op when `z_new=z_base`. That residual need not equal true forward
diffusion noise; the modified path is not asserted to follow the training
marginal. Compare residual norms/distributions and native-step output drift.
Audit `t=0` separately: no division by zero and no extra noise is introduced.

**Bonds require two explicit transport stages.**

- **Stage T1, chemistry/pose engineering:** permit hard element substitutions
  only, with the entry bond matrix, neutrality and atom count unchanged. Carry
  the clean-estimate pose. Require native reconstruction at that pose to recover
  the proposed VAE graph exactly. This can test alternation but is explicitly
  limited to guidance that survives an atom-feature interface.
- **Stage T2, full bond-changing latent proposals:** before dense comparison,
  implement deterministic, seeded graph-to-pose lifting from the current pose:
  restrained UFF on the proposed hard graph, at most 50 iterations, heavy-atom
  displacement <=0.5 Angstrom, unchanged atom map. Require supported force-field
  parameters and native builder recovery of the same complete graph. Record
  before/after geometry and score; reject on any failure. No force-field step
  occurs inside the gradient path. This is an explicit additional operation,
  not a claimed property of the embedding mapping. Every comparison that uses
  T2 must give its matched handoff controls the same operation/budget.

Use per-heavy-atom UFF position constraints with maximum displacement 0.5
Angstrom and force constant 10.0 in the pinned RDKit API's units. Deterministic
minimization starts from the supplied pose, without embedding a new conformer.
Record convergence and the actual displacement; constraint failure, missing
parameters or a need for a new random conformer rejects the handoff. If the
proposed hard graph is unchanged, bypass lifting and preserve the pose exactly.

T2 returns both coordinates and atom features through `F`. Rebuild and score
the hard molecule after lifting; accept the block only if that rebuilt native
objective does not worsen from its admitted baseline. Rejected transport
restores the original `Y_t` exactly. A fresh-pose follow-up is still separate.
If T2 cannot convey bonds reliably, stop the full-graph claim and report T1's
limited result; do not count a desired VAE graph that DiffSBDD never receives.

## 10. Inner optimization and smoke tests

For each block reset optimizer state; use simple gradient descent on the full
fixed-context DBMol objective. Initial step is
`0.25 * g / max(||g||_2,1)` in the descent direction, with a radius 1.0 trust
ball around `z_base`. Try the nominal proposal and up to 7 halvings, requiring
finite same-context nonincrease. Also bound maximum per-row atom/bond TV to 0.02.
Save all proposals, hard graphs, applied/no-op decisions, component derivatives,
forward/backward counts and actual rates. Do not transplant LR 400 or AdamW
moments from a different latent study.

Finite-difference engineering probes use prespecified directions and scales
0.01/0.003/0.001 with at least two adjacent feasible/resolved scales satisfying
`|AD-FD| <= 1e-6 + 0.02*max(|AD|,|FD|)`. Preserve independent forward/reverse AD
checks and any inherited stricter contracts; these prospective FD bounds do
not weaken an existing check. Calibration/inconclusive outcomes remain such.
No derivative is claimed through canonicalization, bond perception or UFF.

Smoke cohort: seed 17/23/29, native500-step sampling, guidance opportunities at
t=0.15/0.10/0.05, up to 5 updates each. First run T1, then qualify T2 separately.
Measure overlap between latent loss-improving directions and changes actually
transported to DiffSBDD. Include bond-only/null transport controls: a bond
change with unchanged atom features/coordinates is invisible to native DiffSBDD.
Zero effective transfers is a feasibility failure, not an improved molecule.

## 11. Matched pilot and literal one-step/five-update test

After T2 and scorer checks, use eight prespecified new seeds 31..38 on BRD4.
Use all 500 native diffusion steps. In the dense test, make a guidance opportunity
after each native transition arriving at levels 20/500 down to 1/500. This is
literally **one diffusion transition, then K hallucination updates** for the
last 20 steps; earlier diffusion is unchanged. No claim is made that this late
schedule is optimal. Broader noise ranges require their own development study.

| Arm | Operation and attempted gradient budget per seed |
| --- | --- |
| D | Original native diffusion, no handoff: zero. |
| H0 | Same 20 eligibility/encoding/lifting checks, K=0 and exact no-op state/RNG restoration: zero. Logs chemistry/VAE coverage without replacing any state. |
| S | Finish native diffusion, then 20 consecutive blocks of up to 5 latent steps with the same scorer/context-rebuild/transport policy: 100. Sequential baseline. |
| A1 | Interleave up to 1 update at each opportunity: 20. |
| A5 | Interleave up to 5 updates at each opportunity: 100. Primary proposed hybrid. |
| A5-exact | Same A5 loop with the zero-training vocabulary conversion. Tests whether learning the mapping helps. |

A5 versus S has the same maximum attempted update count; A1 is a guidance-density
ablation, not a compute-matched efficacy comparison. Report successful steps,
failed opportunities, all scorer forwards/backwards, preparation/lift cost and
elapsed GPU time. A nominal 100-step allowance does not imply 100 backward calls
when a handoff fails. Provide gain-versus-actual-cost curves and matched-cost
prefixes, not a blanket compute-equality claim.

H0 must reproduce D's native tensors/RNG exactly for the same seed. Common random
numbers mean the same step-indexed Gaussian draws across arms, not identical
states after edits.
Use separate RNG streams for mapping, sampling, pose lifting and diagnostics.

Eight seeds on one pocket establish feasibility only. An effectiveness study
would freeze this protocol and use 32 seeds each on six receptor-cluster-separated
held-out pockets, with the required new native contexts and atom-count protocol.
Target IDs/files/split hashes must be sealed before any outcome-based selection;
that expansion is not part of the initial BRD4 pilot or an existing submission.

## 12. Outcomes and decision rules

Report per seed/arm raw output, supported/eligible fraction, raw and strict graph
validity, complete identity agreement at every handoff, novel/unique identities,
atom counts, composition, graph edits, pocket clashes, strain, PoseBusters-style
geometry checks, fixed-context loss components and fresh native affinity.
Keep stereo-stripped graph identities separate from full native stereochemical
structures. Report all failed starts and handoffs in denominators; no validity
filter, fragment retention, top-score selection or seed replacement precedes
the primary comparison.

Independently rebuild **every valid terminal identity** from the pilot plus its
valid unmodified baseline, deduplicate only to avoid identical repeated jobs,
then evaluate frozen native Boltz predictions at pose seeds 17/23. Use ordinary
chemical inputs, no learned embedding substitution. Preserve per-seed/arm
memberships and output pose identities. Report this as fresh-pose model
confirmation, not the fixed-pose objective or measured binding affinity. Docking
and ligand-size-normalized scores are complementary readouts, not proof of binding.

Use paired differences and bootstrap intervals over seed, not individual saved
frames. For later multipocket work bootstrap receptor clusters. Report validity
and fresh-score results jointly; better soft loss with worse valid output is
not success. No efficacy threshold is selected from eight pilot seeds. Advance
only if there is nonzero faithful transport, no unresolved mandatory native
failure, and evidence of worthwhile cost/quality tradeoffs versus S and A5-exact.

## 13. Execution, runtime and storage

All model, RDKit/OpenBabel, optimization, fitting and numerical tests run only
on allocated DTU HPC. Local work is source/AST/JSON/hash inspection and planning.
See the [account-independent HPC guide](../../../docs/HPC_GUIDE.md).

| Future stage | Proposed allocation cap / internal cap | Required output |
| --- | --- | --- |
| Assets/codec/contracts | CPU4 slots,45min /40min | Source/checkpoint/schema/runtime manifests, strict loads and exact graph/atom-map contracts. |
| Native DiffSBDD and boundary replay | Exclusive GPU, 4 slots,1h /55min | Original and stepwise replay, exact injection, RNG/coordinate/vocabulary controls and timing. |
| Adapter fitting and held-out audit | Exclusive GPU, 4 slots, 4h /3h50 | Frozen identities, training/development curves, sealed selected head and untouched audit. |
| Scorer/context and T1/T2 qualification | Exclusive 80GB GPU, 4 slots, 2h /1h50 | Fresh native/component/gradient/context and actual transfer evidence. |
| Three-seed smoke | Exclusive 80GB GPU, 4 slots, 4h /3h50 | All attempted handoffs and profiler; no full-pilot claim. |
| Matched pilot / independent follow-up | Serial allocations,each <=24h /23h | Authenticated chunk resumes, exact RNG/model/context identities, complete arm/seed memberships. |

These are caps, not runtime estimates. Before each next stage, forecast from the
slowest measured per-operation time with factor 1.5 plus startup/checkpoint margin.
If the sealed stage cannot fit, stop and repartition it before execution; retain
all prespecified cases and do not quietly shrink the cohort after seeing outcomes.

Use one GPU-owning PyTorch process for resident VAE, DiffSBDD and Boltz, with a
CPU Mosaic orchestrator. Two CUDA workers may be incompatible with exclusive-
process allocations. Create a **new isolated runtime**, never mutate current
VAE/Mosaic environments. DiffSBDD's inspected environment requests Python 3.10.4,
PyTorch 2.0.1/CUDA 11.8 and Lightning 1.8.4; Mosaic's main Python requirement differs.
First establish native reference receipts in the original-compatible runtime,
then export/load the pure PyTorch diffusion core and all noise-schedule buffers
in the combined runtime and pass output/step parity. Do not silently port
Lightning or drop checkpoint tensors. If coexistence/parity fails, resolve that
engineering dependency before fitting or proposing shared GPU workers.

Recheck actual quota and all active campaign reservations before staging or
launching. Require >=10GiB **additional unreserved** headroom for the initial
scope, or a measured larger requirement; own artifact cap initially 2GiB.
The current Mosaic ledger already records storage-admission failures. Do not
infer free capacity from older DBMol snapshots, reduce another campaign's
reserves, or authorize cleanup through this plan.

## 14. Implementation ownership and order

Only this README and [source-audit.json](source-audit.json) exist for this plan.
Future owned paths, created with their required README files:

| Proposed path | Responsibility |
| --- | --- |
| `configs/protocol-v1.json` | All settings above, exact assets, splits, cohorts and acceptance rules. |
| `scripts/assets.py`, `codec.py` | Authenticated model export/load, exact graph/slot/native-frame maps. |
| `scripts/bridge.py`, `prepare_pairs.py`, `fit_bridge.py` | Constrained adapter, grouped paired data, fitting/audit and exact control. |
| `scripts/native_steps.py`, `pose_lift.py`, `scorer_context.py` | Step replay/noise, T1/T2 handoff, independent native pose/context preparation. |
| `scripts/alternate.py`, `evaluate.py`, `ops.py` | State machine/arms, independent terminal follow-up, sealing/submission/collection. |
| `tests/`, `hpc/`, `runs/<unique-attempt>/` | Allocated contracts, bounded entrypoints and immutable attempt receipts. |

Mosaic shared owners are `src/mosaic/small_molecule/state.py`, `pocket.py`,
`boltz2/{native,prepare,dbmol}.py`; DBMol supplies the graph codec, VAE loader,
decoder feature capture and inspected Boltz reductions. Package minimal exact
source dependencies with hashes; avoid an implicit import from a changing sibling
checkout and avoid sealing the entire dirty tree with unrelated studies.
Do not modify current campaign validators to admit this new experiment.

Implementation order: (1) asset/codec and step replay, (2) exact forward map and
native-pose scorer, (3) train/audit the constrained head, (4) T1 then T2 return
checks, (5) three-seed smoke, (6) matched dense pilot, (7) independent follow-up.
The first executable proof is **one correct round trip and one native diffusion
transition**, not a long optimization run.

No preparation, training or launch CLI exists yet. Future `ops.py` must expose
explicit `prepare`, `submit --stage`, `status`, `collect` and `report` actions
with configuration/output arguments and allocated launch authentication. Document
actual commands, interpreters, source/config/checkpoint hashes, resource requests
and authorized paths in the first immutable run record before submission. The
command names here are an implementation contract, not runnable instructions.
