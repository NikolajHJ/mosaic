# Atom–pair decoder architecture comparison

Status, 29 September 2026: four full workflows and four matched fixed-`z`
diagnostics are complete; other architecture arms remain RUN or PEND. The
latest outcome table and immutable run identities are in the experiment
ledger. These results concern graph reconstruction and raw chemistry-code
validity, not binding, synthesis or a final architecture ranking.
Adopted from the 28 September 2026 discussion.
Execution evidence belongs in
[`05_experiment_ledger.md`](05_experiment_ledger.md).

## Question and scope

The existing 128-dimensional latent / 128-wide decoder predicts all atom,
charge and bond logits in one pass. Its pair features are symmetric functions of
two atom hidden states, but there is no persistent hidden state for a pair and no
update from predicted bonds back to the atoms. The proposed comparison asks
whether a larger molecule latent, a narrower atom state, residual updates and a
few rounds of **bidirectional atom–pair refinement** improve held-out graph
reconstruction and hard-sample behavior. This is an AE/VAE decoder-architecture
study expressly requested after the historical AE/VAE runs. It does not train a
separate activity, chemistry-validity, synthesis or other in-house reward/prior
branch. Later scorers remain frozen external checkpoints
under [`23_frozen_model_plugin_contract.md`](23_frozen_model_plugin_contract.md).

The graph contract stays at 50 heavy-atom slots, with `NO_ATOM`, nine real
elements, five formal-charge categories `(-2,-1,0,1,2)` and five bond
categories. Keep the existing canonical slot order, data filtering and losses
unless an explicitly separated ablation changes one of them. The current Boltz
adapter has fixed occupancy and fixed pose: better variable-size decoding would
not make variable-size Boltz optimization supported automatically.

## Architecture to implement and compare

The **historical reference** is the existing `ModelConfig(hidden=128,
latent=128,layers=3)` model. `GraphDecoder` expands `z` through a `2*hidden`
global layer to 50 atom states, refines each state independently, then predicts
atom and charge categories and symmetric bond logits from pairwise sum,
absolute difference and product. It has approximately 1.75 million decoder
parameters; source037's HPC parameter-count check found **1,746,452** exactly.

The **width/residual control** should use the larger latent and narrower atom
states selected for the new decoder, with residual node refinement but without
persistent pair-state rounds. It tests whether latent width, altered capacity or
the residual path alone explains an observed gain. Where practical, match its
decoder parameter count closely to the pair-refinement candidate and report the
remaining difference. Merely matching dimensions while giving one model far
more parameters would weaken the comparison.

The implementation in progress uses `ModelConfig.decoder_variant` values
`legacy`, `residual` and `pair_refine`. The intended new-model settings are
`latent=256`, `hidden=96` per atom, `global_hidden=320`, `pair_hidden=48` and
`attention_heads=4`; `pair_refine` uses `refinement_rounds=2`, while the
`residual` control executes zero pair-refinement rounds. Source037's HPC
parameter-count check found **1,694,964** decoder parameters for `residual`
and **1,872,476** for the two-round `pair_refine`. These are verified model
construction counts, **not trained quality outcomes**.

The **two-round atom–pair candidate** should retain `z` as a whole-molecule
vector and create atom states `h_i` and symmetric hidden pair states `e_ij`.
For one molecule, pair states have shape `[50,50,D_pair]`; with a batch, the
tensor is `[B,50,50,D_pair]`. This is a hidden representation, distinct from
the five-category bond-probability output `[B,50,50,5]`. In each round:

1. Read soft atom-type and charge predictions along with atom and pair hidden
   states. A charge head may use the same atom state plus its soft atom-type
   context; subsequent updates can pass charge context back to atom and pair
   states. Do not replace the five-category formal-charge target with a scalar.
2. Update each pair from both endpoint states and its old pair state. Build the
   pair function to be symmetric under endpoint exchange, or symmetrize its
   result explicitly. The pair state is a learned bond context, not an
   attention probability and not yet a categorical bond.
3. Aggregate neighbor/pair information into each atom through masked attention
   or edge-conditioned messages. Attention weights across neighbors govern
   information flow; the categorical bond head must still normalize across the
   five **bond types for each pair**.
4. Apply residual node and pair updates, with normalization/stabilization as
   needed. Keep self-pair outputs fixed to `NONE` and gate final bond
   probabilities by endpoint occupancy once, at the common graph boundary.

Use a global trunk width independent of `D_atom`: the intended 256-dimensional
latent should pass through the separate 320-wide global layer rather than an
automatic `2*D_atom=192` bottleneck from the 96-wide atom state. The saved
configuration must fix widths, heads and round weights. The adopted candidate
must be no more than **150% of the current decoder's trainable count**, or
**2,619,678** at this baseline. Both
source037 new variants pass this parameter-count gate. Also report encoder and
whole AE/VAE counts, since a widened latent changes them too. Pair memory,
attention intermediates and wall/GPU time matter even when the trainable
parameter count is small.

A **three-round variant** is conditional: run it only if the two-round model
passes numerical/contract gates and gives a reason to spend the extra compute.
Record why it was launched and compare it with the two-round model under a
matched data and training/evaluation protocol. Do not select the round count on
the test split.

### Additional global bond-context candidate (HPC CPU contracts passed)

The next user-authorized candidate adds **one global 96-dimensional
bond-context vector** alongside the 50 atom-slot states. It is a hidden
whole-molecule representation, not a 51st atom: `GraphSpec.max_nodes` remains
50, the context vector has no `NO_ATOM`/charge label, does not contribute to
atom count, and creates no 51st row or column in the bond tensor. A shared
global 320-wide state receives a residual `320 → 512 → 320` update before
projecting to 51 hidden 96-feature vectors. The first 50 become atom-slot
states; the last is the global bond context. The source040 `global_bond`
implementation normalizes this vector and broadcasts it to the **final bond
head for every atom pair**. It does not enter the atom or charge heads or
the two existing atom–pair attention rounds directly. Pair-specific
information still comes from endpoint and pair states, so broadcasting the
same global context does not make all bonds identical. The graph output
remains 50 × 50 with symmetric categorical bonds.

Source040 also defines `deep_pair_refine`: the same
deeper residual trunk and 50 atom slots, **without** the 51st global bond
context. This is the direct control for the additional context. Compare it
with source037 `pair_refine` to test trunk depth, then compare source040
`global_bond` with `deep_pair_refine` to test the context while holding trunk
depth fixed. Source037 `residual` remains a broader width/residual control.
The source037 immutable source, checkpoints and budgets were unchanged by
this addition. The source040 local archive is
`outputs/source-releases/source-040-global-bond-context.tar.gz`, SHA-256
`65c3a220be0ca666feb5667c52276f3fb4347a252b34bd0a8dc8595c9b953e71`.
Its first HPC upload attempt timed out before SSH authentication. It was
subsequently uploaded and extracted at
`/work3/s234818/latentmol/releases/source-040-global-bond-context` with a
matching archive SHA-256. Selected CPU regression job **29510432** is
completed: **57 passed, five skipped in 47.78 seconds**. This CPU suite alone
does not establish trained-model quality.

Keep the original **2,619,678-decoder-parameter** ceiling unless the user
revises it. The residual 320→512→320 block adds roughly 0.33 million weights
by its matrix sizes alone, before normalization, the extra 96-feature
projection slice and enlarged bond head. The source040 config/contract tests
include a parameter-cap check, but the exact decoder and whole-model counts
were not reported in the selected-suite summary. The HPC parameter-cap
assertion passed for the new candidate. Returning to width 320 keeps the
subsequent dense slot projection smaller than a 1024-wide trunk would. If
the cap is exceeded, reduce or factorize the new branch rather than silently
raising the agreed budget.

Source040 includes matched AE and VAE smoke/full YAMLs for `deep_pair_refine`
and `global_bond`, with latent 256, atom/context width 96, global width 320,
trunk expansion 512, pair width 48, four heads and two refinement rounds.
The full YAMLs preserve seed 17, batch 64, the 60+20 validation-governed
schedule up to 120 epochs, the protected 215,000/7,347/7,347 data split,
the existing reconstruction loss and the VAE beta 0.001/20-epoch warmup.
Four source040 two-epoch GPU engineering smokes were submitted as
jobs **29510438–29510441** (deep-trunk AE, global-bond AE, deep-trunk VAE,
global-bond VAE). Four same-arm-dependent full-data jobs were then submitted:
deep-trunk AE **29510443** after 29510438, global-bond AE **29510444**
after 29510439, deep-trunk VAE **29510449** after 29510440, and global-bond
VAE **29510451** after 29510441. By 29 September, deep-trunk AE had
completed and its matched fixed-`z` diagnostic had run; the other three
source040 full jobs remained PEND. For an interpretable comparison, train matched AE and VAE versions under the
same schema, split, losses, checkpoint selection and diagnostic protocol as
the source037 pair model. Report both parameters and actual GPU cost, then
compare reconstruction, whole-graph validity and fixed-`z` draws on the same
held-out cohort. The deep-trunk control separates the effects of shared depth
and global context. As with the existing decoder, attention/feedback can coordinate soft predictions but
does not make the final categorical hard draws a joint autoregressive law.
Source040 passed its selected CPU contract suite, including the cap,
gradient and symmetry checks. A later source042 HPC assertion established
**2,242,044** decoder parameters for `global_bond`; the exact deep-trunk
control count is not reported here. The completed deep-trunk AE test and
fixed-`z` outcomes are in the ledger; they do not rank the remaining arms.

### `wide_global_bond`: direct 512-wide output and residual bond context (CPU suite passed)

This is a **separate candidate**, not a revision to the immutable
source040/041 jobs above. Interpret the user's later reference to a "520"
block as the previously specified **512-wide** block unless corrected. The
`wide_global_bond` shared path is `z(256) → 320 → 512`, retains width 512 for the
final projection into 50 atom-slot vectors and one 96-feature global bond
vector, and adds an explicit `z → 512` skip into that shared path. Unlike
source040's `320 → 512 → 320` block, it does not narrow back to 320 before
the slot projection. The extra vector remains whole-molecule bond context,
not a 51st atom or an additional row/column of the graph.

The bond-context path also receives a direct `z → 96` skip and
passes through one residual block containing two 96→96 fully connected layers
before its final use. Keep the existing symmetric bond head, two atom–pair refinement rounds,
atom/charge heads, categorical graph contract and occupancy gating unchanged.
The source042 HPC contract suite verified the two skip locations and gradient
paths. The comparison
to source040 `global_bond` isolates the effect of
retaining a wide shared output and giving bond context a residual path,
although those two changes occur together unless a further ablation is run.

The requested direct 512-to-51×96 output projection intentionally makes
this candidate exceed the earlier 150% decoder-parameter ceiling. Treat
that ceiling as **revised for this candidate only**; do not retroactively
change the budget or interpretation of source037/040 jobs. HPC count assertions
confirmed **3,191,548** decoder parameters for `wide_global_bond`, versus
**2,242,044** for source040 `global_bond` and the former ceiling of
**2,619,678**. Record encoder and whole-model counts, memory and GPU time before
claiming any efficiency or quality benefit. Any AE/VAE comparison must use
the same graph schema, protected split, sampling/evaluation budgets,
validation checkpoint rule and fixed-`z` diagnostics as the earlier arms.
Source042 is the immutable implementation release: local
`outputs/source-releases/source-042-wide-global-bond.tar.gz`, HPC
`/work3/s234818/latentmol/releases/source-042-wide-global-bond{,.tar.gz}`,
verified archive SHA-256
`bd5c282741f3981c77fc81d9ec277880f3e379c42c9f2bef40d3cefc503cda90`.
Its selected CPU suite **passed** as job **29510497**: 62 passed and five
skipped in 49.69 seconds. It verified exact decoder counts, direct
`z → 512` and `z → 96`/context-MLP gradients, atom-output isolation and
bond symmetry. This is an implementation contract result, not a trained
model or chemical result. Four matched configs exist:
`configs/arch_wide_global_bond_{ae,vae}_{smoke,full}.yaml`. Two-epoch GPU
engineering smokes were submitted as AE **29510498** and VAE **29510499**;
the full-data jobs AE **29510500** and VAE **29510501** were submitted with
same-arm `done(...)` dependencies and were PEND at the last scheduler check.
These submissions establish no training or chemistry outcome.

### `stable_deep_pair_refine` after the deep-trunk VAE failure (captured-state gate passed)

Source040 deep-trunk VAE job **29510449** failed during epoch 7 with a
non-finite training loss. Exact unchanged-source replays **29532730** and
**29532819** reproduced the failing batch and traced activation growth
from the shared global residual block through the slot projection and
pair refinements to a bond-head LayerNorm input near `1.83×10²¹`. The
first observed NaNs were at that LayerNorm; the underlying numerical cause
and a stable training result remain unproven. Details and the retained
failure belong in the [ledger](05_experiment_ledger.md).

The separate source047 variant `stable_deep_pair_refine` inserts
non-affine normalization **after** residual updates in the global, slot,
atom/charge-context, edge and node paths. Its no-affine modules add no
trainable parameters or state keys, allowing historical weights to be
loaded for a controlled failing-state diagnostic while preserving the
original source040 decoder variants' forward behavior. A loaded source040
checkpoint evaluated through this changed forward is **not** an exact
continuation of the source040 training run. Any source047 AE/VAE training
must be tracked as a new candidate with matched data, schedules, checkpoint
selection and fixed-`z` budgets.

Immutable source047 archive SHA-256 is
`1ad89fb80d74f5cb9a2ce06a41e399a84459e4f8cdd6255d6fcc0f2587ec7814`;
four configs exist as
`configs/arch_stable_deep_pair_refine_{ae,vae}_{smoke,full}.yaml`.
Selected CPU architecture/training/graph suite job **29533087** was
completed: **64 passed, five skipped**. Diagnostic-046 job **29533001**
retained the exact epoch-7 failing batch with SHA-256
`95a84843ba9ed6afcba351d4f23b39e15c6689dbb3e513f8e68ce24835b25d76`.
Source047 GPU gate **29533114** then strictly loaded the failed source040
weights for both decoders on the same 64 captured latent vectors and
targets. The old decoder reproduced NaN bond logits/loss; the stabilized
decoder produced finite logits, loss and nonzero finite `z` gradients, with
no non-finite parameter gradients. This validates the **captured failure
state only**. No source047 long-run training stability or chemistry
improvement has been reported.

Matched source047 two-epoch engineering smokes **29533162** (AE) and
**29533163** (VAE) finished with finite training and completed evaluations,
using the same `zinc215k-seed17-v1` cache and corresponding smoke configs.
These subset runs do not establish long-run stability or quality. Full-data
jobs **29533194** (AE) and **29533195** (VAE) were submitted after the smoke
gate; their outcomes are pending. See the [ledger](05_experiment_ledger.md)
for source, config, submit-script and log identities.

### What dependence this does and does not model

At fixed `z`, the current categorical sampler draws each atom, each occupied
charge and each unordered occupied bond from its own final distribution, then
masks absent slots. Shared `z` correlates outputs across molecules, and the
occupancy masks add a limited dependency within a hard graph. A residual
atom–pair refiner can make **the parameters of each distribution** depend on
soft predictions elsewhere in the graph. However, independent final draws from
those distributions are still conditionally factorized at fixed `z` apart from
the occupancy masks. Do not claim an autoregressive or fully joint hard-graph
law from attention alone. Changing the hard sampling law would require its own
implementation, log-probability contract, and matched evaluation.

## Comparison and diagnostic protocol

AE and VAE must share a given architecture, graph schema, data identities,
encoder/decoder implementation, reconstruction loss, seed policy, optimizer,
selection rule and evaluation cohorts. Their established difference remains:
AE decodes the posterior mean without KL; VAE samples the posterior in training
and uses a scheduled Gaussian KL. Widening `z` changes the number of KL terms,
so report total and per-dimension KL, active latent dimensions and posterior
statistics. Keep the first VAE beta/schedule declared and identical across new
variants; treat any beta sweep as a separate experiment. A new architecture
cannot exactly resume an old checkpoint. If initialization from an old model is
attempted, label that as a separate warm-start study, never as the matched
from-scratch comparison.

Use the immutable 215,000 / 7,347 / 7,347 cache from
[`07_full_training_protocol.md`](07_full_training_protocol.md) where possible,
with the original test set sealed until architecture, hyperparameters and
checkpoint roles are selected by training/validation evidence. A short
implementation/smoke gate may precede full training; distinguish its subset,
budget and outcomes from the full-data study. Keep equivalent update and
evaluation budgets for AE and VAE **within each architecture**. Report optimizer
updates and actual GPU time because a pair-state model can cost more per step
despite identical epoch counts. The historical checkpoints are contextual
references, not compute-matched controls against a new training run.

Minimum HPC checks and retained outputs:

- Parameter counts and tensor-shape/symmetry checks, including `e_ij=e_ji`,
  diagonal `NONE`, occupancy gating once, finite outputs and nonzero finite
  gradients from atom, charge and bond objectives to `z` and intended weights.
- Training/validation component losses, learning curves, occupancy/atom-count
  errors and bond-class metrics, including present bonds. Keep the established
  plateau/continuation evidence; do not label a completed budget converged.
- Held-out raw hard decodes: exact graph reconstruction, raw RDKit
  sanitization/connectivity result, first failure reason, element/charge/bond
  correctness, valid changed molecules and uniqueness. Retain invalid and
  rejected attempts in denominators; do not repair silently.
- Fixed-`z` categorical draws as well as argmax outputs, using the **actual**
  decoder sampling law and equal draws/temperatures. Compare charge given
  element and local bond/valence patterns given endpoint chemistry with the
  held-out data. Raw slot-index correlations can reflect canonical ordering,
  so report local or permutation-aware relationships as well.
- AE/VAE latent neighborhoods and interpolation, VAE prior draws, entropy and
  size/charge shifts, all with explicit sampling budgets. Prior-sample validity
  and reconstruction validity are different outcomes. If a later frozen-Boltz
  check is justified, use matched starts/gradient and native hard-candidate
  budgets and retain the fixed-occupancy/pose limitation.

A separate full-budget, read-only fixed-`z` diagnostic is configured to use 128 seeded
validation molecules per validation-selected checkpoint. Encode each molecule
once to its posterior mean for both AE and VAE, then draw 16 categorical hard
graphs at that fixed `z` and record the argmax graph as a separate comparator.
This gives 2,048 attempted categorical graphs plus 128 argmax graphs per
checkpoint, with variable occupancy and no repair or validity-conditioned
resampling. Retain every attempted atom/charge/bond graph, checkpoint and source
identities, failure reasons, raw chemistry-valid counts, exact targets and
unique valid identities. Aggregate element–charge, local element/charge–bond
signatures and bond endpoint-chemistry tables for **all** raw draws, including
invalid ones; the endpoint table must include `NONE` pairs so its denominator
supports `P(bond type | endpoint element and charge)`. Compare with targets and
argmax. Reject a requested molecule budget larger than the chosen split rather
than silently shrinking it. This measures the decoder's actual categorical
sampling law and local dependencies, not binding or a calibrated molecular
likelihood. This diagnostic is implemented in source038 and is **not part of
the source037 full-training jobs**; execution,
chosen checkpoints and results require separate ledger records. Source038's
selected HPC CPU suite passed 52 tests (five skipped), including the pattern
counting and atom/charge gradient-path regressions. An eight-seed × four-draw
end-to-end smoke on a two-epoch pair-AE checkpoint retained all attempts and
found zero valid hard graphs in 8 argmax and 32 categorical attempts. It checks
recording, not the planned full-data architecture comparison.

### Source039 dependent follow-ups after full training

The `architecture_followup` wrapper prepares four separate fixed-`z` runs,
one for each source037 full-data parent: residual AE **29510332**, pair-refine
AE **29510333**, residual VAE **29510334** and pair-refine VAE **29510335**.
Each follow-up depends on its exact successful parent job, uses
the immutable source039 diagnostic release after its own test gate, and reads
the same protected validation split. The source039 archive SHA-256 is
`fc65aeb692e760d1c04eed2fa83559de5a5752d15662f2c260b9173c31b89914`.
An initial CPU-suite launch failed before tests because its uploaded archive
name did not match `run_job.sh`'s expected alias; the corrected-alias suite
passed as job 29510378 (53 tests, five skipped). The four dependent CPU
follow-ups were then submitted as jobs **29510380–29510383** in the arm order
above. By 29 September, residual AE, pair-refine AE and residual VAE
follow-ups had completed with retained fixed-`z` results; pair-refine VAE
was waiting for its running parent. Exact run identities and counts are in
the ledger.

The follow-up must find exactly one completed parent `ae-full-workflow` or
`vae-full-workflow` run under the declared HPC run root by **LSF parent job ID,
source037 archive SHA-256, mode and decoder variant**. It reads that parent's
`selection_before_test.json` and requires a single
`best_validation_reconstruction` checkpoint chosen with
`test_was_used_for_selection=false`. The selected checkpoint path must be the
final recorded training stage's `best.pt`, with bytes matching the sealed
SHA-256. That file may contain an inherited best-validation winner from an
earlier recorded parent stage: its payload `run_id` must belong to **one of
the parent's recorded stages**, and its epoch must equal the sealed selected
epoch. Its saved source archive identity, complete training configuration,
mode and decoder variant must also match the parent, rather than relying on
directory names alone. There is no "latest directory" or test-selected fallback.
The outer source039 wrapper records resolver failures in an `ExperimentRun`,
including a missing or inconsistent parent, before any fixed-`z` draw starts.
`run_study` then checks that
the checkpoint's dataset SHA-256 equals the validation-cache manifest. A
failed parent, missing or ambiguous selection, modified checkpoint, or data
identity mismatch must fail the follow-up and remain in the ledger.

For each selected checkpoint, run the same source038 fixed-`z` method on **128
seeded validation molecules × 16 categorical draws**, plus 128 argmax graphs,
at temperature 1 with variable occupancy. Both AE and VAE use the encoded
posterior mean, avoiding a second posterior-noise source in this comparison.
The source039 wrapper requests CPU execution and retains all raw
attempted graphs, target/argmax/categorical pattern tables, per-seed counts,
chemical failure categories, checkpoint and source hashes, and unique run
paths. Record actual LSF dependencies and outcomes separately for each arm.
These diagnostics assess graph sampling and chemistry-code validity, not
Boltz binding, synthesis or a truly joint categorical graph law.

### Source041 diagnostic compatibility for the new variants (CPU suite passed)

The source039 selector accepts only `residual` and `pair_refine` parents.
A **diagnostic-only** source041 update accepts
`deep_pair_refine` and `global_bond` source040 parents without changing
their training code, source identity, decoder weights or graph schema. It
preserves the same exact-parent LSF ID/source-hash/mode/variant gates,
completed workflow requirement, sealed validation-reconstruction checkpoint
selection, checkpoint-byte and payload provenance checks, and validation
dataset identity check. The four submitted mappings use source040 deep-trunk
AE **29510443**, global-bond AE **29510444**, deep-trunk VAE **29510449**
and global-bond VAE **29510451**. A successful same-arm parent should gate
each CPU follow-up through an explicit scheduler dependency.
For those four new arms, the parent-source identity passed to the selector
must be the **source040 training archive** SHA-256
`65c3a220be0ca666feb5667c52276f3fb4347a252b34bd0a8dc8595c9b953e71`,
distinct from the source041 diagnostic release hash
`2738a237337b1d89eb2bf241aade2d59ba0dcb8d6828d171d7ab6c66a76f1903`.

Each submitted follow-up is configured to run the same
128 encoded validation means × 16 hard draws at temperature 1, plus 128
argmax graphs, retaining every attempt and its checkpoint/source/config
identity. Source041's selected CPU compatibility job **29510456** passed
60 tests (five skipped), including four-variant parent/selection gates.
Four same-arm-dependent fixed-`z` CPU jobs were then submitted:
deep-trunk AE **29510463** after
29510443, global-bond AE **29510464** after 29510444, deep-trunk VAE
**29510465** after 29510449, and global-bond VAE **29510466** after
29510451. By 29 September, deep-trunk AE follow-up had completed; the
other three were still dependent/PEND. Its raw graph results are in the
ledger; submission alone establishes no outcome for the others.
A source041 compatibility test must not be misreported as
new training or as a chemistry improvement.

### Source043 `wide_global_bond` fixed-`z` compatibility (CPU suite passed)

A diagnostic-only selector update accepts the source042
`wide_global_bond` AE and VAE full-workflow parents, jobs **29510500** and
**29510501**, only after successful completion. It must preserve the same
exact-parent job/source/mode/variant gates, sealed validation checkpoint,
checkpoint-byte and payload checks, and protected validation-cache identity.
The **parent training source** is source042 SHA-256
`bd5c282741f3981c77fc81d9ec277880f3e379c42c9f2bef40d3cefc503cda90`;
the separate source043 diagnostic release has SHA-256
`76f9c840569e2c009cd9f27eef67db8ba7a29ea36d351b002232207abe24d215`.
Match the
prior fixed-`z` budget: 128 encoded validation means × 16 temperature-1
categorical draws, plus 128 argmax graphs, retaining all raw attempts and
failure categories. The immutable release is at
`/work3/s234818/latentmol/releases/source-043-wide-bond-followups` and
its selected CPU suite passed as job **29510507**: 63 passed, five skipped
in 48.58 seconds, including the wide-variant exact-parent selector.
Matched fixed-`z` CPU jobs were then submitted: AE **29510512** depends on
`done(29510500)` and VAE **29510513** on `done(29510501)`; both were PEND
at the last scheduler check. No checkpoint resolution or graph result has
been reported.

The package's `ExperimentRun` and HPC wrappers already save per-run source,
configuration and checkpoint identity. Every submitted test or training job,
including failure, needs its own [`05_experiment_ledger.md`](05_experiment_ledger.md)
entry with immutable release/archive hash (including uncommitted source), cache
and split hashes, configuration, LSF job ID, run directory, logs, status and
artifact paths. Submission is not a pass. Keep source releases immutable and
run directories unique; only fill observed fields after inspecting HPC output.

## Interpretation limits

Greater accuracy or valid reconstruction does not by itself show that the new
latent supports better Boltz optimization. Conversely, a factorized sampler's
invalid draws do not prove the pair refiner failed to learn useful relaxed
dependencies. Evaluate hard outcomes and relaxed gradients separately. The
topology-only graph omits stereochemistry and rejects molecules requiring
unsupported chemistry; no training result establishes synthesis, off-target
selectivity or experimental binding.

## Conceptual ligand-architecture review, 10 October 2026

The selected original epoch-89 VAE is `stable_deep_pair_refine`: latent 256,
node width 96, pair width 48, four attention heads and two refinement rounds,
with global width 320 and residual `320 → 512 → 320`. It does **not** use
the optional global 96-dimensional bond-context candidate. Preserve this
checkpoint and freeze its encoder in a prospective first decoder comparison;
no architecture was changed or trained in this review.

[AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w), its
[atom attention](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/network/atom_cross_attention.py)
and [pair modules](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/network/modules.py),
[Boltz-2](https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/) and its
[triangular multiplication](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/layers/triangular_mult.py),
and [Protenix's Pairformer](https://github.com/bytedance/Protenix/blob/main/protenix/model/modules/pairformer.py)
offer pair/atom information-flow ideas. They predict coordinates conditioned
on supplied chemistry; they are not direct substitutes for a VAE predicting
atom, charge and bond categories. [DeepFold](https://pmc.ncbi.nlm.nih.gov/articles/PMC10699847/)
and its [repository](https://github.com/newtonjoo/deepfold) concern AF2-based
protein structure rather than a ligand-category decoder.

The strongest narrow borrowing candidate is one compact gated, symmetric,
occupancy-masked pair triangle/path update per existing round: pair `(i,j)`
can receive information through shared slot `k`. Contract pair states rather
than permanently materializing a three-atom tensor. This is a hypothesis
for a matched ablation, not a validity fix or a proposal to copy dozens of
full-complex layers. Benchmark memory, runtime, reconstruction and unfiltered
generation before adopting it. Precision/compilation and loss/batch options
remain conceptual in [docs45](45_dense_chemistry_100_epoch_continuation.md);
no numerical tests, implementation or submissions occurred in this review.
