# Stage 0 implementation plan: relaxed small-molecule graphs in Mosaic

Plan date: 2026-10-10. Implementation started: 2026-10-12; documentation update: 2026-10-16. Status: all Work 0-5 code and tests authored and merged; independently reviewed. Static checks and local immutable source packaging completed. Numerical acceptance gates A-G remain pending; the dated follow-up below records allocated-HPC gradient testing and pivotal numerical decisions. [The ledger](experiment_ledger.md) records actual receipts.

Source of requirements: [project description](project_desc.pdf), especially Stage 0 (page 2, section 3), the V0 optimizer contract (pages 3-5, section 4), and the minimum outputs (page 7, section 8). The reviewed Mosaic baseline is commit `1eb4af6`. The source PDF SHA-256 is `c9785f45dcd34fbc3feaebcc0bc57905dc606387f76308bb6364eadb40cbeda0`; the user has moved the input to `docs/project_desc.pdf`, and that move is currently uncommitted.

## 1. Outcome and scope

Deliver a small-molecule branch that exposes atom and bond probabilities as the optimized variables and differentiates a scalar derived from frozen Protenix-Tiny outputs back to both variable classes. The same branch must support ligand-only and protein-plus-ligand contexts, with ligand reference geometry disabled and protein features kept native.

Stage 0 ends when those two forward/backward paths, projected updates, and reproducible run artifacts work. It establishes engineering behavior. Stable objective improvement, useful molecular design, and chemical feasibility are later-stage questions.

Follow this dependency order, with independent tasks assigned to the parallel tracks in section 6:

1. Audit and pin the native Protenix ligand feature path.
2. Define the fixed-size relaxed graph and prepared context contracts.
3. Add differentiable feature injection and explicit ligand geometry masking.
4. Expose a resident frozen-model trunk/distogram forward path.
5. Add a strict graph gradient evaluator and projected-probability update loop.
6. Add raw readouts, checkpoint/resume, and the two-mode validation gates.
7. Record the result and hand the branch directly to Stage 1.

Excluded from Stage 0: model training, learned feature bridges, affinity heads, molecular priors, valence/connectivity/sparsity constraints, hardening schedules, charge optimization, variable atom count, conformer updates inside optimization, and broad optimizer/model-depth searches. Discrete molecule conversion is allowed during fixture preparation and detached evaluation only.

This plan concerns the Mosaic repository and the supplied Protenix-Tiny protocol. It does not import the AE/VAE/latent pipeline from the separate DBMol exploratory repository.

### Current implementation receipt, 2026-10-12–13

The user authorized implementation of all planned code components and tests, with progress and pivotal decisions documented. HPC is unavailable for this turn: no HPC connections, uploads, submissions, local numerical tests, preprocessing, model execution, or evaluation are performed. Authored tests and source review are implementation evidence, not passed validation gates. The [2026-10-10 native protein smoke](hpc_protenix_tiny.md) remains historical evidence for a different input/path and cannot validate this graph branch.

| Planned work | Code/documentation status | Numerical acceptance |
| --- | --- | --- |
| Work 0: native audit | Complete: pinned source/runtime guards and consumer table in [the native contract](stage_0_native_contract.md). | Native hard/relaxed parity and actual model execution pending. |
| Work 1: graph state | Complete: T1 commit `8cb6f6c`, shared X/E_upper state, mapping, initialization, projection and authored tests; shared acetanilide fixture and native preparation present. | Tests not executed; gate A pending. |
| Work 2: native injection | Complete: T2 code merged through `4168f0f`/`3470f17`, differentiable float injection, explicit geometry hook and native context checks. | Geometry/protein preservation, occupancy and conformer invariance unmeasured. |
| Work 3: resident Tiny | Complete: T3 pinned offline loader, native/geometry-off trunk paths, graph LossTerm composition and explicit confidence boundary. | Both forward modes and derivative checks pending. |
| Work 4: graph loop | Complete: strict raw graph derivatives, separate probability-space projected steps, fixed/split typed RNG and correctly scored final evaluation. | No trajectory or restart run; gate F pending. |
| Work 5: artifacts/readouts | Complete: context serialization, raw readout/failure retention, immutable checkpoint commit pointers, source/environment bindings, CLI/config, validation suite, tests and future allocation/quota launcher. | Artifact/replay behavior and real validation suite not executed. |

Implementation milestones and validation milestones are separate. M0's shared contract/source audit is complete; M1's code is integrated but its producer/consumer test evidence remains pending; M2/M3 require later actual model execution. Code completion does not close Stage 0 acceptance. [The ledger](experiment_ledger.md) records implementation/static-review evidence without inventing a numerical experiment or submission; [the usage guide](stage_0_usage.md) gives future allocated-HPC commands.

Pivotal decisions: preserve native `ref_mask` rather than remove occupied identity embeddings; gate ligand geometry at its actual encoder consumers; use neutral `Q000` slot names; serialize the native preparation's sampled conformer rather than promise regeneration; inject float element distributions and bond existence while explicitly losing bond order; reuse only one final recycle; return native distograms without fabricating PAE/coordinates. The pinned native 64-class label edges support a conservative whole-bin `<8 Å` contact score, not an exact distance CDF. Details and fidelity exceptions are in the native contract. Stage 1 PAE remains deferred and unsupported by the geometry-off graph result.

The independent reviewer inspected merged implementation slices. Fixes included integer-native element truncation, nested auxiliary metrics under LossTerm composition, raw proposal overflow/failure retention, leaf precision checks, immutable per-step/evaluation checkpoint commit pointers, complete CLI settings, source/environment resume bindings, and strict finite trunk/distogram auxiliary checks. Host NumPy simplex projection is reused outside differentiation; no sequence evaluator sanitizes raw graph values. Numerical correctness of these fixes remains untested in this outage turn.

Final launcher review also separated scheduler evidence from numerical payloads: the parent run may contain LSF stdout/stderr, while an atomic `mkdir RUN/payload` refuses reuse of an existing payload. All numerical outputs and pytest temporary directories stay inside that new payload, preserving earlier runs. The launcher authenticates the source manifest before executing quota helpers and rejects unmanifested execution files. These are authored preservation/authentication boundaries; allocated-HPC behavior remains untested.

Prepared-feature identities hash the actual dtype-normalized resident arrays; the original NumPy importer bytes have a separate pre-cast hash. This avoids naming a pre-cast fingerprint as the identity of the arrays used by the model. Final static checks pass; local release receipts are stored under `results/releases/stage0-20261013/`. Numerical gates A-G remain pending.

### Allocated-HPC gradient-test follow-up, 2026-10-16

The user restored DTU HPC access and requested DBMol-inspired real Protenix gradients into a relaxed molecule. The [focused protocol](protenix_relaxed_gradient_tests.md) and [ledger](experiment_ledger.md) record this bounded engineering extension and every actual attempt. T3 adds an explicitly mapped pocket/anchor contact NLL; T1/T3 test atom and edge-existence tangent derivatives and measured projected descent; T2 replaces a sparse element one-hot matrix multiplication with exact indexed scatter after an A10 TF32 boundary failure; T4 binds context/source/runtime identities and retains raw probes, candidate graphs and failures. The existing acetanilide/20-residue engineering fixture, fixed neutral occupancy and existence-only native bonds remain shared. This test's explicit synthetic pocket does not replace Stage 1's separate prescribed target-contact fallback.

Pivotal numerical decisions:

- Preserve the default-precision failed derivative result. A separately sealed `matmul_precision="highest"` profile reuses the exact authenticated earlier prepared context and frozen model, requiring all nonprecision configuration fields and initial model/context digests to match. Both graph leaves then passed all three finite-difference scales in case 0; the entire attempt still failed a later guard. No epsilon or acceptance tolerance was loosened, and no new conformer was substituted.
- Compare a bond-order-null graph with an unaltered graph through the same standalone primal forward. The previous null guard mixed an autodiff-compiled auxiliary output with a standalone output; retain that failed measurement and record the cross-path difference separately. The null threshold remains `2e-5`, scalar AD/primal parity requires `2e-5`, and a new final standalone-primal decline greater than `1e-7` supplements the three evaluated candidate declines. These controls distinguish a graph perturbation from numerical execution-path differences.
- Keep unit fresh-process fixtures on CPU within the allocated DTU host to avoid exclusive-GPU parent/child contention, and independently rerun all seven feature checks on the allocated A10. All numerical work remains on allocated HPC.

Final actual state: earlier source 02/03 runs passed 124 selected unit checks and seven GPU feature checks but failed their respective native guards; they remain retained. Immutable source-04 job **`29735139` DONE** passed **126 selected unit checks (59.66 s), seven A10 feature checks (25.55 s), and the real native two-case pytest item (207.07 s)**. Both graph leaves in both cases passed all three epsilons. Same-primal bond-order null differences were exactly zero. Each case completed three declining AD-evaluated projected steps at `eta_X=eta_E=0.01`, plus an independent initial-to-final standalone-primal decrease; intermediate standalone-primal losses were not separately evaluated. Model/context digests matched before/after and the authenticated prior input. Results are downloaded and integrity-verified.

| Final mixing case | Initial standalone-primal loss | Final standalone-primal loss | Focused result |
| --- | --- | --- | --- |
| `0.15` | `1.7413299084` | `1.7370867729` | Derivative, null, three-step and final identity checks passed. |
| `0.30` | `1.6747362614` | `1.6332819462` | Derivative, null, three-step and final identity checks passed. |

This completes the focused user-requested Protenix relaxed-graph gradient study, with no binding/chemical-validity claim. It does not close full Stage 0 gates A-G, both-mode acceptance, restart/protein regression or M3. The ledger carries exact source hashes, jobs, commands, retained failures, numerical profile and downloaded evidence; the focused protocol contains the rerun command. All numerical computation occurred on allocated DTU HPC.

### Carbon-chain trajectory follow-up, 2026-10-16

The user requested 100 updates per initialization and a polished report using
`render_protenix_trajectory.py`. The [carbon-chain protocol](carbon_chain_100.md)
records the completed study. Pivotal choices: start fresh from neutral ten-carbon
decane at the same 15%/30% relaxation strengths; retain the synthetic protein,
fixed RNG, highest precision and derivative thresholds; independently measure
the standalone primal at **every** candidate and require both losses to decline
by more than `1e-7`. Retain all accepted states and rejected trials, with raw
chemical readouts detached from optimization. No hardening/repair or new
scientific objective was added to force a discrete molecule change.

Allocated GPU job **29735673 DONE** passed 128 selected unit checks and completed
100 accepted updates per start. Primal losses: 1.844622612 → 1.773505211 and
1.609976530 → 1.119375110. Model and prepared-context digests are unchanged.
Both hard readout trajectories stay connected decane and sanitize at all 101
states; soft probabilities change. This demonstrates longer measured relaxed
descent, not a new chemical identity or binding result. Report rendering uses
separately sealed presentation source and allocated CPU jobs; it never mutates
the tested GPU source or reruns the model. Full Stage 0 A–G/M3 remains pending.

The subsequent user-requested [500-update study](carbon_chain_500.md) starts
fresh from both carbon-chain initializations and reuses the authenticated
100-update context. Its explicit longer-budget replay allows only
`checks.steps` to change, retaining all other inputs and numerical gates.
Reserve 8 GiB free at preflight to cover the longer evidence while keeping
4 GiB headroom. Render on allocated CPU after the GPU job, preserving the
earlier report. This is a requested longer engineering trajectory, not a new
scientific objective or a waiver of full Stage 0 acceptance.

Storage follow-up for the added 50% start: its first allocation correctly
stopped before tests at just below the conservative 8 GiB gate. Recover only
the two verified redundant upload archives from this new campaign, retaining
identical local tarballs, manifests, read-only HPC source and all results.
Recheck the unchanged quota gate and use a new run identity; record the stopped
attempt and report-job cancellation. This bounded transfer-file cleanup avoids
changing numerical/space tolerances or deleting experiment evidence.

Completed follow-up: GPU jobs **29735858/29735926 DONE** delivered 500 accepted
updates for each of 15%, 30% and 50% uniform mixing. Allocated CPU report
**29735927 DONE** compares all 1503 saved states; evidence packaging
**29735961 DONE** and local integrity verification passed for all 4619 files.
The [protocol/result](carbon_chain_500.md) links the downloaded interactive
report and records losses, molecular changes and receipts. All final hard
readouts replace C3 with S3; the 15% and 30% starts also fragment, while the
50% endpoint remains connected. This bounded synthetic-peptide contact study
does not establish binding or complete full Stage 0 A–G/M3 acceptance.

The next user-requested [real-target 1000-update comparison](known_ligand_1000.md)
uses the deposited T4 lysozyme L99A construct with benzene as a known reference
and six carbon slots for a size-matched chain. Crystal coordinates select
contact residue IDs only; the model remains sequence-only, without a crystal
template. Record this distinct target/size/selection and compact serialization
policy rather than comparing its losses directly with the synthetic runs.
The charged CCD JQ1 is not silently neutralized; the original Stage 1 gates
remain pending. Readiness units/preparation passed on allocated HPC; a native
pilot must pass before the 1000-update budget is submitted.

## 2. What the current code already provides

| Existing code | Reuse | Required change or boundary |
| --- | --- | --- |
| [models/protenix.py](../src/mosaic/models/protenix.py): `load_model`, `ProtenixTiny` | Cache-aware resident model loading; Tiny currently selects `protenix_tiny_default_v0.5.0`. | Add a graph-facing wrapper. `target_only_features` currently accepts protein chains only. Verify the selected checkpoint identity rather than assuming all Tiny variants are interchangeable. |
| [losses/protenix.py](../src/mosaic/losses/protenix.py): `get_trunk_state` | Native input embedding and final differentiable trunk iteration. | Audit feature assumptions and recycling semantics. Existing execution stops gradients through preceding recycles, passes `pair_mask=None`, and still runs one final iteration if given zero recycling steps. |
| [losses/protenix.py](../src/mosaic/losses/protenix.py): `protenix_forward_from_trunk` | Direct access to `model.distogram_head(trunk_state.z)`. | Split out a lightweight graph result. Existing forwarding always samples coordinates and runs confidence, then builds protein backbone/atom37 views. |
| [common.py](../src/mosaic/common.py): `LossTerm`, `LinearCombination` | Scalar loss plus auxiliary metrics and composable loss terms. | Graph losses accept graph state and explicit token indices; they do not interpret the state as a 20-amino-acid sequence. |
| [optimizers.py](../src/mosaic/optimizers.py): `projection_simplex` | Existing row-wise Euclidean probability-simplex projection. | Apply separately to atom rows and upper-triangle bond rows. This function uses host NumPy; projection remains outside the differentiated model forward. |
| [optimizers.py](../src/mosaic/optimizers.py): gradient/trajectory conventions | Evaluate, differentiate, update, report loop structure. | Add a graph driver. `_eval_loss_and_grad` assumes one array and replaces nonfinite values; it cannot establish raw gradient correctness. `simplex_APGM` also associates a pre-update score with a post-update best state. |
| [losses/structure_prediction.py](../src/mosaic/losses/structure_prediction.py): `contact_log_probability` | Distogram-to-contact primitive. | Use explicit ligand/protein indices and the required reduction. Existing `BinderTargetContact` uses a different protein-binder reduction and is not the protocol's contact objective. |
| [cache.py](../src/mosaic/cache.py), [pyproject.toml](../pyproject.toml) | Model cache configuration and JAX/Python environment groups. | Capture resolved Git dependencies and checkpoint hashes; `uv.lock` is ignored and currently absent. |

Keep the protein APIs working. In particular, do not force ligand tokens into `StructurePredictionModel`'s PSSM API or fabricate `full_sequence`, backbone, atom37, PAE, or coordinates to satisfy `StructureModelOutput`.

## 3. V0 representation and module contracts

### 3.1 One shared relaxed graph

Represent the differentiated state as a JAX/Equinox pytree with two floating leaves:

| Field | Shape and meaning | Invariant |
| --- | --- | --- |
| `X` | `[N, 6]`, element probabilities in the fixed order `C, N, O, S, F, Cl`. | Every row is nonnegative and sums to one. |
| `E_upper` | `[N*(N-1)/2, 5]`, bond probabilities in the fixed order `NO_BOND, SINGLE, DOUBLE, TRIPLE, AROMATIC`. | One simplex row per unordered pair, using a deterministic lexicographic `i < j` index list. |

Construct `E[N,N,5]` differentiably from `E_upper`, mirror the off-diagonal rows, and set every diagonal to `[1,0,0,0,0]`. Derive the Protenix-facing bond-existence matrix as `B = 1 - E[..., NO_BOND]`. Differentiate through this construction so each stored pair receives the contribution from both symmetric model entries.

All `N` heavy-atom slots are occupied throughout a trajectory. There is no null element, atom-deletion variable, or optimized padding mask. Formal charge is fixed at zero. Ligand `ref_pos[N,3]` is a fixed zero placeholder. Do not optimize coordinates.

Use probabilities directly, with projection after each update. Do not introduce logits or a softmax parameterization. The proposed Stage 1 size is `N=25`; small hard fixtures can use their own fixed `N` for Stage 0 checks. A shape change is a separate compiled context, not a variable-size trajectory.

**Declared feature loss:** V0 sends total bond existence to Protenix, not bond order. Exchanges among SINGLE/DOUBLE/TRIPLE/AROMATIC at fixed `B` must therefore leave this model objective unchanged. Preserve all five states in the shared graph for later chemistry/prior operators; do not claim Protenix currently supplies bond-order-specific guidance. Ligand stereochemistry, hydrogens, variable charge, and reference geometry are also outside V0.

### 3.2 Prepared context and forward result

Prepare a context once per input/shape outside JIT and outside optimization. It owns native static features, protein inputs, model token/atom mapping, ligand and protein token indices, native masks, geometry-gating metadata, fixed atom names, and source identities. Slot names and entity IDs must be stable and must not encode an initial molecule's mutable atom identity or topology.

Use these conceptual interfaces; names may be adjusted during implementation:

```text
prepare_context(ligand_spec, protein_chains, geometry_mode) -> PreparedContext
inject_graph(context, graph) -> native-shaped differentiable features
forward_trunk(model, context, graph, key) -> MolecularTrunkOutput
objective(graph, output, key) -> (scalar_loss, auxiliary_metrics)
projected_step(graph, gradients, eta_X, eta_E) -> updated_graph
hard_readout(graph) -> raw element/bond assignments and conversion status
```

`MolecularTrunkOutput` contains distogram logits/bins and explicit token mappings. It may expose native trunk/pair state when needed. Coordinates and confidence belong to a separate optional path and must never be silently filled with zeros or reported as available by a trunk-only forward.

Both modes use the same injection and objective machinery. Protein-plus-ligand mode adds native prepared protein context. Do not assume the ligand is the first block of tokens; consume the recorded indices.

## 4. Ordered implementation work

The work numbers describe technical dependencies, not a requirement for one person to complete every item serially. Graph/update mechanics, model scaffolding, and artifact handling can progress while the upstream feature audit is underway. The adapter's native behavior and final integrated validation still depend on that audit.

### Work 0: resolve the upstream feature contract before coding the adapter

The checkout references the JAX implementation through the `protenix` dependency from `escalante-bio/protenij`, but does not contain its source. Audit an immutable resolved revision and the matching native featurizer before asserting support for relaxed ligands.

Produce a feature-consumer table with actual field names, shapes/dtypes, locations in upstream source, and whether each field is fixed, differentiable, removed, or retained. Trace at least:

- Element encoding and any discrete element lookups: how can `X` reproduce native one-hot features and interpolate them without `argmax`, host conversion, or integer indexing on mutable identity?
- Bond-existence features and their pair-embedding path: where can fractional `B` enter before a boolean cast, threshold, sparse edge enumeration, or feature cache would disconnect it?
- Ligand tokenization, token/atom mapping, representative atoms, residue types, atom names, reference-space IDs, entity/chain IDs, masks, charge, and any topology-dependent derived fields.
- Reference positions, relative vectors/distances, reference-space equality gates, geometric encodings, frames, templates, and every geometry-derived contribution in the input/atom/pair path.
- MSA/template handling and the legality of an empty protein context; verify an ordinary ligand-only input actually reaches the trunk.

For each identity/topology-derived field, either update it differentiably from the shared graph or justify its fixed neutral/native convention. A hard starting molecule must not leave a hidden fixed bond list or geometry-derived topology in the relaxed forward.

Prefer a native feature override. If upstream casts or combines channels too early, add the smallest explicit, versioned hook at the native embedding boundary, preserving frozen pretrained weights. Record any dependency patch and its hash. Do not silently bypass the bond path or add a trained surrogate. If no faithful differentiable hook exists, record the concrete blocker before advancing.

**Exit evidence:** pinned checkpoint/dependency identities, feature-consumer table, the exact element/bond injection points, and an explicit geometry-gating design for both modes.

### Work 1: implement graph state and preparation

Implement fixed vocabularies, upper-triangle indexing, reconstruction of `E/B`, shape/value assertions, leafwise simplex projection, and a detached hard readout. Keep metadata/static indices out of the differentiated leaves.

Prepare one ordinary neutral fixture, proposed acetanilide (`CC(=O)Nc1ccccc1`), using the native hard-ligand importer. Check it is represented entirely by the supported vocabulary and neutral charge convention. Preserve importer atom order and explicit-hydrogen policy in the fixture manifest. A different compatible fixture can replace it if native tokenization requires this; record the reason.

Store its one-hot graph and construct an interior relaxed version of the same graph by probability mixing. Use the same slot ordering, `N`, static context, and RNG identity throughout the paired checks. Add synthetic mapping fixtures for elements and bond categories absent from the main hard molecule.

Provide the configuration fields needed by Stage 1 without performing its initialization study: carbon-heavy element probabilities, independent probability-space perturbations, expected degree `d0`, and `p0=d0/(N-1)`. Initialize bond noise only on unordered pairs, then project and mirror. Exact baseline values are recorded when Stage 1 starts.

**Exit evidence:** meaningful contract tests for shape, symmetry, self-bond exclusion, simplex membership, and preservation of fixed occupancy.

### Work 2: implement differentiable injection and ligand geometry-off mode

Inject the element distributions through the audited native element representation and `B` through the audited bond-existence pathway. Preserve native shapes and floating dtypes. Prepare static metadata once; the per-step feature update must consist only of differentiable JAX operations.

Define geometry-off as zero contribution from ligand reference geometry, not merely zero coordinates. Mask or disable geometry-derived channels at the audited consumption points, including any nonzero constants or biases generated from a zero displacement/distance. Ligand-ligand and ligand-protein reference-coordinate relations must not invent a shared origin or encode a reference conformer. Retain legitimate metadata unrelated to geometry and the native protein-only reference contributions.

Preserve ligand atom/token occupancy. Do not set a generic `ref_mask` to zero without proving that it gates geometry independently of atom presence. Separate occupancy masks from reference-geometry contribution masks where necessary. Frame validity must remain truthful; do not manufacture valid ligand frames for a later confidence metric.

Test retained native features and protein feature blocks against the prepared native context. Under geometry-off, changing an otherwise unused ligand reference conformer must not change forward output; changing ligand `X` or `B` must change the corresponding injected features. Check the ordinary protein-only path remains native.

**Exit evidence:** one-hot retained-feature parity, explicit geometry ablations, preserved occupancy, and no discrete/host conversions in the graph-to-model differentiated path.

### Work 3: implement the resident frozen Protenix-Tiny path

Load Tiny once before the trajectory, set the resolved model to its supported evaluation/inference mode without disabling differentiation of graph inputs, and keep it warm. In JAX, differentiate only the graph pytree; the model and prepared context are fixed arguments. No model optimizer, parameter update, or `stop_gradient` on injected `X/B` is permitted.

Run native input embedding, the trunk, and the distogram head only. Reuse `get_trunk_state` if the audit confirms compatible masks/features; otherwise factor the smallest shared helper rather than cloning the entire model forward. Leave existing protein prediction behavior intact.

Start with one supported final trunk iteration (`recycling_steps=1`). Record that reused recycling detaches earlier recycle states and gradients flow through the final iteration. Reject unsupported recycle counts explicitly; do not treat zero as a verified zero-work mode. Use a fixed model key for derivative checks and record the RNG policy for trajectories. Avoid carrying stale graph-dependent embeddings between steps.

A selected off-diagonal native distogram output can supply a scalar for engineering derivative tests in ligand-only mode. Label it `diagnostic_only`; it is not a proposed ligand-design score, confidence metric, or Stage 1 objective. Protein-plus-ligand validation can use the exact native contact reduction described in the handoff below.

**Exit evidence:** both modes return finite native trunk/distogram outputs, gradients reach `X` and bond existence, and the cheap path never invokes structure sampling or confidence.

### Work 4: implement one projected-probability graph loop

Use a direct graph value-and-gradient call with auxiliary metrics. Inspect and log raw values/gradients before any transformation. Stop and retain a failure artifact on NaN/Inf; do not use the existing sanitized evaluator to establish correctness.

For the first driver, use projected gradient descent with separate atom and bond step sizes:

```text
(loss_k, aux_k), gradients_k = value_and_grad(loss)(graph_k, key_k)
assert raw loss/output/gradients are finite
X_next = projection_simplex(X_k - eta_X * grad_X)
E_upper_next = projection_simplex(E_upper_k - eta_E * grad_E_upper)
graph_next = (X_next, E_upper_next)
assert graph invariants
record the evaluated graph_k, its loss/gradients, and the proposed graph_next
```

Reuse Mosaic's projection leafwise, with unit simplex mass, no hardening scale, and no logspace/momentum initially. If optional gradient centering/clipping is later needed, record both raw and applied gradients and all settings. Do not force a common clip threshold or step size across the six-category atom and five-category bond leaves.

Evaluate/checkpoint states with unambiguous iteration labels. A best-state record must contain the actual evaluated state and its score. Evaluate the final post-update state separately if reporting its score. Projection and detached readout happen between forward calls; differentiating through the entire optimizer trajectory is unnecessary.

**Exit evidence:** a short integration trajectory preserves all invariants, saves complete restart state, and resumes consistently. Objective improvement or chemistry validity is not a Stage 0 pass criterion.

### Work 5: add readouts and reproducible artifacts

At configured readout intervals, take `argmax(X)` and `argmax(E_upper)` and reconstruct the same graph. Save the raw arrays before attempting detached RDKit conversion/sanitization. Keep disconnected graphs and conversion failures, with their reason and step. Do not repair, delete fragments, substitute molecules, or feed these readouts back into the next relaxed forward.

Create a unique run directory with a small schema:

```text
runs/<run_id>/
  manifest.json              # source/config/environment/model/input identities
  config.json                # exact resolved settings and vocabulary orders
  trace.jsonl                # scores, diagnostics, iteration/RNG/state references
  checkpoints/step_*.npz     # X, E_upper, optimizer/counter/RNG state
  readouts/step_*.json        # raw hard assignments and conversion outcome
  validation.json            # checks, tolerances, commands, passed/failed gates
  failure.json               # present if the run fails; do not discard the run
```

Retain raw probabilities, enough information to reconstruct full `E`, optimizer state (step sizes/counters and any later buffers), typed JAX RNG implementation/key data, objective values, separate atom/bond gradient norms, projection residuals, entropy/maximum probabilities, bond mass/expected degree, and readout status. Trace entries must distinguish evaluated from updated states. Save initial/final states and failed states as well as scheduled checkpoints.

Record source contents including uncommitted patches, configuration hash, native hard/protein input identities, atom/token ordering, geometry-gating revision, Tiny checkpoint hash, resolved upstream Git revisions, resolved environment, device/driver, precision, recycle settings, objective formula/sign, and RNG policy. Capture a dependency lock/export in run artifacts even though this repository ignores `uv.lock`.

Add `docs/experiment_ledger.md` when implementation begins. Record each future validation/run, including commands, execution environment, source/config/input identities, artifact paths, outcome, and retained failures. This planning document is not a numerical experiment or a passed test. DTU HPC is the proposed numerical execution location, using a compatible Linux/JAX GPU environment; site configuration and access must be established at implementation time. This planning task includes no job submission.

## 5. Proposed file layout

Keep the new branch narrow rather than introducing a general plugin framework:

| Proposed file | Responsibility |
| --- | --- |
| `src/mosaic/small_molecule/__init__.py` | Small public API; optional model imports remain lazy. |
| `src/mosaic/small_molecule/contracts.py` | Shared context/result/step contracts and contract version; no upstream model loading. Graph constants and graph type are imported from `state.py`. |
| `src/mosaic/small_molecule/state.py` | Shared graph contract, vocabularies, upper-triangle mapping, initialization, invariants. |
| `src/mosaic/small_molecule/features.py` | Static native context preparation and pure graph feature injection/geometry gating. |
| `src/mosaic/small_molecule/protenix.py` | Frozen resident Tiny wrapper and trunk-only result. |
| `src/mosaic/small_molecule/losses.py` | Explicit-mask molecular objective adapters using `LossTerm`; engineering diagnostics clearly marked. |
| `src/mosaic/small_molecule/optimizers.py` | Strict graph evaluator and leafwise projected updates, reusing existing simplex projection. |
| `src/mosaic/small_molecule/io.py` | Detached readout, manifest/trace/checkpoint serialization and resume. |
| `examples/small_molecule_stage0.py` | One reproducible entry point for both forward modes and validation, separate from a notebook. |
| `examples/configs/small_molecule_stage0.json` | Explicit model, fixture/context, graph, gradient-check, step-size and artifact settings. |
| `tests/test_small_molecule_state.py` | Representation invariants and pair/element identity. |
| `tests/test_small_molecule_optimizers.py` | Raw gradient handling, separate projected updates, and evaluated-state semantics. |
| `tests/test_small_molecule_features.py` | Native feature mapping, geometry gating, and occupancy/protein preservation. |
| `tests/test_small_molecule_protenix.py` | Separately marked real-checkpoint forward/backward and finite-difference checks. |
| `tests/test_small_molecule_losses.py` | Explicit-index objective reduction, distance-bin convention, and loss sign. |
| `tests/test_small_molecule_io.py` | Raw failed readouts and checkpoint/resume fidelity. |
| `tests/test_small_molecule_integration.py` | Tests of producer/consumer boundaries and the assembled two-mode pipeline, with real-checkpoint tests explicitly marked. |
| `tests/data/small_molecule_stage0/` | One shared fixture manifest, hard graph, relaxed graph recipe, atom ordering, and protein-context identities. |

Adjust names as the upstream audit warrants, but keep state, native adaptation, model execution, and detached readout separate. Reuse scalar loss composition and model/projection helpers without rewriting unrelated protein modules.

## 6. Parallel tracks and integration practices

Four implementation tracks can proceed in parallel after a short shared-contract kickoff. The native feature audit is the main technical dependency: it must resolve real ligand injection and geometry gating, but it need not hold up graph mechanics, artifact handling, or a model wrapper developed against native hard inputs. Numerical work remains subject to the execution arrangement in Work 5; this section schedules future implementation, not runs for the present planning task.

### 6.1 Track ownership and dependencies

| Track | Owned implementation and tests | Can start independently | Integration dependency and handoff |
| --- | --- | --- | --- |
| **T1: graph state and projected updates** | `state.py`, `optimizers.py`, `test_small_molecule_state.py`, `test_small_molecule_optimizers.py`. Covers graph construction, initialization, strict gradients, and separate atom/bond projection. | After graph leaf names, vocabulary/pair ordering, and loss-callable signature are agreed. Use a small deterministic analytic test loss for update mechanics. | Provide the actual graph pytree, raw gradient/step result, and evaluated/updated-state semantics to T2-T4. Real gradient evidence waits for T2/T3. |
| **T2: native feature audit and adaptation** | `features.py`, `test_small_molecule_features.py`, and any audited upstream feature hook/patch. Covers Work 0, native hard preparation, graph injection, geometry gating, and preserved protein inputs. | Immediately audit the pinned dependency and prepare the native fixture. Publish findings incrementally rather than waiting for a complete adapter. | Request context fields through the common contract. Deliver the feature-consumer table, native-shaped feature injection, atom/token mappings, and dependency patch identity to T3/T4. Unsupported native assumptions block affected integration. |
| **T3: resident model and objective wiring** | Small-molecule `protenix.py`, `losses.py`, `test_small_molecule_protenix.py`, `test_small_molecule_losses.py`. Own any necessary factored helper changes to the existing Protenix modules. | Inspect/reuse the native loader and trunk, define the trunk-only result, and test objective indexing/reduction with synthetic logits while T2 audits. | Consume T2's real prepared context/injected features and T1's graph. Deliver the graph loss callable `(graph, key) -> (loss, aux)`, explicit output indices/bins, and real Tiny gradient evidence. Stage 1 confidence selection remains deferred. |
| **T4: artifacts and integration** | `io.py`, `test_small_molecule_io.py`, `test_small_molecule_integration.py`, shared fixture files, entry point/config, and public exports. T4 is the integration owner. | Implement raw readout serialization, manifest/checkpoint schemas, and restart/error handling from the agreed graph/RNG/step contracts. | Assemble T1-T3 into the entry point early. Own integration tests, matching score/state/checkpoint records, source-release identity, and the combined acceptance report. |

T4 also coordinates edits to `contracts.py`, the new pytest marker in `pyproject.toml`, and shared configuration. T1 reviews graph/step fields, T2 reviews prepared-context fields, and T3 reviews model-result/loss fields. The documentation owner maintains documentation and experiment history; the modularity reviewer remains a separate role from the implementers and integration owner.

With four implementers, assign one track each. With three, combine T1 and T4 initially, then shift integration work after the graph contract is available. With two, combine T1+T4 and T2+T3. Keep the review roles distinct in all cases; these combinations describe workload, not permission to omit independent review.

### 6.2 Agree one contract before parallel implementations diverge

Publish a small `small_molecule_v0` contract in code and document its meaning. It need not be a general registry or plugin system. All tracks import the same types/constants and load the same fixtures; no private copies of vocabularies, pair maps, masks, or native-input recipes.

The kickoff must settle the following shared conventions:

| Boundary | Convention to agree | Producer and consumers |
| --- | --- | --- |
| Graph | `X`/`E_upper` leaf names, floating precision, fixed occupancy/charge, vocabulary order, lexicographic unordered-pair order, and symmetry/diagonal construction. | T1 produces; T2-T4 consume. |
| Prepared context | Atom-slot/token mapping, explicit ligand/protein index arrays, immutable base features, geometry mode, occupancy masks, and input identity. Actual upstream field details remain provisional until T2's audit. | T2 produces; T3/T4 consume. |
| Model result | Distogram-logit shape, native bin interpretation, token indices, available fields, and truthful absence of coordinates/confidence. | T3 produces; objective adapters and T4 consume. |
| Loss and gradients | Scalar minimization loss, auxiliary metrics, `diagnostic_only` label, graph-only differentiation with frozen model/prepared context, and separate raw atom/bond gradients. | T3 produces the callable; T1 differentiates; T4 records. |
| Step/RNG | Which state was evaluated, which state is the next iterate, whether a key has already been consumed, and the exact RNG state needed for the next evaluation. | T1 produces; T4 persists/restores. |
| Artifacts | Contract/checkpoint schema version, graph/context/model identities, precision/tolerance settings, and incompatible-checkpoint rejection. | T4 produces; every track supplies its identity/settings. |

Use the hard fixture and interior relaxed fixture already specified in Work 1. T2 establishes the native importer ordering; T4 stores the agreed fixtures/manifest, and T1 verifies their graph representation. Include a synthetic token layout with non-leading or interleaved ligand indices to expose accidental first-`N` slicing. Fixtures must not require remote downloads for ordinary contract-test collection.

Keep context preparation, pure injection, model forwarding, graph updates, and detached serialization as separate boundaries. In particular, the optimizer accepts a graph loss callable rather than importing Protenix, and the adapter must not call the optimizer or readout converter. Injection returns updated features without mutating the prepared base context. Model parameters remain resident/frozen while graph-dependent embeddings are recomputed for each new graph.

If the upstream audit requires a contract change, update the shared type/schema, producer, affected consumers, fixture, and boundary tests in one coordinated change. Record the reason and contract revision. Reject incompatible vocabulary/pair ordering, context, geometry mode, checkpoint identity, or schema during resume instead of guessing or silently remapping arrays.

### 6.3 Common development and merge practices

- Use one branch/isolated worktree per active implementation track when working concurrently. Do not have multiple agents edit the same checkout/files without an explicit ownership handoff. Start from a common integration baseline and keep changes small enough to review.
- Each track owns its table-listed files. T4 serializes shared exports/configuration/marker/contract changes; the documentation owner serializes documentation/ledger changes. Assign a single owner to upstream dependency patches. This avoids independent feature-hook variants and conflicting environment edits.
- Use the same resolved Python/JAX/dependency/checkpoint baseline and shared fixture manifest. Dependency changes are a coordinated decision with source/environment hashes, not track-local upgrades.
- Keep imports lazy and model loading explicit. Ordinary contract tests must not download checkpoints or initialize every model stack. Typed signatures, inline shape/mask/gradient comments, scoped Ruff checks, and boundary assertions apply to new code.
- A track handoff includes its source diff/revision, contract revision, fixture IDs, relevant selected test commands/results, failures, and unresolved native assumptions. A mocked result must be labeled as such; a missing real-model check stays pending.
- Merge a thin compatible slice early, then expand it. Run the affected boundary tests on the merged source, since passing tests on separate branches do not establish that their combination works. Re-test when an interface, environment, fixture, model path, or relevant implementation changes; avoid unrelated full-model panels.
- Keep raw failures and evidence in the ledger. The final native acceptance run uses a single immutable combined source release, including uncommitted contents and dependency patches, with a unique run directory. Evidence from an older branch/release cannot validate later merged code.

### 6.4 Tests that make the tracks work together

Module tests establish local behavior. The following integration tests establish compatibility across the boundaries; they supplement the acceptance gates in section 7.

| Test | Tracks joined | What it must catch |
| --- | --- | --- |
| Shared graph -> injection | T1 + T2 | One-hot retained-feature parity, upper-triangle pair identity, floating element/bond injection, fixed occupancy, and unchanged native protein blocks. |
| Explicit indices -> objective | T2 + T3 | Correct contact reduction/sign/bin convention with non-leading/interleaved ligand and protein indices; no assumptions about first-`N` token layout. |
| Graph -> real Tiny -> raw derivatives | T1 + T2 + T3 | Finite-difference atom/bond-existence tangent derivatives, both symmetric entries contributing to a stored edge, and null bond-order directions at fixed existence. A test double cannot prove this native path. |
| Graph A -> graph B -> graph A | T1 + T2 + T3 | A changes to B when identity/existence changes, A reproduces on the second call with the same key, and base features/protein blocks remain unchanged. Detects in-place mutation and stale graph-dependent caches. |
| Geometry-off conformer substitution | T2 + T3 | Changed unused ligand reference coordinates/conformer do not change outputs, occupancy stays present, and native protein reference channels remain active. |
| Evaluator -> projected step -> trace | T1 + T4 | Separate step sizes, simplex invariants, raw nonfinite rejection with a retained failure record, and loss/gradient records tied to the exact evaluated state. |
| Readout -> continued relaxed forward | T1 + T3 + T4 | Raw argmax/conversion, including a failed conversion, does not alter the next `X/E_upper` or introduce a repair into the differentiated path. |
| Checkpoint -> fresh-process resume | T1-T4 | Uninterrupted and resumed continuations match within declared precision tolerances, including the next RNG key, iteration labels, and context/model identities; incompatible schemas/identities fail explicitly. |
| Full two-mode execution | T1-T4 | Ligand-only and protein-plus-ligand graph -> features -> resident Tiny -> scalar -> raw gradients -> projected update -> checkpoint/reload. Verify one model load and no diffusion/confidence in the cheap path. |

Small deterministic test doubles and analytic losses are useful for interface, projection, and serialization tests. Use deliberately permuted indices and deliberate failure inputs rather than tests that merely repeat an implementation's formulas. The final derivative/forward/restart evidence must exercise the real resolved Tiny checkpoint on the assembled code. A model spy can prove calls were excluded or loading was resident; it cannot establish biological/native gradient fidelity.

The proposed commands below illustrate the selected test split after the files and marker exist. Execute numerical tests in the agreed execution environment, record the exact final commands, and do not treat these examples as results:

```bash
# Selected unit/boundary tests, with real-checkpoint tests excluded.
uv run pytest tests/test_small_molecule_*.py -o addopts= --strict-markers -m "not slow and not small_molecule_model"

# Actual Tiny checks; override the repository's default exclusion of slow tests.
uv run pytest tests/test_small_molecule_*.py -o addopts= --strict-markers -m small_molecule_model
```

Register `small_molecule_model` centrally before these commands are used. Parameterize the appropriate integration cases over both forward modes. Confirm the selected native tests actually execute: skipped, deselected, or expected-failure cases remain pending acceptance evidence. Set tolerances and RNG policy before judging outcomes; retain test output and source/input identities. No CI service or HPC submission is introduced by this documentation update.

### 6.5 Integration milestones and stopping rule

| Milestone | Parallel progress | Required combined evidence |
| --- | --- | --- |
| **M0: shared-contract kickoff** | Agree graph/callable/output/artifact boundaries and file ownership. T2 starts the upstream audit; T1/T3/T4 can build independent mechanics/scaffolding. | One contract/fixture plan and pinned dependency baseline. Native feature details are still visibly pending. |
| **M1: track contracts usable** | T1 supplies real state/updates; T2 resolves the native feature table/hook; T3 supplies native hard-input trunk output; T4 supplies serialization and shared fixtures. | Producer/consumer contract tests pass on a merged slice, and native audit blockers are resolved for the next milestone. |
| **M2: early complete pipeline** | T4 assembles T1-T3 before each track is polished; fix integration failures at their owning boundary. | Both modes complete one real graph -> objective -> raw backward -> projected step -> checkpoint/reload using the same source/context contract. |
| **M3: Stage 0 acceptance** | Add only the remaining derivative, regression, short-trajectory and resume checks. | Gates A-G pass on the same immutable combined release, with ledger/artifact evidence and declared limitations. Then hand off immediately to Stage 1. |

The likely critical path is T2's audited native injection/geometry path -> T3's real graph gradients -> T4's combined validation. T1 and artifact mechanics should be ready by the time that path is available. A track marked complete or a set of independently passing unit suites does not close Stage 0; the combined M3 evidence does.

## 7. Validation gates and acceptance evidence

These are planned checks, not completed results. Use a small fixture first; avoid multi-target or multi-seed scientific panels. Unit/contract checks establish mechanics, while real-checkpoint checks establish the actual gradient path. A mock alone cannot pass Stage 0.

| Gate | Check | Required evidence |
| --- | --- | --- |
| A. Representation | Synthetic graph projection/reconstruction, fixed occupancy, deterministic pair indexing. | Simplex rows sum to one within the declared precision tolerance; symmetry and diagonal NO_BOND hold exactly by construction; no variable charge/size. |
| B. Native hard sanity | Run the ordinary hard fixture using the resolved native Protenix ligand path. | Finite trunk/distogram outputs; explicit atom/token order and input identities. Native conformer input is a reference case, not the relaxed forward's geometry. |
| C. Geometry-off hard parity | Run the same one-hot graph through the graph adapter with geometry disabled. | Retained element/bond/protein features agree with the corresponding native feature contract. Any output difference from the ordinary conformer case is documented as an intentional geometry ablation. |
| D. Relaxed forward/backward | Run an interior relaxed version in ligand-only and protein-plus-ligand modes. | Finite raw outputs/loss/gradients, nonzero atom-identity and bond-existence tangent gradients, correct shape/masks, and no model parameter updates. |
| E. Perturbations and derivatives | Exchange probability mass within selected atom and bond simplex rows. | Bond-existence perturbations change the native model output/objective; representative autodiff directional derivatives agree with finite differences; bond-order exchanges at fixed existence are invariant. |
| F. Projected trajectory and restart | Run about 5-10 engineering steps, save, then compare uninterrupted with resumed continuation. | Finite raw quantities and graph invariants at each step; complete artifacts and matching replay within declared environment/precision tolerance. No chemical-validity claim. |
| G. Protein regression and cheap execution | Exercise retained protein API and inspect/model-spy the trunk-only path. | Native protein behavior preserved and no diffusion/confidence call in the default graph path. |

For finite differences, use directions that preserve row sum, such as moving mass from C to N or from NO_BOND to SINGLE in one stored unordered pair. Use an interior relaxed state, the same RNG key, identical context, and no projection/argmax/repair inside the compared forward calls. Check that both perturbations remain nonnegative. Compare

`(L(graph + epsilon*d) - L(graph - epsilon*d)) / (2*epsilon)`

against the dot product of the raw autodiff gradient and `d`. Use a few representative directions and an epsilon sweep (for example `1e-3`, `1e-4`, `1e-5` in float32), recording absolute/relative errors and a sensible stable range. Declare tolerances in configuration before judging a run; do not demand nonzero gradients for bond-order directions the existence-only model cannot observe. Avoid nondifferentiable max ties for contact-objective derivative checks.

Mark real-checkpoint tests explicitly, for example `slow` plus a new `small_molecule_model` marker registered in `pyproject.toml`, and select them directly. Keep imports lazy so unit collection does not load/download every model. Future commands should name the graph test files/markers; a repository-wide model test run is unnecessary for this Stage 0 change. Match Python 3.12 and one JAX runtime group to the execution environment.

## 8. Stage 1 handoff and stopping rule

Stage 0 supplies the context/model hooks and artifacts needed to perform the protocol's one quick objective-selection gate. It does not make PAE validity a requirement for completing the cheap graph interface.

At Stage 1 entry, try native mean intra-ligand PAE exactly once as prescribed: compare the ordinary hard conformer path with geometry-off graph input; require finite values, truthful valid ligand frames, and stable nonzero gradients to `X` and bond existence. This needs an explicit optional native structure/confidence path. Do not inherit the current protein-only backbone extraction; audit the native ligand confidence/frame path and return only real supported fields. If any prerequisite fails, retain the failure and use the specified target-contact fallback without modifying the confidence head or inventing a ligand-only proxy.

The PAE loss is the mean of native `PAE_ij` over off-diagonal ligand-ligand token pairs with valid frames. Include both directed pairs `i -> j` and `j -> i`, since PAE is asymmetric, and minimize this loss. An empty valid-pair set fails the gate; it must not produce a zero or fabricated confidence value. Do not substitute ligand pLDDT for this ligand-only objective.

For that fallback, prepare complete BRD4-BD1 from 3MXF with JQ1 removed; do not define a pocket. With explicit ligand-token set `L` and target-token set `T`, implement:

```text
c_ij = sum softmax(distogram_logits_ij)[native distance bins < 8 angstrom]
S_contact = mean over i in L of max over j in T of c_ij
loss = -S_contact
```

The resolved bin convention is now statically audited in [the native contract](stage_0_native_contract.md): 64 labels have 63 native thresholds and a final infinite upper edge. The graph loss sums complete bins with upper edges strictly below 8 Å, excluding the straddling class; this is a declared conservative whole-bin approximation rather than an exact continuous-distance CDF. The existing sequence-binder contact helper is not silently substituted. This target-interaction debugging score does not establish affinity/binding; its model/gradient behavior remains pending. The ligand-only Stage 0 diagnostic scalar must not become a replacement scientific objective.

Stop Stage 0 as soon as gates A-G pass for reproducible ligand-only and protein-plus-ligand forward/backward execution. Hand over the source release, configuration, feature/geometry contract, validation evidence, restartable example, representation limitations, and experiment ledger. Move to Stage 1's one canonical unconstrained trajectory; chemical constraints and learned molecular priors remain later stages.

## 9. Review and documentation responsibilities

Keep a separate reviewer for modularity/reuse and a separate documentation owner during implementation, alongside T4's integration owner. The reviewer checks that protein APIs remain intact, shared graph identity is preserved, native helper reuse is justified, and adapter/readout boundaries are explicit. Implementers write inline shape/mask/gradient comments, with the documentation owner reviewing them and maintaining the common contract explanation, native feature-consumer table, example commands, limitations, contract-change history, and experiment ledger. T4 links the combined acceptance evidence rather than independently approving its own implementation.

On completion, replace proposed gate statuses with evidence links and report remaining unsupported modes. Until then, preserve the distinction between design decisions, upstream assumptions requiring an audit, and actual numerical results.

### 2026-10-16 known-ligand experiment outcome

The scoped [real-target 1000-update study](known_ligand_1000.md) completed on
allocated DTU HPC: 146 selected unit tests, three native pilots, and 1000
accepted updates each for 15%, 30% and 50% uniform starts. It uses the full
T4 lysozyme L99A sequence and a fixed benzene-derived contact pocket, with
six occupied ligand slots. Separate matched hard-input controls and the
interactive trajectory report also completed. The ledger retains immutable
source/input identities, storage-gate failures, bounded recovery and receipts.
The final evidence archive and 15,775 declared files were downloaded and
verified locally by byte hashes; final report browser QA passed.

Per the user's explicit instruction, the report depicts the raw categorical
atom/bond outputs even when chemically invalid. Probability-dependent sizes
and explicit element/slot labels preserve what the optimizer produced;
sanitization only adds an annotation. No repair, filtering or alternative
molecular identity is substituted. This experiment does not close the full
A–G matrix, both-mode/restart validation or milestone M3; those outstanding
acceptance requirements remain as specified above.

### 2026-10-16 requested BRD4 follow-up

The user requested a fresh run at three times the starting learning rate
and 5000 updates using BRD4-BD from the project description. The sealed
[BRD4 protocol](brd4_5000.md) records the pivotal choices: full deposited
3MXF BRD4-BD1 sequence with all residues eligible, the existing whole-target
native contact reduction from section 4.1.2, initial step 0.03 with unchanged
backtracking, and the same six-carbon/three-mixture starts. This is a changed
target and objective relative to T4's pocket NLL. JQ1 is excluded rather than
neutralized or scored as a supported reference. New medium-size artifacts
use the user's separately gated home quota to avoid tight work3 headroom;
model caches and immutable source remain on work3. No existing evidence is
removed. CPU/pilot/resource gates precede the long run, and actual outcomes
belong in the ledger. This scoped follow-up does not close outstanding
PAE, canonical diffuse-start or Stage 0/1 acceptance gates.

### 2026-10-16 requested completely uniform control

The user requested a separate otherwise identical 0.03/5000 BRD4 run with
equal atom probabilities and pair existence near `3/N`. The
[uniform-control protocol](brd4_uniform_5000.md) records six fixed slots,
each element at 1/6, pair existence 0.5, NO_BOND 0.5 and four bonded
categories each 0.125, with zero noise. Run one control, since three exact
copies at the same key would duplicate the computation. Preserve the
original jobs. Native slot/name metadata remains fixed to the same prepared
context, so uniform graph probabilities do not assert a permutation-symmetric
model. Use the existing shared initializer, explicit initialization-replay
authorization, CPU tests and aggregate quota accounting for both studies.
Every new uniform graph still passes native derivative/null/parity gates
at GPU entry before accepting updates. No gate is waived, no jitter added,
and raw invalid/disconnected readouts remain visible.

## 2026-10-16 noisy-uniform initialization control

The user requested a third BRD4 0.03/5000-update study. See
[the noisy-uniform protocol](brd4_noisy_uniform_5000.md). The recorded pivotal
decision is `0.5 * uniform + 0.5 * simplex(N(0,1))` for both atoms and unordered
bonds. Projecting the random component FIRST preserves the positive uniform
floor needed by the unchanged central derivative/null probes. This can change
the baseline bond density and does not imply chemical validity. The two
earlier immutable studies remain unchanged.

Completed preparation: sealed source 01, allocated CPU readiness 29736756
(167 passed), unchanged-library/config/reference and aggregate-quota gates.
Submitted noisy native 29736761 and dependent report 29736762; both pending.
Actual noisy native gates, 5000 updates and final report/results remain open;
submission does not close any scientific acceptance gate.


## 2026-10-16 combined BRD4 comparison report

User requested one report containing all three studies/five trajectories.
[The combined-report protocol](brd4_combined_report.md) records the explicit
initialization-only comparison policy, matched frozen/shared inputs and
per-study source receipts. Allocated CPU readiness 29736903 passed all 13
checks, including synthetic five-case rendering. Combined report/evidence
29736911 is queued behind all three native jobs; no native run or individual
report was changed. An hourly completion follow-up will download, verify and
inspect the actual combined report, then deliver it and stop. All native
trajectories and actual combined-report completion remain pending.


## 2026-10-16 section 4.2 initialization, user-kept six slots

The user requested section 4.2, then explicitly instructed keeping N=6.
[The canonical diffuse protocol](brd4_canonical_diffuse_5000.md) records the
pivotal choices: C=0.75 (others 0.05), d0=2.25, p0=d0/(N-1)=0.45, independent
sigmaX=sigmaE=0.005 noise added BEFORE projection, upper-pair symmetry and
fixed neutral occupancy/geometry-off context. This exposes the existing shared
initializer to the actual experiment rather than introducing another graph
representation. Keep all native settings/rate/5000-update checks unchanged.
Readiness requires CPU tests, exact frozen/source/config bindings and aggregate
quota for four studies plus a combined report. Native acceptance/trajectory
remain pending; a queued job does not close Stage 1 or other acceptance gates.


Completed preparation: canonical source 01 sealed; CPU 29736979 passed all
186 selected tests and source/config/reference/aggregate-quota gates. Native
29736988 and final four-study/six-case report 29736989 are queued. Superseded
pending report 29736911 was intentionally cancelled and confirmed EXIT;
existing GPU runs and individual reports are unchanged. The existing hourly
completion follow-up now targets report-02. Actual native acceptance, 5000
updates and final report/download remain pending.
