# Stage 0 native Protenix contract

Audit/implementation started: 2026-10-12; documentation update: 2026-10-13. Status: **native implementation authored, merged and independently reviewed; numerical fidelity/acceptance pending**. No model run, preprocessing, local numerical test, HPC connection, upload or submission is performed for this implementation turn. Source inspection, AST parsing and lint are static evidence only.

This contract implements the Mosaic small-molecule scope in [the plan](stage_0_implementation_plan.md) and [project description](project_desc.pdf). It does not adopt the separate DBMol AE/VAE scope. The historical native protein smoke is preserved in [its receipt](hpc_protenix_tiny.md); it does not establish ligand-graph gradients or geometry-off fidelity.

## Source and artifact identity

| Binding | Required identity |
| --- | --- |
| Native JAX implementation | `escalante-bio/protenij`, revision `97cf8172f61829681582f06de7052485bc94eae4` |
| Audited archive | 434,462 bytes; SHA-256 `ea7abd10bd9ad2ce1bff0f9bde76969a298d511051fa200fe47cfb5436e43711` |
| Native `protenix/protenij.py` SHA-256 | `37f337839034afd2a814c1ed6b76867b67918f73dd56cdd365e67b8fed882a6d` |
| Tiny model | `protenix_tiny_default_v0.5.0` |
| Asset repository/revision | `nickrb/protenij` / `81a257b61a82a17c41a7c20638460c04c29d97ad` |
| Shared graph/context contract | `small_molecule_v0.1`; checkpoint schema version `1` |

The local audited checkout is `C:/Users/nikol/.codex/dependency_audits/mosaic-stage0-20261012/protenij-97cf8172f61829681582f06de7052485bc94eae4`. It is source material, not a project runtime environment. Runtime guards must authenticate the consumed native files, not accept a matching package name/version alone. The feature/parser/tokenizer/CCD/MSA/padding/backend/configuration consumers are included in the guard design. No upstream training or pretrained-weight modification is introduced.

The five native assets are the CCD CIF, its RDKit molecule pickle, cluster data, Tiny `.eqx`, and Tiny skeleton pickle. Their identities are already recorded in the historical [asset receipt](../results/hpc/20261010-protenix-tiny/20261010-native-tiny-01/asset_receipt.json). The graph loader must verify those exact assets before loading, with all network fallback disabled. Model `.eqx` SHA-256 is `19f5f7af00cda104e8b059289e7e641cecc03309d7e960e6e53d82ad6e3f1f57`.

## One graph and immutable prepared context

Only `RelaxedGraph.X[N,6]` and `E_upper[N*(N-1)/2,5]` are differentiated. Element order is `C,N,O,S,F,Cl`; bond order is `NO_BOND,SINGLE,DOUBLE,TRIPLE,AROMATIC`; pairs are lexicographic `i<j`. `E` is mirrored and its diagonal fixed to NO_BOND. Native bond input is `B=1-E[...,NO_BOND]`. All slots are present; charge is zero. No logits, atom deletion, padding optimization, hydrogens, variable charge, stereochemistry, reference coordinates or bond-order-specific native guidance are added.

The shared `PreparedContext` holds immutable base features, explicit ligand atom/token indices, protein token indices, geometry mask, slot names and identity bindings. Every ligand slot is one heavy atom and one native ligand token; neither ligand-first token order nor contiguous atom indices are assumed. Protein features remain native. Graph-dependent features/embeddings are recomputed each forward; neither a prepared hard bond list nor a previous graph's embeddings become the mutable graph's hidden topology.

Native hard preparation occurs once outside differentiation. Native SMILES import retains existence but loses bond order, so a paired detached RDKit molecule supplies the five-state hard graph, checked against native heavy-atom ordering and edge existence. Atom-order/topology disagreement is an error, not a silent remapping. Formal charges, explicit hydrogen slots and unsupported elements/modes are rejected.

Native conformer generation can be random. The prepared context seals and serializes actual native feature bytes and identities, with `Featurizer(ref_pos_augment=False)` disabling random reference-position rotation/translation. This is not a promise that importing the same SMILES again reproduces identical coordinates. Resume must load the same prepared context rather than regenerate it.

Ligand names become fixed generic slot names `Q000`, `Q001`, ... rather than names such as `C1` that disclose the initial element. This is an intentional retained-feature parity exception; it prevents a fixed atom-name encoding from competing with mutable `X`. Original native-hard features are kept separately for the explicitly labelled conformer baseline, never used as an optimization cache.

## Audited feature consumers

Paths/line numbers below refer to the pinned upstream revision. They describe static source findings; numerical agreement remains a later gate.

| Native field | Shape/type at boundary | Audited consumer | V0 treatment and reason |
| --- | --- | --- | --- |
| `ref_element` | `[N_atom,128]`, floating injected probabilities | `protenix/protenij.py`, atom encoder identity concatenation/`linear_no_bias_f`, approximately lines 1278-1292; encoder ordering in `data/featurizer.py` | Map X into atomic-number-minus-one columns: C=5, N=6, O=7, S=15, F=8, Cl=16. No integer selection/argmax on mutable identity; the pretrained linear path receives probabilities. |
| `token_bonds` | `[N_token,N_token]`, floating | `protenix/protenij.py`, `embed_inputs`/`linear_no_bias_token_bond`, approximately lines 2209-2212; native generation `data/featurizer.py:459-494` | Replace the ligand-ligand block with symmetric B; retain native protein blocks. This native path has no boolean cast. All present bond categories collapse to existence. |
| `ref_charge` | `[N_atom]`, native integer or floating constant | Atom conditioning applies `arcsinh` then its pretrained charge linear map | Set ligand charge to fixed zero; retain protein/native constants. No charge gradient. |
| `ref_mask` | `[N_atom]`, native occupancy/reference identity gate | `protenix/protenij.py:1293` multiplies **all** atom conditioning, including identity | Preserve occupied ligand mask=1. Setting it to zero would remove element/name/charge conditioning as well as geometry. Geometry uses a separate mask. |
| `ref_atom_name_chars` | `[N_atom,4,64]`, fixed encoded names | Native atom conditioning concatenation; `data/featurizer.py` name encoding | Q-prefixed slot names prevent initial-element leakage. Protein names remain native. |
| `ref_pos` | `[N_atom,3]`, fixed input | Native atom single and local-pair geometry consumers `protenix/protenij.py:1272-1274,1300-1334` | Ligand values zero; geometry contributions also gated at every consumer. Preserve native protein values. |
| `ref_space_uid` | `[N_atom]`, static integer | Native local-pair equality/validity channel | Keep static native IDs for token/atom bookkeeping, but gate their geometry contribution for every pair involving a ligand atom. |
| `has_frame`, `frame_atom_index` | `[N_token]`, `[N_token,3]` | Native confidence-frame preparation in `data/featurizer.py:203-234` | Ligand frames truthfully invalid: `has_frame=0`, indices `-1`. The trunk-only output provides no confidence fields. |
| `atom_to_token_idx`, `atom_to_tokatom_idx`, representative indices | Fixed native index arrays | Native atom-to-token aggregation/padding, relative/token features | One native heavy atom per ligand token; `atom_to_tokatom_idx=0` for ligand. Explicit maps drive all slicing and checkpoint identity. |
| `restype`, residue/token/asym/entity/sym IDs | Native static token features | Native token input/relative-position encoding | Native ligand UNK convention and one static entity are retained. Do not encode mutable atom identity/topology in static IDs. |
| `msa`, `profile`, deletion fields | Query-only native-shaped arrays | Native MSA/trunk; `data/msa_featurizer.py:849-883` | Query-only completion uses restype/profile and zero deletions as native nonprotein tokenization does; ligand-only requires no invented protein sequence/MSA. |
| Molecule/permutation metadata | Detached preparation metadata | Excluded from the audited model-consumer path | Preserve ordering identities needed for records; do not feed a stale hard topology or discrete repair into differentiation. |

## Geometry-off hook and reuse decision

Simply setting coordinates to zero is insufficient. Native local pairs still receive `1/(1+||d||²)` and reference-space equality conditioning, so zero positions can say “all atoms coincide” rather than “no geometry”. Native `ref_mask` cannot solve this because it also gates element identity.

The adapter wraps only the frozen input atom encoder, leaving native weights unchanged and preserving identity/charge/name conditioning, atom attention, aggregation and protein behavior. A separate atom geometry mask is 0 for ligand and 1 for protein. Gate the single reference-position term and all three pair contributions—displacement, inverse squared distance and reference-space validity—before they are combined with identity-conditioned terms. Any pair involving a ligand atom has zero geometry contribution; protein-protein native geometry remains active. Occupancy/padding masks remain separate and unchanged.

Reuse Mosaic's existing `get_trunk_state` on that model copy instead of duplicating the full model/trunk. The graph branch supports exactly one final recycle and rejects zero or other counts. Only the graph is differentiated; model/context are fixed. Trunk and distogram are evaluated without diffusion/confidence calls. The original resident native model is retained for hard-input comparison; no existing protein API is replaced.

Conformer invariance, unchanged protein blocks, resident loading, no expensive-head calls, atom/bond tangent derivatives, and A→B→A cache/context immutability are authored validation requirements, not observed outcomes.

The prepared context hashes actual dtype-normalized JAX native/base feature arrays and retains the importer's original pre-cast hash separately. Serialized context files add independent byte-level identities for fresh-process restoration.

## Distogram meaning and loss boundaries

Native distogram labels have 64 classes from 63 thresholds `linspace(2.3125,21.6875,63)`: see `protenix/configs/configs_base.py:374-377` and `protenix/model/loss.py` near line 537. The graph result stores upper class edges as those thresholds plus `+inf`; it does not treat Mosaic's existing 64-point linspace as native bin centers.

For a strict cutoff of 8 Å, sum only classes whose upper edge is `<8`. The class straddling 8 Å is excluded. This is a declared conservative whole-bin contact score, not exact probability that a continuous distance is below 8 Å. Preserve this convention and the raw bins in artifacts.

Protein-plus-ligand loss uses explicit sets L/T and minimizes `-mean_i∈L max_j∈T(c_ij)`. Ligand-only engineering checks use a selected off-diagonal native distogram logit labelled `diagnostic_only`; it is not a new scientific confidence objective. Exchanges among SINGLE/DOUBLE/TRIPLE/AROMATIC at fixed NO_BOND probability are null directions for the existence-only native objective.

`MolecularTrunkOutput` contains real distogram logits/bins and maps, optionally trunk state. It has no fabricated PAE, pLDDT, coordinates, sequence, backbone or atom37 fields. The Stage 1 ligand-PAE gate is deferred: geometry-off ligand frames are invalid and the graph wrapper does not promise a supported confidence path. An explicit unsupported boundary must trigger the documented fallback decision, not a zero/NaN-sanitized score or a modified confidence head.

## Acceptance and implementation handoff

T1 graph/update code is committed as `8cb6f6c`; T2/T3 native code is merged as `4168f0f`/`3470f17`. T4 context/artifact/driver wiring and the combined validation/tests are authored in main. Independent review covered merged boundaries; all numerical tests remain unexecuted. The native public boundaries are:

| API | Boundary |
| --- | --- |
| `prepare_context(smiles, protein_chains, *, cache_dir, input_name, identities)` | Detached native preparation; returns the immutable context and matching hard graph. |
| `inject_graph(context, graph)` | Pure JAX identity/existence injection into native-shaped features. |
| `bootstrap_native_assets(cache_dir, include_model)` | Offline pinned-asset authentication and native path binding. |
| `verify_native_source(package_dir)` / `with_geometry_hook(...)` | Authenticate consumed source and install the frozen input-encoder geometry adapter. |
| `FrozenProtenixTiny.load(cache_dir=...)` | One resident pinned native model plus geometry-gated copy, sharing frozen weights. |
| `forward_trunk(context=..., graph=..., key=..., recycling_steps=1)` | Relaxed graph trunk/distogram output. |
| `forward_native_trunk(context=..., key=..., recycling_steps=1)` | Separately labelled hard/native conformer reference. |
| `build_loss(context=..., loss=..., recycling_steps=1)` | Return `GraphProtenixLoss(graph, key=...)`; model/context are fixed. |
| `NativeContactLoss` / `DistogramDiagnosticLoss` | Explicit-index native output reductions with declared scientific/diagnostic boundary. |
| `confidence_output` / `ligand_pae_gate_prerequisites` | Explicit unsupported confidence/geometry-off frame boundary; no fabricated score. |

The [usage guide](stage_0_usage.md) records actual entry-point/config, resume, test and future allocation commands. Context bytes/hashes, all resolved settings, source/environment/device identities and typed RNG implementation/next unused key are bound to checkpoints. Each checkpoint filename includes step and evaluation; a commit pointer selects only a fully persisted checkpoint/trace prefix. Final post-update state is evaluated separately, so a best/final score belongs to its recorded state. Readout conversion cannot replace graph state. Raw nonfinite loss, outputs, gradients or proposals retain failure evidence and stop before projection; detached diagnostic norms use float64 to avoid float32 overflow.

The independent reviewer prompted fixes for floating element injection into native integer arrays, nested LossTerm auxiliary composition, raw overflow and dtype rejection, checkpoint atomicity and complete resume bindings. New test cases are authored for those boundaries. Selected static AST/Ruff checks are reported in the ledger; final combined static/package receipts are attached separately. The reviewer and documentation owner do not substitute for numerical validation.

All gates A-G remain pending. Contract/unit tests, real-Tiny forward/backward checks, derivative checks, geometry ablations, protein regression, short trajectories and fresh-process resume must be executed later on allocated DTU HPC with exact source/dependency/context/checkpoint identities. No current implementation or static finding establishes chemical validity, binding, a usable scientific optimization objective, or completed M3 acceptance.
