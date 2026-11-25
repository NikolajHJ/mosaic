# Chemical-prior alternatives: gradient interfaces and schema limits

Research comparison, 28 September 2026. The facts below were checked against
official repositories/configurations. The prioritization is a project-specific
recommendation, **not an implemented result or a new job submission**. No public
checkpoint below is established as compatible with our full element, formal-charge,
aromatic-bond and NO_ATOM schema.

**Superseding user direction, 28 September 2026:** do not train a new in-house
chemistry prior or scorer. Use frozen external pretrained checkpoints with
swappable model-input adapters; small supervised input bridges/decoders are
allowed only to align representations of the **same shared canonical molecule**.
The earlier exact-schema energy-critic proposal below is **rejected under this
preference and retained only as historical discussion**, not an active fallback.
See the proposed [shared-graph plugin contract](23_frozen_model_plugin_contract.md).

## Verified capabilities and remaining gaps

| Candidate | Verified public evidence | Implication for this project |
|---|---|---|
| **GraphEBM (DIG)** | Dequantized node/adjacency tensors feed an explicit scalar energy; native Langevin updates differentiate that energy with respect to both tensors. | Useful gradient-interface reference, but no exact-schema pretrained weights are established. The earlier proposal to train our own version is rejected under the current preference. |
| **GDSS, ZINC250k** | Official pretrained score networks operate on continuous node/adjacency tensors. Configuration uses at most 38 nodes and nine element features; preprocessing has no formal-charge or explicit aromatic channel, and occupancy uses hard node flags. | Only a restricted pretrained comparator if its limitations are acceptable. Full-schema retraining is outside current scope; a soft forward does not differentiate hard occupancy. |
| **MoFlow, ZINC250k** | Official pretrained models are linked; the molecular graph flow has differentiable forward and `log_prob` paths. Its ZINC transform uses 38 slots, element/virtual-node encoding and single/double/triple/no-bond categories. | A continuous-density comparator with charge/aromatic/schema gaps. Respect its dequantization distribution; reverse discretization/argmax is a separate barrier. No direct full-schema checkpoint is verified. |
| **CoCoGraph** | Official models use constrained discrete swaps at fixed molecular formula. The parser retains formal charge, kekulizes aromatic bonds and uses parallel edges for bond multiplicity; model inputs include hard graph features. | A possible external hard-sample comparator, not a direct relaxed-graph gradient prior. A scalar time head does not bridge RDKit/NetworkX features or discrete edge indices. |
| **JODO, ZINC250k 2D** | Official 2D checkpoints are linked. Configuration enables formal charges, disables explicit aromatic categories, uses nine atom types/four bonds and at most 38 nodes. | Best initial pretrained comparator to audit for charge support; exact charge encoding, element mapping, kekulization, checkpoint configuration and gradients still need verification. |
| **JODO, MOSES 2D** | Configuration enables aromatic categories but disables formal charges; seven atom types/five bonds, at most 27 nodes. | Does not resolve the charge mismatch merely by providing aromatic bonds. |
| **DiGress** | Official repository links a recovered GuacaMol checkpoint and explicitly says it has not been tested by the author. | A discrete-graph alternative with substantial checkpoint/schema verification still required. A link is not runtime or compatibility evidence. |
| **MiDi** | Joint graph/3D model; official repository reports losing access to some checkpoints, while listing surviving links. | Adds geometry and coordinate semantics beyond this topology-only pipeline; checkpoint availability and exact graph/charge vocabulary require an audit. |
| **DisCo** | Official discrete-state, continuous-time graph-diffusion implementation includes molecular training/benchmark paths and DiGress-derived components. | Useful architectural comparator; no exact-schema pretrained checkpoint has been verified here. A new sampler does not itself remove representation mismatch. |

Sources: [JODO checkpoint listing](https://github.com/GRAPH-0/JODO#checkpoint),
[JODO ZINC configuration](https://raw.githubusercontent.com/GRAPH-0/JODO/main/configs/vpsde_zinc_2d_jodo.py),
[JODO MOSES configuration](https://raw.githubusercontent.com/GRAPH-0/JODO/main/configs/vpsde_moses_2d_jodo.py),
[DiGress checkpoints](https://github.com/cvignac/DiGress#checkpoints),
[MiDi repository](https://github.com/cvignac/MiDi),
[DisCo repository](https://github.com/pricexu/DisCo).
These are current upstream references; any experiment must freeze commits,
checkpoint/configuration hashes and exact preprocessing rather than follow
moving branches.

Additional primary implementation sources:
[GraphEBM energy and Langevin implementation in DIG](https://diveintographs.readthedocs.io/en/latest/_modules/dig/ggraph/method/GraphEBM/graphebm.html),
[GraphEBM paper](https://arxiv.org/abs/2102.00546),
[GDSS official checkpoints](https://github.com/harryjo97/GDSS#pretrained-checkpoints),
[GDSS ZINC configuration](https://raw.githubusercontent.com/harryjo97/GDSS/master/config/zinc250k.yaml),
[GDSS molecular transform](https://raw.githubusercontent.com/harryjo97/GDSS/master/utils/data_loader_mol.py),
[GDSS node flags](https://raw.githubusercontent.com/harryjo97/GDSS/master/utils/graph_utils.py),
[MoFlow official models](https://github.com/calvin-zcx/moflow),
[MoFlow forward, reverse and log-probability code](https://raw.githubusercontent.com/calvin-zcx/moflow/master/mflow/models/model.py),
[MoFlow ZINC transform](https://raw.githubusercontent.com/calvin-zcx/moflow/master/data/transform_zinc250k.py),
[CoCoGraph official repository](https://github.com/manurubo/CoCoGraph),
[CoCoGraph parser/features](https://raw.githubusercontent.com/manurubo/CoCoGraph/main/lib_functions/data_preparation_utils.py),
[CoCoGraph models](https://raw.githubusercontent.com/manurubo/CoCoGraph/main/lib_functions/models.py).
These links were inspected as read-only research; no upstream commit or weight
from these additional alternatives is frozen as a project experiment.

The public DeFoG GuacaMol checkpoint remains a **neutral-compatible interface
test**, not a charge-preserving projector. DBMol's own adapted DeFoG explicitly
models formal charges and aromatic bonds; it is distinct from the public
checkpoint cached for this project. Our source028/030 neutral-interface tests
also failed their complete gradient gate: native cycle features rejected the
tested fractional graph, and frozen-extra edge finite differences did not meet
the declared multi-scale criterion. This is runtime evidence against treating
that interface as ready, not proof that all DeFoG adaptations are impossible.
See [docs18](18_stochastic_defog_projection_protocol.md).
[DBMol Appendix C.3](https://arxiv.org/pdf/2607.19237).
No charge-aware DBMol checkpoint was located in the checked official sources;
that limited search result is not proof of nonavailability.

## Current external-model priority, conditional on evidence

Prioritize a **frozen pretrained model with a documented input bridge**, rather
than a newly trained chemistry critic. JODO ZINC is the leading candidate in
this interface assessment for a restricted fixed-occupancy pretrained audit;
GDSS/MoFlow provide other continuous-input options with known feature gaps.
No audited prior supports the complete soft schema. GraphEBM is excluded as an
active fallback where new scorer weights would be required. These rankings are project-specific
inferences about interface feasibility, not measured quality comparisons.
No new alternative-model experiment is submitted by this research update.

Maintain one canonical graph from the selected molecular decoder and feed every
external branch through its own declared adapter. Prefer deterministic mappings;
train only a small supervised bridge to reproduce a frozen model's native input
or early embedding on paired hard molecules/context, then freeze it. Validate
hard identity and feature parity before soft-path derivatives and hard utility.
Prefer `B_m(G, context)`; the user's small latent decoder is allowed as
`B_m(z, G, context)` when needed, with cycle/identity consistency to shared `G`.
Never fit it to desired validity/binding/reward labels as a new scorer.
Model-specific latent decoders must not independently invent molecules; a small
bridge may not resolve pose/topology-dependent features. The full proposed separation of context,
adapter, frozen forward and objective, plus parity/gradient/native utility gates,
is specified in [docs23](23_frozen_model_plugin_contract.md).

For a frozen external scalar energy model, the desired path is
`z -> frozen decoder -> relaxed element/charge/bond/occupancy tensors -> energy`.
The chain rule returns an energy gradient to `z` without sampling a categorical
graph or constructing an RDKit molecule. The network must actually consume all
relevant tensors; hard masking or ignoring charge would create missing gradient
channels. GraphEBM's published dequantization is not automatically our decoder's
softmax distribution. Training coverage, symmetry, absence gating and noise
levels must be defined for our representation. A GDSS score vector instead
requires a declared noise-level objective or pullback update; it is not already
a calibrated scalar validity score. A MoFlow continuous density likewise is
not the probability that a hardened graph will be chemically valid.

The failed source022/024 AE-initialized denoisers remain negative baselines.
Source022 selected outputs were **0/256 valid** under corruption; the combined
source024 repair reached **2/256**, below its frozen AE control's **4/256**, and
lost substantial clean reconstruction. Their latent-gradient gates also failed.
See [docs16](16_schema_denoiser_pilot.md) and
[docs17](17_denoiser_repair_ablation.md).
Those failures remain relevant when evaluating any external guidance: lower
surrogate loss is not enough. Test clean-input retention, hard validity and
behavior on decoder-like perturbations. An external energy is not automatically
a validity probability, molecular likelihood or synthesis score. Its limitations
cannot be resolved by training a new in-house scorer under the present policy.

For every branch, require input-parity checks, finite-difference checks of the
declared scalar where applicable, nonvanishing latent gradients and **useful
hard-molecule changes** versus frozen baselines at matched cost. Hold out a
development confirmation cohort and preserve all attempted/failing candidates.
A differentiable neural forward is distinct from differentiating its categorical
sampler or RDKit conversions. No gradient test alone establishes useful chemical
guidance, and no branch should enter Boltz combination merely because it loads
or produces a scalar. DiGress, MiDi and DisCo stay alternatives rather than
parallel training commitments.

A broadly reusable fallback is hard-graph sampling under the actual decoder
law followed by compatible frozen native scoring/filtering. A separately tested
score-function gradient `E[(R-b) * gradient_z log q_z(G)]` can avoid differentiating
the external model, with substantial variance/cost and all failures retained.
This is not a learned reward surrogate and does not automatically implement
variable-size Boltz. It remains a proposed option, alongside restricted
pathwise adapters; details are in [docs23](23_frozen_model_plugin_contract.md).
[Discrete-variable gradient estimators](https://arxiv.org/abs/1711.00123).

## Historical energy-critic proposal: rejected under current preference

**Superseded 28 September 2026. Do not execute this proposal under the current
user preference.** It was never an approved configuration or submitted experiment.
The following historical discussion proposed a three-epoch A10
pilot of a scalar energy critic using our full 50-slot element/formal-charge/
aromatic-bond/NO_ATOM contract. Its intended training signal is contrastive
validity or empirical support, **not normalized log density, synthesis
feasibility or a universal chemical-quality score**. This is distinct from
asking the failed AE-initialized denoisers to reconstruct corrupted graphs.

Use only the 215,000-molecule TRAIN reference pool for clean training examples.
Construct invalid graph edits and invalid decoded AE/VAE graphs as negatives,
matching heavy-atom count, charge distribution and tensor softness across
labels where feasible. Include valid TRAIN decoder outputs as positive controls.
Keep valid edited graphs as explicit decoy controls:
an edit or distance from a reference does not by itself make a molecule invalid.
Avoid label leakage through padding, entropy, edit count, batch origin or
model identity. Define negative generation, mixture sizes, selection rules and
held-out development cohorts before running; the original evaluated test split
is not available for adaptive tuning.

An initial proposed discrimination screen is **AUROC at least 0.75 on held-out
decoder outputs**, evaluated within each decoder's own distribution: AE valid
versus AE invalid, and VAE valid versus VAE invalid. Match displacement, hard
atom count, total charge and softness; report size/charge strata and actual
coverage rather than comparing only real training graphs with one model's
invalid outputs. Evaluation uses validation molecules only. This threshold is
exploratory and must be frozen with the protocol;
it is neither a current result nor sufficient evidence to use the critic.
Also require performance on valid edited controls, calibration/threshold
behavior and explicit shortcut probes: hard valid graphs, matched-entropy soft
valid graphs, matched-softness invalid one-edit graphs, and 50:50 tensor mixtures
of unrelated valid molecules. A diffuse mixture must not earn an unjustified
quality reward. Require finite-difference agreement with finite,
useful-magnitude `dE/dz` through each frozen decoder. Evaluate both hard one-hot
and soft decoder inputs; a classifier succeeding only on artificial edits
would not justify latent optimization.

Only after those checks, propose a paired **prior-only** latent screen on the
same starts and budgets for AE/VAE, compared with the normalized local trust
baseline and unchanged starting decodes. Report all-attempt un-repaired
validity, valid-to-valid changes, hard/expected size, probability entropy,
uniqueness/diversity, distance from start, energy and actual cost; retain invalid
outcomes and identity collapse. A falling
energy with unchanged or less plausible hard molecules is a failed usefulness
screen. Source023's **5,437 predicted versus 3,269 true edges over 128 corrupted
graphs**, with **zero valid outputs**, is one concrete prior failure pattern
these diagnostics must expose. Source024's **90/256 clean valid and 37 exact**
versus frozen AE **252 valid and 250 exact**, plus latent-gradient norm about
**2.6e-10**, likewise precludes treating lower reconstruction loss as enough.
Treat the learned critic as a **soft restraint**, combined with independently
evaluated hard graph/chemistry gates; its low score must not become an absolute
veto on an otherwise valid molecule outside its training support. Boltz
combination would remain a later, separately gated experiment.

## JODO ZINC 2D: deeper interface audit

The official parser uses **C, N, O, F, P, S, Cl, Br and I**, stores RDKit formal
charge as a **scalar**, and kekulizes aromatic bonds. `EdgeCom` packs edge
existence and bond order divided by three; ZINC's configuration has no explicit
aromatic channel and at most **38 nodes**. Node-count masks and `nonzero` /
`dense_to_sparse` operations are hard structural choices. These facts allow a
more specific interface proposal, not full compatibility with our categorical
charge, aromatic and 50-slot NO_ATOM representation.
[ZINC parser](https://raw.githubusercontent.com/GRAPH-0/JODO/main/datasets/zinc_dataset.py),
[transforms and masks](https://raw.githubusercontent.com/GRAPH-0/JODO/main/datasets/build_dataset.py),
[model input path](https://raw.githubusercontent.com/GRAPH-0/JODO/main/models/mol_gnn.py).

A **proposed restricted fixed-occupancy forward** would condition each occupied
slot's element probabilities on presence,
`P(element | occupied) = P(element) / (1 - P(NO_ATOM))`, with a positive-occupancy
guard, then apply the checkpoint's centered scaling. Passing unnormalized
nine-element mass would leave the training simplex. For a fixed mask with both
edge endpoints present, map charges to their expectation and nonaromatic bond
probabilities to expected existence and expected order divided by three. This
loses charge uncertainty and bond-category information; explicit aromatic
probability has no lossless mapping through kekulization. Unsupported graphs
must be excluded or studied under a separately declared approximation. The mask
is fixed: no atom-count derivative or birth/death is established.

With `cond_x=None`, a declared noise time and fixed noise draws, one could test
a denoising-consistency scalar through the continuous neural forward. That
would be a **heuristic partial-gradient objective**, not likelihood, calibrated
validity or a derivative through native sampling. It needs exact hard-input
parity, checks of all mapped channels and multi-scale latent finite differences
before useful hard-graph changes are evaluated. The model's loss implementation
is a starting point for that proposed scalar, not evidence it works on our
decoder outputs. [Official losses](https://raw.githubusercontent.com/GRAPH-0/JODO/main/losses.py).

The official checkpoint listing links Zenodo's approximately **3.9 GB
`exp_2d.zip`**; the ZINC evaluation example includes checkpoint-5 overrides.
These assets have **not been downloaded, hash-verified or tested for this project**.
Freeze exact checkpoint/configuration and code hashes before a bounded interface
experiment; passing the interface would still require held-out hard-molecule
utility tests. This remains read-only research, with no new job or adapter
implementation claimed.
[Official checkpoint/evaluation instructions](https://github.com/GRAPH-0/JODO#checkpoint),
[Zenodo record](https://zenodo.org/records/8002902).
