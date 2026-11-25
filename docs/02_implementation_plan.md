# Relaxed Molecular Graph Predictor — Step-by-Step Implementation Plan

> **Superseded work plan (2026-09-26).** The latest user instruction is graph AE
> -> matched graph VAE -> frozen-Boltz2 latent optimization, with all tests,
> training and experiments on DTU HPC. See the active
> [`latent pipeline`](04_latent_pipeline.md) and
> [`experiment ledger`](05_experiment_ledger.md). Statements below naming an older
> protocol as ground truth, prohibiting latent/Boltz2 work, or requiring a small
> KIBA predictor first describe the previous scope and do not override this change.

Executable companion to [`project_desc.pdf`](project_desc.pdf), which is the **ground
truth** for this project. Where this plan and the protocol disagree, the protocol wins.
Rewritten 2026-09-24.

## What this stage is

Build **one small, frozen, differentiable 2D graph predictor** that scores a molecule
against a KIBA target, accepts *relaxed* (soft) atom and bond inputs, and produces
finite, target-dependent gradients with respect to them.

That is the whole deliverable. The predictor is **scaffolding, not the research.** The
research — valence penalties vs. continuous projection, connectivity constraints,
hardening schedules — begins at handoff (protocol §10), against a checkpoint that never
moves again.

## When this stage ends

The moment the eight acceptance checks in [Step 5](#step-5--acceptance-tests-and-the-freeze-gate)
pass. Not when the KIBA metrics look good. **Mediocre-but-usable prediction with stable,
target-dependent gradients is a pass.** Freeze the checkpoint that same day.

---

## Companion documents

| File | Status |
|---|---|
| [`project_desc.pdf`](project_desc.pdf) | **Ground truth.** The protocol this plan executes |
| [`00_glossary.md`](00_glossary.md) | Current. Chemistry, biology and method vocabulary, written for an ML reader. Start here if a term is unfamiliar |
| [`01_related_work.md`](01_related_work.md) | Current. Gap analysis, the validity-mechanism design space, KIBA caveats, reward hacking |
| [`03_hpc_execution.md`](03_hpc_execution.md) | Current. Compute sizing and execution — short version: you probably do not need the cluster |
| [`project.md`](../project.md) | **Superseded.** The earlier Latent-DBMol proposal. Retained as history; not a work plan |

Nothing in this plan depends on DBMol source code, Boltz-2, AlphaFold 3 access, LIT-PCBA
targets, MSAs, or GPU queue contention. Those risks are gone.

---

## Design invariant — read this before writing any code

Everything in this stage exists to serve one property:

> **One forward pass, no discrete branching.** The same code path consumes a hard
> one-hot molecule during training and a soft simplex-valued molecule during
> optimization. Nothing in the forward pass may `argmax`, cast to `long`, index an
> `nn.Embedding`, call RDKit, or branch on whether an input is integral.

Concretely this means embeddings are **matrix products, not lookups**:

```python
h_atom = X @ W_atom        # X: [B, N, K_atom]  simplex rows, one-hot is a special case
e_feat = E @ W_bond        # E: [B, N, N, 5]    ditto
h_atom = h_atom + Q.unsqueeze(-1) * w_charge   # Q: [B, N] continuous scalar
```

Every design decision below follows from this. If a feature cannot be written as a
differentiable function of `(X, Q, E, M)`, it does not go in the model.

### The one purposeful change: decouple the atom variables

DIDgen's atom construction is **valence-coupled** — element identity is entangled with
implicit-hydrogen count and degree. This project requires atom identity, formal charge,
and bond identity to move as **independent** optimization variables (protocol §1, §5).

So: **no valence-derived input features.** No implicit/explicit H count, no atom degree,
no total valence, no ring-membership flags, no hybridization. Those quantities are
*consequences* of `(X, Q, E)`, and making them inputs would (a) break independence and
(b) make the downstream valence-penalty-vs-projection comparison meaningless, because
the constraint being studied would already be baked into the representation.

Heavy-atom graph, implicit hydrogens, and hydrogens never appear as nodes.

---

## Repository layout

```
src/
  data/
    kiba.py             # TDC load + standard split
    featurize.py        # SMILES -> (X, Q, E, M); the only RDKit-dependent module
    build_cache.py      # writes graphs.pt + pairs.parquet
  model/
    gnn.py              # dense edge-conditioned MPNN; soft-input forward
    predictor.py        # trunk + multi-target regression head
  train.py              # the one training run
  validate/
    gradients.py        # finiteness, non-zero, finite-difference agreement
    smoothness.py       # interpolation sweeps
    optimize_smoke.py   # 10-30 step target-specific optimization
    target_specificity.py
  objective.py          # FROZEN handoff interface
tests/
  test_invariants.py    # mask, permutation, symmetry, diagonal, single-path
data/cache/
  graphs.pt             # mol_id -> X, Q, E, M
  pairs.parquet         # mol_id | target_id | kiba_score | split
checkpoints/
  kiba_predictor_v1.pt  # the frozen artifact
reports/
  01_atom_count_audit.md
  02_training_run.md
  03_relaxed_validation.md
```

Small on purpose. If a module does not appear here, ask whether it is scope creep.

---

## Step 1 — Reuse DIDgen/CrippenNet as the scaffold

**Goal:** start from working inverse-design code instead of a blank repository.
**Source:** [`ftherrien/inv-design`](https://github.com/ftherrien/inv-design) —
*"Using GNN property predictors as molecule generators"*, Nature Communications, 2025.

| Task | Deliverable |
|---|---|
| 1.1 | Clone the repo at a **pinned commit**. Build the environment (PyTorch + RDKit). Record exact versions | `environment.yml` / `requirements.txt` |
| 1.2 | Run their existing training and inverse-design example (the logP/CrippenNet path) **unmodified**, end to end. Do not skip this — it is how you learn what "working" looks like before you change anything | one reproduced logP optimization trajectory |
| 1.3 | Write the reuse map: for every DIDgen module, mark **keep / adapt / replace** with one line of justification | `reports/00_didgen_reuse_map.md` |

### Reuse decisions (protocol §3)

| Component | Decision |
|---|---|
| Repository & workflow | **Keep.** Base code, and its direct input-optimization philosophy |
| Fixed-size dense graph handling, padding/masking | **Keep.** Directly needed — optimized variables must be explicit dense tensors |
| Training loop, optimizer setup, validation, checkpointing, logging | **Keep** wherever compatible |
| RDKit parsing / graph-construction utilities | **Adapt.** Reuse rather than building a general cheminformatics pipeline |
| Model scaffold (small GNN) | **Adapt.** Replace only the graph core needed for explicit categorical bond features |
| Atom construction | **Replace.** Valence-coupled → independent (X, Q, E). See the design invariant above |
| Output head | **Replace.** logP scalar → KIBA target-labelled multi-output |
| Gradient-to-input / inverse-design utilities | **Keep.** Later projection and constraint work plugs into this same interface |

**Prefer a small adapter over a clean rewrite.** A rewrite that is 20% nicer and costs a
week is a loss.

**Anti-goals:** do not port DIDgen's valence-coupled atom construction unchanged; do not
substitute a large molecular Transformer or any protein-ligand architecture.

---

## Step 2 — Minimal KIBA preprocessing

**Goal:** a deterministic `(X, Q, E, M)` cache and a pairs table. Nothing else.
**Source:** [TDC DTI / KIBA](https://tdcommons.ai/multi_pred_tasks/dti/) — approximately
117k drug-target pairs, 2,068 drugs, 229 protein targets.

```python
from tdc.multi_pred import DTI
data  = DTI(name='KIBA')
split = data.get_split()        # standard TDC random split
```

### Field handling (protocol §4)

| Field | Use | Processing |
|---|---|---|
| Drug / SMILES | Yes | Parse each **unique** SMILES once with RDKit |
| Target ID | Yes | Map to `0 … T-1`. **Never encode the protein sequence** |
| Protein sequence | **No** | Optional metadata only; never reaches the model |
| KIBA score `Y` | Yes | Regression label. Standardize using **training-split** statistics only, if it helps numerical stability |

### Featurization

- `X`: `[N_max, K_atom]` one-hot atom identity. Element vocabulary derived **once** from
  KIBA and then fixed. `K_atom` **includes a `NO_ATOM` category** — see the decision
  below.
- `Q`: `[N_max]` formal charge, one scalar per atom (`atom.GetFormalCharge()`).
- `E`: `[N_max, N_max, 5]` categorical bonds — `NO_BOND, SINGLE, DOUBLE, TRIPLE,
  AROMATIC`. Dense and **symmetric**; diagonal forced to `NO_BOND`.
- `M`: `[N_max]` atom/padding mask, retained for batching.

Do **not** kekulize — `AROMATIC` must survive as its own category.

### Decision: `NO_ATOM` is a real category in `X` — **decide before freezing**

**Adopted.** Padding is represented as a genuine `NO_ATOM` category inside `X`'s
vocabulary, not only by the external mask `M`.

The protocol is ambiguous here — §4 says to add "padding/no-atom handling only as needed
for fixed-`N` batching", and §5 lists `M` as a separate mask. The two readings have very
different consequences:

| Reading | Consequence |
|---|---|
| Occupancy lives only in `M` | `M` is a **constant input**. Molecule size is fixed for the whole optimization and is not differentiable |
| Occupancy is a category in `X` | Occupancy is a coordinate on the simplex the optimizer already moves. **Molecule size becomes a continuous, differentiable variable** |

Define occupancy and size as:

```python
occ  = 1.0 - X[:, NO_ATOM]      # [N_max], differentiable occupancy in [0, 1]
size = occ.sum()                 # differentiable molecule size
```

**Why adopt it.** It removes the need to commit to an atom count before optimization
begins, which is otherwise a hyperparameter chosen per starting molecule (DBMol fixes `N`
per target from the reference ligand). It costs almost nothing here — one extra
vocabulary entry — and it is the standard one-shot graph generation trick, not a research
risk.

**Why the deadline matters.** `NO_ATOM` changes `K_atom`, which changes the input
dimension, which changes the frozen checkpoint. Step 7 permits retraining only for a
concrete blocker, so retrofitting this after the freeze means either breaking the freeze
discipline or rebuilding. **Build the vocabulary with `NO_ATOM` from day one even if
occupancy is held fixed during early experiments** — it is nearly free now and expensive
later.

Note what this does *not* require: a latent space or a decoder. Differentiable occupancy
is a property of the output parameterization, available directly on `X`. A decoder would
still have to pick its own `N_max` internally, so it relocates the choice rather than
removing it. See [`01_related_work.md`](01_related_work.md) §3.

### Fixed `N_max`

Run **one** atom-count audit over the 2,068 unique drugs before choosing. Report the
distribution and the count dropped at several candidate cutoffs; pick the smallest
`N_max` that drops only a very small tail (a high percentile such as 99.5% is a
reasonable rule). Prefer dropping a handful of oversized molecules over engineering a
second batching system. Record the choice, the cutoff rule, and the exact dropped
molecules in `reports/01_atom_count_audit.md`.

**Anti-goals:** no fingerprints, no molecular descriptors, no conformers, no 3D, no
protein embeddings, no pockets, no protein graphs. No extensive canonicalization or
duplicate harmonization unless RDKit parsing or repeated records demonstrably break
training. No custom split.

### Cache format

| File | Contents |
|---|---|
| `data/cache/graphs.pt` | `mol_id -> (X, Q, E, M)` |
| `data/cache/pairs.parquet` | `mol_id | target_id | kiba_score | split` |

Only 2,068 unique molecules back ~117k pairs, so the graph cache is tiny and the pairs
table indexes into it. (If training turns out slow, run the trunk once per *unique*
molecule per batch and gather — but only if it is actually slow.)

**Store `X` and `E` as int8 category indices, not one-hot** (~9 MB instead of ~174 MB),
and expand to one-hot in the `collate_fn`. This does not violate the single-path
invariant: the prohibition is on index lookups *inside the forward pass*, where gradients
flow. Dataloader-side expansion of constant input data has no gradient passing through
it. See [`03_hpc_execution.md`](03_hpc_execution.md) §3.

Pin the RDKit version and record it next to the cache — featurization must be
reproducible across your laptop and the cluster, or the two disagree silently.

---

## Step 3 — Minimal relaxed-input architecture

**Goal:** the smallest model that satisfies the design invariant. Predictive novelty is
explicitly not a goal (protocol §5).

| Component | Choice |
|---|---|
| Atom identity `X` | `[N, K_atom]` incl. `NO_ATOM`; one-hot in training, probability simplex during optimization |
| Formal charge `Q` | One scalar per atom; integer normally, continuous when relaxed |
| Bond identity `E` | `[N, N, 5]` categorical tensor |
| Occupancy `occ` | `1 − X[:, NO_ATOM]`; derived, not a separate input. Gates messages and pooling |
| Mask `M` | `[N]`; retained for batching. **Not** the size variable — see below |
| Trunk | Small permutation-equivariant **edge-conditioned** message-passing network, **3–4 layers**, modest hidden size |
| Pooling | Occupancy-weighted **mean** over atoms — see the pooling caveat below |
| Head | Molecular embedding → vector of `T` target scores; `target_id` **selects one output** |

Edge conditioning in the dense form, gated by occupancy throughout:

```python
occ    = 1.0 - X[..., NO_ATOM]                        # [B, N]
msg_ij = mlp_e(E @ W_bond) * (h_j @ W_h)              # [B, N, N, H]
h_i    = update(h_i, (msg_ij * occ_j[..., None]).sum(dim=2))
h_mol  = (h * occ[..., None]).sum(1) / (occ.sum(1, keepdim=True) + eps)
```

### Two things occupancy will break if you let it

**1. Gate `E` by `occ_i · occ_j`.** Otherwise a slot can be switched off while its bonds
still carry signal, and the optimizer gets a ghost atom that contributes bonding without
paying the occupancy cost. Enforce the coupling in the forward pass rather than hoping a
penalty catches it.

**2. Prefer occupancy-weighted mean pooling over sum pooling.** With a sum, every added
atom adds a term, so "grow toward `N_max`" raises the score almost for free. KIBA score is
genuinely size-correlated over part of its range, so the model will have learned some of
this and the optimizer will find it immediately. Mean pooling dampens the effect; it does
not remove it, because size-correlated features survive the normalization. Expect
inflation-by-growth as a named exploitation mode and report realized size distributions.

**Anti-goals:** no target embeddings initially; **do not broadcast target information
through the GNN** — the trunk stays molecule-only and all target dependence lives in the
multi-output head; no Pairformer, no 3D coordinates, no attention over a protein, no
sequence models, no foundation-model embeddings. Do not run a GCN/GAT/GIN/Transformer
comparison — that is a separate paper, not this step.

### What "AF3-like" means here (protocol §6)

Only the **ligand variables**. AF3 accepts ligands via SMILES/CCD, and CCD carries
per-atom element and charge plus per-bond connectivity and order. We want the optimizer
acting on the analogous atom/charge/bond variables. The scoring network stays a small 2D
GNN. Architecture, Pairformer, diffusion, 3D coordinates, protein context, and confidence
heads are explicitly **not** part of the similarity we care about.

The payoff: a projection or constraint that works on `X/Q/E` here can later be tested
against a modified Boltz/AF3-class ligand interface **without redesigning the
optimization concept**.

### Invariant tests (`tests/test_invariants.py`)

Write these before training. They are cheap and they catch the failures that silently
corrupt everything downstream.

| Test | Assertion |
|---|---|
| Mask correctness | Changing padded slots (or their count) leaves the score unchanged |
| Permutation equivariance | Permuting atom order leaves the score unchanged |
| Bond symmetry | `E == E.transpose(1, 2)` in, and the model is insensitive to which triangle is written |
| Diagonal | `E[i, i] == NO_BOND` enforced on load and after any update |
| Single path | No `argmax` / `.long()` / `nn.Embedding` / RDKit call in the forward. A one-hot input scores identically whether labelled "hard" or "soft" |

---

## Step 4 — One practical training run

**Goal:** one checkpoint that clearly beats a trivial baseline and tolerates near-integer
inputs. Design it so the first usable checkpoint is the final one (protocol §7).

| Item | Choice |
|---|---|
| Split | Standard TDC **random** split. This is a surrogate check, not a DTI generalization paper |
| Objective | Regression on the selected KIBA target output (gather, then MSE). Standardize `Y` if it helps stability |
| Hard inputs | Exact one-hot `X`/`E`, integer `Q` |
| Relaxed robustness | **In the same run**, apply mild random perturbations around one-hot `X` and `E` for a subset of examples |
| Hyperparameters | DIDgen defaults / one sensible small-GNN config. **No sweep** |
| Stopping | Validation loss / early stopping purely to avoid clear overtraining. Do not chase incremental gains |
| Seeds | **One** main run. Repeat only if clearly unstable or technically failed |

### Relaxed augmentation

On the fly, from the same hard molecule, keeping the same KIBA label:

```python
X_soft = (1 - eps) * X + eps * random_simplex_like(X)   # eps small, e.g. 0.05-0.15
```

The goal is **local smoothness around valid molecules** — not assigning physical meaning
to arbitrary highly-mixed atoms and bonds. Keep perturbations mild.

**The perturbation must cover the `NO_ATOM` coordinate**, on both occupied and padded
slots. If the model only ever sees occupancy at exactly 0 or 1, fractional occupancy is
territory it was never trained on, and `∂score/∂occ` is arbitrary there. A size penalty
applied on top of a meaningless occupancy gradient is still meaningless — this
augmentation is what makes differentiable size real, and no downstream mechanism
substitutes for it.

**Anti-goal:** do not build a second dataset of relaxed states.

### Baselines to beat

Global mean, and **mean-per-target** — the two the protocol names, and gate check 2 is
scored against them.

**Also report a two-way additive baseline** (drug mean + target mean, fit on the training
split). It is not part of the gate, but it is the diagnostic for gate check 7: kinase
inhibitors are promiscuous, so much of KIBA's variance sits in drug-level and
target-level *main effects* rather than drug×target interaction. A model that only
narrowly beats the additive baseline has learned main effects and little interaction, and
its per-target gradients will be near-identical — check 7 then fails for a data reason,
not a modelling one. Catching that here costs one line; catching it later costs a week.
See [`01_related_work.md`](01_related_work.md) §5.

Report all three in `reports/02_training_run.md`, alongside train/valid RMSE, the
per-target breakdown, and the **per-target pair-count distribution** (coverage across the
229 targets is uneven, and it explains most per-target anomalies).

---

## Step 5 — Acceptance tests and the freeze gate

**This is the stopping gate.** None of these require state-of-the-art affinity
prediction (protocol §8).

| # | Check | Minimal acceptance criterion |
|---|---|---|
| 1 | Data path | KIBA loads reproducibly; almost all retained SMILES parse; `X/Q/E/M` batches are deterministic and correctly padded |
| 2 | Ordinary prediction | Training clearly beats a trivial mean-per-target or global-mean baseline; validation behaviour stable enough to use as a proxy |
| 3 | Hard-input gradients | Finite, **non-zero** gradients w.r.t. `X` and `E` |
| 4 | Relaxed forward pass | Mildly softened `X/E` produce finite predictions with no numerical pathologies |
| 5 | Local smoothness | Small interpolations away from a hard molecule change the score smoothly enough for gradient optimization |
| 6 | Short optimization | From a fixed starting molecule, **10–30 gradient steps** move the selected target score in the desired direction |
| 7 | Target specificity | The same starting molecule optimized for **two different target labels** receives measurably different gradients / trajectories |
| 8 | Invariants | `tests/test_invariants.py` passes |

### How to run checks 3–7

- **Gradients (3):** `X.requires_grad_(True)`; backward from the selected output; assert
  finite and non-zero. Add finite-difference agreement along a few random directions —
  cheap, and it catches a wrong gradient that is merely *plausible*.
- **Smoothness (5):** interpolate `X(t) = (1-t)·X_A + t·X_B` between two padded
  molecules and sweep `t ∈ [0, 1]`. Plot score vs. `t`; look for jumps and NaNs, not for
  a particular shape.
- **Short optimization (6):** parameterize with logits and `softmax` so iterates stay on
  the simplex for free; Adam; 10–30 steps. This is a smoke test, **not** the optimizer
  study — resist tuning it.
- **Target specificity (7):** compute `∂score/∂X` for two targets from the same start
  molecule and report **cosine similarity**. Well below 1 is a pass. Choose two targets
  that each have plenty of training pairs — KIBA coverage is uneven across the 229
  targets, and a thinly-observed head will look undifferentiated for a data reason
  rather than a modelling one. Measure this **immediately after training**, not last: it
  is the check most likely to fail.

> **Run everything in fp32. Do not enable mixed precision.** Checks 3, 5 and 7 all measure
> *gradient* quantities — finite-difference agreement, smoothness under interpolation,
> cosine similarity between per-target gradients — and fp16/bf16 degrades precisely those.
> A check that fails on a numerical artifact looks identical to one that fails on a real
> modelling problem, and you will spend days on the wrong one. At this model size mixed
> precision buys nothing anyway. The same goes for any training-acceleration framework
> that enables it by default.

### The gate

**If all checks pass, freeze the checkpoint immediately and stop predictor development.**

- Do **not** block on mediocre-but-usable KIBA benchmark performance if gradient
  behaviour is stable and target-dependent.
- Do **not** retrain merely because a larger hidden size or a different GNN might improve
  validation metrics.

Write `reports/03_relaxed_validation.md` as the short validation report: ordinary
prediction sanity, relaxed interpolation, gradient finiteness, and target-specific short
optimization trajectories.

---

## Step 6 — Freeze and hand off

Once frozen, **every** downstream experiment uses this same checkpoint, so that observed
differences come from the molecular optimization method rather than a moving predictor.

### Handoff interface (protocol §11)

```python
objective = model(X_soft, Q, E_soft)[target_id]
```

> **Deviation to confirm with the supervisor.** The mask `M` is required for correct
> dense batching, so in practice the signature is
> `model(X_soft, Q, E_soft, M)[target_id]`. Either `M` is passed explicitly or it is
> fixed at handoff time for a given starting molecule. Flagging rather than silently
> changing the protocol's stated interface.

Requirements on the frozen artifact:

- `checkpoints/kiba_predictor_v1.pt` — weights, element vocabulary, `N_max`, bond
  category order, `Y` standardization constants, and the DIDgen commit hash, all in one
  file. A checkpoint that cannot reconstruct its own vocabulary is not frozen.
- `src/objective.py` — loads it, sets `eval()`, disables parameter grads, exposes the
  interface above and nothing else.
- Tag the commit. Any later change to the predictor invalidates every downstream result.

### Stage outputs (protocol §11)

1. A small adaptation of the DIDgen codebase accepting explicit `X/Q/E/M` tensors and
   categorical bond features.
2. A minimal KIBA preprocessing script and cache; **no protein preprocessing pipeline**.
3. One frozen target-labelled KIBA predictor checkpoint used by all downstream work.
4. A short validation notebook/report (checks 2, 5, 3, 7 above).
5. The clear handoff interface.

---

## Step 7 — Reopen the predictor **only** for a concrete blocker

Not for a hunch, not for a better metric. The smallest allowed response (protocol §9):

| Observed blocker | Smallest allowed response |
|---|---|
| Vanishing or exploding relaxed gradients | Adjust input scaling/normalization or relaxed-noise magnitude first; retrain the same model **once** if required |
| Soft inputs behave erratically between one-hot corners | Increase mild near-integer augmentation; **keep the architecture unchanged** |
| All target heads give almost identical gradients | Check target-head training and data indexing **first**; only then modestly increase head capacity |
| Immediate extreme proxy exploitation | Tighten the local relaxed training region or add simple out-of-distribution guardrails; **do not build a new DTI model** |

Then freeze again and return to optimization. Log every reopening in
`reports/03_relaxed_validation.md` with the trigger and the change.

---

## After the handoff — where the research actually starts

Out of scope for this plan, recorded so the scaffolding is built to serve it
(protocol §10):

1. **Baseline:** unconstrained / simplex-only relaxed atom and bond optimization.
2. Then valence penalties vs. continuous projection, connectivity constraints, hardening
   schedules, and other graph-validity mechanisms.
3. Hold **starting molecules, target labels, and the frozen predictor** fixed across
   every method comparison.
4. Only once the cheap `X/Q/E` optimization mechanics are convincing do successful ideas
   move to the expensive Boltz/AF3-class setting.

One measurement worth designing for now, because it is the natural primary metric of
that study: the **hardening gap** — `score(relaxed x*) − score(hardened/projected x*)`.
It quantifies how much of the optimizer's apparent gain survives the return to a valid
discrete molecule, and it is exactly what distinguishes hardening schedules from each
other. The frozen interface already supports it; no extra scaffolding needed.

### Controlling size once occupancy is differentiable

A size penalty is the right tool, but the **form** matters more than the presence:

| Form | Effect | Verdict |
|---|---|---|
| `λ · size` | Constant downward pressure; the optimizer grows until marginal score gain equals `λ` | Reprices the exploit rather than removing it. Equilibrium size drifts per target and per starting molecule |
| `λ · (size − size₀)²`, anchored to the **starting molecule's** size | Trajectories stay near their own starting size | **Recommended.** Targets the actual problem — cross-method comparability — and mirrors what DBMol enforces by fixing `N`, and post-hoc with its ±3 / ±5 size-consistency filter |
| Hard clamp or projection onto a size band | Removes the degree of freedom entirely | Use for the main comparison if size drift turns out to dominate |

Two rules that keep this from contaminating the study:

1. **Fix `λ` once, across every method.** If method A needs one `λ` and method B another,
   the mechanism comparison is confounded by `λ` tuning rather than measuring mechanisms.
   Same discipline as the frozen predictor.
2. **Run the main mechanism comparison at fixed occupancy, and make "size free to float"
   its own arm.** Size correlates with almost every affinity-like score, so letting it
   drift inside the main comparison turns it into a confound. As a separate condition it
   is a result.

Worth noting: *size penalty vs. hard size constraint* is structurally the same question as
*valence penalty vs. projection* — soft-during versus hard-after. It is a miniature of the
whole study, on a one-dimensional quantity that is trivial to measure, which makes it a
good warm-up experiment before the full comparison.

---

## Consolidated anti-goals

The protocol's prohibitions, in one place. Every one of these is a way this stage
overruns:

- State-of-the-art affinity metrics, unseen-target generalization, architecture novelty,
  exhaustive hyperparameter tuning.
- Protein sequences, embeddings, pockets, protein graphs, or any protein representation.
- Fingerprints, molecular descriptors, conformers, 3D coordinates.
- Pairformer, diffusion, structural confidence heads, foundation-model embeddings.
- A modern large molecular Transformer or protein-ligand architecture.
- GCN/GAT/GIN/Transformer ablations as a separate study.
- Target embeddings, or target information broadcast through the GNN trunk.
- A custom dataset split or custom dataset construction.
- A second dataset for relaxed states.
- Extensive canonicalization / duplicate harmonization absent a demonstrated problem.
- A second batching system to accommodate oversized molecules.
- Hyperparameter sweeps, multi-seed studies, retraining for better validation metrics.
- Porting DIDgen's valence-coupled atom construction unchanged.

---

## Compute

A 3–4 layer dense GNN over ≤ 2,068 unique molecules at `N_max ≈ 60–70`, trained on ~117k
pairs. This is a **single-GPU, minutes-to-hours** workload; a laptop GPU or one modest
cluster GPU is ample, and CPU-only is viable if slow. Dense `N × N × 5` bond tensors are
the memory driver, and they are small at this `N_max`.

There is no queue-contention problem, no multi-day job, and no job-array design work.
Do not spend time on HPC engineering at this stage — [`03_hpc_execution.md`](03_hpc_execution.md)
was written for a workload that no longer exists.

---

## Effort allocation

Shares are of **this stage**, which should itself be a **minority of the project** —
target ≤ 25–30%, with the balance going to the optimization research in protocol §10.
Calendar weeks depend on dates that are still open (see below).

| Step | Work | Share |
|---|---|---|
| 1 | DIDgen reuse: env, reproduce their example, reuse map | 20% |
| 2 | KIBA preprocessing, atom-count audit, cache | 15% |
| 3 | Representation + model changes, invariant tests | 20% |
| 4 | The one training run | 10% |
| 5 | Acceptance tests, validation report, **freeze** | 25% |
| 6 | Handoff interface, checkpoint hygiene, tag | 10% |
| 7 | Blocker fix | 0% budgeted — contingency only |

Step 5 is deliberately the largest. The gradient and relaxation properties *are* the
product; the KIBA metric is not.

**Do not cut:** the invariant tests (Step 3), the target-specificity check (Step 5 #7),
or checkpoint reproducibility (Step 6).

---

## Open questions

1. **Calendar.** Project start, hand-in date, and ECTS weight — needed to turn the effort
   shares above into real weeks.
2. **Handoff signature.** Confirm the `M` deviation noted in Step 6.
3. **`N_max` and the element vocabulary.** Both are outputs of the Step 2 audit; recorded
   here as deliberately unfixed rather than guessed. **`NO_ATOM` as a real category in
   `X` is decided** (Step 2) — it must be settled before the freeze because it changes
   `K_atom`. Confirm with the supervisor that this reading of protocol §4/§5 is intended,
   since the protocol leaves it ambiguous.
4. **Starting molecules for the optimization study.** Which molecules and which target
   labels form the fixed comparison set? Choosing them before freezing costs nothing and
   lets the Step 5 smoke tests double as the baseline's first data points.
5. **DIDgen reproducibility.** Does `ftherrien/inv-design` run as released, with the
   current RDKit/PyTorch? Unknown until Task 1.2 — this is the only real engineering
   risk in the stage, and it surfaces in week one.
