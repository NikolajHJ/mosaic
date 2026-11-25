# Related Work and Gap Analysis — Relaxed Molecular Graph Optimization

Background for [`project_desc.pdf`](project_desc.pdf) (ground truth) and
[`02_implementation_plan.md`](02_implementation_plan.md).
Unfamiliar chemistry or biology terms are defined in [`00_glossary.md`](00_glossary.md).
Rewritten 2026-09-24 against the current project framing.

**Bottom line up front.** The mechanism this project uses — differentiate a trained GNN
property predictor back to its molecular graph input — is published prior art (DIDgen).
Continuous relaxation of molecular graphs is published prior art (DBMol, MoFlow, the
Gumbel/straight-through lineage). Multi-target regression on KIBA is a solved benchmark
task. **None of stage 1 is novel and it is not meant to be.**

What is *not* settled is the question the frozen predictor exists to answer: **when a
relaxed molecular graph has been optimized off the data manifold, which mechanism should
return it to chemical validity — a valence penalty, a continuous projection, a
connectivity constraint, or a hardening schedule — and does the answer transfer to an
AF3-class ligand representation?** Each mechanism exists somewhere in the literature.
No controlled head-to-head under a fixed scorer and fixed starting points exists.

---

## 1. The direct ancestor: DIDgen / CrippenNet

[**"Using GNN property predictors as molecule generators"**](https://github.com/ftherrien/inv-design)
— Nature Communications, 2025. Code: [`ftherrien/inv-design`](https://github.com/ftherrien/inv-design).

The core move: a GNN trained *only* to predict a property is also, by differentiation
with respect to its input, a generator. No decoder, no latent space, no separate
generative model. Gradient descent runs directly on the graph tensors.

Properties of the released implementation that this project depends on (as stated in the
protocol; **verify against the paper and repo in Step 1.2–1.3** rather than trusting this
summary):

| Property | Why this project needs it |
|---|---|
| Fixed-size dense graph inputs with padding/masking | The optimized variables must be explicit tensors, not a sparse edge list |
| Predictor deliberately made tolerant to **near-integer** inputs | Without this, a relaxed graph lands off-distribution immediately and gradients are meaningless |
| Direct input-optimization workflow already built | The gradient-to-input plumbing is the part that is tedious rather than interesting |
| logP / Crippen output head | Replaced here by a target-labelled KIBA head |
| **Valence-coupled** atom construction | **The one thing to discard.** See below |

### The one purposeful divergence

DIDgen couples element identity with valence/hydrogen count in its atom construction.
This project requires atom identity, formal charge, and bond identity to move as
*independent* variables (protocol §1, §5). That is not a cosmetic change:

- If valence is baked into the atom representation, the optimizer cannot violate valence
  — which sounds desirable until you notice it makes the downstream study impossible.
  **"Valence penalty vs. continuous projection" is only a question if the representation
  permits the violation in the first place.**
- It also means DIDgen's own validity behaviour is not a baseline for this project. Their
  representation solves by construction what this project deliberately leaves open.

Expect this to be the source of most of the adaptation work, and expect DIDgen's reported
validity rates not to transfer.

---

## 2. The design space this project actually explores

The research question is: *relaxed graph → valid molecule, by what mechanism?* Each of
these is a published point in that space. **They have never been compared under a common
scorer.**

| Mechanism | Representative work | What it does | Known cost |
|---|---|---|---|
| **Simplex-only** (clip + renormalize) | DBMol; the protocol's §10 baseline | Keep iterates on the probability simplex, nothing more | Nothing prevents chemically impossible graphs |
| **Generative-prior projection** | DBMol's use of [DeFoG](https://arxiv.org/abs/2410.04263) — sample a noisy discrete graph from the relaxed optimum, denoise from late time `t = 0.8` | Hands validity to a pretrained discrete generative model | Needs a trained graph generative model; the prior can overwrite the optimization |
| **Post-hoc validity correction** | [MoFlow](https://arxiv.org/abs/2006.10137) | Repair the decoded graph after the fact | Non-differentiable; the optimizer never learns about it |
| **Hardening / annealed relaxation** | [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) (Jang et al.), [Concrete](https://arxiv.org/abs/1611.00712) (Maddison et al.), [straight-through estimators](https://arxiv.org/abs/1308.3432) (Bengio et al.) | Anneal a temperature so iterates approach one-hot corners during optimization | Gradient bias; schedule is a hyperparameter with real effect |
| **Stay discrete, differentiate locally** | [DST, ICLR 2022](https://arxiv.org/abs/2109.10469) | A scaffolding tree makes a *discrete* graph locally differentiable | Sidesteps relaxation entirely; different design point |
| **Penalty terms** | Common but scattered | Add a differentiable valence/connectivity penalty to the objective | Weighting is arbitrary; penalties can be traded against the objective |

**The gap.** These are reported in different papers, on different predictors, with
different datasets and different optimizers. Nobody holds the scorer, the starting
molecules, and the target labels fixed and varies only the validity mechanism. That is
exactly what protocol §10 sets up, and the frozen checkpoint is what makes it clean —
**the single most important methodological choice in the project is that the predictor
stops moving.**

### Composite objectives: Mosaic, and the protein-side precedent

[`escalante-bio/mosaic`](https://github.com/escalante-bio/mosaic) (MIT, JAX) is the
closest existing implementation of *optimizing one relaxed design variable against several
frozen models at once*. It optimizes a continuous relaxation of a protein sequence — a
simplex over the 20 amino acids at each position — against a weighted sum of loss terms
drawn from roughly fifteen wrapped models: Boltz-1/2, AlphaFold2, OpenFold3, Protenix,
ESMFold2, ProteinMPNN, ESM-2/C, stability predictors. Loss terms are composable units
combined additively with weights, so adding or swapping an objective is a few lines.
Escalante use it for minibinder design; see
[their write-up](https://blog.escalante.bio/minibinder-design-is-just-not-that-hard/).

**Why it matters here.** The "project one relaxed representation into several frozen
critics and optimize jointly, rather than running them as sequential pipeline stages"
idea is *not* open research — it is built, working and open-source on the protein side.
Anyone proposing it for molecules should cite Mosaic and claim the molecular case, not the
pattern.

**But the transfer is not mechanical, and the reason is the interesting part.** Protein
and small-molecule design differ in exactly the place this project cares about:

| | Protein sequence design | Small-molecule graph design |
|---|---|---|
| Design variable | Sequence over 20 amino acids | Graph: `(X, Q, E)` |
| Is an arbitrary point in the discrete space a real molecule? | **Yes.** Any sequence of standard residues is a synthesizable polypeptide | **No.** Most graphs violate valence, connectivity or charge consistency |
| What discretization risks | Whether it *folds and binds* — quality | Whether it *exists at all* — validity, and then quality |
| Consequence | Relax, optimize, argmax. Validity is free | Needs an explicit validity mechanism: penalty, projection, denoiser, or hardening |

Mosaic can use a comparatively simple relaxation because its output space has **no
chemical validity constraint**. DBMol needs an entire generative projection stage because
a molecular graph space does. That asymmetry is why this project exists and why it is not
merely "Mosaic for molecules".

Consistent with that, the Escalante write-up says little about discretization or hardening
— on proteins those are not pressing. It does report one reward-hacking mode: over-weighting
the ProteinMPNN sequence term drives designs toward homopolymers. Quality degrades under
over-optimization even where validity cannot.

**Practically:** read it as a design reference, do not adopt it. It is protein-only with no
ligand support, it is JAX where DIDgen is PyTorch, and it wraps precisely the expensive
structure models protocol §1 defers. What *is* worth copying is the interface: build the
objective as a list of weighted loss terms over `(X, Q, E, M)` from the start, so that
swapping a validity mechanism in the §10 study is a config change rather than a rewrite.

### DBMol as the nearest sibling

[DBMol, arXiv:2607.19237](https://arxiv.org/abs/2607.19237) relaxes atom and bond types
onto the simplex with continuous formal charges, optimizes through a frozen Boltz-2 for
50 iterations with clipping + renormalization, then projects once at the end via DeFoG.
It is the "relax everything, project at the end" corner of the table above, run at
expensive scale.

Relevant to this project in three ways, and **not** as a baseline to reproduce:

1. It is evidence the overall recipe works, which de-risks the concept.
2. Its projection stage is one concrete mechanism to include in the §10 comparison.
3. Its representation choice — categorical atoms/bonds on the simplex, continuous charges
   — is essentially the `X/Q/E` choice the protocol makes, arrived at independently for
   the same reason (alignment with an AF3-class ligand interface).

Two caveats carried over from the earlier analysis, still worth knowing: DBMol is not the
strongest method on its own held-out table (it wins on not needing reference ligands, not
on raw score), and no public code was found. Neither matters now that reproducing it is
out of scope.

---

## 3. The contrast line: latent-space optimization — and why this project avoids it

Worth knowing precisely so the thesis can say why it went the other way.

- **ChemVAE** (Gómez-Bombarelli et al., *ACS Cent. Sci.* 2018) — the original continuous
  latent space + gradient/BO on a property predictor.
- **[LIMO, ICML 2022](https://arxiv.org/abs/2206.09010)** — VAE latent → decoder emits an
  intermediate *soft* molecular representation specifically to keep a downstream property
  predictor differentiable. Architecturally the closest relative of a latent version of
  this project.
- **[IDOLpro, Chem. Sci. 2025](https://arxiv.org/abs/2405.11785)** — optimizes diffusion
  latents against differentiable scoring functions for structure-based design.

**Why this project optimizes the graph directly instead.** A latent space applies the
chemical prior implicitly and inseparably: the decoder decides what is valid, so you
cannot isolate or compare validity mechanisms. This project needs the validity mechanism
to be an *explicit, swappable component*. Optimizing `X/Q/E` directly is what makes the
§10 comparison possible at all.

Secondary reason: a latent space adds a second trained model whose failures are
confounded with the optimizer's. The protocol's insistence on one frozen predictor is
the same instinct.

State this as a deliberate design choice with a reason, not an oversight.

---

## 4. The eventual destination: AF3-class ligand interfaces

Protocol §6 and §10 are explicit that the cheap surrogate is a sandbox, and that
successful mechanisms move to a Boltz/AF3-class setting later. Background for that
transfer, deliberately kept short because it is **not** stage-1 work:

- [BoltzDesign1](https://www.biorxiv.org/content/10.1101/2025.04.06.647261v1) — inverts
  Boltz-1 for binder design. Useful engineering precedent: it optimizes pair/distogram
  distributions and skips the structure module to cut memory and avoid gradient
  pathologies.
- [Boltz as a strong baseline for atom-level representation learning](https://arxiv.org/html/2602.13249v1),
  code [`hsjang0/boltz-as-FM`](https://github.com/hsjang0/boltz-as-FM) (MIT) — documents
  practical mechanics of running Boltz on ligands, including with no protein.
- [AF3 input specification](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)
  — AF3 takes ligands as SMILES or CCD codes; the CCD path carries **per-atom element and
  formal charge** and **per-bond connectivity and order**. This is precisely why `X/Q/E`
  is the right sandbox representation: it is the same primitive ligand description, so a
  constraint written against it has somewhere to land.

### The known obstacle on the far side: `ref_pos`

Preserved from the earlier analysis because it was expensive to find and it will matter
later. **It is not a stage-1 concern** — the protocol explicitly excludes 3D, Pairformer
and structure prediction from what "AF3-like" means here.

AF3-class featurizers consume per-atom reference-conformer coordinates (`ref_pos`,
alongside `ref_mask`, `ref_element`, `ref_charge`, `ref_atom_name_chars`,
`ref_space_uid`). `ref_element` and `ref_charge` have clean soft analogues; `ref_pos`
does not, because an RDKit conformer requires a *committed discrete* molecule. Every
faithful AF3 reproduction — Boltz, Protenix, Chai — inherits the same fields, so this is
a specification-level obstacle, not a vendor quirk.

Implication for this project's framing: the transfer claim should be scoped to the
**chemical** ligand variables (`X`, `Q`, `E`), which do transfer cleanly. Do not promise
that a mechanism validated here drops into a co-folding model unchanged — the
conformer-dependent features are an open problem between here and there, and saying so is
a stronger thesis position than eliding it.

---

## 5. The surrogate task: KIBA and DTI regression

[KIBA](https://tdcommons.ai/multi_pred_tasks/dti/) (Tang et al., *JCIM* 2014) integrates
Ki, Kd and IC50 measurements for **kinase** inhibitors into a single score. Distributed
via [Therapeutics Data Commons](https://tdcommons.ai/) (Huang et al., NeurIPS D&B 2021);
roughly 117k drug-target pairs over 2,068 drugs and 229 targets. Standard modelling
baselines are DeepDTA (Öztürk et al., *Bioinformatics* 2018) and descendants, essentially
all of which encode the protein sequence — which this project deliberately does not.

Three properties of this dataset matter for the gate in `02_implementation_plan.md` §5:

**1. A random split is the right choice here, and it is not a weakness.** Random splitting
inflates DTI numbers relative to cold-drug / cold-target splits, which is why DTI papers
avoid it. But this project is not making a generalization claim — it needs a *locally
smooth, differentiable* function in the neighbourhood of known molecules, which is
exactly where a random split trains well. The correct posture: use the standard random
split, and never report a DTI performance claim.

**2. Kinase inhibitors are promiscuous, which directly threatens gate check 7.** Much of
KIBA's variance is explainable by drug-level and target-level *main effects* — a potent
type-I inhibitor scores well against many kinases — rather than by drug×target
interaction. A molecule-only trunk with a multi-target head can fit KIBA respectably by
learning "general kinase affinity" plus small per-target offsets, and in that regime
`∂score/∂X` is nearly identical across targets. That is precisely the failure the
protocol's §9 blocker table anticipates ("all target heads give almost identical
gradients").

> **Recommended diagnostic, beyond what the protocol specifies.** The protocol names
> global-mean and mean-per-target as the trivial baselines. Add a **two-way additive
> baseline** — drug mean + target mean fit on the training split. If the model only
> narrowly beats *that*, it has learned main effects and little interaction, and gate
> check 7 will fail for a data reason. Diagnose it there rather than after a week of
> debugging the head. Pick the two targets for check 7 from the well-observed end of the
> target distribution.

**3. Coverage across the 229 targets is uneven.** Thinly-observed targets have poorly
trained heads. Report the per-target pair-count distribution in
`reports/02_training_run.md`; it is one line of code and it explains most per-target
anomalies you will otherwise chase.

---

## 6. Reward hacking and proxy exploitation

The literature most likely to be needed in practice, because protocol §9 lists
"immediate extreme proxy exploitation" as a blocker.

- [Generative Models Should at Least Be Able to Design Molecules That Dock Well
  (JCIM 2023)](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01355) — the canonical
  demonstration that optimizing a scoring proxy produces molecules that score well and
  are otherwise worthless.
- [In Search of Beautiful Molecules (JCIM 2025)](https://pubs.acs.org/doi/10.1021/acs.jcim.5c01203)
  — catalogues concrete exploitation modes: long flexible aliphatic chains inflating
  contact counts, high-logP lipophilic blobs.
- [A data-driven generative strategy to avoid reward hacking in multi-objective molecular
  design (Nat. Commun. 2025)](https://www.nature.com/articles/s41467-025-57582-3).
- [PoseBusters](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901501/) — physical
  plausibility as a *gate* rather than a score. Relevant later, not now.

### The important reframing

A GNN trained on 2,068 molecules will extrapolate wildly once the optimizer moves `X/E`
off the data manifold, and because the representation carries **no valence constraint**,
the cheapest exploits are chemically impossible graphs: over-saturated atoms, degenerate
atom-type mixtures, fully-connected aromatic blobs, or simply filling every slot up to
`N_max`.

**This is the phenomenon the downstream study measures, not a defect to eliminate.** The
whole point of comparing valence penalties against projection against hardening is to see
which one suppresses it at least cost to the objective.

What stage 1 needs is only that exploitation is not *immediate and total* — that there is
a usable window of 10–30 steps before the score becomes meaningless. That is what
protocol §9's response ("tighten the local relaxed training region, or add simple
out-of-distribution guardrails") is for, and why it forbids the tempting overreaction of
building a better DTI model. A better DTI model would be exploited too, one week later.

Predict the exploitation modes in writing *before* running the first optimization. Being
able to say "we expected these three and observed two" is worth more than a clean result.

---

## 7. Novelty verdict — what to claim

**Not novel, do not claim:**

- Differentiating a GNN property predictor back to its graph input — DIDgen.
- Continuous relaxation of molecular graphs onto the simplex — DBMol, MoFlow.
- Temperature annealing / straight-through hardening of discrete structures — Gumbel,
  Concrete, STE.
- Multi-target regression on KIBA — a standard benchmark with stronger published models.
- **Composite multi-model objectives over one relaxed design variable — Mosaic.** Joint
  optimization against several frozen critics instead of sequential pipeline stages is
  built and open-source on the protein side. Claim the molecular case, not the pattern.
- The frozen predictor itself. It is infrastructure.

**Genuinely open, and the reason the project is worth doing:**

1. **A controlled comparison of graph-validity mechanisms.** Fixed scorer, fixed starting
   molecules, fixed target labels; vary only the mechanism. The literature supplies the
   mechanisms and no comparison. The sharpest framing: composite-objective relaxed
   optimization is established where the design space has **no validity constraint**
   (Mosaic, proteins). What that pattern requires where validity *is* at risk — which
   mechanism, applied during or after, at what cost to the objective — is the open
   question, and it is specific to molecules.
2. **The hardening gap** — `score(relaxed x*) − score(hardened/projected x*)` — as the
   metric that distinguishes them. It quantifies how much of the optimizer's apparent
   gain survives the return to a valid molecule. Rarely reported anywhere.
3. **Whether the answer transfers to an AF3-class ligand interface.** Scoped to the
   chemical variables `X/Q/E`; honest about `ref_pos` (§4).

**Recommended framing:** not *"we built a KIBA predictor"*, but *"which mechanism should
return a relaxed molecular graph to chemical validity, and how much of the optimization
survives it?"*

The predictor is the apparatus. The comparison is the thesis.

---

## 8. Reading order

Before Step 1 (reuse):

1. DIDgen paper, then `ftherrien/inv-design` — specifically the atom construction, the
   near-integer noise augmentation, and the gradient-to-input loop.
2. TDC DTI/KIBA documentation — the split API and the exact field names.

Before Step 3 (architecture):

3. AF3 input documentation, the ligand/CCD section only — it is short, and it justifies
   the `X/Q/E` choice in one paragraph of the thesis.

Before the handoff (§10 work):

4. The Gumbel-Softmax and straight-through papers — the hardening-schedule arm of the
   comparison comes straight from them.
5. DBMol §3–4, for the projection arm.
6. The reward-hacking papers in §6 — read these *before* the first optimization run, not
   after the results look strange.
