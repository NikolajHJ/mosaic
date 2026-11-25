# Glossary

Reference for [`project_desc.pdf`](project_desc.pdf) and the other docs in this folder.
Written for a reader who knows machine learning and does not know chemistry or biology.
Assumes GNNs, probability simplices and autograd; explains everything else.

Created 2026-09-24.

---

## Three terms that will mislead you first

**Hallucination** — *not* the LLM sense. In protein and molecule design it means: take a
trained predictor, freeze it, and optimize its **input** by gradient descent until its
**output** looks the way you want. You are "hallucinating" a molecule the model believes
has the property. It comes from AlphaFold2-based protein design. The project title
("Small-Molecule Hallucination") uses this sense. Nothing to do with a model making
things up.

**Projection** — overloaded in these docs, in two genuinely different senses:
1. *Mathematical*: mapping a point back onto a feasible set — e.g. clipping and
   renormalizing a vector back onto the probability simplex. Cheap, closed-form.
2. *DBMol's sense*: the entire stage that converts an optimized relaxed graph back into a
   valid discrete molecule, using a trained generative denoiser. A whole pipeline stage.

When the docs say "valence penalties versus continuous projection", sense 1 is meant.
When they say "the projection stage", sense 2.

**Target** — *not* a regression label. In drug discovery a **target** is the protein you
want a drug to act on. "229 targets" means 229 proteins. The regression label is the
binding measurement between a drug and a target.

---

## Molecules and their representation

**Molecule** — a set of atoms joined by bonds. For our purposes: a graph, where nodes are
atoms and edges are bonds.

**Molecular graph** — the graph abstraction of a molecule. Node features: which chemical
element, what formal charge. Edge features: bond type. This project's `(X, Q, E, M)` is
exactly this in dense tensor form.

**Atom / element** — a node. Its element (carbon, nitrogen, oxygen, sulfur, fluorine,
chlorine…) is a categorical label, which is why `X` is a one-hot / simplex vector.

**Heavy atom** — any atom that is not hydrogen. Molecular graphs almost always contain
only heavy atoms, because hydrogens are numerous, boring, and inferable.

**Implicit hydrogen** — hydrogens not represented as nodes. Their count is *derived* from
element + formal charge + bond orders. **Why it matters here:** this derivation is exactly
the "valence coupling" the protocol tells you to break. If hydrogen count is an input
feature, `X`, `Q` and `E` are no longer independent variables.

**Bond / bond order** — an edge, with a type: single, double, triple, or aromatic. This
project adds a fifth category, `NO_BOND`, so that "no edge here" is representable in a
dense `N × N` tensor. This makes the graph fully connected in tensor form with absence as
an explicit category.

**Aromatic** — a special bond type in rings with delocalized electrons (benzene being the
canonical case). Not a single or double bond; genuinely its own category. Chemically very
common in drugs.

**Kekulization** — rewriting aromatic rings as alternating single and double bonds. RDKit
can give you either representation. **Why it matters here:** this project keeps `AROMATIC`
as its own bond category, so you must **not** kekulize during featurization or the
category vanishes.

**Formal charge** — an integer bookkeeping charge on an atom (usually −1, 0 or +1 in drug
molecules). Not a physical partial charge; a valence-electron accounting convention. This
is the `Q` variable.

**Valence** — how many bonds an atom can form. Carbon 4, nitrogen 3 (or 4 when
positively charged), oxygen 2, and so on. Violating valence makes a molecule that cannot
exist. **Why it matters here:** because `X`, `Q` and `E` are independent optimization
variables, nothing structurally prevents the optimizer from putting six bonds on a carbon.
That is not a bug — it is the phenomenon the valence-penalty-versus-projection study
measures.

**Chemical validity** — whether a graph corresponds to a molecule that could exist: legal
valences, consistent charges, parseable by RDKit. The central quantity this project's
downstream research trades against the objective.

**SMILES** — a text string encoding a molecular graph. `CCO` is ethanol. The standard
interchange format; KIBA ships drugs as SMILES.

**Canonical SMILES** — a deterministic SMILES for a given molecule, so the same molecule
always produces the same string. Used for deduplication.

**Scaffold** — the core ring system of a molecule with side chains stripped (the
Bemis–Murcko scaffold). A *scaffold split* puts whole scaffolds in train or test, which is
a much harder generalization test than a random split.

**Functional group** — a recurring substructure with characteristic chemistry (hydroxyl,
carbonyl, amine…). Vocabulary you will see in papers; not used in this project's
representation.

---

## Three dimensions, and why this project stays in two

**Conformer / conformation** — a specific 3D arrangement of a molecule's atoms. One
molecular graph has many conformers, because single bonds rotate. RDKit generates them
(ETKDG plus a force-field cleanup).

**Why it matters here:** a conformer requires a *committed, discrete* molecule — you
cannot generate 3D coordinates for a probability distribution over graphs. This is the
`ref_pos` obstacle described in [`01_related_work.md`](01_related_work.md) §4, and the
main reason AF3-class models are hard to optimize through with soft inputs. This project
avoids the problem entirely by staying 2D.

**Chirality / stereochemistry** — 3D "handedness". Two molecules can have identical graphs
and be non-superimposable mirror images, with very different biological effects. Not
representable in `(X, Q, E)` and deliberately out of scope here.

**Docking** — placing a ligand into a protein's binding pocket in 3D and scoring the pose
with a physics-based or empirical function. Fast and famously exploitable.

**Co-folding model** — a structure predictor that predicts the 3D structure of a
*complex* (protein + ligand) jointly, rather than a protein alone. AlphaFold 3, Boltz and
Chai are co-folding models. DBMol optimizes through one; this project does not.

---

## Proteins, targets and binding

**Protein** — a chain of amino acids that folds into a 3D shape. Its function follows
from that shape.

**Amino acid** — the monomer units proteins are built from. Twenty standard ones, so a
protein sequence is a string over a 20-letter alphabet. **Any** such string is a
synthesizable polypeptide — which is why protein design has no chemical-validity problem,
unlike molecular graph design. See [`01_related_work.md`](01_related_work.md) §2.

**Residue** — one amino acid within a protein chain. "Pocket residues" means the specific
amino acids lining a binding site.

**Minibinder** — a small designed protein (under ~100 amino acids) that binds a chosen
target. A major application of protein design methods; Escalante's Mosaic targets it.

**Protein sequence vs. structure** — the sequence is the 1D string of amino acids; the
structure is its 3D fold. Predicting the second from the first is what AlphaFold does.
**This project uses neither** — targets are just integer IDs.

**Ligand** — any molecule that binds to a protein. In drug design, the small molecule you
are designing.

**Binding pocket / binding site** — the cavity on a protein surface where a ligand binds.

**Binding affinity** — how strongly a ligand binds its target. The quantity the whole
field is trying to optimize.

**Kinase** — an enzyme that attaches a phosphate group to another molecule, using ATP as
the phosphate source. Humans have roughly 518 of them. They are a major drug target class,
especially in cancer.

**Kinase inhibitor** — a drug that blocks a kinase. Most bind in the ATP pocket.

**ATP-competitive / promiscuity** — because the ATP-binding pocket is structurally similar
across the kinase family, an inhibitor that fits one often fits many. This is
**promiscuity**, and it is the single most important property of the KIBA dataset for this
project: it means much of the score is explained by "is this a potent kinase binder at
all" rather than by the specific drug–target pairing. See
[`01_related_work.md`](01_related_work.md) §5 for why this threatens gate check 7.

**GPCR** — G protein-coupled receptor, a large family of membrane receptors. More than 30%
of approved drugs act on them. Mentioned in DBMol's motivation; not used here.

**De novo design** — designing molecules from scratch, rather than modifying known
compounds or screening an existing library.

**Virtual screening** — computationally scoring a large library of *existing* molecules to
decide which to test experimentally. The opposite approach to de novo design.

**Structure-based drug design (SBDD)** — design that uses the target's 3D structure.
Requires the structure; this project does not do it.

---

## Measuring binding

All of these are experimental measurements of how tightly a ligand binds. You will see
them as regression labels.

**Kd (dissociation constant)** — equilibrium constant for the ligand coming off the
protein, in concentration units (nM, µM). **Lower means tighter binding.**

**Ki (inhibition constant)** — the analogous constant for an inhibitor. Lower is more
potent.

**IC50** — the concentration that produces 50% inhibition. Lower is more potent. Depends
on assay conditions, so it transfers between experiments less well than Ki.

**pKd / pKi** — `−log10` of the molar value. **Higher means tighter.** Models usually work
in this log space because the raw values span many orders of magnitude.

**KIBA score** — Kinase Inhibitor BioActivity. A single score that statistically
integrates Ki, Kd and IC50 measurements, adjusting for the systematic offsets between
assay types (Tang et al., *JCIM* 2014). It exists because the three measurement types are
not directly comparable.

> **Check the sign before you optimize anything.** Different releases and papers apply
> different transformations to the KIBA score, so "higher is better" is *not* safe to
> assume. Plot the distribution when you load it and confirm which direction means
> stronger binding. Getting this backwards silently inverts your entire optimization and
> everything downstream will look like a subtle modelling failure.

**DTI (drug–target interaction)** — the general task of predicting a binding measurement
for a (drug, protein) pair. KIBA is a DTI regression dataset.

---

## Molecular property metrics

**logP** — the octanol/water partition coefficient on a log scale. Measures
**lipophilicity**: how much a molecule prefers fat to water. Important for whether a drug
can cross membranes. High logP is a classic reward-hacking signature, because greasy blobs
score well in many binding proxies.

**Crippen logP** — a fast estimate of logP by summing per-atom contributions (Wildman &
Crippen, 1999). Computable in microseconds by RDKit. **Why it matters here:** this is the
property DIDgen's released predictor was trained on, hence "CrippenNet". This project
replaces it with KIBA.

**QED** — Quantitative Estimate of Drug-likeness, in [0, 1]. Combines several property
desirability curves into one number.

**SA score** — Synthetic Accessibility, roughly 1 (easy to make) to 10 (hard). A heuristic
from fragment frequency plus complexity penalties. Not differentiable.

**Molecular weight (MW)** — sum of atomic masses. Correlates with almost everything,
including many binding scores, which makes it a confound.

**Heteroatom** — any non-carbon, non-hydrogen atom (N, O, S, halogens…).

**Aliphatic** — non-aromatic carbon chains. "Long aliphatic chains" is another documented
reward-hacking signature.

**ECFP / Morgan fingerprint** — a bit vector built by hashing each atom's neighbourhood up
to a radius. ECFP4 means radius 2. The standard molecular feature vector before GNNs.

**Tanimoto similarity** — Jaccard similarity on two fingerprints. The default molecular
similarity measure. Molecular **diversity** is usually reported as mean pairwise Tanimoto
*distance* within a generated set.

**Retrosynthesis / synthesizability** — whether a molecule can actually be made in a lab,
and by what route. A molecule can be perfectly valid and still unmakeable.

---

## Datasets and benchmarks

**KIBA** — the dataset this project uses. Roughly 117k drug–target pairs over 2,068 drugs
and 229 kinase targets, labelled with the KIBA score. Distributed through TDC.

**TDC (Therapeutics Data Commons)** — a Python package and benchmark collection for
therapeutic ML. Provides KIBA with a standard split via `from tdc.multi_pred import DTI`.

**Davis** — the other common kinase DTI dataset, labelled with Kd. Not used here.

**BindingDB / ChEMBL** — large public repositories of binding measurements. Harder to use
directly because assay harmonization and cleaning are substantial work — which is exactly
why the protocol chose KIBA instead.

**ZINC / ZINC-250K** — a large library of purchasable compounds; the 250k subset is a
standard molecular generation benchmark. Appears in the superseded plan and in DeFoG's
training; **not used in this project.**

**LIT-PCBA** — a de-biased virtual screening benchmark with experimentally confirmed
actives and inactives per target (Tran-Nguyen et al., *JCIM* 2020). DBMol's benchmark;
**not used in this project.**

**PDB (Protein Data Bank)** — the public archive of experimentally determined 3D
structures.

**CCD (Chemical Component Dictionary)** — the PDB's standard dictionary of small
molecules, giving per-atom element and formal charge plus per-bond connectivity and order.
**Why it matters here:** this is the ligand description AlphaFold 3 accepts, and it is
what "AF3-like" means in protocol §6 — the `(X, Q, E)` variables are the CCD's variables.

---

## Models, tools and code

**DIDgen** — the name the project protocol uses for the method in *"Using GNN property
predictors as molecule generators"* (Nature Communications, 2025). The idea: a GNN trained
only to *predict* a property becomes a *generator* when you differentiate it with respect
to its input. Code at [`ftherrien/inv-design`](https://github.com/ftherrien/inv-design).
This project's direct ancestor and code scaffold.

**CrippenNet** — as used in the protocol, DIDgen's trained GNN predicting Crippen logP.
This project keeps the architecture style and replaces the head and task with KIBA.
Confirm both names against the paper during Step 1 — they are the protocol's shorthand.

**RDKit** — the standard open-source cheminformatics library. Parses SMILES, computes
properties, generates conformers. The only chemistry dependency in this project, used only
during featurization — **never inside the forward pass.**

**DBMol** — the closest published relative of this project (arXiv:2607.19237). Relaxes a
molecular graph onto the simplex, optimizes it through a frozen Boltz-2, then projects
back to validity with DeFoG. See [`01_related_work.md`](01_related_work.md) §2.

**Boltz-1 / Boltz-2** — open co-folding structure predictors. Boltz-2 also predicts
binding affinity. About 30 seconds per backward pass, which is why this project uses a
cheap surrogate instead.

**AlphaFold 3 (AF3)** — DeepMind's structure predictor for complexes, including
protein–ligand. Weights are access-restricted. Relevant here only for its **ligand input
specification**, which motivates the `(X, Q, E)` choice.

**Chai-1 / Protenix** — other open AF3-class co-folding models. Mentioned as alternatives;
not used.

**Mosaic** (`escalante-bio/mosaic`) — MIT-licensed, JAX. Composite-objective **protein**
design: optimizes a relaxed protein sequence (a simplex over the 20 amino acids per
position) against a weighted sum of losses from ~15 frozen models at once. The closest
existing implementation of the "one relaxed variable, many frozen critics, optimized
jointly" pattern. Protein-only, no ligand support. A design reference here, not a
dependency — see [`01_related_work.md`](01_related_work.md) §2.
Not to be confused with **MosaicML / Composer** (Databricks Mosaic AI), an unrelated
PyTorch training-acceleration library that this project does not need.

**ProteinMPNN** — an inverse-folding model: given a 3D backbone, predict amino acid
sequences that would fold into it. Used inside Mosaic as a loss term; over-weighting it
drives designs toward homopolymers, a documented reward-hacking mode.

**DeFoG** — Discrete Flow matching for Graph generation (Qin et al., ICML 2025). A
generative model over discrete graphs. DBMol uses it as its projection stage: sample a
noisy discrete graph from the optimized relaxed one, then denoise from late time
`t = 0.8`. Not used in this project's stage 1.

**SynFormer** — maps a molecule to a structurally similar molecule that is reachable by
known chemical reactions. DBMol's optional synthesizability post-processing step.

**AiZynthFinder** — open-source retrosynthesis planner; used to check whether a route
exists.

**AutoDock Vina / GNINA** — docking programs. Vina scores are in kcal/mol where **more
negative is better**.

**PoseBusters** — a test suite checking whether a predicted 3D pose is physically
plausible (bond lengths, angles, steric clashes). A validity gate, not a score.

**MoFlow / JT-VAE / DiGress / ChemVAE / LIMO / IDOLpro** — molecular generative and
latent-optimization models referenced for contrast in
[`01_related_work.md`](01_related_work.md). None are used.

**MSA (multiple sequence alignment)** — aligning evolutionarily related protein sequences;
AlphaFold-family models consume them as input. Expensive to compute. **Not needed here**,
which removes a large hidden cost from the superseded plan.

---

## Structure-prediction confidence metrics

You will meet these reading DBMol. None are computed in this project.

**Distogram** — the predicted probability distribution over pairwise distances, binned.
AlphaFold-family models output one. DBMol's contact loss reads it.

**pTM** — predicted TM-score, a self-assessed structural accuracy estimate in [0, 1].

**ipTM** — the interface version of pTM, scoring how well the *interaction between* chains
is predicted. Higher is better. DBMol's confidence loss uses it.

**PAE / iPAE** — Predicted Aligned Error, in Ångströms; lower is better. `iPAE` restricts
it to the interface.

**BCov (BindCov)** — pocket coverage: how much of the intended binding pocket the
generated ligand actually occupies, per AF3. Higher is better. DBMol's headline held-out
metric.

**Boltz2 Success / AF3 Success** — per-target success rates combining affinity and
confidence thresholds. Defined by DBMol, not standard.

---

## Optimization vocabulary

**Inverse design** — given a desired property value, find an input achieving it, rather
than predicting the property from the input. The whole premise of DIDgen and this project.

**Continuous relaxation** — replacing discrete variables with continuous ones so gradients
exist. Here: one-hot atom and bond categories become points on the probability simplex.

**Probability simplex** — for `K` categories, the set of non-negative vectors summing to
1. A one-hot vector is a vertex; a relaxed category is an interior point.

**Relaxed / soft vs. hard / discrete** — "soft" means simplex-valued (a mixture of
categories); "hard" means one-hot (a real molecule). The core tension of the project is
that the optimizer lives in soft space and chemistry lives in hard space.

**Hardening** — driving a relaxed representation back toward one-hot, usually by annealing
a temperature during optimization.

**Straight-through estimator (STE)** — use the discrete value in the forward pass and
pretend it was continuous in the backward pass. A biased but practical gradient.

**Gumbel-Softmax / Concrete distribution** — a differentiable relaxation of sampling from
a categorical distribution, with a temperature knob that interpolates between uniform and
one-hot.

**Discrete flow matching / CTMC** — the machinery behind DeFoG. Generative modelling over
categorical states, where denoising is a continuous-time Markov chain over discrete
states rather than a diffusion over continuous ones.

**Denoiser** — a model trained to map corrupted data back to clean data. Used as a
"projection" operator when you want a learned notion of validity.

**Surrogate / proxy / critic / oracle** — the differentiable model that stands in for the
real objective. Used near-interchangeably in these docs. The "oracle" framing carries an
implication of ground truth that a KIBA GNN does not deserve; prefer *surrogate* or
*proxy*.

**Reward hacking / proxy exploitation** — the optimizer finds inputs that score
excellently on the proxy and are worthless on the true objective. Expected here, and
studied rather than merely avoided. See [`01_related_work.md`](01_related_work.md) §6.

---

## Terms specific to this project

**`X` / `Q` / `E` / `M`** — the four ligand tensors. `X` `[N, K_atom]` atom identity
(one-hot or simplex), `Q` `[N]` formal charge as a continuous scalar, `E` `[N, N, 5]`
bond categories, `M` `[N]` padding mask.

**`N_max`** — the fixed number of atom slots in the dense representation, chosen once from
an atom-count audit over KIBA's drugs. Molecules larger than it are dropped.

**`NO_ATOM`** — an extra category in `X` marking an empty slot. Its presence makes
molecule **size** a differentiable variable rather than a fixed hyperparameter. See
[`02_implementation_plan.md`](02_implementation_plan.md) Step 2.

**Occupancy (`occ`)** — `1 − X[:, NO_ATOM]`, the continuous degree to which a slot holds
an atom. `occ.sum()` is the differentiable molecule size.

**Single-path invariant** — the project's central design rule: one forward pass, with no
`argmax`, no integer casts, no `nn.Embedding`, no RDKit, and no branching on whether an
input is discrete. Embeddings are matrix products so that one-hot and simplex inputs flow
through identical code.

**The freeze** — the moment the predictor passes its acceptance checks and its checkpoint
stops changing. Everything downstream uses that one file, so that method comparisons are
not confounded by a moving scorer.

**Hardening gap** — `score(relaxed x*) − score(hardened/projected x*)`. How much of the
optimizer's apparent gain survives the return to a valid discrete molecule. The natural
primary metric for the downstream study.

**Handoff interface** — `objective = model(X_soft, Q, E_soft)[target_id]`. The single
entry point every optimization experiment calls.
