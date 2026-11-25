# Project Proposal: Latent-DBMol

## Shared Latent-Space Optimization for Structure-Guided Molecular Design

**Status:** Research proposal  
**Primary objective:** Test whether DBMol’s direct soft-graph optimization can be replaced by optimization in a learned molecular latent space.  
**Initial scope:** Fixed molecular size \(N=25\), Boltz-2 as the only task-specific optimization signal, and a pretrained molecular encoder–decoder where possible.

---

## 1. Executive summary

DBMol optimizes soft atom types, bond types and charges directly through a frozen Boltz-2 model. This provides gradients for molecular design, but the optimized representation can move far outside the space of chemically meaningful molecules.

This project proposes replacing that direct representation with a learned canonical molecular latent space:

\[
G
\xrightarrow{E_{\mathrm{mol}}}
z
\xrightarrow{D_{\mathrm{mol}}}
\hat G.
\]

Here:

- \(G\) is a discrete molecular graph.
- \(E_{\mathrm{mol}}\) maps a molecule into the latent space.
- \(z\) is a continuous latent representation.
- \(D_{\mathrm{mol}}\) decodes \(z\) into a molecule or soft molecular graph.

During target-specific design, Boltz-2 is frozen and only \(z\) is optimized:

\[
z_{k+1}
=
z_k-\eta\nabla_z\mathcal L_{\mathrm{Boltz}}
\left(D_{\mathrm{mol}}(z_k),P\right),
\]

where \(P\) is the target protein and pocket.

The decoder should act as a learned chemical prior. Compared with DBMol, this may:

- keep optimization closer to valid molecular space;
- coordinate atom, bond and scaffold changes;
- reduce proxy exploitation through fractional atoms and bonds;
- reduce the amount of chemical repair needed after optimization;
- provide a foundation for later multi-model objectives;
- eventually support variable molecular size.

The first project milestone is deliberately narrower:

> Determine whether fixed-size latent optimization produces more chemically valid molecules while preserving DBMol’s Boltz-2 and held-out performance.

The full shared-representation system should only be developed if this simpler experiment succeeds.

---

## 2. Background

### 2.1 DBMol

[DBMol](https://arxiv.org/abs/2607.19237) performs target-specific molecular design using:

1. A continuous relaxation of atom types, bond types and charges.
2. Gradient optimization through frozen Boltz-2.
3. Sampling of discrete graphs from the optimized probabilities.
4. Chemical repair using DeFoG.
5. Optional mapping to synthesizable analogues using SynFormer.

For \(N\) atoms, the optimized representation is approximately:

\[
X\in\Delta^{N\times 9},
\]

\[
E\in\Delta^{N\times N\times 5},
\]

\[
q\in\mathbb R^N.
\]

The optimization objective is:

\[
\mathcal L_{\mathrm{DBMol}}
=
\mathcal L_{\mathrm{affinity}}
+w_1\mathcal L_{\mathrm{contact}}
+w_2\mathcal L_{\mathrm{confidence}}.
\]

The main benchmark uses reference-ligand-dependent molecular sizes. The paper also evaluates a fixed setting:

\[
N=25,
\]

which is the median reference-ligand size across its seven LIT-PCBA targets.

### 2.2 Main limitation being addressed

DBMol’s optimization variables are only constrained locally:

- atom probabilities sum to one;
- bond probabilities sum to one;
- charges are clipped.

These constraints do not enforce joint chemical consistency. The optimizer may therefore find favourable Boltz-2 inputs that do not correspond to realistic molecules.

The proposed latent representation should encode correlations learned from real molecules:

- which atoms commonly occur together;
- which bonds are compatible with those atoms;
- common scaffolds and functional groups;
- valid ring structures;
- charge–bond compatibility;
- drug-like molecular statistics.

The proposal is not that the latent representation contains more raw information than DBMol’s graph tensors. It should instead provide a better-structured optimization space.

---

## 3. Research questions

### Primary research question

Does optimization in a learned molecular latent space improve the validity and robustness of DBMol-generated molecules without destroying target-specific binding signals?

### Secondary questions

1. Can a pretrained molecular decoder provide an end-to-end differentiable path from \(z\) to Boltz-2-compatible soft molecular features?
2. Does latent optimization reduce the gap between relaxed Boltz-2 scores and scores obtained after discrete decoding?
3. Does latent regularization reduce reward hacking and off-manifold optimization?
4. Does the optimized latent space transfer better to AlphaFold 3 or docking-based evaluation?
5. Can a shared latent space later be aligned with synthesis and ADMET models?
6. Are relative representations useful for aligning independently trained molecular models?
7. Is a direct molecular bottleneck more reliable than mapping \(z\) directly into Boltz-2’s hidden representation?

---

## 4. Hypotheses

### H1: Improved chemical validity

The decoder’s learned molecular prior will make optimized outputs more chemically valid than independently optimized atom and bond probabilities.

### H2: Better optimization-to-decoding retention

The difference between the relaxed optimization score and the final discrete-molecule score will be smaller:

\[
\Delta_{\mathrm{decode}}
=
\mathcal L_{\mathrm{Boltz}}(G_{\mathrm{decoded}})
-
\mathcal L_{\mathrm{Boltz}}(G_{\mathrm{soft}}).
\]

### H3: Better held-out transfer

Latent optimization will sacrifice some extreme Boltz-2 improvement but preserve or improve AF3 pocket coverage, docking scores and molecular validity.

### H4: Reduced off-distribution behaviour

A latent prior or trust-region penalty will prevent optimization from entering regions poorly supported by the decoder’s training distribution.

### H5: A shared space is possible but requires molecular grounding

Contrastive or relative-representation alignment will only be reliable if all aligned model states remain tied to the same decoded molecule.

---

## 5. Scope

### 5.1 Included in the first iteration

- Fixed molecular size \(N=25\).
- The same atom and bond vocabulary as DBMol where possible.
- A deterministic latent vector \(z\).
- A pretrained molecular encoder–decoder if a suitable one is available.
- Boltz-2 as the only task-specific critic during optimization.
- A latent-prior or trust-region regularizer.
- Discrete decoding and rescoring during evaluation.
- Comparison with DBMol’s fixed-\(N\) formulation.
- One-target pilot followed by the seven-target LIT-PCBA evaluation.

### 5.2 Excluded from the first iteration

- Variable molecular size.
- Joint optimization with SynFormer.
- Joint ADMET optimization.
- Off-target selectivity optimization.
- Full probabilistic optimization over \(q(z)\).
- Fine-tuning Boltz-2.
- Experimental synthesis or wet-lab evaluation.
- Exact end-to-end gradients through discrete reaction generation.
- Direct optimization of explicit 3D coordinates.

These should be considered only after latent optimization has been validated.

---

## 6. Terminology

### Canonical molecular latent space

A continuous space \(Z\) in which each point \(z\) represents a molecule or a local distribution of molecules.

### Molecular encoder

\[
E_{\mathrm{mol}}:G\rightarrow Z.
\]

### Molecular decoder

\[
D_{\mathrm{mol}}:Z\rightarrow G.
\]

For optimization, the decoder should expose soft outputs:

\[
D_{\mathrm{soft}}(z)
=
(X(z),E(z),q(z)).
\]

### Boltz bridge

The module that converts the decoder’s soft molecular output into the early ligand features accepted by the modified Boltz-2 model.

### Model adapter

A learned mapping connecting \(Z\) to another model’s representation space.

### Semantic consistency

The requirement that every adapter interprets \(z\) as representing the same molecule decoded by \(D_{\mathrm{mol}}\).

---

## 7. Proposed architecture

### 7.1 Recommended first implementation

The safest initial architecture keeps a molecular graph as the common bottleneck:

\[
z
\rightarrow
D_{\mathrm{soft}}(z)
\rightarrow
(X,E,q)
\rightarrow
\text{Boltz-2}
\rightarrow
\mathcal L_{\mathrm{Boltz}}.
\]

The same \(z\) is also discretely decoded:

\[
z
\rightarrow
D_{\mathrm{discrete}}(z)
\rightarrow
G.
\]

This gives the complete computational path:

\[
z
\rightarrow
\text{soft molecular graph}
\rightarrow
\text{Boltz-2 early ligand features}
\rightarrow
\text{protein–ligand representations}
\rightarrow
\text{binding loss}.
\]

This approach uses Boltz-2’s existing differentiable soft-embedding modifications from DBMol. It does not initially require learning an independent mapping directly into Boltz-2’s hidden space.

### 7.2 Why not start with a direct hidden-state adapter?

The proposed direct adapter would be:

\[
A_B:Z\rightarrow H_B,
\]

where \(H_B\) is Boltz-2’s earliest dense ligand representation.

However, the relevant Boltz-2 representation is unlikely to be one fixed vector. It may contain:

- per-atom embeddings;
- pairwise atom–atom features;
- bond information;
- charges;
- token metadata;
- possibly stereochemical information.

For fixed \(N\), a direct adapter might need to produce:

\[
H_{\mathrm{atom}}\in\mathbb R^{N\times d},
\]

and

\[
H_{\mathrm{pair}}\in\mathbb R^{N\times N\times d_p}.
\]

This introduces a serious semantic-consistency risk: \(A_B(z)\) might describe a different molecule from \(D_{\mathrm{mol}}(z)\).

The direct adapter should therefore be a later experiment, not the first implementation.

### 7.3 Composed mapping into Boltz-2

The first version still provides a mapping from \(Z\) to Boltz-2’s space, but it is explicitly grounded:

\[
A_B(z)
=
\operatorname{Embed}_{\mathrm{Boltz}}
\left(D_{\mathrm{soft}}(z)\right).
\]

This guarantees that Boltz-2 sees the same molecular representation that the decoder claims \(z\) represents.

---

## 8. Candidate molecular encoder–decoders

No candidate should be assumed to work without an implementation audit.

### 8.1 VFMol

[VFMol](https://pubs.acs.org/doi/10.1021/acs.jcim.5c03005) combines variational latent-space modelling with discrete flow matching.

Code: [DHUDBlab/VFMol](https://github.com/DHUDBlab/VFMol)

Advantages:

- already contains a VAE-style latent representation;
- already combines latent modelling and flow-matching generation;
- conceptually close to the proposed architecture;
- includes property-guided generation.

Unknowns:

- whether the full \(z\rightarrow\) graph sampling path is differentiable;
- whether pre-sampling logits can be exposed;
- whether discrete intermediate states break the gradient path;
- whether its atom vocabulary matches DBMol;
- whether it supports an exact \(N=25\) mode;
- whether reconstruction and validity are sufficient for this task.

VFMol is the conceptually preferred candidate but must pass the differentiability audit.

### 8.2 MoFlow

[MoFlow](https://arxiv.org/abs/2006.10137) learns an invertible mapping between molecular graphs and continuous latent variables.

Code: [calvin-zcx/moflow](https://github.com/calvin-zcx/moflow)

Advantages:

- continuous and approximately invertible graph–latent mapping;
- one-pass decoding;
- easier gradient flow from \(z\) to graph tensors;
- fixed maximum-size graphs fit the first-iteration setting;
- reported exact reconstruction of training molecules.

Disadvantages:

- older model;
- may require post-hoc validity correction;
- uses graph padding and a fixed maximum size;
- may generate lower-quality molecules than newer flow-matching methods;
- vocabulary and preprocessing may need modification.

MoFlow may be the easiest proof-of-concept decoder because differentiability is more important than state-of-the-art sample quality in the first experiment.

### 8.3 Junction Tree VAE

[JT-VAE](https://proceedings.mlr.press/v80/jin18a.html) generates molecules through a scaffold tree followed by molecular assembly.

Code: [wengong-jin/icml18-jtnn](https://github.com/wengong-jin/icml18-jtnn)

Advantages:

- chemically meaningful latent space;
- high validity;
- structured scaffold-level generation;
- widely studied for latent molecular optimization.

Disadvantages:

- autoregressive and discrete decoding;
- difficult to differentiate through the complete decoder;
- not naturally expressed as dense \(N\times N\) soft graph tensors;
- older code and dependencies.

JT-VAE is useful as a generation or black-box latent-optimization baseline, but it is unlikely to be the best first choice for end-to-end Boltz gradients.

### 8.4 DeFoG

[DeFoG](https://arxiv.org/abs/2410.04263) is the graph denoiser used by DBMol.

Code: [manuelmlmadeira/DeFoG](https://github.com/manuelmlmadeira/DeFoG)

DeFoG is not natively a decoder of the form:

\[
z\rightarrow G.
\]

It operates as:

\[
G_t,t\rightarrow G_1.
\]

Using it as the canonical decoder would require:

- adding a latent-conditioning mechanism;
- training or fine-tuning the model;
- defining how \(z\) controls global molecular identity;
- maintaining differentiability through its discrete sampling process.

For that reason, DeFoG should not be the first decoder unless an existing latent-conditioned version becomes available.

### 8.5 Decoder-selection gate

Before implementing Boltz-2 integration, the coding agent must evaluate each candidate against the following requirements:

| Requirement | Required? |
|---|---:|
| Public code and checkpoint | Yes |
| Compatible licence | Yes |
| Continuous latent representation | Yes |
| Fixed-\(N=25\) support | Yes for iteration 1 |
| Differentiable \(z\rightarrow\) soft graph path | Yes |
| Atom and bond logits accessible | Yes |
| Formal-charge representation | Strongly preferred |
| Valid discrete decoding | Yes |
| Reconstruction of input molecules | Yes |
| GPU-compatible PyTorch implementation | Strongly preferred |

If no pretrained model satisfies these requirements, a small fixed-\(N\) graph VAE may need to be trained specifically for the project.

That is the main practical uncertainty in the proposal.

---

## 9. Data

### 9.1 Molecular pretraining data

Start with ZINC-250K because:

- it is used to train DBMol’s DeFoG component;
- it contains drug-like molecules;
- comparison with DBMol is easier;
- preprocessing and molecular vocabulary are already established.

For the first fixed-size model, construct a subset containing exactly 25 heavy atoms:

\[
\mathcal D_{25}
=
\{G\in\mathrm{ZINC250K}:N_{\mathrm{heavy}}(G)=25\}.
\]

If this subset is too small, alternatives are:

1. Use a narrow range and pad to \(N_{\max}=25\).
2. Use an explicit “no atom” category.
3. Train separate decoders for nearby sizes.
4. Pretrain on all sizes but condition the decoder on \(N=25\).

The simplest first option is exact \(N=25\).

### 9.2 Molecular vocabulary

Target vocabulary:

\[
\mathcal X=
\{\mathrm C,\mathrm N,\mathrm O,\mathrm F,\mathrm P,\mathrm S,
\mathrm{Cl},\mathrm{Br},\mathrm I\}.
\]

Bond vocabulary:

\[
\mathcal E=
\{\text{no bond},\text{single},\text{double},
\text{triple},\text{aromatic}\}.
\]

Charge vocabulary:

\[
\mathcal C=\{-1,0,+1\}.
\]

If the pretrained decoder uses a different vocabulary, do not silently discard categories. Either:

- restrict the dataset;
- learn a documented mapping;
- or retrain the output heads.

### 9.3 Evaluation data

Use the same seven-target LIT-PCBA protocol as DBMol.

For the initial controlled comparison:

- use \(N=25\) for every target;
- use the same pocket residue definitions;
- use the same protein sequences and MSAs;
- do not use reference-ligand geometry or interactions;
- use the same candidate-generation budget.

### 9.4 Splitting strategy

For training the molecular latent model:

- use scaffold-based train/validation/test splits;
- ensure exact duplicate removal;
- record canonical SMILES and molecular graph hashes;
- report reconstruction on both random and scaffold-held-out splits.

For later representation alignment:

- use a separate scaffold-held-out alignment test set;
- prevent the adapters from being evaluated only on molecules seen during training.

---

## 10. Soft graph interface

The decoder should expose:

\[
L_X(z)\in\mathbb R^{N\times K_X},
\]

\[
L_E(z)\in\mathbb R^{N\times N\times K_E},
\]

\[
L_C(z)\in\mathbb R^{N\times K_C}.
\]

Convert these logits to soft categories:

\[
X(z)=\operatorname{softmax}(L_X(z)/\tau),
\]

\[
E(z)=\operatorname{softmax}(L_E(z)/\tau),
\]

\[
C(z)=\operatorname{softmax}(L_C(z)/\tau).
\]

The expected formal charge is:

\[
q_i(z)=
\sum_{c\in\{-1,0,+1\}}c\,C_{i,c}(z).
\]

Bond tensors must be symmetric:

\[
E_{ij}=E_{ji}.
\]

The diagonal must always correspond to no bond:

\[
E_{ii,\text{no bond}}=1.
\]

The decoder interface should be:

```python
class MolecularDecoder(Protocol):
    def decode_soft(
        self,
        z: torch.Tensor,
        temperature: float,
    ) -> SoftMolecularGraph:
        ...

    def decode_discrete(
        self,
        z: torch.Tensor,
        strategy: str = "argmax",
        num_samples: int = 1,
    ) -> list[DiscreteMolecularGraph]:
        ...
```

The graph structures should be explicit:

```python
@dataclass
class SoftMolecularGraph:
    atom_probs: torch.Tensor      # [B, N, K_atom]
    bond_probs: torch.Tensor      # [B, N, N, K_bond]
    charge_probs: torch.Tensor    # [B, N, K_charge]
    expected_charges: torch.Tensor  # [B, N]

@dataclass
class DiscreteMolecularGraph:
    atom_types: torch.Tensor
    bond_types: torch.Tensor
    charges: torch.Tensor
    smiles: str | None
    valid: bool
```

---

## 11. Boltz-2 integration

### 11.1 References

- [Boltz-2 paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)
- [Official Boltz repository](https://github.com/jwohlwend/boltz)

### 11.2 Required behaviour

Boltz-2 must remain frozen:

\[
\nabla_{\theta_{\mathrm{Boltz}}}\mathcal L=0.
\]

Gradients must flow through:

\[
z
\rightarrow
D_{\mathrm{soft}}(z)
\rightarrow
\text{Boltz embeddings}
\rightarrow
\mathcal L.
\]

### 11.3 Soft atom embeddings

For a normal discrete atom type:

\[
h_i=W_{\mathrm{atom}}[x_i].
\]

For a soft atom distribution:

\[
h_i=X_iW_{\mathrm{atom}}.
\]

### 11.4 Soft bond embeddings

For a normal discrete bond:

\[
h_{ij}=W_{\mathrm{bond}}[e_{ij}].
\]

For a soft bond distribution:

\[
h_{ij}=E_{ij}W_{\mathrm{bond}}.
\]

### 11.5 Charges

Use expected charges from the decoder:

\[
q_i=
\sum_c c\,C_{i,c}.
\]

### 11.6 Parity requirement

For one-hot atom and bond inputs, the modified Boltz-2 path must reproduce the original discrete path.

Required test:

\[
\left|
f_{\mathrm{Boltz}}(G)
-
f_{\mathrm{Boltz}}(\operatorname{onehot}(G))
\right|
<\epsilon.
\]

This should be checked separately for:

- affinity outputs;
- binder logits;
- confidence outputs;
- distograms;
- final loss.

Do not begin latent optimization until one-hot parity is demonstrated.

### 11.7 Memory optimizations

Follow DBMol where possible:

- freeze all Boltz-2 parameters;
- use a single recycle;
- enable gradient checkpointing;
- stop gradients through unnecessary structure-generation components;
- cache protein MSAs and fixed preprocessing;
- batch latent starting points only when memory permits;
- use mixed precision only after verifying gradient stability.

---

## 12. Optimization objective

### 12.1 Boltz objective

Replicate DBMol’s objective:

\[
\mathcal L_{\mathrm{Boltz}}=
\mathcal L_{\mathrm{affinity}}
+w_1\mathcal L_{\mathrm{contact}}
+w_2\mathcal L_{\mathrm{confidence}}.
\]

Use the same weights as DBMol initially.

### 12.2 Latent trust-region loss

Prevent \(z\) from moving arbitrarily far from its initialization:

\[
\mathcal L_{\mathrm{trust}}
=
\|z-z_0\|_2^2.
\]

This is preferable to only using \(\|z\|^2\), because it defines a local optimization region around a valid starting molecule.

### 12.3 Latent-prior loss

For a standard Gaussian latent prior:

\[
\mathcal L_{\mathrm{prior}}
=
\|z\|_2^2.
\]

If the model provides a learned latent density \(p(z)\), use:

\[
\mathcal L_{\mathrm{prior}}
=
-\log p(z).
\]

### 12.4 Decoder-confidence loss

If the decoder provides likelihood or uncertainty:

\[
\mathcal L_{\mathrm{decoder}}
=
-\log p_{\mathrm{decoder}}\left(D(z)\mid z\right).
\]

Alternatively, penalize high output entropy or ensemble disagreement. This must be used cautiously because forcing low entropy too early can stop exploration.

### 12.5 Initial total objective

\[
\mathcal L(z)=
\mathcal L_{\mathrm{Boltz}}
+\lambda_{\mathrm{trust}}\mathcal L_{\mathrm{trust}}
+\lambda_{\mathrm{prior}}\mathcal L_{\mathrm{prior}}.
\]

Start with only these terms.

Do not add synthesis, ADMET or selectivity losses until this objective has been validated.

---

## 13. Latent initialization

Test at least three initialization strategies.

### 13.1 Prior initialization

\[
z_0\sim\mathcal N(0,I).
\]

Advantages:

- genuinely de novo;
- simple;
- broad exploration.

Risk:

- prior samples may decode unreliably.

### 13.2 Encoded random valid molecule

\[
z_0=E_{\mathrm{mol}}(G_0),
\]

where \(G_0\) is randomly sampled from the molecular training distribution without target-specific information.

Advantages:

- begins at a valid, supported latent point;
- enables trust-region optimization.

Risk:

- introduces scaffold bias from the seed.

### 13.3 Encoded carbon chain

This most closely resembles DBMol, but the decoder may not reconstruct a 25-carbon chain reliably, and such a molecule may be outside its ordinary training distribution.

Use it only as an ablation.

### 13.4 Recommendation

Use random valid ZINC molecules as the primary initialization strategy. Run multiple seeds but separately report:

- one-start performance;
- multi-start performance;
- equal-Boltz-call performance.

---

## 14. Optimization algorithm

Start with Adam rather than copying DBMol’s learning rate.

Suggested sweep:

\[
\eta\in
\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},
10^{-2},3\times10^{-2}\}.
\]

Additional settings:

- gradient clipping;
- 10-, 25- and 50-step runs;
- decoder temperature sweep;
- early stopping based on relaxed-loss saturation;
- trust-region coefficient sweep;
- deterministic seeds.

Pseudocode:

```python
z = torch.nn.Parameter(z0.clone())
optimizer = torch.optim.Adam([z], lr=config.optim.lr)

for step in range(config.optim.num_steps):
    optimizer.zero_grad()

    soft_graph = decoder.decode_soft(
        z,
        temperature=temperature_schedule(step),
    )

    boltz_batch = boltz_bridge.build_batch(
        protein=protein,
        pocket=pocket,
        soft_graph=soft_graph,
    )

    outputs = frozen_boltz(boltz_batch)

    boltz_loss = compute_dbmol_loss(outputs, pocket)
    trust_loss = ((z - z0) ** 2).mean()
    prior_loss = latent_prior_loss(z)

    total_loss = (
        boltz_loss
        + lambda_trust * trust_loss
        + lambda_prior * prior_loss
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([z], max_norm)
    optimizer.step()

    if step % discrete_check_interval == 0:
        evaluate_discrete_checkpoint(z.detach())
```

---

## 15. Discrete checkpoint evaluation

The optimizer may improve the soft representation while worsening the actual decoded molecule. Therefore, periodically perform:

\[
z_k
\rightarrow
G_k=D_{\mathrm{discrete}}(z_k)
\rightarrow
\mathcal L_{\mathrm{Boltz}}(G_k).
\]

Record:

- relaxed Boltz loss;
- discrete Boltz loss;
- molecular validity;
- decoder confidence;
- latent distance from \(z_0\);
- latent prior probability;
- molecular similarity to the initial molecule.

These discrete checks should normally be performed without gradients.

They require extra Boltz calls and must therefore be reported separately when comparing compute budgets.

Possible early-stopping rule:

> Stop when the relaxed loss improves but the best discrete loss has not improved for \(m\) checkpoints.

---

## 16. Stage-one experiments

### Experiment 0: Decoder capability audit

Goal:

> Determine whether a candidate decoder is suitable before integrating Boltz-2.

Tests:

- checkpoint loads;
- encode/decode round trip;
- exact graph reconstruction;
- validity;
- atom and bond vocabulary;
- fixed-\(N\) behaviour;
- soft-logit availability;
- gradient flow from logits to \(z\);
- GPU memory use;
- inference speed.

Deliverable:

```text
reports/decoder_audit_<model>.md
```

Go/no-go conditions:

- end-to-end gradients exist;
- no hidden categorical sampling cuts the gradient path;
- at least acceptable reconstruction and validity;
- output vocabulary can be mapped without major information loss.

### Experiment 1: Decoder-only latent optimization

Optimize simple differentiable molecular properties before involving Boltz-2.

Examples:

- expected molecular weight;
- expected heteroatom count;
- expected bond count;
- expected aromatic-bond fraction.

Purpose:

- confirm gradients behave sensibly;
- visualize latent trajectories;
- detect decoder collapse;
- verify that small latent steps produce controlled graph changes.

### Experiment 2: Boltz one-hot parity

Pass the same molecule through:

1. ordinary discrete Boltz-2;
2. modified one-hot soft Boltz-2.

Compare all required outputs.

Failure here blocks the rest of the project.

### Experiment 3: Gradient sanity

For a valid latent point \(z_0\):

1. compute \(\nabla_z\mathcal L\);
2. take a very small step;
3. verify that the relaxed Boltz loss decreases;
4. decode the updated latent point;
5. check whether the discrete loss changes in the expected direction.

Also compare automatic gradients with finite differences along random latent directions.

### Experiment 4: Single-target pilot

Select one LIT-PCBA target with relatively stable DBMol performance.

Run:

- unconditional decoder sampling;
- random latent search;
- latent gradient optimization without prior;
- latent gradient optimization with prior;
- DBMol direct soft-graph optimization.

Use a reduced candidate count for initial debugging.

### Experiment 5: Full fixed-\(N\) benchmark

Run on all seven DBMol targets with:

\[
N=25.
\]

Use the same:

- pocket definitions;
- optimization-step budget;
- Boltz-2 configuration;
- candidate count;
- validity filtering;
- final evaluation protocol.

---

## 17. Baselines

The following baselines are required.

### B0: Unconditional molecular decoder

Sample molecules without Boltz-guided optimization.

### B1: Random latent search

Sample the same number of latent points as the number of evaluated optimization states.

This tests whether gradients outperform simple search.

### B2: DBMol fixed-\(N\)

Original soft-graph optimization with \(N=25\).

### B3: Latent optimization without regularization

\[
\mathcal L=\mathcal L_{\mathrm{Boltz}}.
\]

### B4: Latent optimization with trust region

\[
\mathcal L=
\mathcal L_{\mathrm{Boltz}}
+\lambda_{\mathrm{trust}}
\|z-z_0\|^2.
\]

### B5: Latent optimization with prior and trust region

Full proposed first-stage system.

### Optional B6: Zeroth-order latent optimization

Use evolutionary search or Bayesian optimization over \(z\).

This determines whether differentiating through the decoder is actually beneficial.

---

## 18. Evaluation metrics

### 18.1 Decoder metrics

- exact reconstruction;
- atom accuracy;
- bond accuracy;
- charge accuracy;
- validity;
- uniqueness;
- novelty;
- scaffold diversity;
- latent interpolation validity;
- local latent smoothness.

### 18.2 Optimization-retention metrics

Define:

\[
\Delta_{\mathrm{decode}}
=
\mathcal L_{\mathrm{discrete}}
-
\mathcal L_{\mathrm{soft}}.
\]

Report:

- mean and median \(\Delta_{\mathrm{decode}}\);
- correlation between soft and discrete scores;
- fraction of trajectories where both improve;
- fraction where the soft score improves but the discrete score worsens;
- best discrete score along the trajectory;
- final discrete score.

This is one of the project’s most important evaluations.

### 18.3 Chemical metrics

- RDKit validity;
- connectedness;
- valence validity;
- aromaticity consistency;
- charge validity;
- molecular-weight distribution;
- logP;
- QED;
- synthetic accessibility score;
- fingerprint diversity;
- scaffold diversity;
- novelty relative to training data.

### 18.4 Target-specific metrics

Use the DBMol protocol where possible:

- Boltz-2 Success;
- Boltz predicted affinity;
- Boltz binder probability or logit;
- Boltz pocket coverage;
- ipTM and relevant confidence values.

### 18.5 Held-out metrics

Where available:

- AF3 pocket coverage;
- AF3 Success;
- AF3 interface predicted aligned error;
- AF3 target-pocket distance;
- AutoDock Vina;
- GNINA;
- PoseBusters validity.

[PoseBusters](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901501/) can detect clashes and physically implausible ligand geometries.

### 18.6 Synthesis metrics

Not optimized in iteration one, but evaluate afterward:

- SynFormer route success;
- AiZynthFinder route success;
- route length;
- building-block availability;
- estimated synthesis cost.

### 18.7 Computational metrics

- Boltz forward/backward calls;
- wall-clock time;
- peak GPU memory;
- decoder time;
- candidate generation time;
- valid molecules per GPU-hour.

---

## 19. Fair comparison protocol

All comparisons must specify whether they are matched by:

- number of Boltz calls;
- wall-clock time;
- number of decoded candidates;
- number of valid evaluated candidates.

Primary comparison:

> Equal number of Boltz forward/backward calls.

Secondary comparison:

> Equal total wall-clock budget.

For the full DBMol-style protocol:

- 50 optimization steps;
- 2,048 generated candidates;
- invalid molecules removed;
- 100 valid molecules randomly selected;
- no Boltz-based selection of the final 100.

Do not compare one DBMol trajectory with many latent restarts without reporting the additional compute.

---

## 20. Provisional success criteria

These thresholds should be fixed before running the final experiments.

Suggested criteria:

1. End-to-end latent gradients pass numerical checks.
2. One-hot Boltz parity is within defined numerical tolerance.
3. The correlation between soft and discrete Boltz scores is meaningfully positive.
4. Latent optimization produces a higher valid yield than direct soft-graph optimization.
5. Boltz-2 Success is no more than 0.05 below DBMol, or is better.
6. Held-out AF3 pocket coverage is not more than 0.03 below DBMol.
7. Vina score is not more than 0.5 kcal/mol worse on average.
8. Molecular diversity remains within 0.05 of DBMol.
9. Optimization does not collapse to one scaffold family.
10. Improvements reproduce across multiple seeds.

These numbers are provisional and should be revised based on pilot variance.

---

## 21. Direct Boltz-space adapter: second-stage experiment

If the graph-grounded approach succeeds, test a direct adapter:

\[
A_B:Z\rightarrow H_B.
\]

### 21.1 Extracting Boltz representations

Use forward hooks to identify the earliest ligand-only dense representation before protein–ligand interaction layers.

The selected state should:

- depend on atom types, bonds and charges;
- not yet depend strongly on the target protein;
- preserve atom-level structure;
- have stable dimensions for fixed \(N\).

Do not assume that a single pooled vector is sufficient.

### 21.2 Training pairs

For each molecule \(G\):

\[
z=E_{\mathrm{mol}}(G),
\]

\[
H_B(G)=
\operatorname{ExtractBoltzEarlyState}(G).
\]

Train:

\[
A_B(z)\approx H_B(G).
\]

### 21.3 Alignment losses

Regression:

\[
\mathcal L_{\mathrm{MSE}}
=
\|A_B(z)-H_B(G)\|^2.
\]

Cosine loss:

\[
\mathcal L_{\mathrm{cos}}
=
1-\cos(A_B(z),H_B(G)).
\]

Contrastive loss:

\[
\mathcal L_{\mathrm{InfoNCE}}
=
-\log
\frac{
\exp(\operatorname{sim}(A_B(z_i),H_B(G_i))/\tau)
}{
\sum_j
\exp(\operatorname{sim}(A_B(z_i),H_B(G_j))/\tau)
}.
\]

Use hard negatives with similar scaffolds, molecular weights and fingerprints. Randomly unrelated negatives may make the task too easy.

### 21.4 Semantic consistency

Add:

\[
\mathcal L_{\mathrm{consistency}}
=
\left\|
A_B(z)
-
H_B(D_{\mathrm{discrete}}(z))
\right\|^2.
\]

Also require downstream-output agreement:

\[
f_{\mathrm{Boltz}}(A_B(z),P)
\approx
f_{\mathrm{Boltz}}(D(z),P).
\]

### 21.5 Adapter go/no-go test

The adapter must reproduce Boltz outputs on scaffold-held-out molecules.

If representation similarity is high but affinity, confidence or contact outputs differ substantially, the adapter is not useful.

---

## 22. Relative representations

Reference:

[Relative Representations Enable Zero-Shot Latent Space Communication](https://arxiv.org/abs/2209.15430)

### 22.1 Anchor set

Choose a diverse set of anchor molecules:

\[
\mathcal A=
\{A_1,\ldots,A_K\}.
\]

Possible selection method:

1. Compute molecular fingerprints.
2. Cluster by scaffold and chemical similarity.
3. Choose cluster medoids.
4. Balance molecular size, charge and functional groups.

Suggested initial sizes:

\[
K\in\{128,256,512,1024\}.
\]

### 22.2 Relative coordinates

For model \(i\):

\[
r_i(G)=
\begin{bmatrix}
\operatorname{sim}(h_i(G),h_i(A_1))\\
\vdots\\
\operatorname{sim}(h_i(G),h_i(A_K))
\end{bmatrix}.
\]

Train or evaluate:

\[
r_{\mathrm{mol}}(G)
\approx
r_{\mathrm{Boltz}}(G)
\approx
r_{\mathrm{syn}}(G).
\]

### 22.3 Recommended use

Initially use relative representations as:

- an alignment regularizer;
- a diagnostic of cross-model geometric agreement;
- a way to detect whether models organize molecules similarly.

Do not initially make the relative representation the decoder’s only input. Anchor similarities may not preserve enough information for exact molecular reconstruction.

### 22.4 Main risk

Boltz-2, a molecular decoder and a synthesis model may organize molecular similarity differently:

- Boltz-2 emphasizes protein interaction;
- a molecular decoder emphasizes structural plausibility;
- a synthesis model emphasizes reaction accessibility.

Relative geometry may therefore disagree for legitimate reasons.

---

## 23. CLIP-style multimodel alignment

Reference:

[CLIP](https://arxiv.org/abs/2103.00020)

For every molecule \(G\), obtain:

\[
z_{\mathrm{mol}}=g_{\mathrm{mol}}(E_{\mathrm{mol}}(G)),
\]

\[
z_B=g_B(H_B(G)),
\]

\[
z_S=g_S(H_S(G)).
\]

Train positive pairs from the same molecule and negative pairs from different molecules.

Combined loss:

\[
\mathcal L_{\mathrm{CLIP}}
=
\mathcal L_{\mathrm{mol},B}
+
\mathcal L_{\mathrm{mol},S}
+
\mathcal L_{B,S}.
\]

Contrastive alignment alone is insufficient because it preserves broad similarity rather than exact molecular identity.

It should be combined with:

\[
\mathcal L_{\mathrm{representation}}
=
\lambda_{\mathrm{CLIP}}\mathcal L_{\mathrm{CLIP}}
+\lambda_{\mathrm{relative}}\mathcal L_{\mathrm{relative}}
+\lambda_{\mathrm{reconstruction}}\mathcal L_{\mathrm{reconstruction}}
+\lambda_{\mathrm{cycle}}\mathcal L_{\mathrm{cycle}}.
\]

---

## 24. Probabilistic latent optimization

Only implement after deterministic \(z\) succeeds.

Represent the design distribution as:

\[
q(z)=
\mathcal N(\mu,\operatorname{diag}(\sigma^2)).
\]

Use the VAE reparameterization trick:

\[
z=\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
\]

Reference:

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

Optimize:

\[
\begin{aligned}
\mathcal L(\mu,\sigma)
={}&
\mathbb E_{z\sim q}
\left[
\mathcal L_{\mathrm{Boltz}}(D(z),P)
\right]\\
&+
\beta D_{\mathrm{KL}}
\left(
q(z)\,\|\,p(z)
\right).
\end{aligned}
\]

Advantages:

- naturally generates multiple candidates;
- models uncertainty;
- can preserve exploration;
- may reduce dependence on one local optimum.

Risks:

- extra Boltz calls per step;
- variance collapse;
- noisy gradients;
- one Gaussian may not represent several distinct scaffold families.

A mixture model or population of Gaussian latents may eventually be preferable.

---

## 25. Later multi-objective extensions

Once the canonical molecular space is validated, additional critics can be connected.

Possible objective:

\[
\begin{aligned}
\mathcal L(z)=\;&
\lambda_B\mathcal L_{\mathrm{Boltz}}
+\lambda_S\mathcal L_{\mathrm{synthesis}}\\
&+\lambda_A\mathcal L_{\mathrm{ADMET}}
+\lambda_O\mathcal L_{\mathrm{off-target}}\\
&+\lambda_U\mathcal L_{\mathrm{uncertainty}}
+\lambda_P\mathcal L_{\mathrm{prior}}.
\end{aligned}
\]

Potential systems:

- [SynFormer](https://www.pnas.org/doi/10.1073/pnas.2415665122) for synthesis.
- [ADMET-AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11226862/) for ADMET properties.
- [DrugCLIP](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8bd31288ad8e9a31d519fdeede7ee47d-Abstract-Conference.html) for rapid protein–molecule compatibility.
- [ConPLex](https://www.pnas.org/doi/10.1073/pnas.2220778120) for drug–target interaction prediction.
- [DynamicBind](https://www.nature.com/articles/s41467-024-45461-2) for ligand-dependent protein conformations.
- [RxnFlow](https://arxiv.org/abs/2410.04542) for reaction-based molecular generation.
- [SynCoGen](https://arxiv.org/abs/2507.11818) for joint synthesis-aware graph and 3D generation.

For practical optimization, constraints may be preferable to a weighted sum:

\[
\min_z\mathcal L_{\mathrm{binding}}(z)
\]

subject to:

\[
\begin{cases}
\text{synthesis probability}(z)\geq\tau_s,\\
\text{ADMET}_k(z)\in[a_k,b_k],\\
\text{off-target score}(z)\leq\tau_o,\\
\text{uncertainty}(z)\leq\tau_u.
\end{cases}
\]

---

## 26. Immediate technical risks

### Risk 1: No suitable pretrained decoder

The required combination may not exist:

- continuous latent representation;
- high-validity molecular decoding;
- accessible soft graph logits;
- exact end-to-end gradients;
- compatible atom and bond vocabulary;
- fixed-\(N\) control;
- public checkpoints.

Mitigation:

- audit VFMol and MoFlow first;
- train a small fixed-\(N\) graph VAE if necessary;
- do not begin full Boltz integration before this is settled.

### Risk 2: Discrete operations cut gradients

The decoder may contain:

- categorical sampling;
- argmax;
- autoregressive branching;
- discrete stopping decisions;
- reaction selection.

Mitigation:

- expose pre-sampling logits;
- use expected soft graph states;
- use Gumbel–Softmax only if necessary;
- consider straight-through gradients as an ablation;
- use zeroth-order latent optimization if exact gradients are unavailable.

Reference:

[Gumbel–Softmax](https://openreview.net/forum?id=rkE3y85ee)

### Risk 3: Soft outputs remain chemically meaningless

Even a learned decoder can output soft mixtures during optimization.

Mitigation:

- keep \(z\) near encoded valid molecules;
- use a latent trust region;
- periodically decode and rescore;
- monitor decoder uncertainty;
- compare against discrete checkpoints.

### Risk 4: Optimizer exploits the decoder

Gradient descent may find latent vectors where:

- decoder logits appear favourable;
- the discrete molecule is invalid;
- the decoder behaves unpredictably;
- model adapters disagree.

Mitigation:

- latent-density regularization;
- adversarial latent validation;
- decode-and-rescore checkpoints;
- scaffold-held-out testing;
- compare several decoder checkpoints;
- stop when relaxed/discrete disagreement grows.

### Risk 5: Boltz early state is not a simple molecular embedding

The early representation may be distributed across atom, token and pair tensors.

Mitigation:

- use the graph-grounded route first;
- inspect the Boltz source with forward hooks;
- document all injected tensors;
- require exact parity when injecting ordinary molecule states.

### Risk 6: Atom ordering mismatch

Molecular graphs are permutation-invariant, but tensor implementations are not.

Mitigation:

- use canonical atom ordering;
- preserve atom maps through preprocessing;
- verify the ordering used by Boltz;
- compare graphs using typed graph isomorphism;
- explicitly test aromatic systems and symmetric molecules.

### Risk 7: Boltz proxy exploitation remains

The latent decoder may reduce chemical invalidity without solving reward hacking.

Mitigation:

- use AF3, Vina and physical checks for evaluation;
- report proxy and held-out metrics separately;
- avoid selecting final molecules using held-out evaluators;
- later add uncertainty or multiple critics.

### Risk 8: Computational cost

Fifty Boltz backward passes take approximately 30 minutes in the DBMol setup.

Mitigation:

- single-target pilot;
- reduced steps during debugging;
- cached MSAs;
- batched starts when possible;
- checkpointing;
- one recycle;
- strict experiment tracking;
- surrogate pre-optimization only as a later study.

---

## 27. Hard no-go conditions

Pause or redesign the project if any of the following occurs:

1. No candidate decoder exposes a differentiable \(z\rightarrow\) soft graph path.
2. A custom decoder cannot reach acceptable reconstruction and validity.
3. Modified Boltz-2 fails one-hot parity.
4. Gradients through the decoder are zero, unstable or unrelated to finite differences.
5. Soft and discrete Boltz scores are effectively uncorrelated near valid latent points.
6. Latent optimization consistently produces invalid molecules despite strong trust-region constraints.
7. The method improves only relaxed Boltz scores and consistently worsens discrete or held-out scores.
8. The direct Boltz adapter cannot reproduce standard Boltz outputs on held-out molecules.
9. Required GPU memory exceeds available hardware with no practical reduction strategy.
10. The chosen decoder licence or checkpoint cannot legally be used.

These outcomes would not prove that latent optimization is impossible, but they would invalidate the selected implementation.

---

## 28. Repository structure

```text
latent-dbmol/
├── README.md
├── pyproject.toml
├── environment.yml
├── configs/
│   ├── decoder/
│   │   ├── vfmol.yaml
│   │   ├── moflow.yaml
│   │   └── custom_graph_vae.yaml
│   ├── boltz/
│   │   └── boltz2_dbmol.yaml
│   ├── experiments/
│   │   ├── decoder_audit.yaml
│   │   ├── single_target.yaml
│   │   └── lit_pcba_7target.yaml
│   └── targets/
├── src/
│   └── latent_dbmol/
│       ├── data/
│       │   ├── zinc250k.py
│       │   ├── lit_pcba.py
│       │   ├── graph.py
│       │   └── vocab.py
│       ├── decoders/
│       │   ├── base.py
│       │   ├── vfmol_adapter.py
│       │   ├── moflow_adapter.py
│       │   └── custom_graph_vae.py
│       ├── boltz_bridge/
│       │   ├── batch_builder.py
│       │   ├── soft_embeddings.py
│       │   ├── hooks.py
│       │   └── losses.py
│       ├── alignment/
│       │   ├── adapters.py
│       │   ├── contrastive.py
│       │   ├── relative.py
│       │   └── anchors.py
│       ├── optimization/
│       │   ├── latent_optimizer.py
│       │   ├── schedules.py
│       │   ├── regularizers.py
│       │   └── checkpoints.py
│       ├── evaluation/
│       │   ├── molecular.py
│       │   ├── boltz.py
│       │   ├── docking.py
│       │   ├── synthesis.py
│       │   └── metrics.py
│       └── utils/
│           ├── logging.py
│           ├── reproducibility.py
│           └── serialization.py
├── scripts/
│   ├── prepare_zinc25.py
│   ├── audit_decoder.py
│   ├── test_boltz_parity.py
│   ├── optimize_single_target.py
│   ├── run_lit_pcba.py
│   └── evaluate_candidates.py
├── tests/
│   ├── test_decoder_shapes.py
│   ├── test_graph_symmetry.py
│   ├── test_gradient_flow.py
│   ├── test_boltz_parity.py
│   ├── test_atom_ordering.py
│   └── test_roundtrip.py
└── reports/
```

---

## 29. Example configuration

```yaml
experiment:
  name: latent_dbmol_single_target
  seed: 42

molecule:
  num_atoms: 25
  atom_vocab:
    - C
    - N
    - O
    - F
    - P
    - S
    - Cl
    - Br
    - I
  bond_vocab:
    - none
    - single
    - double
    - triple
    - aromatic
  charge_vocab: [-1, 0, 1]

decoder:
  name: moflow
  checkpoint: null
  freeze: true
  soft_temperature:
    initial: 1.0
    final: 0.25
  discrete_strategy: sample

boltz:
  checkpoint: null
  num_recycles: 1
  gradient_checkpointing: true
  freeze: true
  detach_structure_module: true

objective:
  affinity_weight: 1.0
  contact_weight: 2.5
  confidence_weight: 2.5
  trust_weight: 0.01
  prior_weight: 0.001

optimization:
  optimizer: adam
  learning_rate: 0.003
  num_steps: 50
  max_grad_norm: 1.0
  discrete_check_interval: 5
  num_initializations: 1

sampling:
  num_candidates: 2048
  num_valid_for_evaluation: 100
  final_selection: random_valid

evaluation:
  run_boltz: true
  run_af3: false
  run_vina: true
  run_synformer: false
```

---

## 30. Testing requirements

### Unit tests

1. All probability tensors have correct shapes.
2. Atom and bond probabilities sum to one.
3. Bond probabilities are symmetric.
4. No self-bonds are produced.
5. Expected charges are correct.
6. Gradients reach \(z\).
7. Decoder parameters remain frozen.
8. Boltz parameters remain frozen.
9. One-hot inputs reproduce standard Boltz behaviour.
10. Canonical atom ordering is preserved.

### Integration tests

1. Encode a molecule.
2. Decode its soft graph.
3. Build a Boltz batch.
4. Run forward and backward.
5. Confirm nonzero finite gradients in \(z\).
6. Take one optimizer step.
7. Decode a discrete molecule.
8. Run RDKit validity checks.
9. Save and reload the complete optimization state.

### Reproducibility tests

- fixed seeds;
- deterministic preprocessing;
- configuration saved with each run;
- Git commit recorded;
- dependency versions recorded;
- target files hashed;
- decoder and Boltz checkpoints hashed.

---

## 31. Logging

Every optimization step should log:

```text
step
total_loss
affinity_loss
contact_loss
confidence_loss
trust_loss
prior_loss
latent_norm
distance_from_initial_latent
gradient_norm
decoder_temperature
atom_entropy
bond_entropy
soft_validity_proxy
discrete_validity
soft_boltz_score
discrete_boltz_score
soft_discrete_gap
wall_clock_time
peak_gpu_memory
```

Save:

- \(z_k\);
- decoder logits;
- discrete molecules;
- SMILES;
- Boltz outputs;
- optimization configuration;
- random seeds.

---

## 32. Recommended development order

### Milestone 1: Environment and decoder audit

- Reproduce candidate decoder sampling.
- Reproduce published or repository metrics.
- Test differentiable decoding.
- Select the decoder.

### Milestone 2: Fixed-\(N\) latent model

- Build the \(N=25\) molecular dataset.
- Train or adapt the latent model if necessary.
- Validate reconstruction, validity and latent smoothness.

### Milestone 3: Boltz bridge

- Implement soft atom and bond embeddings.
- Reproduce one-hot Boltz outputs.
- Implement DBMol losses.
- Verify gradients.

### Milestone 4: Toy latent optimization

- Optimize simple graph properties.
- Optimize a single Boltz output for one molecule.
- Measure the soft–discrete gap.

### Milestone 5: Single-target experiment

- Compare against unconditional generation, random latent search and DBMol.
- Tune learning rate, temperature and trust-region strength.
- Freeze settings before the full benchmark.

### Milestone 6: Seven-target benchmark

- Run the fixed-\(N=25\) LIT-PCBA protocol.
- Perform held-out evaluation.
- Report multiple seeds and compute.

### Milestone 7: Representation alignment

- Extract early Boltz ligand states.
- Train direct adapters.
- Add relative representations.
- Compare direct hidden-state mapping with the graph-grounded route.

### Milestone 8: Multi-objective extension

- Add synthesis, ADMET and selectivity critics individually.
- Measure gradient conflict.
- Move to constrained or Pareto optimization if necessary.

---

## 33. Expected contributions if successful

1. A molecular latent-space alternative to DBMol’s direct graph relaxation.
2. A controlled comparison between explicit soft-graph and learned-manifold optimization.
3. A method for passing structure-predictor gradients through a molecular decoder.
4. Measurements of the relaxed-to-discrete objective gap.
5. A framework for aligning molecular generative and predictive models.
6. Evidence about whether relative representations work across molecular models with different objectives.
7. A foundation for future variable-size and multi-objective molecular design.

---

## 34. Expected negative result if unsuccessful

A scientifically useful negative conclusion would be:

> Current molecular latent decoders do not preserve sufficiently accurate and differentiable molecule identity for structure-predictor-guided optimization. Direct soft-graph optimization, despite its chemical weaknesses, offers a more reliable gradient interface.

The project should therefore preserve failed trajectories, gradient diagnostics and decoder-disagreement measurements rather than reporting only successful examples.

---

## 35. Information required before implementation

The proposal can be developed without additional information. Actual implementation would benefit from:

1. The exact DBMol paper version intended for replication.
2. A DBMol source-code repository or author implementation, if available.
3. The exact Boltz-2 commit and checkpoint to use.
4. Access to the seven LIT-PCBA target files, pocket residues and MSAs.
5. Available GPU type, memory and total compute budget.
6. Whether AlphaFold 3 evaluation is accessible.
7. Whether SynFormer and AiZynthFinder should be installed for final evaluation.
8. A preferred first decoder, if one has already been selected.
9. Whether the objective is a short student project, thesis project or full research paper.

If no DBMol code is available, the first coding task must be a careful reimplementation of its differentiable Boltz-2 modifications and fixed-\(N\) baseline.

---

## 36. Primary references

### Core method

- [DBMol: Design of High-Affinity, Target-Specific Small Molecules through Structure Prediction Models](https://arxiv.org/abs/2607.19237)
- [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)
- [Official Boltz repository](https://github.com/jwohlwend/boltz)
- [DeFoG: Discrete Flow Matching for Graph Generation](https://arxiv.org/abs/2410.04263)
- [Official DeFoG repository](https://github.com/manuelmlmadeira/DeFoG)
- [SynFormer: Generative AI for Navigating Synthesizable Chemical Space](https://www.pnas.org/doi/10.1073/pnas.2415665122)
- [Official SynFormer repository](https://github.com/wenhao-gao/synformer)

### Candidate latent models

- [VFMol: A Discrete Flow Matching Variational Autoencoder for Molecular Graph Generation](https://pubs.acs.org/doi/10.1021/acs.jcim.5c03005)
- [Official VFMol repository](https://github.com/DHUDBlab/VFMol)
- [MoFlow: An Invertible Flow Model for Generating Molecular Graphs](https://arxiv.org/abs/2006.10137)
- [Official MoFlow repository](https://github.com/calvin-zcx/moflow)
- [Junction Tree Variational Autoencoder](https://proceedings.mlr.press/v80/jin18a.html)
- [Official JT-VAE repository](https://github.com/wengong-jin/icml18-jtnn)

### Representation alignment

- [Relative Representations Enable Zero-Shot Latent Space Communication](https://arxiv.org/abs/2209.15430)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Categorical Reparameterization with Gumbel–Softmax](https://openreview.net/forum?id=rkE3y85ee)

### Possible extensions

- [RxnFlow](https://arxiv.org/abs/2410.04542)
- [SynCoGen](https://arxiv.org/abs/2507.11818)
- [ADMET-AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11226862/)
- [DrugCLIP](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8bd31288ad8e9a31d519fdeede7ee47d-Abstract-Conference.html)
- [ConPLex](https://www.pnas.org/doi/10.1073/pnas.2220778120)
- [DynamicBind](https://www.nature.com/articles/s41467-024-45461-2)
- [PoseBusters](https://pmc.ncbi.nlm.nih.gov/articles/PMC10901501/)
- [GuacaMol molecular-generation benchmark](https://arxiv.org/abs/1811.09621)

---

## 37. Final recommendation

Begin with the smallest experiment that tests the central claim:

\[
\boxed{
z
\rightarrow
\text{differentiable soft graph decoder}
\rightarrow
\text{frozen Boltz-2}
}
\]

Use:

- fixed \(N=25\);
- one molecular decoder;
- one target;
- one latent vector;
- Boltz-2 plus a latent trust region;
- periodic discrete decoding and rescoring.

Do not begin with direct hidden-state alignment, relative representations, SynFormer or multi-objective optimization.

The project should proceed to those stages only if the initial experiment shows that latent optimization:

1. preserves a useful gradient;
2. produces valid decoded molecules;
3. retains Boltz-guided improvements after decoding;
4. performs competitively with DBMol under equal compute.

The most important unresolved question is not whether a shared latent space can be mathematically defined. It can.

The critical question is:

> Can a pretrained molecular decoder provide a sufficiently accurate, smooth and differentiable connection between that latent space and the exact molecular features used by Boltz-2?

That should be tested before investing in the full multimodel architecture.