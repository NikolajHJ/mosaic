# Molecular graph VAE research handoff

**Evidence through 19 November 2026.** This is a working reference for an agent picking up the molecular VAE in DBMol_exploratory or integrating its decoder into Mosaic. It follows the project from its first autoencoders to the 60-million-molecule run, including experiments that failed or stayed exploratory. For a group meeting, use the [two-page illustrated briefing](molecular-vae-research-journey.pdf). For decisions about the code, follow the linked configs, experiment records and saved outputs. Some raw artifacts live in the workspace's ignored `outputs/` directory or on DTU and may be absent from a fresh clone.

This copy is published in Mosaic. The commands and experiment paths below use the root of the DBMol_exploratory research checkout. DBMol citations jump to a [source index](#dbmol-source-index) with the original file and section. The frozen epoch-45 weights are available from the [Mosaic model card](../../models/dbmol-vae/README.md).

## Where the project stands

We set out to optimize a small molecule by moving one continuous vector. The decoder turns that vector into probabilities over a variable-size molecular graph. A frozen protein scorer can then differentiate a binding or contact objective through the decoder to the vector. An encoded starting ligand and a fresh Gaussian draw give two very different starting conditions. We ultimately want both to yield chemically meaningful edits, including changes in size, with improvements that survive fresh scoring of the resulting discrete molecule.

| If you are working on… | Start here | Why |
| --- | --- | --- |
| A latent molecule-design baseline | Frozen **epoch 45** | Extensive independent generation, perturbation and native-scoring results. It has also been packaged for Mosaic. |
| The effect of scaling training | **Epoch 46** and the saved **partial epoch 47** | The same graph vocabulary and matched evaluation panels expose opposite trends for reconstruction and unconditional generation. |
| The 50-atom failure | The **47.68M-row checkpoint**, epoch-46 reference and the fixed validation panel | The loss change starts at this exact boundary. Using the completed epoch-46 model as the student would answer a prevention question instead of measuring recovery. |
| The smaller VAE | **435,400-parameter V100 benchmark** | Architecture, training code and measured costs exist. The benchmark discarded temporary weights. A full-run quality comparison is still open. |
| Whether the representation helps search | The matched **direct graph / latent** and **fresh scoring** experiments | A changing latent, a valid graph and a better relaxed objective are intermediate results. The useful outcome is a distinct, valid hard graph with a reproducible fresh improvement per unit of scoring budget. |

Epoch 45 reached **99.87% exact reconstruction** on 10,000 held-out molecules and **75.41% raw argmax validity** on 10,000 Gaussian draws. The 60M run improved held-out PubChem exact reconstruction from **395 to 480 of 512** between epoch 46 and partial epoch 47, while valid prior argmax graphs fell from **190 to 6 of 256**. A checkpoint can be useful for encoding known molecules and poor at unconditional initialization. This is the most consequential distinction in the branch. [10k evaluation](#dbmol-source-001) · [Large-corpus evidence](#dbmol-source-002) · [Search review](#dbmol-source-003)

**Jump to a question**

- [What do the graph, latent and gradient represent?](#representation-and-model)
- [Which measurements answer which questions?](#reading-the-metrics)
- [What did we learn from the first models and the chemistry continuations?](#early-experiments)
- [Which chemical failures received a gradient, and which search worked?](#how-the-chemistry-training-objective-evolved)
- [How did atom order, interpolation and dataset membership affect the experiments?](#ordering-interpolation-and-data)
- [Why did the 60M run fill all 50 slots?](#the-epoch-47-failure)
- [What exactly is the recovery loss and how can we tell if it works?](#the-recovery-continuation)
- [Is the compact VAE a better option?](#the-compact-vae)
- [What happened when we actually optimized molecules?](#latent-molecule-design)
- [Which artifacts, commands and comparisons should a new agent use?](#working-from-the-saved-state)

## Representation and model

The schema is a heavy-atom 2D graph with a maximum of 50 slots. Its element categories in saved order are empty, C, N, O, F, P, S, Cl, Br and I. Formal charges are −2, −1, 0, +1 and +2. Bond categories are none, single, double, triple and aromatic. The schema uses implicit hydrogens and drops stereochemistry in the canonical molecular identity. The latent therefore cannot specify an enantiomer or a 3D pose. A scorer that needs coordinates must obtain them from a separate declared source. Read category order from the checkpoint spec when mapping into another model. [GraphSpec and graph tensors](#dbmol-source-004) · [Encoding, hard validity and identity](#dbmol-source-005)

**There are three useful views of the decoder output.** Confusing them has direct consequences for atom count, bond probability and gradients.

| View | Shapes for one molecule | Used for |
| --- | --- | --- |
| Raw logits (`DecoderOutput`) | Atom `[50,10]`, charge `[50,5]`, bond `[50,50,5]` | Cross-entropy, stable occupancy logits, conditional bond probabilities. The shared bond view averages both orientations before softmax. |
| Differentiable marginals (`RelaxedGraph`) | Same category axes, with normalized probabilities | Passing continuous features to a scorer. Slot occupancy is one minus its empty probability. Off-diagonal conditional bond probabilities are gated by the product of both endpoint occupancies. Missing mass goes to no bond. |
| Discrete graph (`HardGraphBatch`) | One atom and charge label per slot and one bond label per pair | Exact reconstruction, chemical validation, detached repair targets and fresh scoring. The standard argmax path takes conditional bond argmax and applies the hard occupied-pair mask. |

Thus a mostly empty slot can have a plausible **conditional** bond type while its **marginal** chance of a bond remains low. Passing the marginal into an adapter and multiplying by both endpoint occupancies again changes the objective. Conversely, using the conditional prediction as a real edge to an absent atom changes the represented graph. The hardening rule also matters for ambiguous slots. The most likely individual category can be empty even when the *sum* of probabilities of all real elements exceeds 0.5. Report any alternative occupancy threshold explicitly. [Conversion and hardening](#dbmol-source-004) · [Categorical sampling](#dbmol-source-006)

### The large triangle VAE

```text
hard molecular graph, in the encoder's expected slot order
    -> 4 bond-aware encoder message blocks, atom width 96
    -> flatten 50 x 96 -> 320 -> 2 residual 320-wide updates
    -> 256 means and 256 log variances
    -> sample z for training, or use the posterior mean to reconstruct

z, 256 coordinates
    -> direct projection to 50 x 96 atom states
    -> initialize symmetric 50 x 50 x 48 pair states
    -> 4 separately parameterized residual refinement rounds
       pair update + gated triangle paths + pair-aware atom attention
       soft atom and charge feedback into subsequent computations
    -> atom / charge / symmetric bond logits
    -> relaxed graph for gradients, discrete graph for chemical evaluation
```

The encoder accounts for **2,173,216** parameters and the decoder **1,768,644**, for **3,941,860** total. The production architecture uses a direct latent-to-slot projection. Earlier global trunks, global bond-context vectors and wider hidden layers describe earlier comparisons. Each large-model round has four attention heads, a 96-wide atom update, a pair MLP with 96-wide hidden layers and a 24-wide triangle bottleneck. A pair state is a feature vector for a possible relationship. Its width 48 is independent of the number of atom slots and the five eventual bond labels. [Architecture and confirmed counts](#dbmol-source-007) · [Encoder](#dbmol-source-008) · [Refinement and heads](#dbmol-source-009)

For pair `(i,j)`, the triangle update uses learned projections of `(i,k)` and `(k,j)` through intermediate slots `k`. It gates by occupancy, excludes self-paths, normalizes by eligible intermediate occupancy and symmetrizes the result. The implementation contracts pair tensors directly. It can express that two candidate bonds meeting at a third atom inform another pair. This supplies context before the final categorical choices. Separate draws from those heads can still combine into an invalid graph. [TrianglePairUpdate](#dbmol-source-010)

During VAE training a sampled encoding is `z = mu + exp(logvar / 2) * epsilon`. Reconstruction gradients can reach both encoder and decoder. The standard KL is summed over latent dimensions and averaged across molecules. In the large real-molecule branch, cross-entropy supervises every atom slot with reduced padding weight, charge over occupied atoms, and bonds over occupied upper-triangular pairs. The effective size penalty comes from the checkpoint/config for the experiment. The partial-47 run used zero explicit real size-loss weight even though the shared loss class has a nonzero default. During a design run that optimizes one latent, the decoder and scorer remain frozen and only that vector is stepped. [Shared model](#dbmol-source-011) · [Reconstruction reductions and KL](#dbmol-source-012) · [Epoch-47 objective audit](#dbmol-source-013)

## Reading the metrics

| Metric or event | How to read it |
| --- | --- |
| Exact reconstruction | Compare the graph decoded from a known molecule with the original under the declared graph identity. The main held-out tables typically compare canonical, stereo-stripped identities after posterior-mean argmax. Compact graph-mode evaluation additionally recognizes verified label-preserving slot permutations. |
| Valid reconstruction | A known molecule was encoded and its output satisfies the chemistry checks. A different valid molecule can pass. Always report exact reconstruction separately. |
| Unconditional raw validity | Draw a fresh latent from the declared prior and decode. Count every attempted latent, including empty, disconnected and unsanitizable outputs. State the decoding law and whether there are multiple categorical draws per latent. |
| Argmax versus categorical generation | Argmax picks the most likely label at each decision and handles padding. Categorical decoding draws from the distributions and samples each undirected bond once. Different validity on the same latents is expected. Phase-specific temperatures, rejection sampling or repairing afterward change the question. |
| Raw versus strict teacher validity | Native inspection checks graph labels, RDKit sanitization, a single component and radical status. Strict teacher eligibility additionally checks serialized SMILES and representability through the project's round trip. The first failing stage provides a diagnostic category. |
| Repair coverage or cost | Coverage is the fraction of attempted candidates with the declared repair outcome. Keep already-valid, repaired, censored and errored counts separate. A finite search cost is a cost to a witnessed repair in that budget. A missing witness leaves the true nearest valid graph unknown. |
| Diversity | Record distinct valid canonical identities relative to all attempts, and scaffold diversity when available. Hundreds of sanitized copies of a small family can look strong in a validity-only summary. |
| Size and occupancy | For a raw hard graph count occupied slots. For the relaxed graph sum the slot occupancy probabilities. Report the distribution, median and fraction hitting 50. A mean can hide a minority stuck at the ceiling. Counts after a fixed-size selection or repair describe that procedure. |
| Scorer improvement | A relaxed score tests the differentiable surrogate. A fresh score on the discrete candidate tests transfer through hardening and any geometry rebuild. Predicted contact, affinity and confidence have different units and directions. None measures experimental activity. |

These quantities also have different denominators. A training window of 1,000 updates at 16 fresh priors per update contains 16,000 varying draws. A fixed independent panel might have only 128 shared latents. The first can flag trouble quickly, while the second supports paired comparisons. Similarly, a study sampling four hard graphs from each latent has more graphs than independent latent draws. Save denominators, seeds, membership, failure categories, temperature and use of repair next to each claimed result. [Checkpoint evaluator](#dbmol-source-014) · [Native graph inspection](#dbmol-source-005) · [Strict classification](#dbmol-source-015) · [Graded metrics](#dbmol-source-016)

## Early experiments

### What the first experiments selected for

The early ZINC experiments compared a deterministic graph autoencoder (AE) with a variational autoencoder (VAE). An AE assigns a code to each input graph. A VAE predicts a Gaussian mean and log variance for each input. During training the sampled code is the mean plus Gaussian noise scaled by the predicted standard deviation. Both learn categorical graph reconstruction. The VAE adds a KL penalty against a standard Gaussian prior. This supplies a simple distribution to sample during generation. Whether that prior reaches parts of the decoder that produce valid, diverse, editable molecules is an empirical question. [Training losses](#dbmol-source-012) · [Historical architecture comparison](#dbmol-source-017)

The ZINC split contained 215,000 training, 7,347 validation and 7,347 test molecules after filtering unsupported representations and duplicates. Selection took place on validation before evaluation on the random held-out split. One training seed and different stopping epochs limit comparisons. The pair-refinement VAE reconstructed **7,103/7,347** held-out molecules exactly, compared with **7,064/7,347** for its AE counterpart. Yet only **18/1,024** standard-normal draws from that VAE decoded to valid molecules. Across VAE variants with results the corresponding range was **8–18/1,024**. Drawing several categorical outputs at a code obtained by encoding a real molecule often reproduced that molecule. Conditional decoding around known inputs and generating from a new random vector should therefore have separate columns in any results table. [ZINC split, checkpoint selection and comparisons](#dbmol-source-017) · [Fixed-code sampling protocol](#dbmol-source-018)

| Early decoder experiment | What the evidence supports |
| --- | --- |
| Residual and pair refinements | The residual VAE reconstructed 6,587/7,347 test molecules exactly. Pair refinement increased that count to 7,103 on its own validation-selected checkpoint. Each saw the same split. |
| Separate global bond context | A path from the molecular latent to the bond calculations bypassed the atom states. The wide AE reached 7,169/7,347 exact. Its place in the study was as a strong reconstruction comparator. |
| Additional depth and stability | The original deep-pair VAE failed in epoch 7. A separately trained stabilized deep-pair variant reached 7,159/7,347 exact. At equal epoch 60, the stabilized VAE reconstructed 6,920 exactly versus 5,313 for the wide VAE. These families have different parameter counts. |
| Continue the wide VAE | Restoring the model, optimizer and random streams allowed the 120-epoch campaign to reach 180 epochs. Across the two validation-selected checkpoints, exact test reconstruction increased from 6,890 to 6,948. The latter produced 8/1,024 valid prior draws. More steps helped reconstruction while generation remained poor. |

The full table names checkpoint roles separately. For example, the epoch-86 stabilized VAE won its validation-reconstruction comparison. An epoch-89 stabilized checkpoint selected for a different role became the parent of later decoder-only work. Substituting one changes the experiment even when architecture names match. [Six-family results](#dbmol-source-017) · [Wide continuation config](#dbmol-source-019) · [Stabilized family config](#dbmol-source-020)

There was a reason to keep a VAE alongside better deterministic reconstruction baselines. An early 128-dimensional AE/VAE study found an **11.51 percentage-point** advantage for VAE intermediate validity on 223 paths where both models produced valid endpoints, with three intermediate attempts per path. Its anchor-clustered interval was 7.69–15.41 points. The same study tested a density estimate in latent space. On the VAE, optimizing with that estimate gave **49.65%** valid decodes versus **70.70%** for the local trust-region control. Being in a region of high empirical embedding density was an unreliable substitute for measuring graph quality in this setup. Neither result establishes semantic interpolation, stronger binding, or transfer to the later 256-dimensional representation. [Paired latent study](#dbmol-source-021) · [Cross-study interpretation](#dbmol-source-017)

### Why we introduced a random training branch

The MOSES triangle VAE started with joint encoder/decoder training on **1,688,539** supported training molecules, plus 20,000 each for validation and test. Its initial objective was reconstruction plus KL with zero explicit size-loss weight. Gaussian posterior noise during reconstruction is distinct from drawing new vectors independently from the standard Gaussian. The latter appeared in evaluation during this phase. At epoch 25, **4,088/4,096** posterior-mean argmax reconstructions were chemically valid. Only **47/4,096** raw prior argmax graphs and **73/16,384** raw prior categorical draws were valid. The categorical observations came from four draws at each prior point. These results motivated explicit training around the prior. Simultaneous differences in data, architecture and training prevent attributing this trajectory to triangle updates alone. [MOSES corpus, original objective and epoch-25 measurements](#dbmol-source-007)

## How the chemistry training objective evolved

### A successful proxy exposed a sulfur shortcut

In a separate decoder-only ZINC campaign, the stabilized epoch-89 encoder remained frozen. Dense chemistry supervision used posterior codes from real molecules and fresh Gaussian vectors. On 4,096 shared prior points the original decoder produced 64 valid argmax graphs. After 100 decoder-training epochs it produced **840/4,096**. Categorical validity rose from 124 to 2,743 of 16,384 draws and exact held-out reconstruction rose from 7,160 to 7,182 of 7,347. Meanwhile expected sulfur count per random graph rose from **0.391 to 17.58**, expected carbon fell from **16.48 to 3.68**, and expected heavy-atom count stayed near 22. These are soft population expectations over all draws. The model found a chemically unrepresentative way to improve validity. [Matched decoder evaluation](#dbmol-source-022)

The maximum-valence penalty explains an incentive in the objective. For a neutral atom with incident bond order five, its excess above a carbon cap of four is one, while its excess above a sulfur cap of six is zero. The optimizer can reduce this term merely by shifting element probability toward sulfur. The calculation establishes a direction favored by that component. Multiple terms share a decoder, so it does not establish what fraction of the measured drift came from that direction. Later neighbor-aware rules gave a second caution. In a matched local pilot from a common 823/1,024 valid start, a control retained 734 valid after ten updates while the added neighbor term retained only 611. Fewer local valence errors coincided with more disconnected graphs. The group kept that pilot out of the production recipe. [Sulfur mechanism audit](#dbmol-source-023) · [Neighbor-loss pilot](#dbmol-source-024)

### A repair witness gives a gradient after the search

The triangle chemistry continuation started from epoch 35. For a student decode, the search took its argmax graph and held occupied slots, elements and formal charges fixed. It explored bond edits using detached decoder probabilities to prioritize candidates. A successful witness had to pass the configured graph, connectedness, chemistry and representation checks. Differentiating a categorical loss that increases the student’s probability of that witness requires no derivative through the search or chemistry toolkit. The initial target loss included **all** atom slots plus occupied charges and unordered occupied bond pairs. Thus a search that only *edits* bonds can still send a direct supervised gradient to atom logits. That distinction becomes important again at epoch 47. [Search and objective specification](#dbmol-source-025) · [Repair target and loss implementation](#dbmol-source-026)

The objective combined real reconstruction and KL, real and synthetic repair terms, distributional element and charge penalties, and occupancy guidance on fresh random vectors from the frozen epoch-35 decoder at the same coordinates. A population composition penalty can limit widespread drift while leaving individual molecules compositionally distinct. Treating the dataset average as a mandatory formula for each small molecule would answer a different question. The campaign benchmarked 1 real to 3 random examples and alternative execution settings. The eventual frozen-reference validation from epoch 36 used **256 real and 512 prior examples**. Preserve coefficients and reduction rules separately from these counts. In particular, averaging a per-branch loss differs from pooling all graph decisions across both branches. [Weights and denominators](#dbmol-source-027) · [Actual reference-validation settings](#dbmol-source-028)

The original teacher returned no repair target for already-valid graphs or for graphs without a witness inside the budget. It recorded those outcomes separately. The former is a valid raw output. The latter is censored search with unknown distance to the nearest valid graph. A subsequent experimental branch escalated bounded search and considered partial-progress signals for censored graphs. Keep raw validity, witness coverage among eligible graphs and the loss denominator separate when reading a study. In one saved epoch-36-branch training row, 15 of 330 eligible priors were censored, or 15 of 512 scheduled priors. Neither percentage is the chemical failure rate of the decoder on independent random draws. [Failure categories and measured denominators](#dbmol-source-029) · [Execution and runtime comparisons](#dbmol-source-030)

### What “distance to validity” means in this implementation

For a search that edits a fixed set of graph labels, one supported witness cost is the number of changed labels plus a nonnegative probability-regret term. Changing a label from the raw choice with probability `p_old` to a label with probability `p_new` incurs `1 + w * max(0, log(p_old) - log(p_new))`, with a numerical probability floor. Changing one ambiguous bond from a 0.51 choice to its 0.49 alternative costs less than replacing a 0.999 choice with a 0.001 alternative. The implementation records edit count and regret separately. The search takes categorical decisions on a detached graph. The differentiable training loss scores returned labels under the live decoder. Minimizing a witnessed target's CE and minimizing the mathematical distance to the set of all valid graphs are different operations. [Witness cost and search](#dbmol-source-031) · [Search-gradient study](#dbmol-source-032)

The search comparison on two frozen epoch-43 panels contained 512 prior graphs, including 123 invalid ones. Beam at radius three repaired 122/123 in 10.293 seconds of discovery work. Structural proposals with a beam fallback repaired 123/123 in 2.822 seconds. A* at radius three repaired 119/123 in 4.056 seconds, and on its 119 shared successes lowered mean witnessed cost from 2.7013 to 2.5457. The A* heuristic counts a lower bound on necessary edits from disconnected components and excess valence, then sums the cheapest available distinct bond-edit costs. That argument assumes fixed atoms and charges and one change per unordered bond. The frontier and edit-radius budgets can still cut off the search. A* improved some costs, while structural proposals provided the best coverage and time here. This is the basis for the epoch-45 choice. [Methods, costs and interruptions](#dbmol-source-033) · [A* implementation](#dbmol-source-034)

For comparing near-valid outputs offline, the default graded quality profile gives equal weight to local consistency, connectedness and repair proximity. It reports a 0–100 summary from the component scores. Other profiles exist and a missing positive-weight component can block a summary. An agent tuning the training loss should inspect the three underlying values and separately report the raw valid fraction. Report strict validity or valid chemical identity directly whenever an actual discrete molecule is required downstream. [Graded quality profiles](#dbmol-source-016)

### What the epoch-36 replays actually diagnosed

A replay reproduced historical fixed-panel raw validity of **641 → 674 → 357** out of 1,024 along the same 100 updates. At update 96 the full step lowered total loss yet moved valid count from 590 to 531. The reconstruction gradient norm on decoder parameters was roughly 28 times the prior-repair norm. A quarter-sized trial displacement from the same pre-step weights kept 580 valid, with fewer atom- and bond-label flips. This identifies a local objective tradeoff in shared parameters. It also explains why checking gradients for finite values, or observing decreasing total loss, could coexist with rapid loss of prior validity. Across 200 updates the quarter-step alone and a weak current-valid self-target both lost validity while each still reconstructed 256/256 held-out molecules exactly. [Replayed steps and independent self-target control](#dbmol-source-035)

Self-targeting had another weakness. Once a live graph became invalid, it ceased supplying the positive target that could preserve its earlier identity. Stronger self-targeting passed a short probe but failed longer monitoring. Changing the no-bond weighting and normalization produced a severe regression from 149 to 12 valid outputs on 256 fixed prior points in eight updates, despite passing derivative tests. A better source of positive labels was the frozen original epoch-36 model, evaluated at the same latent independently of the student’s current validity. The accepted new term used full categorical NLL for strictly valid frozen targets, weighted by 0.1 and averaged across all scheduled prior examples. The original frozen epoch-35 occupancy reference remained separate. [Self-guidance strength](#dbmol-source-036) · [Bond-weighting regression](#dbmol-source-037) · [Frozen reference and its objective](#dbmol-source-038)

After 2,500 disposable updates the frozen-reference version moved fixed-panel prior validity from **641 to 665 of 1,024**, categorical validity from **2,091 to 2,323 of 4,096**, and held exact reconstruction at **256/256**. Sulfur frequency went from 1.5070% to 1.5089%. The production continuation restarted from the saved epoch 36, preserving its optimizer history. It reached **753/1,024** valid at epoch 40 with 256/256 exact reconstruction. These longitudinal results give the reason to preserve the epoch-35 occupancy and epoch-36 target references when reproducing this particular campaign. [2,500-update proof](#dbmol-source-038) · [Production epoch-40 checkpoint](#dbmol-source-039)

### What carried through epoch 45

Epoch 44 changed both the fraction of the AdamW parameter displacement applied after updating the moments, from 0.25 to 0.125, and the synthetic batch composition. At full ramp a logical batch contained 256 real molecules, 384 independent prior draws and 128 local perturbations around real codes. The local branch used chemistry supervision. A local perturbation did not acquire the original molecule as a compulsory reconstruction target. Epoch 45 retained that mixture and added structural repair proposals. The proposals were 3.69 times faster in the serial CPU search benchmark, while complete epochs 44 and 45 took 9.94 and 9.97 hours. A faster search kernel can have little effect on total time once training, parallel search and evaluation share the allocation. [Local training protocol](#dbmol-source-040) · [Structural search comparison](#dbmol-source-041) · [Completed training records](#dbmol-source-042)

On the fixed 1,024-point training monitor, epoch 45 rose to **769** raw valid graphs from epoch 44’s **759**. The larger, fresh comparison gives a better picture of the plateau. Across 10,000 prior draws the argmax valid rates at epochs 43, 44 and 45 were **75.49%, 74.88% and 75.41%**. Categorical valid rates were **64.35%, 64.26% and 63.30%**. On 10,000 real validation molecules exact reconstructions were **99.84%, 99.88% and 99.87%**. Use epoch 45 as a frozen reproducible starting point. Those measurements give little evidence that extending the same recipe for another few epochs would by itself solve generation or latent editability. [Fresh checkpoint comparison and protocol](#dbmol-source-001)

## Ordering, interpolation and data

### Atom ordering and reconstruction identity

A chemical graph survives a consistent renumbering of its atoms. Our original encoder flattened 50 ordered atom states before predicting the latent distribution. Changing input order could therefore change the vector and decoded molecule. Three questions need separate tests. Does a permutation of one input reproduce the same posterior? Does the resulting discrete graph equal the original molecule? Do corresponding atoms in **different** analogues occupy matching slots? Canonical ordering solved the first problem for supported inputs. The third still matters for interpolation. [Canonical pilot and raw comparisons](#dbmol-source-043)

We initially tried learned invariant pooling and a reconstruction objective that permitted atom matching. The first short pilot produced zero exact reconstructions on 2,048 held-out molecules. A follow-up added topology-sensitive messages and a longer encoder warm-up against the frozen ordered teacher. It still produced zero exact reconstructions. Finite, unclipped gradients and similar training and validation code errors gave little support for a simple clipping or held-out-overfitting explanation. A useful counterexample came from two chemically valid six-carbon graphs with different connectivity. Every node had degree three and identical initial features. Local messages and pooled moments gave the same vector for both, despite different triangle structure. This demonstrates a concrete information loss in that encoder family. Its frequency in the training set remains unknown. [Counterexample](#dbmol-source-044)

The successful correction maps each supported discrete graph into the dataset's canonical atom order **before** applying the trained ordered encoder. It copies the full encoder and decoder weights. The mapper establishes a bijection, appends empty slots, and verifies every element, charge and both axes of the bond matrix. Cached graphs may bypass repeated mapping after the cache convention, RDKit version and sampled canonical labels have been checked. Keep this discrete RDKit step outside the differentiable route from a live latent to a soft graph to a protein scorer. The encoder adapter explicitly rejects soft inputs or calls requiring gradients with respect to the input graph. [Mapping](#dbmol-source-005) · [Encoder adapter and cache checks](#dbmol-source-045)

Job **29883264** ran 250 encoder-only and 750 joint updates. The parent and candidate reconstructed the same 2,044 of 2,048 held-out molecules exactly. All 1,024 shuffled trials reproduced their unshuffled candidate posteriors and outputs. Of those, 1,020 matched the original molecule, versus zero with the uncorrected parent on shuffled input. On this particular 10,000-draw panel, argmax validity was 75.55% for the parent and 75.48% for the candidate. Categorical validity was 63.05% and 63.04%. This justified integrating the canonical adapter and retaining the shuffled-input regression. It established neither global latent smoothness nor atom correspondence across analogues. [Matched outcome](#dbmol-source-046)

### What the interpolation experiments actually tested

Start with two real molecules A and B. Encode their posterior means and decode a straight line between the means. For every arm, report whether it reconstructs the correct endpoint molecules. On the original 128-pair frozen evaluation, all canonical endpoint pairs reconstructed correctly, and 66 midpoints were valid. Reordering B into A's frame before encoding made 126 midpoints valid, but only 38 pairs still decoded to both correct endpoint identities. Among the 30 pairs whose canonical map was already the identity, all 30 canonical midpoints were valid. The pairs requiring reordering were a different cohort, so the difference cannot measure a pure permutation effect. [Frozen paths and endpoint controls](#dbmol-source-047)

**Blended labels.** The first training pilot used 400 updates from epoch 45 per arm, 32 real reconstruction examples per update, and a shared frozen prior anchor. Its treatment added two mapped pairs per update at nine interior fractions from 10% to 90%. It mixed the endpoint labels independently for each categorical atom, charge and bond target. The treatment used 18 additional latent decodes per update. All 128 evaluation pairs retained both correct endpoints. Mean divergence from the soft targets dropped from 0.02140 to 0.02002 versus the control, yet valid argmax midpoints fell from 68 to 64. The exact target marginals themselves produced only 42 valid argmax midpoints out of 128. Compatible choices of atoms and bonds can get separated when each label is mixed independently. On the same pairs, a follow-up adding chemistry loss at interpolated points recovered two midpoints to reach 66/128, versus 68/128 for the control. A finite repair gradient on a bad midpoint did not establish a general benefit from retraining the shared decoder. [Matched targets and follow-up](#dbmol-source-048)

**Actual valid intermediates.** We next built A → B → C triplets with an observed valid graph B. Each endpoint had a verified single atom or bond edit to B, with disjoint edit sites. The treatment trained the midpoint of the A and C means against B's graph. The training triplets were observed molecules. The evaluation used held-out centers and derived, verified endpoints screened against training and reserved test identities. Treat its 64 selected paths as a specific task.

| Matched 400-update comparison | Control | Valid-intermediate supervision |
| --- | ---: | ---: |
| Valid midpoint | 48/64 | 40/64 |
| Exact intermediate at midpoint | 12/64 | 11/64 |
| Exact held-out reconstruction | 2,044/2,048 | 2,040/2,048 |
| Valid prior argmax, 10,000 draws | 75.62% | 68.99% |
| Valid prior categorical sample, same 10,000 latents | 63.27% | 52.51% |

At a nominal weight of 0.1 the median ratio of the weighted midpoint loss to ordinary reconstruction was 47.7. Median pre-clip gradient norm rose from 0.222 to 4.260. Of 1,368 previously valid argmax prior draws lost by the treatment, 857 changed only bond labels. Another 705 draws gained validity. All eight net midpoint losses occurred among triplets with nonidentity maps. The code verified that the *chemistry* mapped between A, B and C, then trained against B in its own canonical slot frame. A correct graph target still left both a slot-assignment problem and an oversized gradient. The records support both mechanisms without identifying a separate causal effect for each. [Triplet builder](#dbmol-source-049) · [Training and paired failure audits](#dbmol-source-050)

**Aligned slots and capped gradients.** A 200-update follow-up required matching canonical atom slots across all triplet members. It capped the weighted interpolation gradient at 25% of the base gradient norm before shared optimizer clipping. The cap activated on 195 updates and the median applied weight was 0.00709. Both arms had 59/64 valid midpoints, the same 2,044/2,048 exact reconstructions, and closely matched prior validity. Exact B at the midpoint was 31/64 for the control and 32/64 for treatment. This study largely removed the earlier prior-validity cost. It left no clear reason to add the loss to the large continuation. The selected triplets, effective gradient scale and training budget changed together. [Config](#dbmol-source-051) · [Paired results](#dbmol-source-052) · [Runner](#dbmol-source-053)

For proposed nearest-neighbor losses, measure how often a qualified pair actually exists under the specified search. In a 500-query, 100,000-reference fingerprint probe, 66 closest references had equal atom count, 24 passed atom mapping, and 13 passed the stricter similarity and edit criteria. Allowing one extra terminal atom on an otherwise unchanged aligned core added nine candidates, moving pre-similarity coverage to 33/500. This measures exact top-one matches in a sampled library. Searching for the *nearest eligible* neighbor is a different selection scheme. [Coverage probe and optional-atom limitations](#dbmol-source-054)

### Which molecules enter each training and evaluation set?

| Dataset consumed | Training graphs | Validation / test | How to interpret the split |
| --- | ---: | --- | --- |
| Triangle VAE's MOSES cache | 1,688,539 | 20,000 / 20,000 | Custom supported, deduplicated split of the MOSES inventory. Official benchmark membership differs. |
| MOSES plus ChEMBL 37, epoch 46 | 3,670,698 | 70,000 / 70,000 | Preserves MOSES assignments. Adds 50,000 ChEMBL graphs per holdout from reserved scaffold families absent from MOSES. |
| PubChem-expanded sparse parent | 81,409,264 | 70,000 / 70,000 | Complete packed eligible training inventory, including 77,738,566 newly added PubChem identities. |
| Size-balanced view used for epoch 47 | 60,000,000 | 170,000 / 170,000 | Index view into the packed parent. Adds 100,000 distinct PubChem-only identities per holdout. |

The ChEMBL audit read 2,897,819 source rows. Converting to the stereo-free graph schema, filtering and deduplicating produced 2,174,584 distinct passing graphs. Of these, 92,425 already occurred in MOSES. The remaining 2,082,159 included 612,327 distinct ring scaffolds absent from MOSES. Leading rejection reasons were failed graph round-trip, over 50 heavy atoms and disconnected structures. These counts refer to structural identities. The VAE reconstructs element, formal charge and bond labels. Assay targets and 3D coordinates lie outside the cache schema. [MOSES membership](#dbmol-source-007) · [ChEMBL audit and counts](#dbmol-source-055) · [Combined dataset and exclusions](#dbmol-source-056)

The PubChem build streamed 124,860,638 rows, deduplicated canonical graphs and screened against existing identities and the reserved ChEMBL scaffold families. Full membership and source attribution occupy about 0.86 GiB compressed. The 13.56 GiB packed cache stores occupied atoms and charges, sparse undirected edges, canonical identities and offsets. It builds padded tensors on read. The separate 60M index view adds about 230 MiB. A failed cache build exposed an RDKit edge case where recanonicalizing already-canonical aromatic text changed its spelling. The working path parses the stored identity directly and checks that the decoded graph round-trips to it. Keep that contract when porting data loading. [Ingest, failure replay and published cache](#dbmol-source-002) · [Packed dataset reader](#dbmol-source-057)

“Balanced” means distributing the available unique rows as evenly as feasible. Counts 15–31 have about 2.286M training graphs per bin. Scarcer bins exhaust availability, including 29 one-heavy-atom graphs and 241,482 at 50. Both extra 100,000-row holdouts use spare PubChem rows in the 15–31 range. They are identity-disjoint but can share scaffold families across splits apart from the existing ChEMBL exclusions. When reporting new-PubChem reconstruction, name the cohort and its size range. It cannot by itself establish extrapolation to rare sizes, novel scaffolds or Gaussian-prior generation. The training histogram puts only about 0.40% of real graphs at the 50-atom cap. [Per-size counts](#dbmol-source-058) · [Membership selection](#dbmol-source-059) · [Indexed dataset](#dbmol-source-060)

## The epoch 47 failure

Epoch 46 completed one pass over 3,670,698 MOSES–ChEMBL training molecules with a fresh Gaussian latent for each real molecule. Epoch 47 switched to the 60M view, 240 real molecules and 16 fresh priors per logical update, real reconstruction and KL, and an atom-preserving repair loss on the priors. Several things changed together, including the corpus, the number of optimizer updates per pass, the sampling ratio and the removal of earlier frozen-reference and composition guidance. The late checkpoint improved reconstruction on two newer validation sources as unconditional generation collapsed. [Epoch-46 recipe](#dbmol-source-056) · [Original epoch-47 config](#dbmol-source-061) · [60M study](#dbmol-source-002)

| Matched measurement | Epoch 46 | Partial epoch 47 at 47,682,960 real molecules |
| --- | ---: | ---: |
| Exact, on 512 added PubChem validation molecules | 395 | 480 |
| Valid, on those 512 PubChem molecules | 442 | 488 |
| Exact, on 256 scaffold-held-out ChEMBL molecules | 79 | 173 |
| Valid, on those 256 ChEMBL molecules | 94 | 174 |
| Exact, on 64 historical MOSES validation molecules | 64 | 64 |
| Raw valid argmax, on 256 shared Gaussian vectors | 190 | 6 |
| Raw valid, one categorical draw on those 256 vectors | 157 | 6 |
| Raw argmax at the 50-slot cap, same 256 vectors | 0 | 249 |

The PubChem change includes 88 newly exact and three formerly exact reconstructions lost. The ChEMBL change includes 95 gained and one lost. The earlier partial checkpoint, after 27,811,440 training molecules, already filled 229 of 256 random graphs and produced seven valid argmax molecules. Across the 60M training membership, only 241,482 graphs (0.40%) have 50 atoms. Copying the data histogram cannot account for the prior behavior. The stable result on the small historical MOSES panel also illustrates why a reconstruction panel needs appropriate size and source coverage. [PubChem membership and result](#dbmol-source-062) · [ChEMBL result](#dbmol-source-063) · [Paired prior results](#dbmol-source-064) · [Earlier checkpoint comparison](#dbmol-source-003)

### Where and how the size drift appears

On the 256 random vectors, mean hard counts after the initial readout and the four refinement rounds were **31.25 → 25.76 → 27.68 → 49.83 → 49.86**. Swapping in the epoch-46 third round on a separate set of 96 removed all 93 cap hits, but valid outputs increased only from one to four. The reciprocal swap did not reproduce the full failure on the other backbone. On 64 actual PubChem encodings, the same late decoder averaged 22.19 atoms and hit the cap zero times. Shuffling those encodings independently within each latent coordinate preserved the marginal distributions but sent 59/64 graphs to the cap. These interventions implicate learned interactions in the refinement network and the joint latent structure. A single bad output bias or a universal 50-atom preference does not explain this pattern. [Intermediate readouts and reciprocal swaps](#dbmol-source-013)

The original prior teacher could edit bonds while holding every sampled slot, element and charge fixed. A successful repair nevertheless trained the model against the **whole graph**, including its own atom choices. On a full student graph each occupied target pushes the empty-slot logit downward, even though the search supplied no evidence about occupancy. Already-valid or censored graphs contributed zero to this prior term. The coefficient 0.375 multiplies a mean over the prior batch. Changing to a 15:1 sampling ratio does not divide it by another 15. The ratio changes exposure and gradient variability. Many labels from a dense graph can dominate the count of easy decisions while one incorrect bond still invalidates the entire molecule. [Original loss and normalization](#dbmol-source-026) · [Mechanism and controls](#dbmol-source-013)

Three pieces of evidence help distinguish mechanism from historical causation.

1. Removing direct atom/charge targets reduced the predicted increase in mean soft atom count from a local repair-gradient step by about **86% at epoch 46**, but only **4% at late epoch 47**. Bonds and occupancy still share decoder weights. These are local directional derivatives for the tested samples. The optimizer used AdamW across many steps.
2. On four independent real reconstruction minibatches at the healthy checkpoint, the measured real-data gradient moved probe occupancy up on one batch and down on three. This leaves the initiating historical change unresolved. Confidently attributing the entire collapse to the sampling ratio, model size or a particular batch would go beyond these results.
3. Substituting only epoch-46 occupancy at the late model's final readout on another fixed 128-vector panel removed cap hits, but the 35 valid graphs comprised only **17 distinct** molecules, with 30 of the 35 containing only carbon as a heavy element. A full covariance Gaussian fitted on 512 real encodings still produced **114/128** cap hits. Restoring sizes, broad chemistry and diversity are three separate tests. [Gradient, substitution and fitted-prior experiments](#dbmol-source-013)

A practical workaround for initializing a **separate design study** was to draw founders from epoch 46 and encode them through the later model. Of 256 original draws, 190 gave distinct valid epoch-46 founders. Decoding epoch-47 posterior means gave 144 distinct valid structures on that original denominator, 99 exactly preserved founders, 45 valid edits and zero cap hits. Keep the original graph identity alongside the re-encoding. That experiment measured generation and identity preservation. Protein scores were never used to select or validate those edits. [Founder handoff records](#dbmol-source-065)

## The recovery continuation

The user selected a **recovery attempt from the affected weights during epoch 47**. A matched fork from healthy epoch 46 would answer whether a loss prevents collapse. The current branch instead asks whether genuine PubChem and ChEMBL reconstruction gains can survive while the existing model relearns plausible random samples. This choice makes the existing partial checkpoint the student and the completed epoch-46 checkpoint a frozen reference.

For a logical batch, keep reconstruction and KL on 240 true training molecules. Draw 16 new standard-normal vectors from the saved independent prior generator. Decode the same vectors with the student and reference. The new prior objective is

```text
0.375 * structural bond repair
+ 1.0 * soft occupancy cross-entropy against frozen epoch 46
+ 1.0 * conditional chemistry cross-entropy on eligible epoch-46 graphs
```

- **Bond repair.** Search on the current student graph with the structural teacher. A successful witness supplies labels on the unordered occupied pairs. Average their conditional bond cross-entropy per graph, then average across all scheduled prior rows. Rows without a repaired target contribute zero to this particular term. The bond-only target removes explicit supervision on the student's own element, charge and occupancy choices.
- **Occupancy.** For every slot of every vector form `u = logsumexp(real_element_logits) - no_atom_logit`. Use binary cross-entropy with logits against the detached epoch-46 occupancy probability for that same vector and slot. The derivative with respect to `u` is the student's occupancy minus the reference's occupancy, before averaging. Thus a confidently occupied slot can still receive a strong corrective gradient if the reference predicts absence. Each vector can have a different target count. The parent need not give a chemically valid hard graph for this term to apply.
- **Reference chemistry.** Take the frozen reference's hard graph at the same vector. Check eligibility by chemical validity and the representational round trip. On accepted rows use element CE over the real-element classes on occupied slots, charge CE on those slots, and bond CE over unordered occupied pairs. The component weights are 1, 1 and 2. No-bond has class weight 0.1. Multiply the combined term by the fraction of eligible batch rows. Excluding the empty class from the element softmax removes its direct conditional-label gradient. The shared refinement parameters can still influence it indirectly.

The live forward/backward pass stays differentiable through the student logits. The detached search, chemistry checks and all frozen reference operations sit outside this path. This matters when porting the objective to Mosaic or rewriting it against other tensor types. Distinguish the reference atom/charge/bond term from the structural repair branch, whose bond normalization is a different reduction. Preserve the original resume checkpoint, AdamW state, train-row cursor, permutation and generators when asking about this run. [Implemented prior objective](#dbmol-source-066) · [Shared repaired-bond reduction](#dbmol-source-067) · [Recovery config](#dbmol-source-068) · [Authenticated training runner](#dbmol-source-069)

### What has already been demonstrated

The short recovery experiment used a frozen encoder, four fresh vectors and four real molecules per step, new AdamW state, and 24 matched updates. Occupancy supervision plus real reconstruction alone changed a baseline of **125/128 full, 7/128 valid and distinct, 61/64 exact** to **0/128 full, 5/128 valid and distinct, 60/64 exact**. Adding conditional chemistry on valid parent graphs gave **0/128 full, 9/128 valid and distinct, 58/64 exact**. The healthy reference on those same vectors had 91/128 distinct valid molecules. The count failure can be reversed. The small pilot offers little evidence yet for substantial recovery of chemical validity. Its optimizer, real/prior ratio and update budget differ from production. [Pilot and protocol](#dbmol-source-013) · [Saved pilot metrics](#dbmol-source-070)

A separate CPU smoke test exercised a complete logical 240+16 update with the actual partial-47 weights and AdamW moments. It used a CPU microbatch of eight and a training sample from the corpus. All 16 student priors received bond targets, all 16 received occupancy targets, and 11 reference graphs passed the chemistry/schema check. Unweighted repair loss was 0.01019, occupancy BCE 7.764 and conditional chemistry 5.150. Real reconstruction was 0.02446, KL 86.91 with coefficient 0.001, and pre-clip gradient norm 70.48. Student weights changed and the frozen reference checksum was preserved. This verifies the path and the intended eligibility accounting. One CPU update cannot tell us the A100 throughput, sustained stability or final quality. [Actual-checkpoint CPU result](#dbmol-source-071)

### How the next result should be read

The recovery panel fixes **128 Gaussian vectors and 64 held-out PubChem validation molecules**, seed `2026111909`. It records the starting state, offsets 10 and 100 from the segment start, global update multiples of 1,000, and the segment endpoint. The ground-truth prior baseline on this specific panel is epoch 46 **0 full, 91 valid and distinct, median 22 atoms** versus partial epoch 47 **125 full, 7 valid and distinct, median 50**. Of the 121 invalid graphs at handoff, 99 failed sanitization and 22 were disconnected. These 128 draws are distinct from the 256-draw comparison above. Use the actual GPU panel's starting row for paired reconstruction and comparison on that device. The smoke-test reconstruction loss is a scalar over training examples and cannot substitute for that held-out measure. [Monitoring implementation](#dbmol-source-072) · [Fixed prior baseline](#dbmol-source-073)

If full graphs fall, inspect valid and distinct molecules, failure reasons and exact real reconstruction on the **same** inputs. A sharp rise in valid graphs with few distinct identities or mainly all-carbon outputs calls for examining conditional chemistry. If cap hits fall but gaps in connectivity grow, inspect the pair and bond losses before adding another size term. If the early panel looks encouraging, repeat the comparison on a larger independent panel and show both hard argmax and categorical generation. A change of one molecule on the 64-row reconstruction panel is 1.56 percentage points. Counts, paired gains/losses and a longer time series convey more than a claim that reconstruction was “preserved.” These are interpretation criteria for a bounded experiment.

## The compact VAE

The 435k model tests whether a smaller representation can support the same design work at lower cost. It has **238,360 encoder** and **197,040 decoder** parameters. It uses a 128-D latent, 24 atom features, 18 pair features and a 12-feature triangle bottleneck. Four message layers, four decoder refinement rounds and four attention heads remain. The encoder uses canonical graph order and a residual 128-wide readout. The graph-matched reconstruction variant can recognize an atom permutation as correct. For an incorrect prediction the approximation chosen to align labels can affect the CE gradient. Matching and canonicalization should be recorded independently. [Architecture, training recipe and graph-matching semantics](#dbmol-source-074) · [Compact config](#dbmol-source-075) · [Verified matcher](#dbmol-source-076)

| Full optimizer updates on one 32 GB V100 | Batch | Real molecules/s | 60M examples at that rate |
| --- | ---: | ---: | ---: |
| Reconstruction + KL | 256 | 242.84 | 68.63 h |
| Reconstruction + KL | 512 | 258.24 | 64.54 h |
| Plus 15:1 prior guidance | 256 | 90.96 | 183.24 h |
| Plus 15:1 prior guidance | 512 | 74.41 | 223.97 h |

All four conditions processed the same 4,096 measured real examples after warmup. Both batch sizes fit natively. For this benchmark the chemistry arm combines bond-only structural targets with a population atom-count distribution term. The 60M projections exclude evaluation, checkpointing and queue waits, and a mature model may produce a different repair workload. These recipes and allocations differ from the large-model production run. In particular, the compact benchmark's count-distribution term and the new production model's frozen per-slot BCE answer different research questions. [Timing outputs and assumptions](#dbmol-source-074) · [Population objective](#dbmol-source-067)

The next useful compact-model result would compare held-out reconstruction, unfiltered random validity, scaffold diversity, size and editability at matched data exposure. Model size by itself explains neither the large model's confidence around an encoded graph nor the observed interaction between its prior targets and occupancy. Training resources and the number of calls to structural search belong in a cost comparison. The V100 timing experiment supplies a real budget estimate for such a pilot.

## Latent molecule design

The design experiment updates one molecular latent vector, `z`. Keep the encoder, decoder and protein scorer frozen. A real founder enters through its encoded posterior mean. A random start can bypass the encoder entirely. The decoder produces probabilities for elements, charges and undirected bonds. The scorer's derivative with respect to those probabilities flows back through the decoder to `z`. A chemistry objective can contribute its own gradient at the same graph. This use of the VAE makes editability an independent research question. A model can reconstruct every founder in a panel and still require a large displacement before any founder changes identity. [Latent-design experiment index](#dbmol-source-077) · [Checkpoint perturbation study](#dbmol-source-001)

Keep four observations separate when reading or extending a trajectory.

| Observation | What the saved value measures |
| --- | --- |
| Relaxed loss | A continuous model prediction on the probabilities, with the optimization context and temperature at that update. |
| Fixed-slot hard graph | The highest-probability labels under the prescribed occupied slots. Changing a slot mask can change the molecule even with identical decoder logits. |
| Raw hard graph | The decoder's own highest-probability atoms, including `NO_ATOM`, plus charge and bond labels. Retain invalid outputs and their failure reasons in the denominator. |
| Fresh score | A separate model prediction after re-importing a discrete valid molecule and rebuilding its features or conformer under the declared scoring protocol. Compare with the founder under the same fresh protocol. Experimental binding requires separate evidence. |

### Boltz-2, the DBMol objective and DeFoG

An early panel tested whether carbon-chain initialization would trap the model in carbon-only chemistry. Across ten proteins we encoded ten chain lengths (C6 through C15) per protein, kept the original occupied mask fixed, and optimized only the latent with Boltz's binary affinity probability. Of 100 trajectories, 94 completed 100 updates and six stopped at the original derivative gate. Among the 94 completers, 93 final argmax graphs were valid and 72 had a valid identity different from the starting chain. Only three replaced carbon with a heteroatom. The model often changed bonding while retaining the starting elemental composition. This panel used a frozen pose, the historical clipped step and an affinity-only objective. Report every denominator if comparing it with a later composite-loss or variable-size campaign. [Protocol, per-target counts and failure evidence](#dbmol-source-078)

The later DBMol-style affinity term minimizes the mean native continuous prediction minus the mean raw binder logit. Both terms have default weight one. The full fixed-pose objective weights affinity by 1, pocket contacts by 2.5 and confidence by 2.5. Contacts combine coverage across the pocket with the strongest ligand contact. On the validated path, the native affinity checkpoint keeps its own differentiable trunk. A second checkpoint supplies the structural trunk for contacts and confidence. The affinity and confidence heads use detached pose coordinates while gradients still flow through ligand features and trunk activations. Diffusion can produce a pose separately. An agent rebuilding this experiment should preserve its coordinate source, pocket and ensemble reduction. [Exact equations, gradients and precision history](#dbmol-source-079) · [Composite scorer](#dbmol-source-080)

The adapter maps element probabilities to native element features. Charges become an expected formal charge. The model receives bond-existence probability and, where supported, an expectation over typed bond embeddings. The fixed-occupancy policy removes `NO_ATOM` from the element softmax on occupied slots and masks self and absent bonds. It supplies no direct binding gradient to the empty-slot logit. A forward or gradient check using that path therefore says nothing about differentiating through atom insertion. Native pointwise automatic-differentiation checks did pass. Historical finite differences also encountered ReLU branch changes, precision limits and cancellation between the contact terms. The FP64 contact readout helped particular comparisons. No experiment isolated one numerical defect that explains every reported discrepancy. [Graph policy](#dbmol-source-081) · [Native feature adapter](#dbmol-source-082) · [Contact validation](#dbmol-source-083)

DeFoG used a different contract. Sample a discrete candidate, map it into the checkpoint vocabulary, start the late-time denoising procedure from the candidate, and accept only an eligible, aligned target graph. Hold the selected target fixed while differentiating its negative log probability under the live decoder. Target refresh can change the guidance from one interval to the next. The gradient reaches the VAE through the decoder's probability of that fixed target. The implementation does not differentiate through the sampled denoising trajectory. Charge projection in the supported historical variant was lossy and explicitly enabled. In a six-update integration test, eight eligible native outputs supplied two selected targets and both gradients reached the shared latent. [Repair interface and vocabulary](#dbmol-source-084) · [Integration smoke and limits](#dbmol-source-085)

The longer comparison used the older frozen epoch-89 decoder, from a separate training lineage. Its Boltz-only control stopped under the archived Armijo rule after 225 accepted updates and 226 attempts. The DeFoG arm accepted all 500 requested updates but produced zero valid endpoint candidates out of nine. A large target loss can move the relaxed distribution without delivering a valid maximum-probability graph. Keep the control's true endpoint and the invalid guided candidates when analyzing this study. The later best-finite-trial policy belongs to subsequent runs. [Matched study and saved offline trajectory](#dbmol-source-086) · [Later step policy](#dbmol-source-087)

### Protenix and the first full epoch-45 design trajectory

The first long Protenix run started from a random latent and used a fixed 25-slot BRD4 context, a Protenix contact score and structural repair on the decoded graph. The chemistry weight stayed at 1 through update 20,000, rose geometrically to 10 by 30,000 and then held. Temperature fell from 0.7 to 0.4 over 50,000 updates. Noise proportional to the binding-gradient scale fell from 0.4 to zero by update 30,000. One can reproduce the settings independently of the cluster campaign controller from the study config. [Training-independent optimization code](#dbmol-source-088) · [Exact trajectory configuration](#dbmol-source-089)

Job 29875634 completed 50,000 updates in 5.90 GPU hours. The relaxed contact score rose from 0.151316 to 0.245647 and peaked much earlier at 0.345866 on update 27. The endpoint was valid and differed from the start at four element labels and twelve unordered bond labels. Across the trajectory, 16,289/50,000 source graphs were valid and 50 distinct valid identities appeared. By the saved 30k, 40k and 50k observations the argmax identity was the same. Temperature, noise and chemistry weight changed together. The run demonstrated molecular editing, while its endpoint score alone could establish neither the contribution of each schedule nor improved binding of the final discrete molecule. An auxiliary bridge fitted to hidden decoder features could also improve its own mapped score while the directly decoded score fell. Check the original graph path whenever trying a new scorer-specific decoder or bridge. [First trajectory and interactive report](#dbmol-source-090) · [Hidden-bridge comparison](#dbmol-source-091)

### Sharp categorical predictions, temperature and step length

At temperature 1, six illustrative encoded molecules had a median of 100% of occupied element predictions and present bond predictions above 0.99 for the winning label. Held-out encoded molecules also resisted random perturbations. At epoch 45, four perturbations for each of 1,000 held-out anchors yielded zero valid changed identities at each of four tested scales. Random Gaussian starting points changed far more readily. Temperature rescaling of the same logits preserves the argmax. Its role during optimization is to change the probabilities and the gradients, which can lead the next latent to a different graph. [Saturation and perturbation evidence](#dbmol-source-003)

The native gradient measurements point to a step-length problem for some starts. At radius 0.1, over five saved/prior Protenix centers, median neighboring contact-gradient cosine was 0.995 at temperature 1.0, 0.985 at 0.7 and 0.969 at 0.4. At 0.4 only two of five steps changed the raw hard graph. At radius 1.0 no descent step beat both random controls. A second experiment examined four encoded holdouts. Normalized descent reached a valid fixed-slot edit in 4/4, versus 0/8 random rays. Only one descent ray reached an edit by radius 1. These small panels support local coherence alongside saturation and loss of directional accuracy farther away. A universal shattered-gradient explanation is inconsistent with the measured small-radius cosines. [Native landscape and scorer configuration](#dbmol-source-092) · [Boundary measurements and trajectories](#dbmol-source-093)

Two comparisons are especially useful when choosing an optimization protocol.

* Warming from 1.5 to 1.0 versus constant 1.0 increased the starting gradient norm by 6.6–39 times on three encoded founders. All six encoded trajectories retained their starting graph over 32 updates. Among fresh-scored edits with valid baseline scores, 3/5 warm edits and 7/8 constant-temperature edits improved. Two warm candidates failed conformer embedding. These counts make a single “best temperature” premature. [Matched recipes and fresh scores](#dbmol-source-094)
* Searching along the normalized gradient at equal cost, 31 Protenix calls per trajectory, committed a valid changed molecule for two of three encoded founders. The previous update rule produced none. The MOSES founder's committed edit improved fresh contact loss by 0.02959. The ChEMBL edit failed fresh conformer embedding. The invalid prior founder favored the previous rule, with seven distinct valid fixed-slot edits versus zero committed under normalized search. Record trial-only and committed discoveries separately. [Bounded search result](#dbmol-source-095) · [Search implementation](#dbmol-source-096)

### Atom counts in native models

Scaling a slot's element mass to zero failed to reproduce actual deletion in all three tested native contexts. Its other inputs, including reference position and incident bonds, still carried evidence of its presence. A soft 25-atom budget then tested prebuilt topologies. Both arms selected the same 24-atom panel molecule, but the unmodified decoder stayed at 25 atoms for every one of 65 observations in each arm. The expected count stayed inside the penalty-free band. That experiment cannot show that the budget causes raw count changes. [Deletion-equivalence diagnosis](#dbmol-source-097) · [Native atom-budget result](#dbmol-source-098)

The successful raw-count experiment evaluated the current discrete molecule at its actual size, proposed a scorable addition and deletion from the decoder, and used their native score differences as coefficients on live occupancy probabilities. Each latent trial was re-decoded unmasked. The candidate only counted as realized if the new hard graph justified it. Across 24 matched starts, guidance changed count for 15 starts versus 9 with random search. Random search evaluated 1,591 cheap directions versus 90 guided ones and obtained a lower mean fresh loss, 2.58340 versus 2.61254, at a similar native-call spend. The completed larger campaign logged 20,000 accepted updates across 6,498 trajectories, including 9,027 chemistry changes, 3,327 additions and 2,018 removals. Restarts create new baselines. These observations show that a VAE can change its own molecule size under this proposal and verification procedure. They do not establish superiority over inexpensive random search. [Raw-count method and comparison](#dbmol-source-099) · [Final stage evidence](#dbmol-source-100)

### Comparing an adaptive controller with a different representation

The matched adaptive-search comparison kept the epoch-45 representation and 25 occupied slots fixed. Three BRD4 seeds gave mean best freshly scored hard contact losses of 2.11675 under adaptive restarts, 2.18843 for independent runs and 2.28281 for continuous trajectories. Each arm received 48,096 relaxed calls per seed and the same fresh allowance. Actual fresh attempts differed as candidates became available. A separate three-founder pilot allowed eight fresh attempts per founder and arm. Mean best fresh loss was 2.45124 for direct graph edits and 2.45228 for latent restarts. The first comparison supports the adaptive controller within a VAE search. The second leaves open whether the VAE adds value over a cheaper proposal representation. The implementation and receipts for the adaptive runs live in the restart-search worktree linked by the cross-repository review. [Matched comparisons, code locations and archive chemistry](#dbmol-source-101)

## Working from the saved state

### Identify the right checkpoint

| Role | File or lookup | SHA-256 |
| --- | --- | --- |
| Frozen MOSES epoch 45 for molecular design | `source156-vae-epoch45-structural-attempt002/epoch0045/epoch_0045.pt` under the run root | `5c4946c15515e70bd01281eea7b74693c18598c4b47f6724fcc9898deb1302be` |
| Completed combined epoch 46, frozen teacher in the recovery branch | `vae-pubchem60m-prior15-20261106a/combined-02/checkpoint-000325498.pt` | `b239143056b561d5b08249c5b28e0366722c06d34f3bf94e1faf7acdfead4109` |
| Affected partial epoch 47, student at the transition | `vae-pubchem60m-prior15-v100-20261111b/sparse-02/checkpoint-000524177.pt` | `408b6904281fbe5d9f92e8b40c62e301d099b0fdad87109ef0ca5d90ce172922` |

The HPC run root for those relative names is `/work3/s234818/latentmol/runs/`. The 60M view digest is `64247d04ec1f94d00320bb1e5a0d9c5540419eb37830050d3100775bf4385e18`. The recovery branch's source and destination config digests are recorded in its receipt. A file named for update 524177 has a **completed-epoch field of 46**, a nonzero row of 47,682,960 and `epoch_complete=False`. It is inside epoch 47. At 240 real molecules per update, exactly 12,317,040 real examples or 51,321 updates remained at cutover. Epoch numbers in the earlier ZINC lineage have independent meanings. A portable set of epoch-45 model assets in Mosaic is appropriate for inference. A continuation that needs historical AdamW moments and generators should start from the designated training checkpoint. [Parent paths](#dbmol-source-102) · [Recovery recipe and source identity](#dbmol-source-068) · [Cross-repository branch map](#dbmol-source-103)

### Locate the next panel without changing the run

At **11:07 CET, 19 November 2026**, LSF listed A100 job **30112466** as pending in `gpua100` and callback **30112467** as pending in `hpc`. The recovery directory had produced no panel file at that observation. The old pending GPU worker and its callback had already been retired. The branch's scientific status at that time was **production code exercised locally and submitted, recovery unmeasured on HPC**. For a later handoff, read the scheduler and the newest segment's outputs again. The two-page PDF is a dated summary. [Cutover receipt](#dbmol-source-104) · [Run plan](#dbmol-source-105)

On the configured Windows workstation the documented connector can do the read-only check below. A job may roll out of the scheduler's recent-history window. The callback receipts and the numbered output directories can locate a later allocation. The pending CPU callback handles the checkpoint and schedules a successor allocation if needed. [Current DTU transport](#dbmol-source-106) · [Continuation chain](#dbmol-source-107)

```powershell
$dtu = 'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe'
& $dtu exec "bash -lc 'bjobs -u s234818'"
& $dtu exec "bash -lc 'ls -ld /work3/s234818/latentmol/runs/vae-pubchem60m-e47-parent-guidance-20261119a/sparse-*'"
```

Under each new `sparse-NN/` look for the following, in the order that answers your question.

1. **`recipe_handoff.json`** records the authenticated change from the old training recipe at the first boundary. The new config ties the old SHA, row, update and data population together. Ordinary later resumes use the new config identity.
2. **`quality-panel.jsonl`** contains the direct answer to whether quality changed. For the first segment, compare rows at global updates **524177, 524187 and 524277**. Read `prior_at_capacity`, `prior_valid`, `prior_unique_valid`, `prior_size_histogram`, `prior_failures`, `reconstruction_exact` and `reconstruction_valid`, alongside panel size, seed and validation-row membership. The monitor uses private RNGs.
3. **`training.jsonl` and `progress.json`** describe the training cursor and fresh batches. Fields such as `parent_valid_priors`, `prior_already_valid`, `repair_statuses`, `prior_occupancy_bce` and `prior_conditional_ce` diagnose what was supervised. The student and reference counters answer different questions. The absolute scalar loss also changes at the recipe boundary because new terms were added.
4. **`latest.json`, checkpoint and `summary.json`** distinguish a saved boundary from the most recently logged optimizer step. A normal wall-time pause can produce a resumable checkpoint. The next allocation should read the durable cursor. The 60M order is regenerated from the original seed and epoch, and the independent prior generator is restored from checkpoint state.

For example, after downloading the small panel file, the following reads the actual quantities and preserves the per-row denominators. It does not load a model. Use the first segment for the three offsets above or supply the corresponding updates for another segment.

```python
import json
from pathlib import Path

panel = Path("quality-panel.jsonl")
updates = {524177, 524187, 524277}
for line in panel.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if row["updates"] in updates:
        print(row["updates"], {
            "full": (row["prior_at_capacity"], row["prior_points"]),
            "valid": (row["prior_valid"], row["prior_points"]),
            "distinct_valid": row["prior_unique_valid"],
            "exact": (row["reconstruction_exact"], row["validation_points"]),
        })
```

### Reproduce existing analyses

Run the following from the repository root in its installed environment, with PyTorch and RDKit for molecular analyses. The recovery diagnostic below accepts explicit copies of two authenticated checkpoints and saved input panels. It evaluates frozen interventions on CPU and requires a new output directory. Choose your local checkpoint paths before running it. A ready-made panel file is indexed by identity in the saved experiment records. [Recovery diagnostic](#dbmol-source-108) · [Graph evaluation workflow](#dbmol-source-109)

```sh
python -m pytest -q tests/training/test_prior_parent_guidance.py tests/test_combined_vae_continuation.py
python experiments/vae/occupancy_recovery.py --help
python experiments/vae/occupancy_recovery.py --parent EPOCH46.pt --student PARTIAL47.pt --train outputs/runs/vae-occupancy-recovery-20261119a/train-panel.json --validation outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json --output NEW_OUTPUT_DIRECTORY
```

The first command covers the trainer and parent-guidance logic on fixtures. The actual restored-state smoke, eight-vector logit tests, fixed-panel comparisons and native scorer experiments provide different evidence. Use a sealed release's recorded command if reproducing the exact historical environment matters. To inspect the chemistry of a saved design run, prefer its offline report generator and original snapshots. The reports keep invalid graphs and per-update probabilities, avoiding accidental selection of only the attractive intermediates. [CPU smoke with real weights](#dbmol-source-071) · [Saved-output report workflow](#dbmol-source-110)

### Where a code change belongs

| Question or change | First implementation to inspect | Evidence to compare afterward |
| --- | --- | --- |
| Change the representation, soft masking or hardening | [Graph contract](#dbmol-source-004) and [sampling](#dbmol-source-006) | Conditional versus marginal bonds, undirected symmetry, diagonal and padded behavior, raw size histogram |
| Change the atom, charge or pair pathway | [Model configuration](#dbmol-source-011), [refined decoder](#dbmol-source-009), [triangle module](#dbmol-source-010) | Checkpoint compatibility, per-round occupancy, exact graph recovery and raw prior generation |
| Rework input permutation or training-label alignment | [Canonical input ordering](#dbmol-source-045), [matched reconstruction](#dbmol-source-076) | Shuffled identity and posterior, both bond axes, exact and valid outputs |
| Change the teacher, distance or guidance weights | [Structural proposals](#dbmol-source-111), [probability-weighted repair](#dbmol-source-031), [large prior loss](#dbmol-source-066) | Raw versus strict validity, already-valid/censored rows, reach of gradients, occupancy and element composition |
| Change population size guidance in the compact model | [Compact prior terms](#dbmol-source-067) and [benchmark](#dbmol-source-112) | Distribution over sizes, categorical and argmax histograms, useful chemistry and cost per real molecule |
| Change a frozen external scorer | [Boltz objective](#dbmol-source-079), [Protenix experiment index](#dbmol-source-090), [model interfaces](#dbmol-source-113) | Same declared graph at each branch, score sign, shape, native support, frozen weights, and fresh actual molecule and pose |
| Change a resume or fixed panel | [Combined trainer](#dbmol-source-069) and [occupancy monitor](#dbmol-source-072) | Cursor, real/prior RNG, optimizer moments, unchanged fixed membership and output identity |

### Decisions the current evidence supports

**For a proposal about “more data,” specify its intended benefit.** The partial 60M run is positive evidence for reconstruction of held-out molecules from the two additional sources. It is negative evidence for direct Gaussian generation under the original 15:1 loss. The joint change in data and supervision prevents assigning either effect solely to corpus size. If a future agent needs a causal answer, compare the two losses from the same healthy checkpoint with matched examples, priors, update counts and optimizer state. Include reconstruction by source and atom count, including a deliberately sampled rare-size validation panel if claiming performance near 50 atoms.

**For a proposal about smoothness, define a useful edit.** A locally continuous relaxed graph can keep the same hard identity for many steps. A larger step can cross several identity boundaries and produce an invalid structure. Specify the starting distribution, radius, temperature, fixed-slot policy, success denominator and comparison direction. Measure how far it takes to reach a changed valid graph, whether the fresh score improves, and what happens on random directions at the same budget. The failed blended-interpolation experiment and more selective triplet study are the historical controls for any new interpolation loss.

**For a proposal about chemical validity, name the failure mode.** Empty graphs, disconnected graphs, impossible valence, representation round-trip failures, excessive sulfur, constant 50-atom graphs and repeated all-carbon identities require different diagnostics. A better local-valence score can coexist with more disconnected graphs. A repair search may produce a valid witness even when the raw graph remains bad. Quote the prevalence of each class, the rows on which a term applies and whether the signal comes from the student's graph, a search witness, a previous checkpoint or the actual training molecule.

**For design integration, put a price on each discovery.** Keep the parent identity, latent, scored mask, raw graph and fresh graph attached to every branch of an adaptive search. Charge all scorer calls, failed conformers and restarts to the appropriate arm. For variable-size runs, include additions, removals, realized changes that survive unmasked decoding, and fresh score differences within comparable size ranges. The technically stronger result is a larger number of diverse, valid, independently rescored improvements than a direct graph-edit or random baseline under the same controller. The existing evidence makes that a well-posed next comparison. It remains an open result. [Cross-codebase review and comparator design](#dbmol-source-103) · [Representation comparison and alternatives](#dbmol-source-003)

**If proposing an alternative to a VAE, identify the property it would buy.** A motif or junction-tree decoder could constrain fragment assembly but would need a way to expose useful derivatives or to enter the adaptive search as a discrete proposal. A parent-plus-edit model could learn coordinated substitutions, additions and deletions directly. A learned nonlinear prior would address unconditional initialization, with a separate test for whether it also improves movement near known founders. These are candidate experiments. The nearest ready comparison is already available through direct graph edits under the same controller. [Alternatives and criteria](#dbmol-source-114)

For new work use the current exploratory and gradient policies. The historical gate that stopped a particular campaign explains why a recorded arm ended. The user’s present task and the active recipe determine which checks apply now. [Active instructions](#dbmol-source-115) · [Research policy](#dbmol-source-116) · [Gradient validation policy](#dbmol-source-117)

## DBMol source index

Paths are relative to the DBMol_exploratory research checkout. A suffix beginning with `#` identifies the cited section. Paths under `outputs/` refer to saved research artifacts.

| Reference | DBMol path |
| --- | --- |
| <a name="dbmol-source-001"></a>001 | `experiments/vae/checkpoint-comparison.md` |
| <a name="dbmol-source-002"></a>002 | `experiments/vae/large-corpus.md` |
| <a name="dbmol-source-003"></a>003 | `experiments/vae/representation-search-review.md` |
| <a name="dbmol-source-004"></a>004 | `src/latentmol/representation.py` |
| <a name="dbmol-source-005"></a>005 | `src/latentmol/chemistry.py` |
| <a name="dbmol-source-006"></a>006 | `src/latentmol/sampling.py` |
| <a name="dbmol-source-007"></a>007 | `docs/52_large_corpus_triangle_vae.md` |
| <a name="dbmol-source-008"></a>008 | `src/latentmol/models/encoder.py` |
| <a name="dbmol-source-009"></a>009 | `src/latentmol/models/refined_decoder.py` |
| <a name="dbmol-source-010"></a>010 | `src/latentmol/models/triangle.py` |
| <a name="dbmol-source-011"></a>011 | `src/latentmol/models/autoencoder.py` |
| <a name="dbmol-source-012"></a>012 | `src/latentmol/training/losses.py` |
| <a name="dbmol-source-013"></a>013 | `experiments/vae/occupancy-collapse.md` |
| <a name="dbmol-source-014"></a>014 | `src/latentmol/evaluation/checkpoint_comparison.py` |
| <a name="dbmol-source-015"></a>015 | `src/latentmol/chemistry_classification.py` |
| <a name="dbmol-source-016"></a>016 | `src/latentmol/evaluation/quality.py` |
| <a name="dbmol-source-017"></a>017 | `docs/29_architecture_comparison_20261001.md` |
| <a name="dbmol-source-018"></a>018 | `docs/25_architecture_results_20260929.md` |
| <a name="dbmol-source-019"></a>019 | `configs/arch_wide_global_bond_vae_continue180.yaml` |
| <a name="dbmol-source-020"></a>020 | `configs/arch_stable_deep_pair_refine_vae_full.yaml` |
| <a name="dbmol-source-021"></a>021 | `docs/11_stage1_latent_study.md` |
| <a name="dbmol-source-022"></a>022 | `docs/45_dense_chemistry_100_epoch_continuation.md` |
| <a name="dbmol-source-023"></a>023 | `docs/84_sulfur_bias_in_validity_guidance.md` |
| <a name="dbmol-source-024"></a>024 | `docs/88_local_neighbor_loss_pilot.md` |
| <a name="dbmol-source-025"></a>025 | `docs/53_atom_preserving_triangle_continuation.md` |
| <a name="dbmol-source-026"></a>026 | `src/latentmol/atom_preserving_chemistry.py` |
| <a name="dbmol-source-027"></a>027 | `docs/53_atom_preserving_triangle_continuation.md#loss-weighting-and-the-composition-prior` |
| <a name="dbmol-source-028"></a>028 | `docs/74_frozen_valid_reference_guidance.md#exact-candidate-objective-and-identities` |
| <a name="dbmol-source-029"></a>029 | `docs/63_failed_repair_guidance.md` |
| <a name="dbmol-source-030"></a>030 | `docs/66_failure_guidance_execution_speed.md` |
| <a name="dbmol-source-031"></a>031 | `src/latentmol/probability_repair.py` |
| <a name="dbmol-source-032"></a>032 | `docs/40_search_repair_gradient_study.md` |
| <a name="dbmol-source-033"></a>033 | `experiments/chemistry-guidance/README.md#probability-aware-a` |
| <a name="dbmol-source-034"></a>034 | `src/latentmol/guidance/repair_astar.py` |
| <a name="dbmol-source-035"></a>035 | `docs/70_epoch36_collapse_mechanism.md` |
| <a name="dbmol-source-036"></a>036 | `docs/71_valid_output_guidance_iteration.md` |
| <a name="dbmol-source-037"></a>037 | `docs/72_bond_weighted_target_guidance.md` |
| <a name="dbmol-source-038"></a>038 | `docs/74_frozen_valid_reference_guidance.md` |
| <a name="dbmol-source-039"></a>039 | `docs/75_frozen_reference_epoch40_continuation.md` |
| <a name="dbmol-source-040"></a>040 | `docs/89_latent_neighborhood_continuation.md` |
| <a name="dbmol-source-041"></a>041 | `experiments/chemistry-guidance/README.md` |
| <a name="dbmol-source-042"></a>042 | `experiments/vae/README.md` |
| <a name="dbmol-source-043"></a>043 | `experiments/vae/invariant-pilot.md` |
| <a name="dbmol-source-044"></a>044 | `outputs/hpc-results/vae-canonical-invariant-pilot-20261103-a01/regular-cage-counterexample.json` |
| <a name="dbmol-source-045"></a>045 | `src/latentmol/datasets/ordering.py` |
| <a name="dbmol-source-046"></a>046 | `outputs/hpc-results/vae-canonical-invariant-pilot-20261103-a01/completed-comparison.json` |
| <a name="dbmol-source-047"></a>047 | `experiments/vae/interpolation.md#completed--3-november-2026-124024-cet` |
| <a name="dbmol-source-048"></a>048 | `experiments/vae/interpolation.md#midpoint-mechanism-and-chemistry-follow-up` |
| <a name="dbmol-source-049"></a>049 | `src/latentmol/datasets/graph_triplets.py` |
| <a name="dbmol-source-050"></a>050 | `experiments/vae/interpolation.md#failure-analysis` |
| <a name="dbmol-source-051"></a>051 | `experiments/vae/epoch45-slot-aligned-interpolation-pilot.json` |
| <a name="dbmol-source-052"></a>052 | `experiments/vae/interpolation.md#slot-aligned-pilot-results` |
| <a name="dbmol-source-053"></a>053 | `src/latentmol/training/interpolation_pilot.py` |
| <a name="dbmol-source-054"></a>054 | `experiments/vae/chembl-comparison.md#closest-neighbor-coverage-probe` |
| <a name="dbmol-source-055"></a>055 | `experiments/vae/chembl-comparison.md#audit-run` |
| <a name="dbmol-source-056"></a>056 | `experiments/vae/combined-training.md` |
| <a name="dbmol-source-057"></a>057 | `src/latentmol/datasets/bulk_cache.py` |
| <a name="dbmol-source-058"></a>058 | `outputs/hpc-results/pubchem-large-corpus-20261105a/cache-and-view-20261106T094305211844Z.json` |
| <a name="dbmol-source-059"></a>059 | `src/latentmol/datasets/bulk_split_policy.py` |
| <a name="dbmol-source-060"></a>060 | `src/latentmol/datasets/bulk_view.py` |
| <a name="dbmol-source-061"></a>061 | `experiments/vae/epoch46-pubchem60m-prior15-v100-benchmark.json` |
| <a name="dbmol-source-062"></a>062 | `outputs/runs/vae-review-20261119-47682960/pubchem-val-summary.json` |
| <a name="dbmol-source-063"></a>063 | `outputs/runs/large-vae-chembl-heldout-20261119a/summary.json` |
| <a name="dbmol-source-064"></a>064 | `outputs/runs/vae-review-20261119-47682960/fixed-panel.json` |
| <a name="dbmol-source-065"></a>065 | `outputs/runs/vae-occupancy-geometry-20261119a/founder-handoff.json` |
| <a name="dbmol-source-066"></a>066 | `src/latentmol/training/prior_repair.py` |
| <a name="dbmol-source-067"></a>067 | `src/latentmol/training/compact_prior.py` |
| <a name="dbmol-source-068"></a>068 | `experiments/vae/epoch47-pubchem60m-parent-guidance.json` |
| <a name="dbmol-source-069"></a>069 | `src/latentmol/training/combined_continuation.py` |
| <a name="dbmol-source-070"></a>070 | `outputs/runs/vae-occupancy-distill-20261119a/summary.json` |
| <a name="dbmol-source-071"></a>071 | `outputs/runs/vae-occupancy-production-smoke-20261119a/summary.json` |
| <a name="dbmol-source-072"></a>072 | `src/latentmol/training/occupancy_monitor.py` |
| <a name="dbmol-source-073"></a>073 | `outputs/runs/vae-epoch47-guidance-baseline-20261119a/summary.json` |
| <a name="dbmol-source-074"></a>074 | `experiments/vae/compact-benchmark.md` |
| <a name="dbmol-source-075"></a>075 | `configs/vae_compact_triangle.yaml` |
| <a name="dbmol-source-076"></a>076 | `src/latentmol/training/matching.py` |
| <a name="dbmol-source-077"></a>077 | `experiments/latent-design/README.md` |
| <a name="dbmol-source-078"></a>078 | `docs/35_boltz_panel_results.md` |
| <a name="dbmol-source-079"></a>079 | `docs/77_dbmol_boltz_loss_implementation.md` |
| <a name="dbmol-source-080"></a>080 | `src/latentmol/boltz/dbmol_model.py` |
| <a name="dbmol-source-081"></a>081 | `src/latentmol/graph_policy.py` |
| <a name="dbmol-source-082"></a>082 | `src/latentmol/boltz/adapter.py` |
| <a name="dbmol-source-083"></a>083 | `docs/54_contact_precision_and_branch_validation.md` |
| <a name="dbmol-source-084"></a>084 | `docs/37_defog_repair_guidance.md` |
| <a name="dbmol-source-085"></a>085 | `docs/49_defog_minimal_integration_smoke.md` |
| <a name="dbmol-source-086"></a>086 | `docs/60_long_boltz_defog_comparison.md` |
| <a name="dbmol-source-087"></a>087 | `docs/62_full_budget_best_trial_optimization.md` |
| <a name="dbmol-source-088"></a>088 | `src/latentmol/design/latent_protenix.py` |
| <a name="dbmol-source-089"></a>089 | `experiments/latent-design/protenix-vae/full.json` |
| <a name="dbmol-source-090"></a>090 | `experiments/latent-design/protenix-vae/README.md` |
| <a name="dbmol-source-091"></a>091 | `experiments/latent-design/protenix-vae/embedding-comparison.md` |
| <a name="dbmol-source-092"></a>092 | `experiments/latent-design/protenix-vae/landscape.md` |
| <a name="dbmol-source-093"></a>093 | `experiments/latent-design/protenix-vae/editability.md` |
| <a name="dbmol-source-094"></a>094 | `experiments/latent-design/protenix-vae/editability.md#warm-start-result` |
| <a name="dbmol-source-095"></a>095 | `experiments/latent-design/protenix-vae/editability.md#bounded-search-result` |
| <a name="dbmol-source-096"></a>096 | `src/latentmol/design/protenix_step_search.py` |
| <a name="dbmol-source-097"></a>097 | `docs/79_protenix_deletion_and_count_selection.md` |
| <a name="dbmol-source-098"></a>098 | `experiments/latent-design/protenix-vae/atom-budget.md` |
| <a name="dbmol-source-099"></a>099 | `experiments/latent-design/protenix-vae/raw-occupancy.md` |
| <a name="dbmol-source-100"></a>100 | `outputs/runs/vae-review-20261117/raw-occupancy-stage005.json` |
| <a name="dbmol-source-101"></a>101 | `experiments/vae/representation-search-review.md#which-search-results-transfer-to-a-real-decoded-graph` |
| <a name="dbmol-source-102"></a>102 | `experiments/vae/epoch45-combined-training-v100.json` |
| <a name="dbmol-source-103"></a>103 | `docs/project-review-2026-11-18.md` |
| <a name="dbmol-source-104"></a>104 | `outputs/hpc-submissions/vae-pubchem60m-e47-parent-guidance-20261119a/cutover.json` |
| <a name="dbmol-source-105"></a>105 | `outputs/hpc-submissions/vae-pubchem60m-e47-parent-guidance-20261119a/plan.json` |
| <a name="dbmol-source-106"></a>106 | `docs/HPC_CONNECTION.md` |
| <a name="dbmol-source-107"></a>107 | `scripts/hpc/vae_segment_chain.py` |
| <a name="dbmol-source-108"></a>108 | `experiments/vae/occupancy_recovery.py` |
| <a name="dbmol-source-109"></a>109 | `docs/workflows.md` |
| <a name="dbmol-source-110"></a>110 | `docs/59_interactive_optimization_reports.md` |
| <a name="dbmol-source-111"></a>111 | `src/latentmol/guidance/repair_proposals.py` |
| <a name="dbmol-source-112"></a>112 | `src/latentmol/training/compact_benchmark.py` |
| <a name="dbmol-source-113"></a>113 | `docs/23_frozen_model_plugin_contract.md` |
| <a name="dbmol-source-114"></a>114 | `experiments/vae/representation-search-review.md#alternatives-if-the-current-representation-loses-that-comparison` |
| <a name="dbmol-source-115"></a>115 | `AGENTS.md` |
| <a name="dbmol-source-116"></a>116 | `docs/EXPLORATORY_RESEARCH_POLICY.md` |
| <a name="dbmol-source-117"></a>117 | `docs/GRADIENT_VALIDATION_POLICY.md` |
