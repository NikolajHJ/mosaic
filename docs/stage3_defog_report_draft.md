# Stage 3 · Guiding relaxed molecular graphs with custom repair, DeFoG and projection

## How do we tell an invalid molecule which bonds to change?

Protenix can assign a promising pocket-contact score to a molecule that falls into fragments when we pick its most likely bonds. A pass/fail chemistry check identifies the problem. To keep optimizing, we also need a direction. We designed the custom loss to give us explicit control over the edits a repair may propose, how it ranks them and how strongly its gradient competes with Protenix.

**Choosing a target.** Start with the current *hard graph*, obtained by selecting the most likely category for each atom and bond. For an invalid graph, the structural search holds occupied slots, elements and formal charges fixed. It proposes connected backbones and can remove or downgrade excess bonds. A candidate must be connected, pass RDKit sanitization and survive conversion back into our graph schema. Each changed bond costs one plus a confidence penalty. At the unit penalty weight used here, the penalty is the natural log of the old category's probability divided by the proposed category's probability, with a small probability floor in the search. The algorithm keeps the cheapest verified structural candidate it discovers. A bounded search can miss a cheaper repair or find no complete target. Its fallbacks can use partial edits, local valence and connectivity penalties. [Structural search](https://github.com/NikolajHJ/mosaic/blob/bcef814cdb07ed056a47747f47f458e59d6d97e6/src/mosaic/small_molecule/chemistry/structural.py) · [Stage 3 repair configuration](../experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_repair_comparison.py)

**A two-fragment example.** Suppose either of two possible single bonds could join the fragments, and either change alone would produce a valid molecule. At site A the optimizer gives “no bond” probability 0.52 and “single” probability 0.46. At site B they are 0.90 and 0.08. Both bridges require one edit. A costs **1.12** and B costs **3.42** under the search rule. If it finds both, the search picks A because the optimizer already regards that bond as plausible. These probabilities illustrate the calculation.

**Turning the target into a gradient.** For one update we freeze the chosen target and minimize negative Protenix contact plus the weighted **average negative log probability of the target's choices**. At 25 occupied neutral atoms the average covers 25 element rows, 25 fixed-charge rows and 300 atom pairs. Raising the probability of “single” at A lowers the loss. The hard graph gains the bridge only when that choice wins its row. Each update can choose a fresh target. A graph that already passes receives zero repair loss at that step. The gradient flows through the probabilities, with search and chemistry checks outside differentiation. [Repair loss implementation](https://github.com/NikolajHJ/mosaic/blob/bcef814cdb07ed056a47747f47f458e59d6d97e6/src/mosaic/small_molecule/chemistry/loss.py)

## What did tuning the custom loss change?

In one early sparse-start run at weight 1, the search found a valid target at all **30,001** recorded states, yet the optimizer's argmax never became valid and connected. Late in the trajectory a fluorine still had two bonds and the proposed repair required three bond edits. Over the final 5,000 updates the median ratio of repair to Protenix bond-gradient norms was about 0.06. Averaging the complete-target loss over 350 decisions can weaken the signal on the few bonds that determine validity. This motivated trying stronger repair. [Recorded trajectory and gradient analysis](https://github.com/NikolajHJ/mosaic/blob/bcef814cdb07ed056a47747f47f458e59d6d97e6/experiments/brd4-jq1/protenix-random-chem/runs/20261102-results-interpretation-01/README.md)

The early logit-space experiments used learning rate 0.3. We tried constant repair weights 1 and 10, delayed onset, a chemistry-first period and adaptive weighting from repair and Protenix gradient norms. We compared an edit-by-edit search with a search that first builds a connected backbone. Other trials varied initialization, temperature and gradient-scaled noise. One 50,000-update protocol held repair at 1 through update 20,000, increased it geometrically to 30 by update 30,000, cooled the softmax from 0.7 to 0.4 and annealed added noise. The combined changes leave cooling's individual contribution unresolved. [Strategy comparison](https://github.com/NikolajHJ/mosaic/blob/bcef814cdb07ed056a47747f47f458e59d6d97e6/experiments/brd4-jq1/protenix-random-chem/runs/20261031-chemistry-strategies-01/README.md) · [Annealing protocol](https://github.com/NikolajHJ/mosaic/blob/bcef814cdb07ed056a47747f47f458e59d6d97e6/experiments/brd4-jq1/protenix-random-chem/runs/20261103-sparse-annealed-seeds-01/README.md)

The Stage 3 runs compared two repair weights on the same five-fragment starts, with 500 probability-space updates, learning rate 60, unit Protenix weight, temperature 1, and projection off. Both weights used the same search limit of six structural candidates and 1,024 bond operations per candidate, plus the same fallback. GPU models differed between arms. The larger weight retained more valid states but explored fewer distinct valid molecules and gave a lower best valid contact on each seed. [Matched experiment and run records](../experiments/learned-prior/protenix-defog-composition-30000/repair-pair-20261118.md)

| Seeds 17, 23 and 42 | Repair weight 1 | Repair weight 10 |
|---|---:|---:|
| Valid connected states, including initialization and repeat visits | 174 / 1,503 | 819 / 1,503 |
| Valid connected endpoints | 1 / 3 | 3 / 3 |
| Distinct valid molecules per seed | 63, 54, 53 | 14, 10, 13 |
| Mean of each run's best valid hard Protenix contact | 0.352 | 0.288 |

### How the three methods guide the same graph

- **Custom repair** supplies a specific verified target when search succeeds. Cross-entropy encourages the optimizer to realize it. The edits, acceptance rules and edit costs are inspectable, and the repair term switches off once the hard molecule qualifies.
- **DeFoG** supplies soft atom and bond targets from a model trained on molecules. Squared error pulls the probabilities toward those predictions. A learned preference can distinguish arrangements that both satisfy our rules, and continue guiding a valid graph. The most probable categories can still combine into a fragmented molecule.
- **Connected projection** acts after the gradient update. It redistributes bond probabilities to limit valence, attempts to bridge components, and reapplies the valence rule. It can accompany either loss. We then check whether the resulting hard molecule is connected and sanitizes. [Stage 2 operators](../src/mosaic/small_molecule/Stage2/operators.py)

## Why stop after one DeFoG query?

Our first approach treated DeFoG as a molecule generator. In the 500-update Boltz/DBMol run we drew four candidate graphs from the VAE decoder every five accepted updates. Each supported candidate entered DeFoG at model time about 0.8. We sampled the remaining 447 flow transitions to an output molecule, chose the first eligible output, and moved the VAE latent variable in the direction that raised the decoder's probability of that target. Across 100 refreshes the run selected 100 distinct slot-aligned target graphs. **The target changed in all 99 successive pairs, even though the input argmax stayed identical in 81.** Both the candidate draw and the stochastic continuation could move the target.

To isolate the continuation, we ran both targeting methods on the same frozen checkpoint, five identical starting graphs across arms, and the same model time. Each input received eight full continuations and eight one-query evaluations. All **140/140 within-input pairs** of sampled endpoints differed, by a median of 11 atom or unordered-bond labels. In the one-query arm even the probabilities before argmax were identical on every repeat. Sampling retained a different strength: 39/40 unfiltered endpoints were valid and connected. Reading out the most likely categories from the one-query teacher gave a valid connected graph on 3/5 inputs. The 28 pairs within each control share draws. [Protocol, per-control ranges and raw results](../experiments/learned-prior/protenix-defog-composition-30000/target-repeatability-20261125.md)

The denoiser's first forward pass already predicts the probabilities of the clean atom and bond categories. The sampler would use these to draw the next state. We take the probabilities directly, before that draw. This lets a near tie between single and double bonds give the optimizer credit for either. In the reported Protenix experiments we compare the current graph probabilities with this clean prediction using squared error, hold the target fixed for one update and query again on the next graph. These are observable atom and bond distributions. No hidden DeFoG layer or VAE representation enters this loss.

The switch removed sampling variation at fixed input and reduced 447 denoiser calls to one per target. Local smoothness and optimization quality still had to be tested. The earlier relaxed-input pilot found that a pyridine atom crossing an argmax tie changed a teacher row by 0.499 in total variation for a 0.0002 input change. The first 1,000-update Protenix pilot kept the same valid carbon chain throughout. Our later BRD4 experiment measured the benefit of adding this teacher to projection. The historical Boltz and later Protenix studies used different scorers and optimizers, so they cannot isolate the effect of the targeting change on the best contact score. [Locality evidence and optimization context](../experiments/learned-prior/protenix-defog-composition-30000/target-repeatability-20261125.md)

## DeFoG guidance on relaxed graphs

Adding DeFoG to Stage 2 projection helped Protenix find higher-scoring, chemically valid molecules in all nine matched seed and initialization combinations we tested. The benefit was clearest when we retained the best molecule visited during optimization. Only three of the nine guided runs still had a valid, connected molecule at the final update.

## A learned prior on the relaxed graph

Stage 2 uses explicit rules to limit atom valence and encourage connectivity. Those rules leave many possible choices of atom type and bond order. We asked whether a model trained on molecular graphs could guide the optimizer among those choices. We used the frozen GuacaMol checkpoint of DeFoG, a discrete flow-matching model, to supply a molecular prior. Frozen Protenix-Tiny supplied the pocket-specific contact objective. [Project description, §6](https://github.com/NikolajHJ/mosaic/blob/2eeeba20261633cd297ff79b15b527465ead890e/docs/project_desc.pdf) · [Qin et al., ICML 2025](https://proceedings.mlr.press/v267/qin25d.html)

We optimized the molecule directly as probabilities over six elements at 25 atom positions and five bond categories for each atom pair. These Protenix experiments used no VAE encoder, decoder or latent variable. At each update we queried DeFoG once on the current graph. Its graph transformer received the relaxed probabilities, while auxiliary features that required discrete inputs were computed from the current argmax. DeFoG returned predicted clean atom and bond distributions. We held those predictions fixed for one gradient update, then queried the model again. The loss combined negative Protenix contact with a weighted squared distance to the predictions, averaged separately over atoms and atom pairs. All arms restored each probability row to sum to one. Arms with *connected projection* also applied Stage 2 valence constraints and attempted to bridge disconnected components. [Optimizer implementation](../experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_campaign.py)

## What DeFoG added to projection

We scored the raw argmax at initialization and after every update. A molecule qualified if it passed RDKit sanitization and all 25 atoms belonged to one connected component. Our primary outcome was the **highest hard Protenix contact score among qualifying molecules** in each run. We scored the discrete graph with the same pocket and model key as its relaxed counterpart. Higher contact is better under this model. We also counted valid states and checked the final graph, because the optimizer can lose a molecule it found earlier.

The main comparison used one BRD4 pocket, learning rate 60, 500 updates and unit Protenix weight. Three seeds were each tested with a sparse connected start, a dense random start and a fragmented start. The broader sweep varied teacher weight and time, projection, cooling and the way the teacher target was constructed. The table compares the selected constant weight of 0.5 and teacher time of 0.8 against projection alone. For the fragmented seed-17 comparison, we used both members of the completed rerun because the original control stopped early. [Sweep design](../experiments/learned-prior/protenix-defog-composition-30000/README.md) · [Run and recovery record](../experiments/learned-prior/protenix-defog-composition-30000/run-20261113.md)

| Outcome across nine matched runs per method | Connected projection | Connected projection + DeFoG |
|---|---:|---:|
| Mean best valid hard contact per run | 0.284 | **0.398** |
| Valid, connected states across the trajectories | 54 / 4,509 (1.2%) | **2,563 / 4,509 (56.8%)** |
| Runs ending valid and connected | 0 / 9 | **3 / 9** |
| Mean absolute difference between relaxed and hard contact, all states | **0.021** | 0.039 |

DeFoG improved the best valid score in all nine pairs. Guided runs visited between 7 and 468 distinct valid molecules each, so the increase in valid states included changes of molecular identity. The absolute score gap grew, however. The prior improved the molecules we could retrieve while leaving the relaxed score a less faithful estimate of the discrete score. Each trajectory contributed 501 correlated observations. The nine comparisons represent three seed blocks on one pocket, and weight 0.5 was selected using this sweep.

At the lighter weight of 0.0625, the comparison with and without projection was particularly informative. Without Stage 2 graph operators, DeFoG found **zero new valid molecules in nine runs**. The only qualifying observations were the three initially valid graphs, the same outcome as with the teacher switched off. With connected projection at that weight, all nine runs found a new valid identity, although none ended valid. This establishes the role of projection at 0.0625. The matched weight-0.5 standalone test on three fragmented starts is reported below. [Stage 3 results and methods](https://github.com/NikolajHJ/mosaic/blob/e1b57fc2cab68e17a39c366aa3fbb8c21a0ff2a9/docs/defog_status_2026-11-19/agent_reference.md)

At weight 0.0625 with projection, holding the initial DeFoG predictions fixed for all 500 updates gave a mean best valid contact of 0.347, compared with 0.318 when the predictions were refreshed each step. The benefit of recomputing the teacher at each update therefore remains uncertain.

Earlier runs explain why connectivity sometimes appeared easier to retain. From a connected carbon chain at learning rate 0.3, projected DeFoG at weight 0.0625 kept 1,998 of 2,001 states valid and connected and visited 68 distinct molecules. In a separate 10,000-update pair with the same chain start and the same cyclic weight schedule, raising the learning rate from 0.3 to 150 reduced valid connected states from 9,958 to 39. Retaining a connected starting graph and assembling one from fragments therefore warrant separate results. [Historical benchmark tables](https://github.com/NikolajHJ/mosaic/blob/e1b57fc2cab68e17a39c366aa3fbb8c21a0ff2a9/src/mosaic/small_molecule/Stage3/BENCHMARKS.md)

## Comparison with the custom structural repair loss

Our custom loss searches for an atom-preserving repair of the current hard graph. A successful repair supplies a target for the relaxed probabilities. The repair loss falls to zero when the current graph qualifies. We compared unprojected repair at weight 1 with DeFoG at weight 0.5 and connected projection on five seeds. Three seeds also received unprojected repair at weight 10. All methods ran for 500 updates at LR 60 from identical five-fragment starts and used matched within-seed Protenix keys and a 0.0001 uniform mixture. We verified matching starting tensors and keys. The new weight-10 runs used an A100, while the historical optimization runs used other GPU types. [Experiment protocol and results](../experiments/learned-prior/protenix-defog-composition-30000/repair-pair-20261118.md)

| Seed | Best valid contact, DeFoG + projection | Repair 1 | Repair 10 | Valid states / 501, in method order | Final component counts, in method order |
|---:|---:|---:|---:|---|---|
| 17 | 0.335 | 0.347 | 0.296 | 47 / 63 / 328 | 4 / 1 / 1 |
| 23 | 0.404 | 0.352 | 0.263 | 440 / 57 / 249 | 1 / 2 / 1 |
| 42 | 0.391 | 0.359 | 0.304 | 38 / 54 / 242 | 4 / 3 / 1 |
| 71 | 0.363 | 0.423 | — | 101 / 57 / — | 6 / 1 / — |
| 101 | 0.507 | 0.349 | — | 243 / 60 / — | 6 / 3 / — |

On the five matched seeds, projected DeFoG gave a higher best valid hard score than repair 1 on three and averaged 0.400 versus 0.366. Both found a valid connected molecule on every seed. DeFoG finished valid once, and repair 1 finished valid twice. Without seed 101, their mean best scores were 0.373 and 0.370. With projection held on for both losses, DeFoG led repair 1 on two of the three original seeds. The five-seed comparison evaluates complete optimization recipes.

In a post hoc comparison at an equal candidate budget, the mean best score among the first 20 distinct valid molecules per run was **0.345 for projected DeFoG and 0.359 for repair 1**. The full 500-update outcome also rewards methods that discover more candidates. [Five-seed results](../experiments/learned-prior/protenix-defog-composition-30000/repair-pair-20261118.md#21-november--completed-five-seed-comparison)

Repair 10 found a valid connected molecule on 3 of the three starts and ended valid on 3. On these same starts, mean best contact was 0.376 for projected DeFoG, 0.352 for repair 1 and 0.288 for repair 10. Weight 10 visited 14, 10 and 13 distinct valid molecules on these seeds, compared with 63, 54 and 53 at weight 1.

## Rescoring the frozen shortlists

We selected up to five distinct valid canonical SMILES per run by original hard contact and kept their exact ordered atom and bond tensors. Five unused Protenix keys (7001–7005) were shared across all shortlisted graphs. The primary fresh-key score is the average for the original rank-1 molecule. As a secondary measure we averaged all shortlisted molecules within each run and then averaged runs equally. We also replayed the original keys on the same A100. All 16 runs, including the three projected repair controls, entered rescoring.

| Method | Paired seeds | Original winner, archived score | Original winner, fresh mean | Full shortlist, fresh mean |
|---|---|---:|---:|---:|
| DeFoG 0.5 + projection | 17, 23, 42, 71, 101 | 0.400 | 0.400 | 0.352 |
| Repair 1 | 17, 23, 42, 71, 101 | 0.366 | 0.366 | 0.334 |
| DeFoG 0.5 + projection | 17, 23, 42 | 0.376 | 0.376 | 0.329 |
| Repair 1 | 17, 23, 42 | 0.352 | 0.352 | 0.333 |
| Repair 10 | 17, 23, 42 | 0.288 | 0.288 | 0.269 |
| Repair 1 + projection | 17, 23, 42 | 0.335 | 0.335 | 0.298 |

The maximum within-molecule range over the fresh keys was zero. The largest absolute difference between an archived score and its original-key replay was 8.9e-7. Every fresh-key evaluation exactly matched its original-key replay on the same A100. Changing the key supplied no independent score variation for these graphs in this inference configuration. Projected DeFoG gave the higher mean best score on the five-seed comparison. On the three shared starts, repair 10 retained connectivity at the endpoint every time, with lower best scores on all three. Fresh keys preserved these rankings, but evaluating one fixed pocket and model leaves performance on other targets open.

[Candidate-level scores](../results/hpc/20261121-repair10-fresh-keys-01/scores.csv) · [Frozen shortlists, keys and archive identities](../results/hpc/20261121-repair10-fresh-keys-01/selection.json) · [Verified scorer and results](../results/hpc/20261121-repair10-fresh-keys-01/completion.json)

## Standalone DeFoG at weight 0.5

The three fragmented starts at seeds 17, 23 and 42 used LR 60, 500 updates, a constant teacher time of 0.8 and the same probability mixture as the matched projected runs. Only the Stage 2 graph operators were disabled. Initial tensors, original scoring keys, model and pocket digests matched. The new runs used an A100, and the historical projected runs used other GPU models.

| Seed | Projection | Valid connected states | Distinct valid molecules | Best valid hard contact | Final components | Final valid |
|---:|---|---:|---:|---:|---:|---|
| 17 | Off | 0/501 | 0 | — | 6 | No |
| 17 | Connected | 47/501 | 47 | 0.335 | 4 | No |
| 23 | Off | 0/501 | 0 | — | 2 | No |
| 23 | Connected | 440/501 | 440 | 0.404 | 1 | Yes |
| 42 | Off | 0/501 | 0 | — | 7 | No |
| 42 | Connected | 38/501 | 36 | 0.391 | 4 | No |

Without projection, 1,501 of 1,503 graphs were fragmented. All 574 graphs that passed sanitization had multiple components. Seed 23 briefly reached a single component twice, but both graphs failed sanitization. With connected projection, all three seeds found valid molecules, accounting for 525 of 1,503 states (34.9%). Seed 23 also ended valid. For this BRD4 search, DeFoG at weight 0.5 supplied useful guidance alongside connected projection. On its own it yielded no valid candidate from the three fragmented starts. Returning the best valid molecule visited preserves the benefit of the projected search. If the final iterate itself must be valid, custom repair at weight 10 achieved that outcome on all three starts, with lower best contact scores. [Run configuration](../results/hpc/20261121-defog05-no-projection-01/plan.json) · [Paired results](../results/hpc/20261121-defog05-no-projection-01/comparison.json)
## Does DeFoG point toward a valid edit?

We took 12 distinct valid 25-atom graphs from the BRD4 shortlists. For each graph we removed a bridge and, in a separate copy, raised a bond order to create a valence violation. The clean graph was a control. At teacher time 0.8, we ranked 1,200 possible single-bond substitutions by the gain in their margin over the current label under the one-query guidance direction. We counted any substitution that RDKit verified as producing a connected valid graph with the same atom labels. [Protocol and raw results](../experiments/learned-prior/protenix-defog-composition-30000/teacher-mechanism-20261125.md)

| Outcome | Deleted bridge | Valence violation |
|---|---:|---:|
| Median rank of the best verified repair | 28.5 | 3 |
| A verified repair ranked in the top ten | 1 / 12 | 12 / 12 |
| Valid and connected after moving 25% toward the teacher | 0 / 12 | 0 / 12 |
| Valid and connected after moving 75% toward the teacher | 0 / 12 | 2 / 12 |
| Valid and connected at the teacher argmax | 0 / 12 | 1 / 12 |

High repair rank shows that the gradient points toward a useful edit. Hard validity still depends on the size of that change and edits to the other bonds and atoms. All clean inputs stayed valid after a quarter- or half-step. Eight stayed valid after a three-quarter-step. The experiment shows stronger evidence for identifying local valence defects than for reconnecting fragments.

### Stage 3 criteria

- **Improve hard validity and connectivity.** Guided projection increased the number of valid states from 54 to 2,563 out of 4,509. Three of nine guided runs finished valid. The three fragmented starts with no projection yielded no valid state.
- **Reduce the soft-to-hard gap.** The mean absolute gap grew from 0.021 to 0.039.
- **Retain contact quality and exploration.** The selected teacher improved best valid contact in all nine matched pairs and yielded 7 to 468 distinct valid identities per run. The earlier 1,000-update carbon-chain pilot retained one identity throughout.

## Does the teacher need to follow the changing graph?

At weight 0.5 and time 0.8, we compared the archived live teacher with two controls using the same fragmented starts, connected projection, learning rate 60 and 500-update budget. The frozen control kept the initial prediction throughout the run. The shuffled control queried every update, permuted prediction rows within each current winning category, and matched the live gradient norm separately for atoms and bonds on the same graph. Four runs have finished. Seed 42 was still queued on 26 November. [Protocol and verified results](../experiments/learned-prior/protenix-defog-composition-30000/teacher-mechanism-20261125.md#26-november--first-four-completed-controls)

| Seed | Teacher | Best valid hard contact | Valid states / 501 | Valid end |
|---:|---|---:|---:|:---:|
| 17 | Archived live | 0.335 | 47 | No |
| 17 | Frozen initial | 0.335 | 441 | Yes |
| 17 | Shuffled live | 0.379 | 117 | No |
| 23 | Archived live | 0.404 | 440 | Yes |
| 23 | Frozen initial | 0.334 | 431 | Yes |
| 23 | Shuffled live | 0.472 | 451 | Yes |

The frozen target kept 872 of 1,002 states valid and finished valid on both seeds with one query per run. The shuffled teacher found the highest-scoring candidate on both seeds. Mean best contact was 0.426 shuffled, 0.369 live and 0.334 frozen. The shuffled control retained category-level information and responded to the current graph. These two seeds suggest the original row assignment was dispensable for finding a high-scoring valid molecule in this setup. The live runs are historical and seed 23 used different V100 variants across the archived and new arms.
