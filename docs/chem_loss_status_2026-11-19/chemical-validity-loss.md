# Chemical validity during molecular design

Technical reference for agents working on DBMol. Evidence reviewed through 19 November 2026. The main application here is latent optimization of a ligand against the BRD4 pocket using a frozen epoch-45 VAE and a frozen Protenix scorer. Earlier decoder-training and chemistry-only experiments are identified explicitly. [One-page group handout](chemical-validity-loss.pdf) · Experiment home (`experiments/chemistry-guidance/README.md`)

Source and evidence paths are relative to the root of the full DBMol exploratory checkout. This status snapshot on `main` contains the guide and its companion PDF.

## How to choose a starting point

| Question | Where the evidence points |
|---|---|
| How can an invalid molecule give the optimizer a direction? | Search outside autograd for a valid, slot-aligned repair. If found, differentiate the probability of its labels through the current decoder. |
| Can an atom change? | Current repair holds this step’s atom and charge labels fixed. The next latent update can change both. A separate mixed-search implementation also allows atom and charge replacements during repair. |
| Does forming a functional group earn a discount? | The implemented preference compares with a fixed starting reference. Losing a protected feature costs more and restoring it releases that cost. A newly formed group earns no intrinsic reward and receives no automatic protection. |
| Which tested weighting is worth comparing with current guidance? | At atom 2 / structure 2, more steps and endpoints were valid on the native pilot. At 2 / 0, all six endpoints contained zero sulfur. Neither improved every measured outcome. Structure weight 0.5 was sensitive to both atom weight and execution. |
| Which question remains open? | Whether a molecule keeps an *identified* aromatic ring or functional group during binding optimization. All three native starting molecules had nonaromatic rings and lacked the three recognized functional-group cores. |

## Follow one latent update

1. **Decode.** A latent vector `z` passes through the frozen VAE decoder. It returns categorical logits for an atom type and a formal charge per slot, plus a bond type per pair. The native study fixes 25 occupied slots. Symmetric bond types include an explicit absent-bond class. Temperature and the occupancy mask define one probability distribution over these decisions.
2. **Score the proposed molecule.** Protenix consumes these *current probabilities*. Its gradient can change atom types or connectivity at the next step. Independently take the most likely label at each decision to obtain a discrete graph. The raw graph is what we count when reporting validity.
3. **Find a repair when needed.** Detach the labels and probabilities for the discrete search. A candidate that passes the chemistry and representation checks becomes a *witness*, a specific valid target reachable by the search. A bounded failure is recorded as *censored*. It means the search used its budget without finding a witness. It says nothing conclusive about whether a valid repair exists.
4. **Compute the chemistry signal.** If a witness exists, increase the probabilities of its labels. If the starting graph for this step already passes, or the search ends without a witness, the repair term is zero. An enabled preservation term still acts in both cases. The witness stays fixed during differentiation. The next step can select a different witness.
5. **Move the latent.** Combine the two gradients and bound the displacement. Re-decode and repeat. The native scorer never receives the repaired molecule as a replacement for the current proposal. Molecular validity at step `t+1` is an observation, even if the target at step `t` was valid. Update and gradient composition (`src/latentmol/design/latent_protenix.py:175`) · Stable target likelihood (`src/latentmol/repair_guidance/loss.py:49`)

### The scalar being minimized

The chemistry check matters at two points. Raw inspection builds a molecule from the predicted labels, checks connectivity, sanitizes it with RDKit and rejects radicals. Repair acceptance also checks whether the candidate’s canonical molecular string can be encoded in the shared graph schema. A starting reference has the stricter requirement that its canonical representation preserve the supplied labels and slot mapping. These distinctions explain how a graph can be recorded as raw-valid yet be ineligible as an initial reference or excluded from a subsequent fingerprint calculation. Raw inspection (`src/latentmol/chemistry.py:168`) · Repair verification (`src/latentmol/guidance/repair_proposals.py:184`) · Reference mapping (`src/latentmol/chemistry.py:224`)

For the fixed-search study, use the following notation. `r` is the verified repair for the current step, `s` is the initial reference graph, `q_d` is the current categorical probability for decision `d`, and `K` is the number of occupied slots. The code evaluates target log probabilities stably from logits under the same sampling law.

```text
L_total(z) = L_contact(z) + λ × [L_repair(z; r) + L_preserve(z; s)]

L_repair   = mean over d of −log q_d(r_d)      if there is a witness
           = 0                                otherwise

L_preserve = α × (wA × A + wQ × Q + wM × M)
A          = mean over occupied atoms of [1 − q_atom(initial element)]
Q          = mean over occupied atoms of [1 − q_charge(initial charge)]
M          = mean over protected decisions of [1 − q_d(initial label)]
```

The repair mean includes all `K` element labels, all `K` charge labels and each of the `K(K-1)/2` unordered occupied pairs once. At `K=25` the denominator is 350. Unchanged labels and absent bonds enter the likelihood too. Thus a target found solely by editing bonds still supplies gradients for atoms and charges. Treating the objective as cross-entropy on edited bonds alone changes its meaning. In the controlled experiment `λ = 1`, `α = 0.05`, `wQ = 0`, temperature was 1 and there was no added noise. Baseline omitted the entire preservation term. A zero independent charge weight still permits the motif term to penalize a charge change inside a protected group. Loss and denominator (`src/latentmol/repair_guidance/loss.py:49`) · Preservation implementation (`src/latentmol/guidance/exchange.py:218`) · Exact sweep config (`experiments/chemistry-guidance/preservation-protenix.json`)

The native contact objective estimates, for each ligand atom, the highest predicted probability of contact within 8 Å with a member of the protein pocket. It takes the negative mean over ligand atoms, so lower is better. The fixed-context geometry-off interface transmits element probabilities, expected formal charges and bond presence. Single, double and aromatic edges with identical presence have the same direct native bond feature. The typed graph, decoder and chemistry checks can still respond differently. A hard rescore uses the discrete molecule in this same context. Measured affinity and a fresh conformer or pose are separate readouts. Native feature map and objective (`src/latentmol/guidance/mosaic_protenix_worker.py:30`)

## Discrete search cost and differentiable loss

The word “cost” has led to two different experiments. First identify the configuration key. The generic design runner treats them as mutually exclusive. Configuration and dispatch (`src/latentmol/design/latent_protenix.py:67`) · Comparison arm selection (`src/latentmol/design/exchange_protenix.py:26`)

| Mode | Candidate generation and ranking | Gradient used after search |
|---|---|---|
| Current guidance | Atom identities, formal charges and occupancy stay fixed during repair. Structural proposals can join disconnected pieces or prune or downgrade bonds. A bounded bond-edit beam is the fallback. | Verified-target mean NLL when a witness exists. |
| Mixed search, `exchange_guidance` | Proposes atom, charge and bond replacements at fixed occupancy. The additive variant uses replacement probabilities plus atom, charge and protected-reference costs in ranking complete candidates. | Verified-target mean NLL plus continuous reference mismatch. The same configured weights enter the two calculations with different normalizations. |
| Current search plus `preservation_guidance` | Calls the current structural search with its existing proposals, depth, ranking and verifier. | Adds continuous reference mismatch to current guidance. This is the comparison that isolates the effect of the preservation terms. |

Current fallback settings were up to three single-bond edits, width four and up to 32 checks per depth, within the common 96-candidate budget. A structural proposal may coordinate many bond changes in one candidate. Current ranking includes unit edit cost and a positive log-probability regret. The mixed native sweep used up to two edits, width four and up to 48 checks per depth. It shares opportunities among edit families and can continue after finding its first valid candidate. Diagnostic priority and bounded proposal order affect which candidates get evaluated. Equal check budgets do not equate these search spaces. Neither search certifies a globally nearest repair. Structural search (`src/latentmol/guidance/repair_proposals.py:166`) · Mixed search (`src/latentmol/guidance/exchange.py:154`) · Mixed configuration (`experiments/chemistry-guidance/exchange-protenix.json`)

For the mixed `one_minus_probability` variant, the cost of a completed candidate `c`, starting from the *current* raw graph `g`, is

```text
sum over labels changed from g of [1 − p(new label)]
  + wA × number of changed atoms
  + wQ × number of changed charges
  + wM × number of protected decisions in c that disagree with initial s
```

The first three terms compare candidate to current graph. The last compares candidate to initial reference. At replacement probability 0.25, an unprotected atom exchange costs 1.25 with atom weight 0.5 and 2.75 with atom weight 2. An unprotected bond replacement at that probability costs 0.75. If that bond originally matched a protected reference label, breaking the match adds another 0.5 at structure weight 0.5. Restoring a previously lost label decreases the candidate's reference-mismatch count by one. An edit can therefore reduce the total reference cost while still paying its ordinary edit cost. Constant surcharges influence the witness selected. The continuous reference term supplies the gradient that can act even when the search requests no edits. The earlier `log_regret` variant has a different base cost. Candidate cost and counting (`src/latentmol/guidance/exchange.py:116`) · Additive-cost validation (`experiments/chemistry-guidance/atom-exchange-linear.md`)

An infinite atom surcharge can describe forbidding atom substitutions *inside* a search. Taking the mixed weight to infinity would also amplify the reference term when enabled and would retain the mixed proposal schedule and edit radius. Current guidance is recovered by selecting current search and zero reference weights. Freezing atom identities through optimization itself would impose an additional constraint. Fixed-search control validation (`experiments/chemistry-guidance/preservation-protenix.md#controlled-native-comparison`)

## What “structure” means in this implementation

Build the reference once from valid, slot-aligned labels. RDKit identifies its rings and the following functional-group cores. The code then freezes the union of the marked atom slots and unordered internal pairs. In the native study the reference is the unperturbed starting molecule. In the chemistry-only tests it is the valid molecule *before* perturbing its latent. Reference construction (`src/latentmol/guidance/exchange.py:71`) · Start selection (`src/latentmol/design/exchange_protenix.py:39`)

| Reference feature | Match rule | Which decisions count |
|---|---|---|
| Ring | RDKit perceived atom rings, aromatic or nonaromatic | Atom identity, formal charge and every pair inside the ring, including absence of extra chords |
| Amide core | `[NX3][CX3](=[OX1])` | The matched N, C and O and their internal pairs |
| Acid or ester core | `[CX3](=[OX1])[OX2]` | The matched C and both O atoms and their internal pairs |
| Sulfonyl core | `[SX4](=[OX1])(=[OX1])` | The matched S and two O atoms and their internal pairs |

Overlapping motifs count a shared label once. Attachments outside the core are usually free of the group penalty unless another protected motif includes them. If no motifs match, the motif mean is zero. A new amide at a previously unprotected site gains no special discount. It may still benefit other objectives, such as producing a valid graph. Once formed, it can disappear without a newly acquired protection cost. A new group at an overlapping site can change the cost because of the *original* protected decisions there. Reference masks and distance (`src/latentmol/guidance/exchange.py:87`)

**Aromaticity requires two measurements.** A chemically valid conversion can destroy an aromatic reference ring. Conversely, a raw single/aromatic label discrepancy can normalize to the same aromatic molecule. The initial reference requires a label-preserving canonical representation. Raw validity, canonical chemical identity, exact mapped categorical retention and presence of any aromatic ring answer different questions. In particular, counting a new aromatic ring elsewhere as retention of the founder ring obscures the actual change. Compare mapped identities after chemical perception and report literal label mismatches separately. Graph inspection and canonical mapping (`src/latentmol/chemistry.py:168`) · Aromaticity audit (`experiments/latent-design/aromatic-rings.md`)

**Why a nominal motif weight of 0.5 can be weak.** The three native reference graphs had 235, 80 and 176 protected decisions, including 158, 35 and 104 reference nonbonds. If one protected bond goes from probability one to zero and every other probability stays fixed, the extra continuous loss is `0.05 × 0.5 / D_M`. That is 0.000106, 0.000313 and 0.000142 respectively. Even complete disagreement on all protected decisions contributes at most 0.025 through the structure term at this weight. The separate atom term can add cost if an atom changes. Highly confident categorical probabilities can also leave small local gradients. The arithmetic establishes dilution of an individual decision. It does not identify the primary cause of a particular trajectory failure. Per-reference counts and ring retention (`outputs/preservation-protenix-20261112a/review-interactive-20261119/ring-retention.json`)

## How the loss evolved

### A local valence rule exposed an atom-type shortcut

An older penalty used the excess of incident bond order above the maximum allowed for an element and charge. Hold neutral charge and bond orders fixed at a sum of five. A carbon cap of four gives excess one, a sulfur cap of six gives excess zero. A model could reduce that penalty simply by moving probability from C to S. Sulfur can legitimately have expanded valence, so indiscriminately capping it would also damage valid chemistry. A separate decoder-training test added a neighbor term at weight 0.1. At matched update 10, local valence violations decreased from 89 to 36 against control, but disconnected graphs rose from 172 to 352 and the valid count fell from 734 to 611 out of 1,024. A better local metric can accompany worse whole molecules. Sulfur mechanism audit (`docs/84_sulfur_bias_in_validity_guidance.md`) · Neighbor training pilot (`docs/88_local_neighbor_loss_pilot.md`)

### Whole-graph repair improved target discovery

On two frozen epoch-43 CPU panels, 123 of 512 raw prior decodes were invalid. The structural proposals followed by the existing beam found valid targets for 123/123, compared with 122/123 for the beam alone. Serial discovery took 2.677 versus 9.889 seconds. Of 122 commonly repaired cases, 119 gave identical target labels. The benchmark measured discovery under fixed atom identities. It did not include an optimization trajectory or establish that a later decode would reach the discovered target. Repair benchmark and stress cases (`experiments/chemistry-guidance/README.md#fixed-atom-repair-proposals`)

### Chemistry-only optimization tested exchange and anchoring

The first mixed pilot used log-regret cost, 16 perturbed reference molecules and 16 chemistry-only updates. Atom 2 / structure 2 gave 16/16 valid endpoints and 1.125 mean protected changes versus 3.50 for current repair. Atom 5 / structure 2 yielded 15/16 valid endpoints. The follow-up implemented the requested additive cost and used 48 distinct references over three seeds, 32 updates, LR 3, unit temperature and a 0.5 displacement cap. Of the references, 46 had rings and nine had amides. Categories overlapped. Acid/ester and sulfonyl patterns were covered by fixtures only. First pilot (`experiments/chemistry-guidance/atom-exchange.md`) · Additive study and controls (`experiments/chemistry-guidance/atom-exchange-linear.md`)

| Additive chemistry-only arm | Valid endpoints | Protected changes, mean | Endpoint Morgan distance |
|---|---:|---:|---:|
| Current bond-only | 48/48 | 4.646 | 0.9053 |
| Atom 0.5 / structure 0.5, search costs only | 47/48 | 4.500 | 0.9034 |
| Atom 0.5 / structure 0.5, continuous term enabled | 48/48 | 2.500 | 0.9003 |
| Atom 2 / structure 0.5, continuous term enabled | 48/48 | 2.250 | 0.8965 |

The 0.5 / 0.5 arm selected no atom edits, no charge edits and 14 bond edits across accepted repairs. Its improved preservation therefore cannot be attributed to repeatedly replacing atoms in repair targets. In a controlled carbon-star fixture, an atom surcharge of 0.5 selected sulfur at all four tested replacement probabilities, whereas weight 2 selected bond repair at all four. Yet one real chemistry-only trajectory at 2 / 0.5 temporarily gained two sulfur atoms. Candidate ranking and latent drift are different mechanisms. The small decrease in mean fingerprint distance also accompanied fewer protected changes. Fixture, matched endpoints and transient sulfur (`experiments/chemistry-guidance/atom-exchange-linear.md`)

### Native Protenix exposed a different tradeoff

The native pilots ran 500 updates at LR 70 from each of three screened valid, sulfur-free, 25-atom molecules. They held the frozen epoch-45 decoder, initial latent and occupancy mask fixed across arms within a start. The model inputs, contact context and pocket were common. The mixed arms used charge and structure weights 0.5. The fixed-search arms used zero standalone charge weight. These are conditional results on valid starts and correlated trajectories. Native configs and starting procedure (`experiments/chemistry-guidance/exchange-protenix.json`) · Controlled sweep (`experiments/chemistry-guidance/preservation-protenix.json`)

| Mixed-search pilot | Current | Atom 0.5 / structure 0.5 | Atom 2 / structure 0.5 |
|---|---:|---:|---:|
| Valid source graphs / 1,500 | 1,184 (78.9%) | 766 (51.1%) | 373 (24.9%) |
| Valid step-500 endpoints / 3 | 0 | 1 | 1 |
| Mean extra sulfur atoms during the path | 0.523 | 0.372 | 0.078 |
| Distinct parseable valid molecules | 715 | 480 | 257 |
| Mean pairwise Morgan distance | 0.8657 | 0.8610 | 0.8697 |
| Final relaxed contact loss, mean | −0.436 | −0.370 | −0.353 |
| Selected valid hard contact loss, mean | −0.539 | −0.413 | −0.268 |
| Invalid inputs without a witness | 7 | 200 | 466 |

Current guidance gave the strongest selected hard contact score on all three starts. After search *did* find a valid witness, the next source was valid in 162/310 cases for current guidance, 137/534 for mixed 0.5 and 112/661 for mixed 2. The sets of inputs differed by arm. In a separate same-input replay of 99 stratified saved states, current search recovered 2/12 sampled censored cases for mixed 0.5 and 7/12 for mixed 2. Seven of those nine recoveries needed a third edit. This isolates a search-depth effect on specific inputs and leaves additional variation in target choice and latent response. Completed native mixed-search study (`experiments/chemistry-guidance/atom-exchange-protenix.md`) · Stratified replay (`experiments/chemistry-guidance/preservation-protenix.md#completed-same-input-cpu-replay`)

The baseline can still add sulfur. Of 60 stepwise sulfur increases in the three original controls, 53 followed exactly zero chemistry gradient. Seven followed a bond-only repair target. The first two sulfur gains for seed 111310, at steps 1 and 16, followed binding-only updates. This is direct evidence about the gradients preceding those additions. The historical maximum-valence shortcut is a separate result. Positive drift also does not imply every molecule or every slot became sulfur. Recorded sulfur transitions (`outputs/exchange-protenix-20261111a/collected-20261119/repair-audit.json`)

### The controlled native sweep isolated reference preservation

All seven arms used current repair discovery and shared initial molecules. In the next table, each cell reads **batch / interactive**. Each execution contributed three trajectories per setting. The mixed study contributed nine additional executions, including three baseline traces byte-identical to the batch controls. Altogether there were 51 completed executions of 500 updates and three independent starting structures. A source-state count of 1,500 per arm per execution is useful for describing time spent valid. Treating those observations as independent trials would understate uncertainty. Batch measurements (`outputs/preservation-protenix-20261112a/review-batch-20261119/batch-metrics.json`) · Interactive measurements (`outputs/preservation-protenix-20261112a/review-interactive-20261119/loss-details.json`)

| Atom / motif | Valid sources % | Valid endpoints per 3 | Sulfur gain, path mean | Distinct valid molecules | Morgan distance | Final relaxed contact | Selected hard contact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current | 78.9 / 67.0 | 0 / 1 | .523 / .746 | 715 / 506 | .8657 / .8787 | −.436 / −.344 | −.539 / −.436 |
| 0.5 / 0 | 69.7 / 45.4 | 1 / 1 | .231 / 1.019 | 412 / 306 | .8783 / .8722 | −.307 / −.337 | −.422 / −.404 |
| 0.5 / 0.5 | 72.5 / 62.2 | 3 / 1 | .516 / .309 | 593 / 516 | .8682 / .8735 | −.301 / −.364 | −.525 / −.496 |
| 2 / 0 | 76.5 / 50.8 | 1 / 2 | .413 / .088 | 714 / 390 | .8764 / .8726 | −.529 / −.363 | −.612 / −.369 |
| 2 / 0.1 | 66.0 / 55.8 | 2 / 1 | .611 / .046 | 668 / 311 | .8910 / .8634 | −.391 / −.336 | −.445 / −.377 |
| 2 / 0.5 | 50.5 / 78.9 | 1 / 3 | .354 / .160 | 326 / 610 | .8676 / .8718 | −.426 / −.408 | −.471 / −.537 |
| 2 / 2 | 86.4 / 88.2 | 3 / 3 | .509 / .551 | 625 / 733 | .8608 / .8692 | −.467 / −.345 | −.445 / −.383 |

**Validity.** At 2 / 2 each of the six matched trajectories improved in fraction of valid source graphs and all six endpoints passed. The controls gave one valid endpoint across six. Mean selected hard contact was weaker in both executions. At 2 / 0.5 in batch, 740 of 743 invalid sources failed sanitization and three were disconnected. Just 75 of those 743 searches failed to produce a witness. A repair signal can be present through an extended invalid portion of a trajectory. Full interpretation and repair counts (`experiments/chemistry-guidance/preservation-protenix.md#fixed-search-outcomes`)

**Sulfur and diversity.** At 2 / 0 all six endpoints had zero hard sulfur gain although sulfur appeared transiently. The baseline ended at +0.67 sulfur atoms on average in each execution, while 2 / 2 ended at +1.00 / +0.67. The 2 / 0 arm had the best batch hard score, largely from one start, and a weaker interactive mean than current. With atom 0.5, adding structure 0.5 raised validity and improved mean selected hard contact relative to atom-only in both executions. With atom 2 the same addition changed validity by −26.0 percentage points in batch and +28.1 interactively. At 2 / 0.5 the pool of distinct parseable molecules moved from a control of 715 to 326 in batch and from 506 to 610 interactively. Pairwise fingerprint diversity stayed much closer. Diversity of the survivors and yield of distinct survivors both matter. Weight comparison figure (`outputs/preservation-protenix-20261112a/review-interactive-20261119/weight-comparison.png`)

**Ring survival.** The three founders had 7, 4 and 4 detected rings. At 2 / 2, zero of those 15 batch rings and three of those 15 interactive rings retained every mapped atom, charge and internal pair label at endpoint. With the looser criterion that all original cycle edges still exist, the counts were zero and five. The baseline preserved one ring exactly in batch and zero interactively. Individual rings can overlap, and both analyses included invalid endpoints. Among distinct valid molecules, occurrence of any aromatic ring at 2 / 2 fell from 37/715 to 16/625 in batch and from 86/506 to 5/733 interactively. The founders had no aromatic ring to preserve. A separate frozen-prior audit found aromatic rings in 236/7,541 valid argmax molecules versus 9,808/10,000 held-out MOSES molecules. This motivates screening or deliberate selection of aromatic founders for a retention study. Endpoint ring audit (`outputs/preservation-protenix-20261112a/review-interactive-20261119/ring-retention.json`) · Aromaticity across other campaigns (`experiments/latent-design/aromatic-rings.md`)

**Execution sensitivity.** All three control starting latents matched exactly and initial relaxed scores agreed to within 0.00000006. For seed 111210, first-step binding gradient norms were 0.5202 and 0.1602 with zero chemistry gradient. Both displacements hit the 0.5 cap, their cosine was 0.9886, and their difference had norm 0.0754. The hard graphs differed after one update. For the other two controls, first displacement differences were smaller than 0.000004 and the first observed molecular differences appeared later. Of 1,500 updates per setting and execution, 1,289–1,493 reached the cap. LR 70 describes the requested step prior to clipping. Precision, hardware and native execution environment were not isolated as causes. For a new comparison, retain within-start pairing and record the numerical environment of each repeat. First-update replay (`outputs/preservation-protenix-20261112a/review-interactive-20261119/baseline-step-replay.json`)

## Reading the results from disk

| Metric | Definition used here | Read this record |
|---|---|---|
| Trajectory validity | Fraction of pre-update hard graphs passing the recorded whole-molecule check, steps 0–499. Keep the inspecting routine and schema version with a comparison. | `arms/latent/candidates.jsonl`, keyed by step |
| Endpoint validity | Raw hard graph at step 500. Count all starts, including invalid outcomes. | Last candidate row |
| Sulfur drift | Hard sulfur count minus that run's step-0 count. Compute the mean over steps 0–499, maximum through 500, and endpoint change separately. Include invalid graphs. Expected sulfur sums the per-slot sulfur probabilities. | `chemistry-observations.json` |
| Repair availability | Already valid, witness found or censored. The current search also checks schema roundtrip, so an occasional graph that passes raw inspection can still trigger search. | `arms/latent/metrics.jsonl` and per-step search receipts |
| Protected distance | Count of differing mapped atom identities, charges and internal pairs against the fixed starting reference. This hard integer differs from the averaged soft training signal and from the number of whole motifs still intact. | Observation rows plus the saved reference motif slot lists in the cohort's `run.json` |
| Distinct molecules and diversity | Canonical valid SMILES across steps 0–500 pooled per execution and arm. Reparse for fingerprints. Use radius-2, 2,048-bit Morgan fingerprints and mean pairwise 1 − Tanimoto. Keep Murcko scaffold count if assessing scaffold breadth. | Candidate rows, cohort summaries and RDKit-parsing exclusions |
| Final relaxed and selected hard contact | Final relaxed loss can be reported even for an invalid endpoint. For a hard score covering every trajectory, find the valid step with the lowest relaxed loss, then rescore its one-hot graph in the original context. That selection optimizes the relaxed criterion among valid candidates, without exhaustively evaluating all hard scores. | `arms/latent/metrics.jsonl` and `hard-readouts.json`, roles `initial`, `endpoint`, `best_relaxed_valid` |
| Step size and technical status | Requested vs applied displacement, cap hits, separate binding and chemistry norms and available cosines. Completion checks include finite required paths and frozen model integrity. | `arms/latent/updates.jsonl`, metrics and `result.json` |

One baseline graph passed raw inspection but its canonical SMILES could not be reparsed. Preserve its original raw-valid status when reproducing reported validity, and exclude it transparently from Morgan calculations. This is why 716 distinct recorded-valid baseline strings gave 715 parseable molecules. All fixed-search arm pools here fit below the summarizer’s 1,000-candidate sampling threshold. A future larger pool may be sampled for fingerprint distance while its exact unique count is retained. An invalid endpoint has no hard endpoint score. Leaving that field missing avoids biasing the reported mean with a fabricated zero. Native readout selection (`src/latentmol/design/exchange_protenix.py:61`) · Cohort aggregation (`experiments/chemistry-guidance/summarize_exchange.py:12`) · Fingerprint convention (`experiments/chemistry-guidance/benchmark_exchange.py:43`)

## A discriminating next experiment

**Primary question.** If a starting molecule contains an aromatic ring or amide that we want to keep during binding optimization, can an explicit group-level penalty improve retention of that particular mapped group at acceptable validity, sulfur exposure, diversity and contact? Treat a benefit as a paired difference across independent starting structures. The experiment below is proposed work.

1. **Select founders that address the question.** Use valid, representable, slot-aligned 25-atom founders with identified groups. Include aromatic carbocycles, supported heteroaromatics and amides, plus examples of acid/ester or sulfonyl cores where available. Record the exact atom mapping, starting sulfur and saturation of the reference probabilities. Include matched founders lacking the tracked group to measure spontaneous formation. Stratify by scaffold and group. Retention of a legitimate sulfur-containing motif calls for monitoring added sulfur relative to its own baseline.
2. **Change one thing first.** Reuse the current search, scorer, initial latents, budgets and within-device controls. At a common atom weight, compare no motif term, the present per-decision mean and one proposed term per designated motif. Repeating current 0 / 0 and the earlier 2 / 2 setting gives interpretable baselines. Compare on the same founders and include a repeat block to quantify execution sensitivity. Keep dropouts, invalid endpoints and search failures visible.
3. **Specify the group term in chemical terms.** For an amide, mark N–C, C=O and their atom identities. For an aromatic ring, specify mapped cycle edges and aromaticity-equivalent chemistry. One soft proposal is to take the largest reference mismatch among a group's defining decisions, then average over designated groups. This gives a single broken defining bond weight even inside a large scaffold. It also concentrates gradients on the worst decision, so a smooth maximum and a simple mean over defining edges are meaningful alternatives. Calibrate effective scale on the same saved examples before comparing weights. Overlap handling, charge/protonation conventions, and whether extra internal chords violate the intended chemistry should be explicit.
4. **Keep retention, restoration and formation separate.** Record survival at each step, first loss and later restoration of each original mapped group. Count new occurrences elsewhere separately. If the scientific goal is to *encourage formation*, compare an explicit formation incentive on initially motif-negative founders. If the goal is to *keep a group after it forms*, test a separate rule that remembers where and when it first formed. That rule adds path-dependent state and needs its own persistence and replay checks. Rewarding arbitrary motif counts would also reward repeated copies, so report copy counts and scaffold diversity in that experiment.
5. **Measure the tradeoff on molecules.** Report paired exact group retention at step 500 and among selected valid candidates, along with raw validity, yield, hard and expected sulfur trajectories, fingerprint diversity, and hard contact. Distinguish retaining original positions from retaining an equivalent chemical motif after canonicalization. Rescore a size-matched sample of promising valid molecules with fresh geometry to test whether fixed-context contact gains survive. Estimate uncertainty across founders or scaffold blocks. Use trajectory steps to describe dynamics within each run.

For a fast diagnosis before a larger pilot, use a few saved examples representing a broken defining bond, a restored amide, an equivalent aromatic representation, an unprotected new motif and an absent internal chord. Compare scores and gradients at identical decoder outputs for each rule. Then check whether a short matched latent trajectory increases group survival. The first check isolates what the objective rewards. The second captures the decoder's response to that reward. Current test fixtures (`tests/guidance/test_exchange.py`) · Existing matched replay (`experiments/chemistry-guidance/audit_exchange_search.py`) · Gradient validation policy (`docs/GRADIENT_VALIDATION_POLICY.md`)

## Implementation and reproduction

The principal implementation paths are `latentmol.guidance.exchange` for fixed references and optional costs, `latentmol.guidance.repair_proposals` for current repair discovery, `latentmol.repair_guidance.loss` for stable target likelihood, and `latentmol.design.latent_protenix` for gradient composition and latent updates. The comparative native runner is `latentmol.design.exchange_protenix`. The experiment home contains the configs and summarizer. When an analysis needs historical equivalence, use the source archived with that run. Package map (`src/latentmol/README.md`) · Experiment records and configs (`experiments/chemistry-guidance/README.md`)

From a repository environment with the project installed, these focused tests exercise cost arithmetic, reference masks, summary denominators and the comparative runner. The recorded fixed-search validation reported 80 focused checks, including same-input targets and NLL, zero-weight gradient parity on valid and invalid fixtures, and reference persistence on resume. That is evidence about the implementation on those fixtures. The native between-execution difference remains a separate observation. Fixed-search validation record (`experiments/chemistry-guidance/preservation-protenix.md#controlled-native-comparison`)

```sh
python -m pytest tests/guidance/test_exchange.py tests/guidance/test_exchange_summary.py tests/design/test_exchange_protenix.py -q
```

For a new allocated CPU replay of an existing cohort, supply the actual saved cohort, matching checkpoint and a fresh output directory. The script stratifies by original search status. To compare weights using native gradients, use an allocated GPU, the prepared native BRD4 context and its Mosaic Python environment. The invocation below runs the seven fixed-search arms for one screening seed. A comparable three-start execution uses the other two configured seeds, 111310 and 111410, in separate destinations. The launcher enforces an allocation, verifies the checkpoint hash, and rejects an existing output directory. These are reproduction templates, with asset and output paths to supply for a new study. Replay entry point (`experiments/chemistry-guidance/audit_exchange_search.py`) · Native runner (`src/latentmol/design/exchange_protenix.py:139`) · Allocation and transport (`docs/HPC_CONNECTION.md`)

```sh
python experiments/chemistry-guidance/audit_exchange_search.py \
  --cohort COMPLETED_COHORT --checkpoint EPOCH45.pt \
  --output NEW_AUDIT --per-status 12

python -m latentmol.design.exchange_protenix \
  --config experiments/chemistry-guidance/preservation-protenix.json \
  --checkpoint EPOCH45.pt --mosaic-config NATIVE.json \
  --mosaic-python MOSAIC_PYTHON --seed 111210 --output NEW_OUTPUT
```

The complete aggregate reports and per-start audits are available for follow-up analysis without rerunning Protenix. The batch and interactive reports must retain their separate execution identities. The original mixed report was rebuilt from completed records after the historical summarizer failed to fingerprint the one unparseable baseline string. Mixed-search report (`outputs/exchange-protenix-20261111a/collected-20261119/comparison.json`) · Batch fixed-search report (`outputs/preservation-protenix-20261112a/review-batch-20261119/batch-report.json`) · Interactive fixed-search report (`outputs/preservation-protenix-20261112a/review-interactive-20261119/comparison.json`) · Project research policy (`docs/EXPLORATORY_RESEARCH_POLICY.md`)
