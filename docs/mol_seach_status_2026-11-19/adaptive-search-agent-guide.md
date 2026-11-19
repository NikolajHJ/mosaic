# Adaptive search for small molecules

**Agent research guide · 19 November 2026 · DBMol_exploratory · `codex/protenix-restart-search`**

The experiment asks a practical question. Given a fixed number of gradient updates, should we spend them following one molecule, starting over repeatedly, or returning to promising molecules discovered along the way? On the human BRD4 first bromodomain, returning to an updated archive gave the lowest fresh contact loss and the most structurally diverse low-loss molecules on all three seeds of our 48,000-update benchmark. Choosing which molecules to put in the archive becomes especially important when we also care about synthesis. The evidence here is computational and pocket-specific. [Two-page group brief](adaptive-search.pdf) · [Corrected comparison](#source-01) · [Current synthesis-aware protocol](#source-02)

For the algorithm, start below. For a decision about compute, go to [the matched benchmark](#does-feedback-earn-the-compute). For synthesis or docking guidance, go to [the synthesis results](#adding-synthesis-difficulty-to-the-objective) or [docking results](#docking-as-a-second-binding-readout). To set up a new study, use [the runnable baseline](#the-runnable-brd4-baseline), [the file and result map](#how-to-reproduce-and-inspect-a-search), and [the proposed next experiments](#which-experiment-to-run-next).

In this two-file edition, citations lead to the [source register](#source-register). It gives the path and, where relevant, the section in the research checkout.

## What actually happens in a wave

1. **Choose the starting state.** A founder is the first molecule or latent in a lineage. Depending on the experiment, it comes from a random draw or a prior run. For a saved parent we restore the latent vector and its mask of occupied atom slots. That vector holds information about which labels were close calls. A canonical molecular string alone cannot recover this particular state.
2. **Follow a differentiable objective.** The frozen VAE decoder turns the latent into probabilities for atom identities, formal charges and bonds. Protenix uses the relaxed graph to predict contacts with the pocket. The optimizer takes gradients back to the latent, adding the configured chemistry, valence and optional synthesis terms and capping the step length. The decoder and scorer weights stay fixed. A small shift in probabilities can leave every winning label unchanged for many steps, then flip a single bond and produce a new molecule. The 25-slot standard fixes the active mask for the child.
3. **Observe every state.** At each saved update the argmax (the most probable label) gives a discrete graph. Check it for connectivity and chemical validity. Store the graph, latent, source branch, step and score history. A good state can be an intermediate. Its descendant at update 500 may be worse, invalid, or identical to a molecule already scored.
4. **Spend a limited number of fresh scores.** Rank eligible distinct hard candidates, then reconstruct and score the selected molecules with the campaign's hard-graph procedure. The fresh score gives an archive entry a comparable binding readout. Spending fresh scores on every visit would spend the same budget repeatedly on recurring graphs.
5. **Allocate the next wave.** A parent policy chooses eligible archived states. A branch either continues one exactly, takes a small random displacement in latent space, or uses another configured proposal. Run the next group of children and repeat. A promising intermediate can therefore have several children, and an older parent can be selected again. The archive persists even if a later branch gets worse. [Runner and frontier rules](#source-03) · [Latent update and chemistry implementation](#source-04)

The search tree records **which saved state actually initialized a child**. A branch may revisit the same molecular identity at different steps or in a different lineage. Collapsing those visits into a single node across the whole experiment would lose parentage. For a root-to-leaf trace, follow each child's recorded parent back to its exact source update and append the child's observations. The focused report implements this and exposes the per-generation best valid molecule. The path chart shows the recorded per-state loss. Fresh scores exist only at selected states. [Interactive tree and molecular playback](#source-05) · [Report behavior and regeneration](#source-06)

## How to read a score or a count

| Recorded quantity | What it answers | What to watch |
| --- | --- | --- |
| Relaxed contact loss | Does the differentiable protein model favor this *probability-valued* graph? | Useful for steering and shortlisting. Its numerical scale and rank can differ from a fresh score on the argmax. |
| Raw hard validity | Is the graph of winning labels connected and accepted by the chemical checks? | Report the fraction of valid **observations** as well as distinct valid identities. A trajectory parked on one valid graph can have perfect validity. The graph schema omits stereochemistry. |
| Fresh Protenix contact loss | How well does a selected hard molecule score once rebuilt for evaluation? | Lower is better in this BRD4 readout. Score failures, including conformer failures, consume attempts. The 2.30 cutoff is specific to this campaign. |
| Audited hard SA | How difficult does a fragment/complexity heuristic judge this hard molecule to be? | Lower is easier on the familiar roughly 1–10 scale. The audited cutoff uses the project's pre-smoothing value. Keep the package SA alongside it. The two can differ dramatically. |
| Strict joint hit | Does one valid hard molecule have audited SA ≤5 **and** fresh Protenix loss ≤2.30? | SA ≤6 is a separately reported secondary cutoff. A confirmed strict hit also passes the contact threshold on each of two held-out geometry seeds. |
| Distinct, diverse, or new | Did we see a different identity, a different structural neighborhood, or an identity absent from earlier data? | Deduplicate canonical identities first. The main diversity readout uses a greedy Morgan radius-2 fingerprint panel with pairwise Tanimoto ≤0.60. “New” is relative to a named historical reference. Report within-seed and pooled counts separately. |
| Native Vina / route found to stock | Does a separate docking program agree, or can the planner reach the declared building blocks? | More negative Vina is better in the same receptor and box. A planner outcome depends on its stock and search budget. Experimental affinity and laboratory synthesis require their own evidence. |

Three different controls are easy to conflate. A **near-valid** start is an invalid hard graph with a saved bounded repair witness. A **near-miss** synthesis founder is already valid and easy by audited SA, but narrowly misses the fresh binding cutoff. An **independent start** depends on the study: the large matched benchmark accepted raw Gaussian draws, while an earlier small lineage study screened them for validity. [Near-valid design](#source-07) · [Founder definitions](#source-02) · [Benchmark initializations](#source-01)

## Does feedback earn the compute?

The corrected benchmark asks where to spend 48,000 gradient updates on one BRD4 pocket. All three methods used the same frozen epoch-45 decoder and Protenix scorer, 25 occupied slots, learning rate 60, step cap 0.25, chemistry and valence weights of 1, and zero optimization noise. Within a seed, all methods ran on the same GPU. The primary outcomes were the lowest fresh loss, the mean loss of the three best distinct molecules, and the number of structurally distinct low-loss molecules. For the last measure, the analysis takes molecules with fresh loss ≤2.30 and greedily retains candidates with Morgan fingerprint Tanimoto similarity ≤0.60 to every molecule already retained. [Benchmark design](#source-01)

**Adaptive** and **independent** began with the same 12 Gaussian latent draws, with occupancy fixed from the decoder. Their first wave therefore supplies a shared baseline. From wave two, adaptive requested four archive parents and three 500-update children per parent, including two exact continuations across the wave. Independent drew 12 new latents each wave and optimized each for 500 updates, accepting the initial draw regardless of hard validity. **Continuous** took the first shared draw and followed one trajectory for 96 consecutive 500-update windows, carrying forward the terminal latent and mask. Scoring a window left that trajectory intact. [Benchmark preset](#source-08)

Each method used 48,096 relaxed scorer calls per seed. Each 500-update window could nominate up to six distinct hard candidates, for up to 72 fresh attempts per wave and 576 per method per seed. Earlier versions matched everyone to the smallest shortlist. When continuous revisited familiar molecules, that rule deprived adaptive of the evaluations used to select its next parents. The corrected version gave each arm its own allowance, recorded unused calls, and charged failures as attempts. Use jobs **29998584, 29998585 and 29998586** for this comparison. [Final records](#source-09)

| Seed | Method | Best loss ↓ | Top-three mean ↓ | Diverse low-loss molecules | Fresh attempts / successes |
| --- | --- | ---: | ---: | ---: | ---: |
| A · 541101 | Adaptive | 2.12258 | 2.12678 | 106 | 532 / 527 |
| A | Independent | 2.18616 | 2.19023 | 26 | 576 / 562 |
| A | Continuous | 2.28249 | 2.29597 | 2 | 82 / 82 |
| B · 651101 | Adaptive | 2.09676 | 2.09981 | 214 | 566 / 559 |
| B | Independent | 2.17370 | 2.20008 | 25 | 564 / 552 |
| B | Continuous | 2.26136 | 2.29463 | 1 | 34 / 34 |
| C · 761101 | Adaptive | 2.13091 | 2.15719 | 97 | 474 / 464 |
| C | Independent | 2.20542 | 2.20710 | 16 | 563 / 555 |
| C | Continuous | 2.30458 | 2.31640 | 0 | 54 / 54 |

Adaptive won all three readouts in each seed. Its mean best loss was 2.11675, compared with 2.18843 for independent and 2.28281 for continuous. Adaptive used 1,572 fresh attempts across seeds, independent 1,703 and continuous 170. The corresponding successful scores were 1,550, 1,669 and 170. The claim concerns identical updates and available scoring budgets. Consumed scoring work and elapsed time differed. In seed C, adaptive reached its best by wave three, yet its diverse low-loss yield rose from 40 to 97 by wave eight. A flat best-so-far curve can conceal gains in the set of choices available to a chemist. [Per-seed curves and cost accounting](#source-10)

## Which use of a branch slot helped?

**Updating parents versus reusing the same founders.** A smaller study started from three disjoint panels of three previously scored parents. Adaptive and fixed-parent arms shared first-cycle latents and matched perturbation seeds. In later cycles, one arm promoted newly scored elites and the other returned to the original panel. A third arm drew independent starts screened for an initially valid graph. Five cycles of six 96-update branches gave each arm 2,880 updates and 90 actual fresh attempts per panel. Adaptive beat fixed on best and mean top-three loss in all three. Mean best losses were 2.22593 adaptive, 2.25552 fixed and 2.21533 independent. Independent won the best single score in two panels. Summing within-panel diverse low-loss yields gave 32 adaptive, seven fixed and ten independent. These results support feeding discoveries back into the archive. The shorter budget and valid-start screening change what the small independent comparison measures. [Study design](#source-11) · [Panel A](#source-12) · [Panel B](#source-13) · [Panel C](#source-14)

**More parents versus more children.** At 12 children per wave, all allocations shared the initial 12 draws, six waves, 500 updates per child and 36,000 updates per seed. From wave two they requested different numbers of parents. This study matched actual fresh attempts within each seed, giving each allocation 419 in A, 430 in B and 392 in C. Each allocation won best loss once. The highest diverse yield also moved between allocations. Treat branching factor as a choice to revisit when the objective or available founders change. [Configuration](#source-15) · [Per-seed scores and successful evaluations](#source-01)

| Parents × children | Best loss A / B / C | Diverse low-loss count A / B / C |
| --- | --- | --- |
| 6 × 2 | 2.15770 / **2.09097** / 2.17456 | 102 / 134 / 42 |
| 4 × 3 | 2.13539 / 2.10123 / **2.13091** | 78 / 157 / 87 |
| 3 × 4 | **2.12226** / 2.11417 / 2.14031 | 83 / 96 / 103 |

## What 120,000 updates bought

Each long run used four parents, learning rate 60, and 500 updates per child. Wide allocated six children per parent for ten waves. Deep and near-valid allocated three children per parent for twenty waves. Near-valid permitted a repairable invalid starting state to take a continuation slot. This was one run per configuration, on different GPU families with different numbers of fresh scores. [Long-run specification](#source-16)

| Search | Best fresh loss (wave) | Top-three mean | Diverse low-loss count | Fresh attempts / successes |
| --- | ---: | ---: | ---: | ---: |
| Wide | 2.09213 (8) | 2.10295 | 177 | 1,043 / 1,041 |
| Deep | 2.08135 (15) | 2.08397 | 389 | 1,265 / 1,159 |
| Near-valid | 2.05132 (7) | 2.05801 | 398 | 1,036 / 978 |

Near-valid improved from 2.06081 in wave three to 2.05132 in wave seven. The remaining 78,000 updates added 244 diverse candidates. All 19 near-valid starts recovered validity within 18 updates, with a median of one. Their best direct fresh loss was 2.12706. All four final elites traced to screened latent starts from the lowest-ranked founder. The aggregate win therefore gives little evidence that repair produced the winning molecule. In four paired replays, the exact valid continuation produced the better descendant every time, averaging 2.15841 against 2.47342. Both arms received 2,000 updates and 24 attempts. The exact starts already scored better than the saved repair witnesses, so the replay answers which available slot to spend in those waves. Separating repairability from starting quality would require comparable starting scores. [Near-valid run](#source-17) · [Wide run](#source-18) · [Deep run](#source-19) · [Paired replay](#source-20)

**Two controls put the long-run result in perspective.** In an eight-molecule geometry check, the wave-seven incumbent still beat the wave-three incumbent on all three held-out seeds. The mean gap was 0.00786, compared with 0.00949 at discovery, and 29 of 32 attempted scores succeeded. In a separate three-parent pilot, discrete atom substitutions and bond reattachments gave a mean per-parent best of 2.45124, compared with 2.45228 for gradient-based latent restarts. Each arm had eight successful fresh scores per parent. The latent arm found the best single molecule (2.31108). Direct edits enumerate and validate hard proposals without a gradient. That comparison was shelved to keep the research focused on gradient-guided design. [Geometry replay](#source-21) · [Graph-edit results and scope decision](#source-03)

## What the restart experiments established

### The early dense graphs came from the perturbation

The first scheme gave the winning category half of each atom and bond row’s probability mass and redistributed the rest. For 25 occupied atoms there are 300 unordered atom pairs, most of which initially carry no bond. Taking mass away from every winning category simultaneously turned many absent bonds into plausible ones. Expected bond count rose from 28.6 to 160.6. The graph from the original screenshot went from 27 hard bonds to 235 after one update and back to 27 after the next. Across 24 branches, 21 produced a graph with at least 100 bonds. All 213 such observations occurred within the first 18 updates as the temporary logit offset faded. The sparse graph and screened latent alternatives stayed below 41 hard bonds throughout that panel. [Dense-graph diagnosis and restart definitions](#source-22)

Dense states also exposed a weakness in the chemistry loss. The bounded graph-repair procedure could fail to find a valid repair and then contribute zero repair gradient. All sampled frames with at least 100 bonds had zero chemistry loss. Even the sanitized descendants of one such trajectory contained four to eight three-membered rings. Record hard validity, ring statistics and independent fresh scores alongside training loss. In 80 eligible shortlists, relaxed ranking picked the eventual fresh winner only 13 times. [State and shortlist audit](#source-22)

### How much noise preserves a parent yet changes the path?

A radius-0.03 displacement of the archived latent retained the parent’s exact canonical SMILES in 35 of 36 starts across 12 parents. By update 64, none of 30 sibling pairs with two valid endpoints had the same molecule. Eight pairs had temporarily rejoined. Untouched controls reproduced every saved hard label. In three 500-update persistence panels covering several parent-based perturbations, all 34 jointly valid sibling endpoint pairs differed. Late structural similarities often approached those of independent starts. Distinct endpoints establish divergence, while similarity to the founder and improvement over independent starts answer separate questions. [Doses, path metrics and persistence panels](#source-23)

The 500-update timing comparison used two parents, two children per parent and 24 successful fresh scores per treatment. Mean improvement from the parent’s fresh loss was 0.242 with no noise, 0.278 with a radius-0.03 parent kick, 0.251 with 1% update noise fading to zero by update 100, and 0.217 with both. The update-noise-only arm found the lowest single score and had the highest fraction of valid saved states. A subsequent two-parent panel measured gains of 0.155, 0.240 and 0.311 at parent radii 0, 0.003 and 0.03. These small panels support a modest parent kick and keeping update noise opt-in. The records retain the seed, dose and whether the initial hard molecule changed, so agents can distinguish nearby latents that decode identically. [Timing results](#source-03) · [Parent-dose replication](#source-21)

In the raw latent test, radius 0.03 means adding 0.03 times a unit random direction to the vector. It is a distance in latent space. A separate **screened** proposal tests several directions at radii 0.03, 0.10 and 0.25, favoring a valid graph with one to three changed hard labels and retaining the valid parent as a fallback. For a graph-space alternative, increasing the no-bond weight eightfold at 25% mixing reduced the initial soft bond mass and gave the best mean fresh gain (0.294) in a three-parent panel. It also lowered the fraction of valid observations at both tested mixing levels. These findings argue for recording what a child actually inherited and where it ended, alongside the nominal perturbation settings. [Screened proposal](#source-24) · [No-bond weighting experiment](#source-21)

### Chemistry strength, step size and diversity

In the original LR 150 tests, constant chemistry weights 1 and 10 and a late ramp from 1 to 10 produced best fresh losses of 2.0636, 2.1589 and 2.1001. This single-seed comparison used the early dense restart. Later, repair-only arms had 54–58% raw-valid observations against 92–98% with repair and valence-excess guidance. LR 70 beat LR 30 in four of five paired 46,000-update arms. In a separate LR 70 comparison with update noise off, a displacement cap of 0.5 beat 0.25 on five of eight parents. Mean gains were 0.182 and 0.122 at 96 fresh attempts per cap. Check actual displacement when interpreting nominal LR changes. The original constant-1 run hit its cap on 9,496 of 9,500 early updates. [Original schedules](#source-25) · [LR comparison](#source-26) · [Cap panels](#source-03) · [Step audit](#source-22)

Diversity can be tested cheaply by reselecting a final panel from the same archive. Among six selected molecules, the largest pairwise Morgan similarity fell from 0.712 to 0.392 at a mean fresh-loss cost of 0.00343. Both panels retained the best molecule. During search, an optional duplicate check compares the starting latent, occupied mask and logit offset so different latents with the same SMILES remain eligible. A paired pilot of guaranteed best-parent selection and duplicate refill selected identical sources and seeds in all four waves. Neither mechanism triggered, leaving their effects unmeasured. [Final-panel comparison](#source-27) · [Start deduplication and pilot](#source-03)

## Adding synthesis difficulty to the objective

These BRD4 experiments ask whether a search can produce molecules that pass two tests on the **same valid argmax graph**. The strict discovery threshold is audited synthetic accessibility (SA) ≤5 and fresh Protenix contact loss ≤2.3. Report SA ≤6 separately. SA estimates difficulty from fragments and structural complexity. Fresh contact loss measures a model's response to the decoded molecule. A low score on each earns further evaluation. Neither measurement supplies a synthesis route or a binding measurement. [Scoring rules and implementation](#source-28)

**Use audited SA for historical comparisons.** The RDKit 2024.09.6 implementation applied a discontinuous logarithmic transformation above a scaled score of 8. One difficult nine-ring graph went from 8.001 to 1.161. Keep both the pre-smoothing score and the package output. In the 16-panel audit, the search visited 7,921 distinct valid nonfounders and successfully fresh-scored 694. Apparent distinct hits at SA ≤5 and loss ≤2.3 fell from 41 to zero after correction. At SA ≤6 they fell from 85 to six. Historical trajectories retain the effects of the old selection rule. Later campaigns produced additional, genuinely qualifying hits. [Audit, denominators and corrected examples](#source-29)

| Intervention | Result | Implication |
| --- | --- | --- |
| Weak or strong frozen synthesis-model gradients | Valid trajectory states averaged 99.80% without guidance, 94.68% with weak SynFrag and 57.57% with strong SynFrag. Weak and strong DFRscore gave 98.03% and 74.10%. | Stronger gradients can consume much of a branch on invalid graphs. The audited strong DFR candidates at SA ≤5 had fresh losses of 2.464–2.677. |
| Draw the latent toward a bank of easier molecules | Raising local attraction from 0.025 to 0.2 increased distinct visited SA ≤6 molecules from 1/12/1 to 105/63/77 by seed. Mean validity fell from 99.58% to 90.60%. | Neither strength produced a joint SA ≤6/loss ≤2.3 hit across 54 fresh attempts per condition. Counting visits alone favors the stronger setting. |
| Wave-focused guidance, synthesis-aware parent choice and weak attraction | A four-wave focused/Pareto arm reached SA 5.560 and loss 2.22655. A later binding-ranked arm with weak attraction yielded two strict hits on one of three seeds, at 4.756/2.284 and 4.960/2.258. Their fingerprint similarity was 0.516. | The two-hit seed retained 98.22% validity versus 98.99% without attraction. Joint gains can occur with a small validity change. Their frequency across new founder panels remains open. |

The early four-wave parent-selection example originally reported an SA improvement from 6.127 to 4.217. The audit gives 8.154 to 8.023 for that pair. Preserve the published historical settings when reconstructing runs and use the corrected scores for conclusions about chemistry. [Initial pilots](#source-30) · [Completed follow-up panels and audit](#source-31)

The comparison also tested GASA, DeFoG, within-child schedules and entire waves with stronger synthesis guidance followed by a recovery wave. Whole-wave focus alone failed to improve best or top-three binding after wave one in the original panel. GASA preserved observed validity but gave no clear SA advantage. DeFoG retained high validity in its panel while worsening best novel binding loss from 2.06081 to 2.13525. Treat the pre-audit SA summaries as historical and recompute the joint endpoint on the archived hard graphs before adopting a schedule. [Wave-focus comparison and alternative models](#source-32) · [SA audit](#source-31)

**A change in endpoint changes the conclusion.** Retrospectively applying the strict SA ≤5 and loss ≤2.30 rule to the binding-only 48k runs gave adaptive 0/0/3, independent 1/1/1, and continuous 0/0/0 distinct hits across seeds. The three adaptive hits appeared within 13 updates of one child. The three independent hits had mutual fingerprint similarities of 0.10–0.17. Including the width runs gives nine distinct strict hard molecules in total, seven after the diversity filter. This is why the candidate portfolio and the discovery objective should both be specified before comparing newly proposed policies. [Retrospective identity audit](#source-33)

### Starting near a useful trade-off

Using the same 4,500-update search, three perturbation seeds tested three choices of initial parents. The following counts exclude founders and identities in the study's frozen history. Each row reuses the same three parent molecules across seeds. [Founder comparison](#source-34) · [Pooled identities and history](#source-35)

| Founders | New distinct SA ≤6, loss ≤2.3, by seed |
| --- | --- |
| Valid archive molecules with SA ≤5 and fresh loss just above 2.3 | 6 / 3 / 7 |
| Strongest fresh binding scores from the earlier search | 3 / 0 / 0 |
| Accessible library molecules aligned to the same slots | 0 / 0 / 0 |

One near-miss parent had SA 3.940 and fresh loss 2.34121. At update 191 of wave one, its child reached SA 3.976 and loss 2.28668. It retained the parent's favorable SA while moving across the binding threshold. Across the three seeds, near-miss parents yielded nine unique qualifying molecules. Five first appeared in wave one and four in wave two. Seven shared a carbon attached to two hydroxyls and two nitrogens. Count identities and structurally diverse hits separately when describing this family. [Founder trajectories](#source-34) · [Structural queries and counts](#source-36)

Candidate selection also matters. Of 445 previously unscored SA ≤5 identities, a shortlist ranked by relaxed loss with fingerprint diversity supplied 2/24 strict discovery hits. A disjoint, founder- and SA-matched random sample supplied 0/24. With loss ≤2.4, the counts were 12/24 and 4/24. Replacing the final 500-update continuation with an independent start contributed a new structurally diverse strict hit on **zero of three** paired seeds. Both arms reached eight, zero and zero strict identities by seed, with the same three pooled held-out confirmations. Actual discovery calls were 36/35/31 for standard and 42/40/37 for the replacement. Retain the terminal continuation in this preset. [Archive screen](#source-37) · [Final-slot decision](#source-38)

### Two confirmed leads and the route-planning check

Two molecules identified retrospectively in binding-only searches passed the strict contact threshold on both independent geometry seeds. Each row tracks a single hard identity. [Archived source molecules](#source-39) · [Held-out scores](#source-40)

| Source | Audited SA | Discovery loss | Loss at seed 2203 | Loss at seed 3209 |
| --- | ---: | ---: | ---: | ---: |
| Adaptive, job 29998586, wave 5, step 478 | 4.940 | 2.23848 | 2.23804 | 2.23867 |
| Six parents × two children, job 29994169, wave 5, step 169 | 3.149 | 2.25233 | 2.25689 | 2.25327 |

AiZynthFinder used ZINC building-block stock, a maximum depth of six, 100 iterations and 75 seconds per molecule. Both leads used all 100 iterations without reaching stock. Aspirin and paracetamol each yielded a one-step route. An earlier blinded, binding-matched panel yielded 0/24 routes, including 0/6 in the audited SA ≤5 group, and solved both controls. The stock, policies and search budget limit these results. Laboratory feasibility remains unresolved. RetroGNN's frozen classifier gave the low-SA group the most favorable median logit, but the uniformly unsolved research panel cannot validate its within-panel ranking. [Latest route receipts](#source-41) · [Blinded planner and classifier study](#source-42)

## Docking as a second binding readout

The parent–child example in the group brief illustrates the reason for this second readout. Founder 3 and its wave-one child retain the phenyl and pyrazole rings, while the child closes another ring. Their Vina scores go from −7.833 to −5.289 kcal/mol as the fresh Protenix loss improves from 2.34121 to 2.28668. The separate co-crystal ligand JQ1 redocked at −7.878 kcal/mol and 0.690 Å heavy-atom RMSD in the prepared 3MXF pocket. Reproducing one known pose is a setup check. The archive also contains stereochemically unspecified molecules and timed-out dockings, so retain the number of attempted and successful preparations. [Source identities, poses and scores for the illustrated pair](#source-43) · [Docking setup and control](#source-44)

Native AutoDock Vina evaluates hard molecules at the BRD4 3MXF pocket. More negative scores are favorable. In job 30110682, docking five eligible parents changed the third parent from −3.255 to −3.650 kcal/mol. The selected child stayed valid for 501/501 observations and visited 45 identities. All ten with SA ≤6 had already been fresh-scored. Neither arm added a new joint hit at SA ≤6/loss ≤2.3 after 4,500 updates each. This distinguishes changing a parent from improving its descendants. [Vina branch and scoring receipts](#source-45)

TacoGFN provides a differentiable proxy through atom probabilities and existing nonaromatic bond orders. A replayed founder reproduced the upstream hard score and gave a latent directional derivative of 0.260 by autograd and 0.260 by finite differences. Agreement with native docking was 35/45 pairwise rankings across ten molecules and 6/10 within five eligible parents. In the paired 500-update Protenix search, guidance was capped at 5% of the binding-gradient norm. Both completed pairs supplied six successful fresh scores and six successful native redockings per arm. [Proxy validation and paired design](#source-46)

| Founder | Distinct valid nonfounders, off → guided | Best native Vina, off → guided (kcal/mol) | Guided updates |
| --- | ---: | ---: | ---: |
| 1 | 34 → 68 | −3.776 → −5.086 | 500/500 |
| 2 | 14 → 17 | −4.567 → −4.231 | 467/500 |

The adapter uses the current valid hard topology as an anchor for the soft features. It can differentiate through probabilities on the anchored atoms and supported bonds. The discrete choice of topology has no gradient through this adapter. Invalid anchors can skip guidance. Twenty equivalent SMILES traversals of one molecule spanned 0.478 in predicted score, which makes consistent canonical ordering important. In the Protenix arm 5% is an upper bound on the added gradient norm. A comparison of local native-Vina maps on one fixed pose is a separate exploratory signal. It measures the interaction contribution at those coordinates. [Gradient bridge, traversal checks and native map probes](#source-46)

Neither pair yielded a strict SA/contact hit in its scored panels. At 11:15 CET on 19 November, founder 3 and all three **Boltz2** docking-guidance pairs were queued. Boltz2 uses different archived founders, 31 occupied slots, LR 1 and a fixed-pose objective. Its treatment scales TacoGFN to exactly half the Boltz2 latent-gradient norm on usable updates. Compare that arm to its own unguided control. Pooling it with the 25-slot, LR 60 Protenix pairs would confound model, founder and dose. [Timestamped results and queue snapshot](#source-47) · [Boltz2 guidance design](#source-48)

## The runnable BRD4 baseline

Use the **binding-only adaptive arm** of the 48k study when the question is whether feedback improves binding predictions. Use the **three-wave synthesis-aware preset** when the question is whether a modest budget can find easier, low-contact-loss descendants of existing archive molecules. These are different experiments. An absent archive of replayable near-misses changes the starting problem. [Binding benchmark](#source-08) · [Synthesis protocol](#source-02)

| Part of the synthesis-aware run | Current setting and rationale |
| --- | --- |
| Founders | Three connected, replayable hard molecules with audited SA ≤5 and fresh loss in (2.30, 2.40]. Rank by fresh loss and greedily select founders at pairwise Morgan similarity ≤0.60. Restore the saved latent and mask and check that they decode to that hard identity. |
| Optimizer | 25 occupied slots, frozen epoch-45 decoder, LR 60, maximum latent step norm 0.25, temperature 1, chemistry and valence weights 1. Weak frozen SynFrag guidance has a 0.05 relative gradient cap. Per-update noise is zero. |
| Wave 1 | Start one perturbed child from each of the three founders. Record both the source identity and the identity actually obtained at the child start. |
| Wave 2 | Protect the best eligible parent, then fill using the Pareto trade-off between fresh loss and audited SA and the configured diversity rule. The fresh binding ceiling is 2.35, the parent similarity ceiling is 0.85, and points from the same trajectory are spaced by 100 updates. Inspect any logged fallback. Perturb the chosen starts. |
| Wave 3 | Continue the preceding branches using the terminal or recovery state recorded by the policy. The terminal-slot replacement pilot supports retaining this allocation in the current preset. |
| Per-child shortlist | For replayable, previously unscored hard identities, take up to six. Start with SA ≤5, rank by relaxed contact loss and require pairwise fingerprint similarity ≤0.65. Fill open places with SA in (5, 6] under the same rule. Keep the number left empty. |
| Allowances | Three children **total** per wave × three waves × 500 = **4,500 optimizer updates per arm**. Three founder calls plus up to 54 child calls = **57 discovery evaluations**. Up to three distinct finalists × two held-out geometry seeds = **six additional confirmation calls**. Confirmation occurs after parent selection ends. |
| Novelty | The 19 November preset freezes 59,211 historical hard identities, expanded from an earlier 9,481. Keep each run's original exclusions when reproducing it. Compare methods against one common frozen reference for a new retrospective analysis. A molecule can be new to a run and already known to the branch. |

The parent, shortlist and reported-diversity thresholds serve different purposes. Looser parent eligibility can preserve promising branches while the final 0.60 evaluation panel demands larger structural differences. Pareto ranking retains trade-offs where another eligible molecule has yet to improve both objectives at once. The most attractive chemistry in the library can still start too far from the pocket objective. For a new protein, calibrate the founder window against its own reference panel. [Preset, including controller and fallback settings](#source-49) · [Historical reference receipt](#source-50)

## How to reproduce and inspect a search

From the checkout root, inspect the resolved experiment before allocating a GPU. These commands print the configuration and planned budget. The first reproduces the current synthesis-aware setup. The second resolves the long deep design.

```powershell
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-synthesis-standard.json --arm seed-1 --plan-only
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-search-lr60-long.json --arm deep --plan-only
```

| Task | Where to start |
| --- | --- |
| Define a new arm or a selector | Change a config variant and resolve it with the [central runner](#source-51). It accepts repeated `--set SECTION.FIELD=VALUE` overrides. The [runner guide](#source-03) describes trusted hooks for parent selection and shortlisting and which study families use dedicated runners. |
| Check parent selection and perturbations | [Frontier, spacing, ancestry and diversity](#source-52) · [Child initialization and execution](#source-53) · [Screened latent and graph proposals](#source-24) |
| Check parity between arms | [Matched-search controller](#source-54) · [Budget comparison by completed wave](#source-55) · [Fresh-score and width results](#source-01) |
| Inspect synthesis or docking guidance | [Synthesis wave policy](#source-56) · [Audited hard SA](#source-57) · [TacoGFN gradient adapter](#source-58) · [Native Vina evaluator](#source-59) |
| Trace a molecule | Read `run.json` for the seed, resolved settings and founders, then the wave plan for the source branch/step and perturbation. Per-child `metrics.jsonl` and `raw-blocks/` contain trajectory observations and replayable data. Per-wave `archive.json` and final `result.json` carry scoring and budget records in the general runner. The specialized paired studies also write arm-level selections and confirmation files. [Run outputs and schemas](#source-03) |
| Rebuild a report | Register the completed job and result location in the [run registry](#source-60). Collect playback and tree snapshots for the same list of job IDs, then render with the [report builder](#source-61). Repeated `--job` or `--exclude-job` selects the portfolio. [Exact commands](#source-06) |

For a resumed general search, take the last **completed wave** in `search-state.pt`. It includes founders and archived latents/masks, the invalid pool, attempted identities, previously used starts, source-step history and budget summaries. Pin its hash and use matching model assets and resolved scientific settings in a new output segment. A resumption should start at the following wave. Compare lineage and cumulative costs before joining the two segments in a report. For a live run, reconcile the scheduler row with the terminal result and source receipt. A submission note records the state at its timestamp. [Checkpoint implementation](#source-62) · [HPC transport](#source-63) · [19 November results and receipt provenance](#source-64)

**Account for time as well as calls.** Initialization, compilation, per-update scoring, CPU chemistry, conformer generation and docking can have different costs. Report the GPU model, wall time, completed updates, relaxed calls, fresh attempts, successful fresh scores and native docking attempts separately. On one 32 GB V100, eight children batched with four overlapping CPU workers ran at 2.66 updates/s versus 1.82 for the scalar four-thread baseline. Four batched children reached 2.51 updates/s at about 7,012 MiB peak sampled memory, versus 15,208 MiB for eight. Trajectories differed after 40 updates despite similar initial scores and gradients. Use that as a planning measurement for this stack. [Timing result and conditions](#source-65)

**Keep the Boltz2 work on its own scale.** The related 31-slot BRD4 experiments used a fixed ligand pose, full FP32 Boltz2 DBMol scoring, LR 1 and a separate founder archive. On one A40 panel, 16 × 150 updates produced 40 freshly scored distinct molecules versus 17 for 6 × 400. One V100 panel gave 32 and 35 with essentially equal tenth-best loss at 2,400 updates. In a same-device V100 PCIe follow-up, 75-update children filled 48 score slots versus 32 with 150-update children. Comparing the first 32 actual calls gave identical best and tenth-best scores. These are examples of a search finding more eligible candidates per unit of compute. Shared V100 queue labels alone do not guarantee the same device type for a timing comparison. [Boltz2 design, hardware records and allocation pilots](#source-66)

## Which experiment to run next

The strongest current case is for **keeping the adaptive archive** and measuring how many useful choices it produces. The most consequential uncertainty is whether those choices remain valuable when the founders, pocket or readout changes. The following are proposals, in priority order.

| Question | Smallest informative comparison | Endpoint that would change the decision |
| --- | --- | --- |
| Does feedback transfer? | Repeat the corrected adaptive, raw independent and continuous comparison on another pocket. Use a common set of initial draws within seed and several independently chosen panels. Freeze a target-specific contact threshold on a reference panel. | Compare best, top-three and diverse yield at matched update/score allowances **and** at common elapsed GPU time on the same hardware. Report each seed, including those with zero hits. A consistent advantage at both budgets supports transfer. |
| Is the synthesis gain more than one founder family? | Assemble several disjoint, fingerprint-separated near-miss founder trios. Pair each with a binding-first or route-supported trio. Hold the shortlist, historical reference, update count and confirmation budget fixed. | Compare pooled new confirmed strict hits, diverse hits per panel, the number of founder lineages that contribute, recurring motifs and routes to stock. Consistency across unrelated founders matters more than generating further analogues of one hit. |
| Can docking guidance improve independent redocking? | Finish the three paired-founder comparisons already registered for each backbone. If there is a signal, compare off, 5% and 50% on the **same** scorer and founders with a predeclared definition of cap versus target norm and identical shortlist rules. | Record actual nonzero-guidance steps, valid nonfounders, successful matched redockings, native Vina and the joint SA/contact subset. Favor an extension if the paired docking gains recur without a collapse in the supply of qualifying hard molecules. |
| When should a branch give up its next 500 updates? | Compare a fixed branch length against reassigning a child only after recent windows cease to add new, diverse, low-loss molecules. Use identical founders and an equal global update/scoring allowance. | Report time to the first and kth diverse hit, yield by budget and which branches received the released updates. Plot the increment per wave as well as the incumbent best. A prespecified example is time to ten diverse hits, showing unreached endpoints explicitly. |
| Which hits justify real chemistry work? | Freeze a small, structurally diverse confirmed panel and matched controls. Redock under one receptor protocol, use the same stock/planner budget, and ask a chemist to review plausible disconnections and available building blocks. | Keep computationally confirmed scores, route found to stock, chemist assessment, synthesized material and measured binding as separate outcomes. A solved, reviewable route is a concrete reason to advance a molecule. |

**Decision today.** For another BRD4 synthesis-aware pilot, carry forward small parent perturbations, no per-update noise, fresh scoring of valid intermediates, and the three-wave near-miss preset. Keep the corrected 48k comparison as the binding-only reference. Give new compute first to disjoint founder panels and orthogonal validation. Whole-wave synthesis pressure, expanding invalid starts and stronger docking gradients each need a matched demonstration that the displaced branch slot would have done worse. Record a new outcome with its actual seed, allocation, scoring spend, limitations and links in the [experiment ledger](#source-25). The [exploratory research policy](#source-67) and [gradient validation policy](#source-68) cover the scope of smoke checks for new interventions.

## Source register

Paths below are relative to the root of the research checkout for `codex/protenix-restart-search`. Run `git worktree list` to locate that checkout on this machine. Generated reports and job receipts under `outputs/` live in that checkout’s local outputs. The companion PDF is included beside this guide.

| Reference | Path in the research checkout | Section |
| --- | --- | --- |
| <a name="source-01"></a>Corrected comparison | `experiments/latent-design/protenix-vae/restart-matched-search.md` |  |
| <a name="source-02"></a>Current synthesis-aware protocol | `experiments/latent-design/protenix-vae/restart-synthesis-standard.md` |  |
| <a name="source-03"></a>Runner and frontier rules | `experiments/latent-design/protenix-vae/restart-search.md` |  |
| <a name="source-04"></a>Latent update and chemistry implementation | `src/latentmol/design/latent_protenix.py` |  |
| <a name="source-05"></a>Interactive tree and molecular playback | `outputs/protenix-restart-report-20261110/protenix-restart-report.html` |  |
| <a name="source-06"></a>Report behavior and regeneration | `experiments/latent-design/protenix-vae/README.md` | Rebuild the interactive report |
| <a name="source-07"></a>Near-valid design | `experiments/latent-design/protenix-vae/restart-branch-slot.md` |  |
| <a name="source-08"></a>Benchmark preset | `experiments/latent-design/protenix-vae/restart-matched-search.json` |  |
| <a name="source-09"></a>Final records | `outputs/restart-matched-report-20261116/data/summary.json` |  |
| <a name="source-10"></a>Per-seed curves and cost accounting | `outputs/restart-matched-report-20261116/search-method-comparison.pdf` |  |
| <a name="source-11"></a>Study design | `experiments/latent-design/protenix-vae/restart-lineage-feedback.md` |  |
| <a name="source-12"></a>Panel A | `outputs/restart-finished-20261110/29931143-result.json` |  |
| <a name="source-13"></a>Panel B | `outputs/restart-finished-20261110/29931145-result.json` |  |
| <a name="source-14"></a>Panel C | `outputs/restart-finished-20261110/29931146-result.json` |  |
| <a name="source-15"></a>Configuration | `experiments/latent-design/protenix-vae/restart-matched-width.json` |  |
| <a name="source-16"></a>Long-run specification | `experiments/latent-design/protenix-vae/restart-search-lr60-long.json` |  |
| <a name="source-17"></a>Near-valid run | `outputs/restart-finished-20261110/29943712-result.json` |  |
| <a name="source-18"></a>Wide run | `outputs/restart-finished-20261111/29943696-result.json` |  |
| <a name="source-19"></a>Deep run | `outputs/restart-finished-20261111/29943705-result.json` |  |
| <a name="source-20"></a>Paired replay | `outputs/restart-finished-20261110/29948379-result.json` |  |
| <a name="source-21"></a>Geometry replay | `experiments/latent-design/protenix-vae/restart-followup-20261110.md` |  |
| <a name="source-22"></a>Dense-graph diagnosis and restart definitions | `experiments/latent-design/protenix-vae/restart-lr70.md` |  |
| <a name="source-23"></a>Doses, path metrics and persistence panels | `experiments/latent-design/protenix-vae/restart-noise.md` |  |
| <a name="source-24"></a>Screened proposal | `src/latentmol/design/protenix_restart_proposals.py` |  |
| <a name="source-25"></a>Original schedules | `docs/05_experiment_ledger.md` |  |
| <a name="source-26"></a>LR comparison | `experiments/latent-design/protenix-vae/restart-lr30.md` |  |
| <a name="source-27"></a>Final-panel comparison | `experiments/latent-design/protenix-vae/restart-diversity-lr100.md` |  |
| <a name="source-28"></a>Scoring rules and implementation | `experiments/latent-design/protenix-vae/restart-synthesis-standard.md` | Search and scoring rules |
| <a name="source-29"></a>Audit, denominators and corrected examples | `experiments/latent-design/protenix-vae/restart-synthesis-followup.md` | Audited synthesis scores and validity, 17 November 2026 |
| <a name="source-30"></a>Initial pilots | `experiments/latent-design/protenix-vae/restart-synthesis-pilot.md` |  |
| <a name="source-31"></a>Completed follow-up panels and audit | `experiments/latent-design/protenix-vae/restart-synthesis-followup.md` |  |
| <a name="source-32"></a>Wave-focus comparison and alternative models | `experiments/latent-design/protenix-vae/restart-synthesis-next.md` |  |
| <a name="source-33"></a>Retrospective identity audit | `outputs/restart-joint-retrospective-20261119/pooled.json` |  |
| <a name="source-34"></a>Founder comparison | `experiments/latent-design/protenix-vae/restart-synthesis-founders.md` |  |
| <a name="source-35"></a>Pooled identities and history | `experiments/latent-design/protenix-vae/restart-synthesis-standard.md` | Earlier search results under the joint cutoff, 19 November 2026 |
| <a name="source-36"></a>Structural queries and counts | `outputs/restart-unreported-20261119/chemistry-patterns.json` |  |
| <a name="source-37"></a>Archive screen | `experiments/latent-design/protenix-vae/restart-synthesis-archive-screen.md` |  |
| <a name="source-38"></a>Final-slot decision | `outputs/adaptive-search-handoff-20261119/independent-slot-decision.json` |  |
| <a name="source-39"></a>Archived source molecules | `experiments/latent-design/protenix-vae/restart-archive-hit-confirm.json` |  |
| <a name="source-40"></a>Held-out scores | `outputs/adaptive-search-handoff-20261119/archive-confirmation.json` |  |
| <a name="source-41"></a>Latest route receipts | `outputs/adaptive-search-handoff-20261119/archive-routes.json` |  |
| <a name="source-42"></a>Blinded planner and classifier study | `experiments/latent-design/protenix-vae/restart-synthesis-route.md` |  |
| <a name="source-43"></a>Source identities, poses and scores for the illustrated pair | `outputs/vina-archive-six-20261118/scores.json` |  |
| <a name="source-44"></a>Docking setup and control | `experiments/latent-design/protenix-vae/restart-synthesis-standard.md` | Docking setup and positive control |
| <a name="source-45"></a>Vina branch and scoring receipts | `experiments/latent-design/protenix-vae/restart-synthesis-standard.md` | Completed Vina parent-selection pilot, 19 November |
| <a name="source-46"></a>Proxy validation and paired design | `experiments/latent-design/protenix-vae/restart-synthesis-standard.md` | Can a docking score guide an individual update? |
| <a name="source-47"></a>Timestamped results and queue snapshot | `outputs/adaptive-search-handoff-20261119/summary.json` |  |
| <a name="source-48"></a>Boltz2 guidance design | `experiments/latent-design/protenix-vae/restart-boltz-taco-guidance.json` |  |
| <a name="source-49"></a>Preset, including controller and fallback settings | `experiments/latent-design/protenix-vae/restart-synthesis-standard.json` |  |
| <a name="source-50"></a>Historical reference receipt | `outputs/synthesis-standard-history-20261119a/receipt.json` |  |
| <a name="source-51"></a>central runner | `src/latentmol/design/protenix_search.py` |  |
| <a name="source-52"></a>Frontier, spacing, ancestry and diversity | `src/latentmol/design/protenix_search_frontier.py` |  |
| <a name="source-53"></a>Child initialization and execution | `src/latentmol/design/protenix_search_branches.py` |  |
| <a name="source-54"></a>Matched-search controller | `src/latentmol/design/protenix_matched_search.py` |  |
| <a name="source-55"></a>Budget comparison by completed wave | `experiments/latent-design/protenix-vae/restart-operations/compare_search_budgets.py` |  |
| <a name="source-56"></a>Synthesis wave policy | `src/latentmol/design/protenix_synthesis_waves.py` |  |
| <a name="source-57"></a>Audited hard SA | `src/latentmol/evaluation/synthesis_accessibility.py` |  |
| <a name="source-58"></a>TacoGFN gradient adapter | `src/latentmol/guidance/tacogfn.py` |  |
| <a name="source-59"></a>Native Vina evaluator | `src/latentmol/evaluation/vina_docking.py` |  |
| <a name="source-60"></a>run registry | `experiments/latent-design/protenix-vae/restart-report/runs.json` |  |
| <a name="source-61"></a>report builder | `experiments/latent-design/protenix-vae/restart-report/build.py` |  |
| <a name="source-62"></a>Checkpoint implementation | `src/latentmol/design/protenix_search_state.py` |  |
| <a name="source-63"></a>HPC transport | `docs/HPC_CONNECTION.md` |  |
| <a name="source-64"></a>19 November results and receipt provenance | `outputs/adaptive-search-handoff-20261119/hpc-snapshot.json` |  |
| <a name="source-65"></a>Timing result and conditions | `outputs/restart-finished-20261110/29936883-result.json` |  |
| <a name="source-66"></a>Boltz2 design, hardware records and allocation pilots | `experiments/latent-design/protenix-vae/restart-boltz-search-methods.md` |  |
| <a name="source-67"></a>exploratory research policy | `docs/EXPLORATORY_RESEARCH_POLICY.md` |  |
| <a name="source-68"></a>gradient validation policy | `docs/GRADIENT_VALIDATION_POLICY.md` |  |
