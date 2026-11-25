# BRD4 repair guidance in direct probability space

Does stronger verified bond-repair guidance help the search visit new, connected molecules with better predicted BRD4 pocket contact than their own sparse founder? The frozen Protenix Tiny scorer uses the JQ1 pocket of BRD4 (3MXF, chain A) and the archived geometry-off protocol. Its loss is the negative mean, over ligand atoms, of their highest predicted contact probability to a pocket residue within 8 Å. Lower is better.

## Paired comparison at weights 0.1, 1, 5 and 10

The [original three weights](repair-search-strong.json) and [weight 0.1](repair-search-01.json) use **six generations**. A 500-update founder is followed by five generations with up to four distinct parents and three 200-update children per parent. The search fills twelve slots per child generation, redistributing slots if fewer than four parents qualify. With four parents, each gets one exact continuation and two independently perturbed children. Each seed and weight has 61 trajectories and 12,500 updates. All four weights share each seed’s starting graph, scorer key, GPU family, parent policy and restart settings. See the [prior matched width results](../protenix-vae/restart-matched-search.md) for the seed-dependent ranking that motivated using a middle allocation.

| Seed | GPU queue | Weight 0.1 | Weight 1 | Weight 5 | Weight 10 |
|---:|---|---:|---:|---:|---:|
| 111210 | V100 | 30129506 | 30129343 | 30129344 | 30129346 |
| 111310 | A10 | 30129507 | 30129347 | 30129348 | 30129349 |
| 111410 | L40S | 30129508 | 30129350 | 30129351 | 30129352 |

Each GPU job has a 16-hour wall limit. Total budget is **150,000 updates** across 12 jobs. The [original GPU plan](../../../outputs/protenix-repair-strong-lr60-20261120b/gpu-plan.json) and [weight 0.1 GPU plan](../../../outputs/protenix-repair-low-lr60-20261120c/gpu-plan.json) map job names to outputs. CPU job **30129354** collects the original three weights. The readouts include improved starts, SMILES, hard contact changes, validity, yield and sulfur drift. On sampled states with a verified repair, each trajectory also records how weights 1, 5 and 10 would change the projected update using the same gradients. This measures whether the 0.5 step cap makes two weights behave alike.

The [combined report job 30129512](../../../outputs/protenix-repair-four-weights-20261120c/report-submission.json) collects all four weights after the twelve GPU jobs end. The [combined plan](../../../outputs/protenix-repair-four-weights-20261120c/gpu-plan.json) points to `/work3/s234818/latentmol/runs/protenix-repair-four-weights-20261120c-report/report.md` and the adjacent `report.json`.

Submission used `submit_repair_search.py stage --name protenix-repair-strong-lr60-20261120b --queues gpuv100 gpua10 gpul40s --hours 16 --updates-per-job 12500`, then `submit --name protenix-repair-strong-lr60-20261120b`. The low-weight arm used the same staging settings with `--name protenix-repair-low-lr60-20261120c --config experiments/latent-design/direct-jump/repair-search-01.json`. Fifteen focused CPU tests passed for the original three weights. The earlier, shorter pilot and its reporting job were [cancelled before startup](../../../outputs/protenix-repair-direct-lr60-20261120a/superseded.json).

## First completed four-weight comparison

On 20 November 2026, all four L40S arms for seed 111410 finished their 12,500 updates apiece. All started from the same valid graph, with a discrete predicted BRD4 pocket-contact loss of −0.22809. Lower is better. The eight GPU jobs for the other two founders were still queued.

| Repair weight | Distinct new valid molecules | Valid updates | Best hard contact loss | Where the best arose |
|---:|---:|---:|---:|---|
| 0.1 | 19 | 2.44% | −0.30970 | Fourth child generation |
| 1 | 7 | 1.48% | −0.30428 | Fifth child generation |
| 5 | 2 | 2.50% | −0.27133 | Founder trajectory |
| 10 | 2 | 9.04% | −0.33069 | Founder trajectory, step 464 |

Every new valid identity in the table beat its starting molecule on discrete predicted pocket contact. Weight 10 produced the best score on this founder. Weights 0.1 and 1 improved their best score after branching. At weights 5 and 10 both new valid molecules had already appeared during the founder trajectory. All 20 parent-selection rounds had a valid parent. Vina contributed to ranking in 19 rounds. The final selection round for weight 5 used contact alone.

### Did the children explore different paths?

For seed 111410, each of 188 perturbed children had an unperturbed sibling from the same saved parent probabilities. The restart changed two probability rows by mixing in 0.2% target mass and preserved the initial discrete graph. All 188 pairs eventually had different atom or bond labels at the same update. The median first difference across pairs within each arm was between 10.5 and 24 updates.

| Repair weight | Perturbed children | Different endpoint from exact sibling | Visited a valid molecule the exact sibling never visited | New valid identities after the founder trajectory, across all children |
|---:|---:|---:|---:|---:|
| 0.1 | 42 | 38 | 11 | 18 |
| 1 | 47 | 40 | 5 | 5 |
| 5 | 50 | 1 | 0 | 0 |
| 10 | 49 | 9 | 0 | 0 |

The endpoint comparison includes invalid graphs. At weights 5 and 10 every discrete graph on every perturbed path also appeared somewhere on its exact sibling's path. All 1,341 observed transitions from valid to invalid across these two arms' children removed the same single bond and split the graph into two fragments. The repair search found a one-edit target at every invalid child update in these arms, and every target corresponded to a molecule seen before that generation. On returning to validity the repair loss switches off, allowing contact optimization to break the same bond again. The automatic increase to 1% restart mass never activated because even a single update with different labels counts as a split. [Paired trajectory analysis](../../../outputs/protenix-repair-four-weights-20261120c/seed-111410-relaxed-siblings.json) · [Bond and validity transitions](../../../outputs/protenix-repair-four-weights-20261120c/seed-111410-validity-transitions.json)

### Did extending a child past 200 updates help?

An exact restart from an interior state of a child continues the same optimization path. For seed 111410, aligning the source and restarted trajectories gave identical discrete graphs, canonical SMILES and relaxed contact losses at all 5,587 overlapping child observations. These joins reveal some uninterrupted paths longer than the nominal 200-update segments.

| Repair weight | Child paths observed past update 200 | Previously unarchived valid molecules found on those extensions |
|---:|---:|---|
| 0.1 | 9 | 4 on one path, at cumulative updates 211, 213, 218 and 220 |
| 1 | 8 | 3 on one path, at 319, 320 and 332. The molecule at 319 had the best hard contact loss in this arm (−0.30428). |
| 5 | 4 | 0, with the longest path reaching 417 |
| 10 | 8 | 0, with the longest path reaching 397 |

These seven molecules were already counted in the search results. Selection favored paths with a valid parent, and the extended paths received unequal numbers of updates. The 500-update founder at weight 10 provides a separate example of delayed discovery. Its best hard loss by update 200 was −0.26744, and a new molecule at step 464 improved that to −0.33069. None of the other three founder trajectories found a new valid molecule after step 200. [Exact child continuations](../../../outputs/protenix-repair-four-weights-20261120c/seed-111410-exact-continuations.json) · [Within-wave discoveries](../../../outputs/protenix-repair-four-weights-20261120c/seed-111410-discovery-timing.json) · [Founder trajectories](../../../outputs/protenix-repair-four-weights-20261120c/seed-111410-founder-cutoffs.json)

## Second completed four-weight comparison

On 21 November all four V100 jobs for founder 111210 completed. In increasing repair-weight order (0.1, 1, 5, 10), they found 26, 39, 51 and 43 distinct new valid molecules. Of those, 13, 13, 49 and 43 improved on the starting hard contact loss of −0.18951. Weight 5 reached −0.45985 in the second child generation and weight 10 reached −0.42105 in the first. This founder continued to discover molecules after branching under strong repair guidance. The four jobs for founder 111310 were still queued. [Four-weight receipt](../../../outputs/protenix-repair-four-weights-20261120c/seed-111210-results.json)

## Original pilot design (cancelled)

Each arm optimizes 25 neutral, occupied atom rows and 300 unordered bond rows directly. All arms use projected gradient descent at LR 60, a global proposal norm cap of 0.5, and a per-category probability floor of 0.0001. There is no VAE, learned generator, connectivity operator, per-update noise, or preservation penalty. The [configuration](repair-search.json) gives every setting.

On invalid raw argmax graphs the [repair signal](repair_signal.py) maps the direct vocabulary to the fixed-atom structural search. A verified repair supplies a fixed target for the live probability rows. Its negative log probability is averaged over 25 atoms, 25 fixed neutral charges and 300 bond decisions, including absent bonds. The charge contribution is zero. Already-valid states and invalid states without a verified repair have zero repair loss and distinct log statuses. The repair oracle also runs for weight zero so diagnostics remain comparable. A witness can qualify as a result only if the optimizer independently visits it as a raw argmax.

Each seed has a 500-update founder followed by two generations of four 200-update children, for **2,100 updates per seed and weight**. In each generation, up to two parents contribute two children each. Every eligible valid molecule gets a hard Protenix score and a Vina docking attempt. Parent selection weights their percentile ranks equally, prioritizing successful docking and unexpanded graphs. If every dock fails, hard contact ranks the valid parents. If there are no eligible valid parents, the search ranks up to 20 distinct invalid states by verified repair distance and chemical defects. The child continues from the saved probabilities of the selected state.

One child continues exactly from each parent. The other mixes 0.2% probability mass into one atom row and one existing bond row while preserving the parent's hard graph. The bond perturbation favors no bond. If sibling hard trajectories coincide throughout a generation, the next perturbation from that branch uses 1%. Each branch records its first hard divergence from its siblings. If fewer than two distinct parents qualify, the remaining slots use additional distinct perturbations of eligible parents.

## Readout protocol and original job record

Every visited, distinct valid hard graph, including endpoints, is rescored using the same scorer, slot ordering and random key as its founder. Molecule counts deduplicate canonical, connected, reparsed SMILES. A hit must differ from the founder and have strictly lower hard contact loss. A change of −0.05 means a 0.05 increase in the model's predicted pocket contact score. Per-arm results give the winner, branch and step, loss change, number of distinct valid molecules, validity over optimizer update states, and change in sulfur count relative to the founder. Endpoint sulfur drift is recorded for each final leaf and averaged across those leaves. Founder-only results are available separately.

| Seed | GPU queue | LSF job |
|---:|---|---:|
| 111210 | V100 | 30129256 |
| 111310 | A40 | 30129257 |
| 111410 | L40S | 30129258 |

The three weights share one allocation per seed. The total budget is **18,900 updates**. CPU report job **30129259** depends on all three GPU jobs ending and writes `report.md` and `report.json` to `/work3/s234818/latentmol/runs/protenix-repair-direct-lr60-20261120a-report/`. Per-seed outputs use the same prefix with `-balanced-<seed>`. The [GPU plan](../../../outputs/protenix-repair-direct-lr60-20261120a/gpu-plan.json) identifies the exact locations. At submission on 20 November 2026 all jobs were pending. Twelve focused CPU tests passed, including a finite-difference repair check, loss normalization, parent fallback and end-to-end result selection with a test contact scorer. The [founder verification](../../../outputs/protenix-repair-direct-lr60-20261120a/founders-verified.json) gives the three canonical SMILES, their starting sulfur counts and their connected-component counts.

The [submission script](submit_repair_search.py) stages a new campaign with `stage --name NAME` and queues it with `submit --name NAME`. To rebuild a report from available HPC outputs, run `python experiments/latent-design/direct-jump/repair_report.py --plan PATH/TO/gpu-plan.json --output NEW_DIRECTORY` on a host with the run directories mounted. The [collector](repair_report.py) distinguishes completed, provisional and missing results.
