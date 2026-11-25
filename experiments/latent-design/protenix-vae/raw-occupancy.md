# Native scores on unmodified VAE decodes

Can an occupancy gradient add or remove an atom in the VAE's actual hard
output? At every update the baseline and every trial use the decoder's
unmodified hard graph. The native importer rebuilds the ligand from its
canonical SMILES at its decoded atom count. The run records that identity,
the native atom count and the MOL checksum. Invalid decodes and failed
reference geometry are separate outcomes.

One valid addition and one valid deletion are selected from the decoder's
own logits, closest to the hard occupancy boundary. The candidates change
one slot and its incident bonds. Their native score differences from the
raw baseline multiply the live addition or deletion probability in the
local objective. Both contributions can flow into the same latent. The
scored probe structures stay fixed within that derivative and are rebuilt
at the next latent. A probe only counts as recovered if its chemical
identity appears in the new unmodified decode.

The [DTU config](raw-occupancy-dtu.json) draws up to 32 seed-1730 prior
latents and takes the first four distinct valid starts with both scorable
directions, regardless of which score is better. Each start gets up to four
updates along the gradient and four along its reversal. Trial sizes are
0.25, 0.125 and 0.0625 within a radius of 1.0. A valid raw trial with a
non-worsening score is accepted, including an unchanged decode. A trial
can change several atoms at once. The records distinguish the count delta,
number of occupancy flips, and exact probe match.

Protenix-Tiny evaluates the same BRD4 pocket throughout. The contact proxy
is the mean negative log contact probability across all ligand atom and
pocket residue pairs. The mean gives one loss on the same scale across
counts. Seed 294 is used during updates and seed 1201 for an independent
conformer of each initial and final raw molecule. ETKDG has a 64-iteration
cap per attempt. The pilot has no size penalty or fixed atom mask.

```bash
python -m latentmol.protenix_occupancy.raw_occupancy \
  --config experiments/latent-design/protenix-vae/raw-occupancy-dtu.json \
  --output <new-output-directory>
```

The trajectory accepts a scorer callable, so the same comparison can be
run with Boltz by wiring its
[native SMILES prediction](../../../src/latentmol/boltz/native_rescore.py)
to the exact decoded molecule. The Boltz experiment would declare its own
score, prediction seed and call budget.

## First submission

DTU job **29896354** was submitted on 5 November 2026 at 09:28 CET and
was pending on `gpua10` at the first check. It requests one A10, four cores
and one hour. Native score attempts are capped at 320. The source SHA-256
is `c64d2c792a18c2d0b19250748a6f5a9543f6a66893d8752e4bff4efee366e17f`.
Output is under
`/work3/s234818/latentmol/protenix_occupancy/runs/protenix-raw-occupancy-20261105-a01`.

10 focused CPU tests passed. In a CPU smoke with the frozen epoch-45
checkpoint and a synthetic score of `-heavy_atoms / 30`, the four raw
decodes moved from 23, 24, 22 and 22 atoms to 25, 25, 25 and 25. Across
56 scorer calls, the parsed molecule always matched the reported atom
count. These checks exercise the live latent path and the raw-count
contract.

## Native result

Job **29896354** completed on 5 November 2026 at 10:25 CET after 228
allocated seconds and 52 native score attempts. All four starts completed
both arms. The native fresh endpoint receipts matched the exact raw
decoded SMILES and atom counts.

| Prior draw | Gradient raw atom count | Reversed raw atom count | Gradient fresh loss | Reversed fresh loss |
| --- | --- | --- | ---: | ---: |
| 0 | 23 → 25 | 23 → 23 | 2.71908 | 2.80168 |
| 1 | 24 → 25 | 24 → 24 | 2.50880 | 2.52845 |
| 2 | 22 → 22 | 22 → 21 | 2.81585 | 2.71068 |
| 3 | 22 → 24 → 23 | 22 → 20 | 2.55209 | 2.59584 |

The forward gradient changed raw count in three starts and yielded lower
fresh contact loss than its reversal in three of four. Mean fresh loss
was 2.64895 for the gradient and 2.65916 for the reversal, from a shared
start mean of 2.69890. The reversal won draw 2 by 0.10517. No accepted
count-changing raw molecule exactly matched a scored one-slot probe. The
pilot measures a small, mixed result on a structure-model contact proxy.
[Saved result](../../../outputs/hpc-results/protenix-raw-occupancy-20261105-a01/result.json).

## Expanded comparison

The [expanded DTU config](raw-occupancy-expanded-dtu.json) requests 24
new matched starts from a seed-1731 draw pool. Both arms get up to 12
updates within the original latent radius of 1.0. Two additional small
trial sizes, 0.03125 and 0.015625, let both arms retry when a larger step
lands on an invalid or worse molecule. The original early-stop rules
still apply if there is no usable probe or accepted trial. The native call
budget is 2,000, with a reserve before starting each matched comparison
so both arms and their fresh scores can be collected together.

The primary readout averages fresh losses from seeds 1201, 1202 and
1203 within each start, then compares guided minus reversed loss across
starts. Negative differences favor the guided gradient. Report complete
paired coverage, per-seed results, wins/ties/losses and a descriptive
95% bootstrap interval that resamples starts. Raw count changes and
realized updates are separate outcomes. The four earlier pilot starts
are outside the new comparison.

DTU job **29897059** was submitted on 5 November 2026 at 10:49 CET and
was pending on `gpua10` at the first check. It requests one A10, four
cores and a four-hour wall limit. Eleven focused CPU tests passed.
The source SHA-256 is
`4a5acec20a3b20efe3432bda81caf2bacee11deadf9bc987e0cf8ad774d3fa4e`.
Output is under
`/work3/s234818/latentmol/protenix_occupancy/runs/protenix-raw-expanded-20261105-a01`.

### Expanded native result

Job **29897059** completed on 5 November 2026 in 27 minutes 46 seconds
of allocated time. It finished all 24 matched starts and used 487 native
score attempts. All three fresh geometry seeds were available for each
start and both arms. All 144 arm endpoint scores matched their exact raw
decoded chemistry and atom count.

| Fresh geometry seed | Mean guided minus reversed loss | Guided wins | Reversed wins | Ties |
| --- | ---: | ---: | ---: | ---: |
| 1201 | -0.03974 | 14 | 6 | 4 |
| 1202 | -0.03977 | 14 | 6 | 4 |
| 1203 | -0.03946 | 14 | 6 | 4 |

The paired mean difference after averaging seeds within each start was
-0.03966, with a descriptive 95% bootstrap interval of [-0.07902, -0.00033].
Mean losses were 2.61254 guided and 2.65220 reversed. The interval approaches
zero and the experiment uses one protein pocket and one contact proxy.

| Raw VAE outcome | Guided | Reversed |
| --- | ---: | ---: |
| Starts with any count change | 15 | 10 |
| Starts with an addition | 9 | 3 |
| Starts with a removal | 7 | 8 |
| Net growth at endpoint | 8 | 2 |
| Net shrinkage at endpoint | 7 | 8 |
| Accepted addition steps | 10 | 3 |
| Accepted removal steps | 8 | 8 |
| Accepted updates | 67 | 60 |

One start in each arm had both an addition and a removal. Four guided
count-changing updates and one reversed update recovered the exact
identity of an individual scored probe.
[Saved result](../../../outputs/hpc-results/protenix-raw-expanded-20261105-a01/result.json).

## Call-matched random latent search

Does the occupancy gradient improve fresh scores beyond random proposals
with the same accept-if-nonworsening rule? The
[control config](raw-occupancy-random-dtu.json) binds job 29897059's result
and replays the 24 original latent draws, checking each hard SMILES and atom
count. The saved guided arm supplies the reference endpoints.

```bash
python -m latentmol.protenix_occupancy.raw_occupancy \
  --config experiments/latent-design/protenix-vae/raw-occupancy-expanded-dtu.json \
  --random-control experiments/latent-design/protenix-vae/raw-occupancy-random-dtu.json \
  --output <new-output-directory>
```

Each control gets its guided counterpart's actual distinct optimization
score attempts, including the baseline, occupancy probes, and failed native
geometry attempts. These caps total 182, with 4–13 per start. The control
can use those calls on random proposals. It uses the original five step
sizes, radius 1, exact raw-molecule validity, and first non-worsening
acceptance rule. It can redraw after a stalled direction, up to 256
directions per start. This matches the native call ceilings and allows
extra inexpensive decoder proposals. Actual spend is recorded.

Fresh seeds 1201–1203 are separate from the search budget. The job also
replays fresh starting scores to check comparability with the saved arm.
The paired comparison averages seeds within each start and tracks fresh
loss, chemistry changes, additions, removals and net size changes.

Job **29906282** was submitted to `gpua10` on 6 November 2026 with one A10,
four cores and a two-hour limit. Source SHA-256 is
`5caf033c32884c470d8a5706e69e5b93ad953b204e2b1d5b14ef7c2b73539968`.
Twelve local tests passed. A CPU smoke with the epoch-45 VAE and a mock
scorer reproduced all 24 starting SMILES and counts.

Job 29906282 completed in 20 minutes 10 seconds of allocated time.
Random search used 176 of 182 optimization score attempts and exhausted
the native budget on 22 of 24 starts. The other two reached the direction
limit with three attempts left apiece. All fresh results were available.
The 72 replayed fresh starting scores differed from their archived
counterparts by at most 5.8e-7. The run logged RDKit sulfur force-field
typing warnings.

| Result across 24 paired starts | Guided | Random search |
| --- | ---: | ---: |
| Mean fresh contact loss | 2.61254 | 2.58340 |
| Mean change from starting loss | -0.10703 | -0.13617 |
| Starts with any atom-count change | 15 | 9 |
| Net growth / shrinkage at endpoint | 8 / 7 | 4 / 5 |
| Accepted addition / removal updates | 10 / 8 | 6 / 6 |
| Accepted updates that changed the molecule | 30 | 55 |
| Attempted directions | 90 | 1591 |

On each fresh seed, random search won 14 starts, guided won 6, and 4 tied.
Mean guided minus random loss was 0.02914 with a descriptive paired
bootstrap interval of [0.00004, 0.06517]. The 22 pairs that used their
full call budgets averaged 0.03374. Random search used more decoder
proposals and achieved lower fresh loss under this native-call budget.
The occupancy gradient produced more count changes. This comparison
uses one BRD4 pocket and one random-search seed schedule.
[Interactive comparison](../../../outputs/hpc-results/protenix-raw-random-20261106-a01/interactive-report/index.html)
and [saved result](../../../outputs/hpc-results/protenix-raw-random-20261106-a01/result.json).

## 20,000-update campaign

The [20k configuration](raw-occupancy-20k-dtu.json) budgets 20,000 accepted
guided latent updates across a sequence of prior starts. Each start uses
the same epoch-45 decoder, Protenix-Tiny pocket score, occupancy-gradient
rule and raw-molecule acceptance checks as the expanded pilot. The runner
starts a new trajectory after a stall, eight consecutive accepted updates
with unchanged chemistry, or 64 updates in the current trajectory.
Restart transitions are logged separately. The count of chemistry-changing
updates is reported alongside the number of accepted latent updates.

Every native input continues to use the exact valid decoded molecule at
its actual atom count. Fresh conformers are scored independently from the
optimization geometry. This campaign asks how well the tested local
gradient can be used over a larger compute budget, with early stops and
raw molecule quality visible in the records.

DTU submitted GPU stage **29899019** and dependent continuation job
**29899020** on 5 November 2026 at 14:10 CET. The first stage requests
two hours on one A10. The continuation can submit up to five further
20-hour stages after a successful segment with a valid checkpoint and
positive update progress. The global native-attempt cap is 100,000.
The 20,000-update target also remains subject to the draw and stall budgets
in the config.

The [worker](../../../src/latentmol/protenix_occupancy/long_occupancy.py)
saves a state and events for each stage. Best overall and best
chemistry-changing guided molecules have separate receipts. The
[continuation script](../../../scripts/hpc/raw_occupancy_continue.py)
binds the sealed job plan and verifies the predecessor before resuming.
Transient MOL files use node-local scratch. Stage outputs are under
`/work3/s234818/latentmol/protenix_occupancy/runs/protenix-raw-20k-20261105-a01`.
The source SHA-256 is
`b253f18b12a845ff4c7feb41ec1e54e50cfee8e20d65da47186001180b680497`.

Eleven focused CPU tests passed. A real epoch-45 CPU smoke with a synthetic
size score reproduced the four earlier raw count changes. The two-stage
worker smoke resumed consecutive updates with a mocked scorer, and the
mock scheduler smoke exercised continuation and duplicate submission
protection. The first native stage was pending at submission.

### First 20k segment

Stage **29899019** reached its time boundary with 663 accepted updates
across 198 trajectories. Of these, 284 changed the raw molecule, including
109 addition and 65 removal updates. The continuation queued **29900354**,
which started on 5 November at 19:11 CET.

The mean fresh contact loss over three geometry seeds was 2.65589 for
the first starting molecule, 2.43745 for the stage endpoint and 2.17328
for the best guided molecule. These are selected molecules from different
restarts. For the best guided molecule's own trajectory, the optimization
geometry loss went from 2.38522 to 2.17300. That trajectory's start has
no separate fresh score in the first-stage receipt.
[Stage result](../../../outputs/hpc-results/protenix-raw-20k-20261105-a01/stage-001/result.json).

### Interactive snapshot at 09:15 CET on 6 November

The saved logs show 5,427 of 20,000 accepted updates, including 2,392
hard molecule changes. Heavy-atom count increased on 895 updates and
decreased on 539. Stage 2 was still running. The report shows seven
selected trajectories, including an invalid prior draw, and keeps stage
1 fresh scores separate from optimization contact loss.
[Interactive report](../../../outputs/hpc-results/protenix-raw-20k-20261105-a01/report-20261106-0913/index.html)
and [snapshot summary](../../../outputs/hpc-results/protenix-raw-20k-20261105-a01/report-20261106-0913/report-summary.json).
