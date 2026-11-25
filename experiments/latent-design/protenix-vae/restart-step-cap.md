# Does a larger step cap improve a 500-update Protenix child?

The [cap comparison](restart-step-cap.json) uses four seed panels with
eight distinct archived BRD4 parents. Each parent has two screened starts
and each start is replayed at cap 0.25 and 0.5. The arms share the same
learning rate (70), chemistry and valence weights (1), fixed atom slots,
and optimization noise settings. Each seed job runs both arms on one GPU. Per arm
and seed, the budget is 2,000 updates and up to 24 fresh attempts (12
per parent). The readout records attempted and successful fresh scores
separately, with an equal reduction across arms when a parent runs short
of unique candidates. Duplicate molecules are reserved across parents
within an arm. The complete matched results are in `result.json` and
branch-level details in `paired/batch.json`.

The schedule pilot reuses the first screened start for each seed-D parent.
It compares fixed 0.25 and 0.5 with linear 0.5→0.25, linear 0.25→0.5,
and fixed 0.8. Each arm gets 1,000 updates and up to 12 fresh attempts.
The [initialization pilot](restart-cap-perturbation.json) compares exact
parent, protected sparse mixing, screened latent kick, and independent
Gaussian starts at cap 0.5. It uses three
parents, 1,500 updates and up to 18 fresh attempts per arm, and saves path,
validity, and bond-density data. Both pilots match fresh attempts by parent.

Compare parent-level loss gains and the count and structural diversity of
low-loss candidates, using the measured fresh counts. The primary test
isolates child behavior from subsequent elite feedback. The schedule and
initialization pilots are small exploratory comparisons.

The allocated entry point is `python -m latentmol.design.protenix_restart_study`.
DTU accepted the jobs on 9 November 2026 at 16:23 CET. Each has a two-hour
wall limit and was pending at verification. The sealed submissions use
shared 0.05 optimization noise fading to zero by update 100. The current
presets disable that noise for future launches.

| Job | Updates | GPU queue | DTU ID |
| --- | ---: | --- | ---: |
| Seed A | 4,000 | gpuv100 | 29942484 |
| Seed B | 4,000 | gpua100 | 29942487 |
| Seed C | 4,000 | gpul40s | 29942490 |
| Seed D | 4,000 | gpuv100 | 29942492 |
| Schedules | 5,000 | gpul40s | 29942495 |
| Initialization | 6,000 | gpua100 | 29942485 |

The [cap receipts](../../../outputs/protenix-restart-step-cap-20261109a/)
and [initialization receipts](../../../outputs/protenix-restart-cap-perturbation-20261109a/)
hold the exact commands and checked source archives. Local focused tests
and a five-arm mock smoke verified the schedule and matching behavior.

At 16:47 CET on 9 November, DTU confirmed the six originals had exited
and released six replacements with optimization noise disabled. Child
initialization, caps, update and scoring budgets, and queues are the
same as in the original comparison.

| Job | Original job | Replacement job |
| --- | ---: | ---: |
| Seed A | 29942484 | 29943697 |
| Seed B | 29942487 | 29943704 |
| Seed C | 29942490 | 29943710 |
| Seed D | 29942492 | 29943716 |
| Schedules | 29942495 | 29943720 |
| Initialization | 29942485 | 29943695 |

The [cap replacement records](../../../outputs/protenix-restart-step-cap-zero-noise-20261109a/)
and [initialization replacement records](../../../outputs/protenix-restart-cap-perturbation-zero-noise-20261109a/)
hold the new source and submission receipts.

At 19:42 CET on 9 November, seed C and the schedules job had completed.
Seed C matched 2,000 updates and 24 fresh attempts per cap. Cap 0.25 and
0.5 each won on one parent, with best fresh losses of 2.28228 and 2.34606
across the panel. The 0.5 arm had two failed scores.

The schedule job matched 1,000 updates and 12 fresh attempts per arm.
The 0.25→0.5 ramp had the highest mean gain (0.20574 across two parents).
The fixed 0.8 cap had the lowest individual loss (2.23801), but did not
improve the second parent. Seeds A, B, and D and initialization were
still queued.
