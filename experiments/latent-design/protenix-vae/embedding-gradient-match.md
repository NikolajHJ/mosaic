# Protenix bridge contact calibration

Does calibrating the hidden-to-native bridge on decoded contact and local
contact changes improve the molecules found by latent optimization?

The [config](embedding-gradient-match.json) starts from the trained bridge
used in the original Protenix mapping run. Only the copied bridge is fitted.
The epoch-45 VAE and native Protenix weights and context are frozen.
The fit retains hard embedding supervision and adds the relaxed contact
value and forward secants at radius 0.1. At each center, the teacher labels
the endpoints along the original mapped descent, direct relaxed descent,
and a seeded random direction. Flat gradient directions are omitted.
All teacher evaluations use the same fixed slots, Protenix context and
temperature as the corresponding center. Training uses first-order native
VJPs through the mapped scores at both ends of each secant.

The loss weights are 1 for hard embedding MSE, 1 for contact MSE, and
0.04 for secant slope MSE. AdamW runs four passes at LR 0.0002 over
36 centers. The centers come from 12 fresh prior samples and 24 archived
mapping/direct states at steps 500 and later. Eight separate prior samples
and the two comparison starts are scored before and after fit. The saved
late training states come from the same original seed as the comparisons,
so held-out starts and early windows are distinct from the fit but do not
constitute independent trajectories.

The three arms use the exact saved step-0 and step-180 latents, 200 updates
per arm per start, LR 0.1, no noise and the same chemistry/temperature
schedules. The original mapping, calibrated mapping and direct relaxed
path each run the existing latent optimizer with the same budget. The
comparison records the common decoded relaxed contact every 10 updates,
raw graph validity at each update, and elapsed time. Up to three distinct
valid finalists per arm and start, ranked by their decoded relaxed score,
are rescored as discrete graphs in the same fixed context. These are
contact surrogate scores.

Submitted as **29890568** on 4 November 2026 at 13:09:04 CET. One 32 GB
V100, four CPU slots, 90-minute wall limit. The scheduler showed PEND
at 13:09:17 CET. The 30 focused local cases passed.

The allocated command uses `scripts/hpc/protenix_latent_run.sh` with
`-m latentmol.design.protenix_embedding_calibration` and the [config](embedding-gradient-match.json).
Output is `/work3/s234818/latentmol/runs/protenix-embedding-gradient-match-20261104a`.
The staged source SHA-256 is `90dc97a7231c317fd58892ac0ef2179ef0d3d11e0c238707c78f487dea87ef37`.
[Submission receipt](../../../outputs/hpc-submissions/protenix-embedding-gradient-match-20261104a/submit-1793794142860935600.json).

## Completed comparison

Job **29890568** ran on 4 November from 18:51:49 to 19:16:17 CET and
completed all six 200-update arms. Mean held-out secant slope MSE fell
from 0.0787 to 0.0453 across the ten neighborhoods. It improved at the
step-180 start (0.1013 to 0.0320) and worsened at the initial start
(0.00266 to 0.00628).

| Start | Arm | Mean decoded contact | Final decoded contact | Raw-valid update sources |
| --- | --- | ---: | ---: | ---: |
| Initial | Original bridge | 0.2320 | 0.1886 | 127/200 |
| Initial | Calibrated bridge | 0.2168 | 0.2099 | 42/200 |
| Initial | Direct relaxed | 0.3467 | 0.4022 | 181/200 |
| Step 180 | Original bridge | 0.2253 | 0.2176 | 12/200 |
| Step 180 | Calibrated bridge | 0.2639 | 0.2762 | 71/200 |
| Step 180 | Direct relaxed | 0.4990 | 0.5322 | 65/200 |

At the step-180 start, calibration improved mean contact and raw validity
relative to the original bridge. Both bridge endpoints were invalid, and
their highest-scoring valid observation was the shared 0.3902 start.
From the initial start, calibration lowered both mean contact and validity
relative to the original bridge.

From the initial latent,
the direct relaxed arm's mean contact score rose across successive 50-update
blocks (0.2553, 0.3413, 0.3954, 0.4008). Its valid endpoint reached 0.4022,
up from 0.1513 at the start, with 181 of 200 update-source graphs valid.
Its one-hot endpoint scored 0.3834 in the fixed context. The common starting
graph scored 0.1902 in the same finalist pass. The relaxed score was still
slightly higher at step 200 than at step 180.

From the saved step-180 latent, the direct arm rose from 0.3902 to 0.5322
on the relaxed input, but its endpoint was invalid. The valid candidate at
step 290 scored 0.4093 on the one-hot fixed-context readout, compared with
0.3178 for this arm's starting graph. Both mapping variants ended below
the direct arm on the common decoded relaxed score from both starts.

The one-hot evaluations share the original pocket and fixed context.
Fresh-conformer rescoring was outside the experiment budget.
[Saved result](../../../outputs/hpc-results/protenix-embedding-gradient-match-20261104a/result-20261106T085356589999Z.json),
[direct-arm curves](../../../outputs/hpc-results/protenix-embedding-gradient-match-20261104a/direct-curves-20261106T085426672700Z.json),
and [fit and terminal log check](../../../outputs/hpc-results/protenix-embedding-gradient-match-20261104a/fit-check-1793959333464342400.json).
