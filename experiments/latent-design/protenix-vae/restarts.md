# Do probability-space restarts improve the best valid Protenix candidates?

7 November 2026. Three arms use the frozen epoch-45 VAE, prepared BRD4 context,
25 fixed atom slots, seed 17, temperature 1, latent SGD LR 150, and the existing
latent L2 step cap of 0.5. Record cap hits and the separate gradient norms.
The earlier latent LR comparison tested 150 at the same temperature and cap.
Its completed V100 cohort saw similar mean best fresh gains at LR 3, 10 and
150 (0.1634, 0.1641 and 0.1615 over 16 starts). Its L40S cohort stopped during
fresh scoring on an RDKit distance-bound invariant, so the comparison did
not select a rate across both cohorts. This pilot uses the requested 150.

| Arm | Chemistry weight within each branch |
| --- | --- |
| constant1 | 1 |
| constant10 | 10 |
| late-ramp | 1 through update 375, geometric ramp to 10 at update 500 |

The endpoints reuse the [earlier 50k study](README.md#completed-first-trajectory--3-november-2026-0930-cet).
Validity rose in later phases there while temperature and noise also changed.
Here temperature is fixed and the stochastic perturbation is applied at
branch start. The existing structural-repair loss is zero when a raw graph
is already valid or when the search yields no target.

Cycle 1 is one 500-update trajectory. Cycles 2–5 start three independent
500-update branches per available elite, up to nine per cycle. The maximum
per arm is 37 branches and 18,500 updates. Each perturbed row gives its parent
argmax 0.5 probability and splits 0.5 over the other categories with an
independent Dirichlet(1) draw. This applies to active atoms, their charges, and
one copy of each occupied bond pair. The parent remains the hard argmax at
step zero. A detached additive logit offset implements the exact initial graph
and fades linearly to zero by step 25. Both Protenix and chemistry use the
same graph during the fade. This changes the early graph distribution, including
soft bond presence, while preserving the same latent parameters and objective.

Save every raw hard graph and latent. Each branch shortlists up to 16 new
distinct connected raw-valid molecules by relaxed contact loss. The top three
across successful fresh scores and previous elites form the next archive.
Fresh scores come from the existing Protenix FILE contact readout with
conformer seed 1201. Canonical SMILES deduplicate across cycles. Every
candidate keeps its source step and exact slot-aligned hard labels and latent.
Failed conformers are recorded as unscorable. The specific RDKit distance-bound
error seen in the prior cohort now follows that path. Compare scores within this
pocket and scoring protocol. This study compares chemistry schedules under
the restart policy.

## Command and budget

The allocated entry point is `python -m latentmol.design.protenix_restarts`.
The [campaign config](restarts.json) defines the schedules and budgets, and
the [fresh config](restarts-fresh-dtu.json) binds the independent hard scoring
protocol. The [submission records](../../../outputs/protenix-restarts-20261107a/)
capture the exact wrapper, authenticated source and input checks.

Local verification passed 28 focused tests and a two-cycle CPU smoke with the
authenticated decoder and synthetic scorers. The smoke generated all nine
expected branches after finding three elites, verified independent 50/50
probability mixtures and symmetric bonds, and exercised the finite backward
path. The local pytest cache was unwritable, with the tests completing normally.

## Submission

Submitted 7 November 2026 at 10:47 CET. All three jobs were pending at the
initial scheduler check. Each requests one exclusive 32 GB V100, four CPU
slots, and an eight-hour wall limit. The V100 queue was active and had run
the earlier LR comparison using these assets.

| Arm | DTU job |
| --- | ---: |
| constant1 | 29914616 |
| constant10 | 29914617 |
| late-ramp | 29914618 |

Snapshot SHA-256
`dd9901033b51f9e2a34c2f7f447c9a718c7e953c88c906ba2293f90e13cbcdac`.
Outputs are under `/work3/s234818/latentmol/runs/protenix-restarts-20261107a-<arm>`.
The first native evaluation checks finite attached paths. Per-cycle archives,
validity, candidate coverage, fresh scores, and cap-hit counts are pending.

At 13:58 CET on 7 November, constant 1 completed its first cycle and was
at update 251 of branch five in cycle two. Its first archive's best fresh
loss was 2.19730. Constant 10 was at update 191 in its first cycle. The late
ramp was pending at the V100 per-user limit.
