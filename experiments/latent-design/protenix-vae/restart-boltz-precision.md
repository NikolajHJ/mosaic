# Does BF16 change 200-update design quality?

Compare FP32 and neural-block BF16 optimization from identical saved starts,
then rank both trajectories with one common FP32 full-DBMol scorer. The short
[speed probes](restart-boltz-speed.md) showed faster BF16 updates but could not
establish the effect on final molecule quality.

## Paired protocol

Six jobs use the same six ERK2 latent-perturbation seeds as the speed probes:
**281113-281118**, each derived from the same saved best valid timing-pilot
parent. This is six local perturbations of one parent/target, not six proteins.
Each pair keeps both methods on the same allocated GPU. Arm order alternates
FP32-first / BF16-first to reduce order effects. Independent native processes
load each arm sequentially; there is no parallel-branch confound or MPS sharing.

Each arm executes **200 updates** plus its terminal observation, with LR **1**,
step cap **0.25**, chemistry/valence weights **1**, temperature **1**, fixed
**25 occupied atoms**, full DBMol loss and **no noise during optimization**.
Only the initial parent perturbation is random. Initial latent, occupancy and
decoded graph identities must match between arms. Frozen-model, finite-gradient,
shape, occupancy, output-collision and clean-exit checks remain active.

BF16 is limited to checkpoint-marked neural blocks. Both arms retain every-block
checkpointing and RNG preservation, with TF32 off. Geometry, heads and reductions
stay outside BF16 autocast. No other execution change is tested in this pilot.

## Common FP32 readout

Save every raw trajectory state, including invalid observations. After both arms,
form the union of all distinct valid exact pose-slot graphs and score it once in
a fresh **FP32, TF32-off** process. No relaxed-loss shortlist decides eligibility.
Both arms use those same hard scores; canonical molecules with multiple pose
mappings retain their best FP32-scored mapping. Coordinates remain the prepared
reference pose: this is not a regenerated native pose or a binding-affinity test.

Report best loss, mean top-three/top-20 loss, distinct valid molecules, validity
fraction and terminal outcome at **0, 50, 100 and 200 updates**. Top-k means remain
missing when fewer than k distinct molecules exist. Each checkpoint includes
only graphs already observed in that particular arm, preventing future-candidate
leakage. Any common-time readout uses observed prefixes of the saved trajectories
and is secondary; it does not extrapolate molecular quality from a speed factor.

Budget per job: **400 optimizer updates**, **402 relaxed gradient observations**
and **at most 402 common FP32 hard-scoring calls**, usually fewer after validity
filtering and exact-graph caching. Record per-arm candidate counts, overlap and
actual union calls. The common exhaustive scoring policy does not imply equal
consumed per-arm scoring costs. Six jobs total **2,400 updates**.

## Configuration and submission

[Config](restart-boltz-precision.json), shared runner
`latentmol.design.boltz_speed` (`study="boltz_precision"`), existing campaign
submitter. Four jobs target L40S and two A100; 40 GB and 80 GB A100s are eligible.
Each requests one exclusive GPU, four CPU slots, 20 GiB GPU memory, MPS off and a
four-hour wall limit. A10 is excluded because its sole node has a closed
reservation until 24 November, 15:00 CET; A40 has a smaller occupied pool.
[Queue/input evidence](../../../outputs/boltz-precision-plan-20261116a/queue-recommendation-20261116T1338.json).

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py stage --name boltz-restart-precision-20261116a --config restart-boltz-precision.json --hours 4 --max-updates 400 --max-fresh-attempts 402
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py submit --name boltz-restart-precision-20261116a
```

Source and outputs are new. Existing jobs, caches, checkpoints, sealed releases
and the seven-protein search defaults are not changed by this experiment.

## Submitted 16 November 2026

All six paired jobs were accepted and were **PEND at 13:54 CET**:

| Pair | Perturbation seed | GPU queue | Job | First arm |
| --- | ---: | --- | ---: | --- |
| A | 281113 | L40S | 30078756 | FP32 |
| B | 281114 | L40S | 30078757 | BF16 |
| C | 281115 | L40S | 30078758 | FP32 |
| D | 281116 | L40S | 30078760 | BF16 |
| E | 281117 | A100 | 30078761 | FP32 |
| F | 281118 | A100 | 30078762 | BF16 |

Source archive SHA-256:
`eaf54c04f845fe9698148e71c01c2c10d8f013f2374e21f39db9b67b3e86f6fc`.
Existing saved parent, prepared ERK2 context and model assets passed the staging
checks. Seventeen scoped CPU checks passed, including all-state harvesting,
pose-aware common scoring, no future/cross-arm leakage, invalid/empty outcomes,
matched initialization, staging budgets and shared search/resume regressions.
These validate wiring; native quality outcomes remain unmeasured until execution.

[Commands](../../../outputs/boltz-restart-precision-20261116a/commands.json),
[budget plan](../../../outputs/boltz-restart-precision-20261116a/gpu-plan.json),
[scheduler verification](../../../outputs/boltz-restart-precision-20261116a/status-001.json).
Each `seed-*-submission.json` in that directory retains its scheduler receipt.
