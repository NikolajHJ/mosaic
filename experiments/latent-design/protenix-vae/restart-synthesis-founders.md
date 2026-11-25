# Does starting chemistry determine which joint hits the search finds?

Three matched Protenix searches compare the original binding-first founders,
an accessible library, and previously scored near-misses. The human BRD4
pocket, frozen decoder, relaxed contact gradient, and geometry protocol match
the preceding synthesis campaign. Every arm starts three valid 25-atom
molecules on the same active slots. The nine source latents all reproduced
their saved graphs exactly in a local check against the pinned decoder.

| Founders | Selection |
| --- | --- |
| Binding-first | The original near-valid, deep, and wide winners. Historical fresh losses 2.05132, 2.08135, 2.09213. |
| Accessible library | Lowest audited SA in the aligned bank, then two most dissimilar eligible bank molecules by maximum Morgan similarity to those chosen. SA 1.749, 2.403, 2.684. Binding played no role in choosing them. |
| Archive near-miss | Ranked the fresh-scored follow-up archive by loss among SA ≤5 and 2.3 < loss ≤2.4, skipping similarities above 0.6 to earlier choices. SA 4.910, 4.383, 3.940 paired with fresh loss 2.30248, 2.32096, 2.34121. Exact archive latents from jobs 30078658 and 30078381. |

Each seed runs all three arms on one GPU with matched restart random seeds.
Each arm receives 3 waves × 3 children × 500 optimizer updates = 4,500 updates,
plus three fresh founder calls and up to six fresh calls per child, or 57 in
all. The founder calls share a scoring batch. Each is charged exactly once to
its arm and its result is cached. Failed embed attempts still use an allowance.
Fresh failures do not earn archive entries, and the original exact founder can
still start a trajectory. The panel consumes at most 171 fresh calls per seed.

All use LR 60, step norm cap 0.25, fixed chemistry and valence weights 1, zero
optimization noise, constant weak SynFrag at ratio 0.05, and the same
three-wave policy. The first wave perturbs the assigned founders, the second
selects from the binding/SA frontier with diversity and perturbs its parents,
and the third continues from each preceding branch's terminal or standard
recovery fallback. Hard-molecule SA before upper-tail smoothing controls
shortlisting, selection and thresholds. Receipts retain the native RDKit SA
alongside it.

Compare distinct and fingerprint-diverse yields at audited SA ≤5 and ≤6 with
fresh loss ≤2.3 and ≤2.4. The primary new-to-history output excludes all nine
inputs and every valid identity visited in the 16 previously completed
follow-up optimization panels (7,924 identities before adding the new panel).
The companion yield excludes only the nine inputs. Record fresh successes,
actual attempts and valid-state fractions alongside those counts. A group's
initial chemistry and prior binding differ by construction, so this tests the
practical choice of a starting pool. Three perturbation seeds reuse these nine
structures.

| Seed | GPU | Job | At submission |
| ---: | --- | ---: | --- |
| 1713113 | V100 | 30099918 | PEND |
| 1713117 | A10 | 30099919 | PEND |
| 1713129 | V100 | 30099920 | PEND |

The total allowance is 40,500 optimizer updates and 513 fresh calls.
Six-hour wall limits apply per job. Twenty legacy tests and one new founder
panel test passed, covering independent starts, fresh charging, historical
exclusions and a failed founder score. No completed HPC efficacy result was
available at submission.

The central entry point can inspect any seed.

```powershell
python -m latentmol.design.protenix_search --config experiments/latent-design/protenix-vae/restart-synthesis-founders.json --arm seed-1 --plan-only
```

[Configuration](restart-synthesis-founders.json) ·
[Founder records and preparation](../../../outputs/synthesis-founders-20261117/preparation.json) ·
[Archive-state and bank extraction script](../../../outputs/synthesis-founders-20261117/prepare.py) ·
[Sealed launch inputs and submission receipts](../../../outputs/protenix-restart-synth-founders-20261117a/) ·
[Scheduler verification](../../../outputs/synthesis-founders-20261117/scheduler.json)

## First two completed seeds, 18 November 2026

Seeds 1 and 3 completed all three waves and exited zero on V100. At 10:24 CET seed 2 was pending on A10. Against the frozen historical exclusions, near-miss founders yielded 6 and 7 distinct descendants at audited SA at most 6 and fresh loss at most 2.3. Binding-first yielded 3 and 0, and accessible-library yielded 0 and 0. One near-miss descendant in seed 1 met the tighter SA 5 threshold, at SA 3.97610 and loss 2.28668. At SA at most 5 and loss at most 2.4, near-miss yielded 8 and 4, binding-first 0 and 0, and accessible-library 1 and 0.

Each arm used 4,500 updates per seed. In seeds 1/3, actual fresh attempts were 57/45 for near-miss, 57/57 for binding-first and 25/46 for accessible-library. Valid-state fractions were 68.80%/95.92%, 99.16%/99.56%, and 56.66%/74.21% respectively. Counts are distinct within each arm and seed. [Terminal and scheduler evidence](../../../outputs/synthesis-news-20261118T091637Z/snapshot.json).

The near-miss arms' cumulative new-to-history counts at audited SA at most 6 and fresh loss at most 2.3 progressed from 3 to 4 to 6 over the three waves in seed 1 and from 3 to 7 to 7 in seed 3. Four identities occurred in both seeds, giving nine distinct molecules pooled. All three near-miss founder lineages contributed. The strict SA-at-most-5 hit appeared in wave 1 at update 191. Its parent's audited SA / fresh loss were 3.940 / 2.34121 and its own were 3.976 / 2.28668. This example improved the fresh binding score while preserving similar chemistry. The third perturbation seed was still queued at 10:44 CET.

## Seed 2 moved to V100, 18 November 2026

At 14:01 CET, seed 2 was pending on V100 as **30105078**. Its original A10 job 30099919 exited before starting. V100 had the earliest displayed wall limit among running jobs on the compatible queues checked. The replacement uses the 12 GiB GPU request from the two completed founder seeds. The original sealed source, seed, scoring and optimizer budgets, and output location are preserved. [Scheduler and command receipt](../../../outputs/synthesis-founders-20261117/move-from-30099919.json).

V100 job 30105078 completed all three waves and exited zero. At audited SA <=6 and fresh BRD4 loss <=2.3, the near-miss arm found three molecules new to the historical archive, versus zero in both other arms. At SA <=5 and the same loss cutoff all three arms found zero. The near-miss arm had 98.69% valid trajectory states. [Final seed receipt](../../../outputs/synthesis-founders-20261117/result-30105078.json).
