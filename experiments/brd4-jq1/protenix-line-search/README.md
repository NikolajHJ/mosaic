# Seed 23 line search

Does selecting the strongest decrease across nine rates avoid the contact
spikes and improve strict connected raw validity in the sparse seed 23 run?
The archived control is [job 29880094](../protenix-random-chem/runs/20261103-sparse-annealed-seeds-01/README.md).
It completed 50,000 updates with zero valid connected raw argmax states.

This fresh A100 repeat uses the same initial and noise seeds (23 and 230023),
byte-checked initial logits, model key 17, scorer, structural repair, chemistry
weight, cooling, noise schedules and 50,000-update target. One common noise
direction per logit block is scaled with each candidate LR. The source and
nine candidates are scored at the same temperature, chemistry coefficient
and detached target. The best strict decrease in contact plus weighted repair
wins. Equal minima prefer the larger rate. If no candidate decreases the
objective, the logits stay in place and the global schedules still advance.
Candidates use forward passes. The selected next source is differentiated
under the usual refreshed target and temperature.

The producer reuses the chemistry-screen driver and normal allocated safety
checks. The command is the original held chemistry-screen launcher with the
new plan and one A100 allocation. The exact plan, packaged input hashes and
local dispatch command are in
`results/hpc/20261104-protenix-line-search-01`. Metadata controls only run
locally. The 67-hour request prices all ten primal evaluations and one
gradient evaluation at the measured old per-call cost, with a 25% margin
and setup allowance. This is a conservative estimate pending observed
native timing. All nine scores, selected rates and no-ops are retained.

Compare per-step overshoots, strict raw endpoint and trajectory validity,
128 unfiltered endpoint draws and elapsed/model-call budgets. Local descent
holds for the fixed combined trial objective. The displayed contact may rise,
especially when chemistry competes or the graph cools. Contact is a proxy,
and the line-search arm uses more scorer calls per update.

Job 29890096 was accepted and released on A100. Observed PEND with 0 committed updates at 2026-11-04T10:32:09.235442+00:00. See [dispatch verification](dispatch-verification.json).
