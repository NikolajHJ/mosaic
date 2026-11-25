# Three generations of BRD4 allocation search

Does the allocation policy that produces a stronger valid panel at equal GPU
time also give the next generation better starting points? The [configuration](restart-boltz-generations.json)
uses the current BRD4/JQ1 fixed-pose full FP32 DBMol objective, LR 1, and three
fresh seed pairs (2935011, 3135031, 3335051). All six selected runs request a
single 32 GB V100 for up to 24 hours.

Each policy gets **3,600 updates and up to 48 fresh scores per generation**,
for **10,800 updates and up to 144 fresh scores per arm per seed**. This is
64,800 updates and at most 864 fresh scores across the six selected runs.
3,600 permits whole cohorts for every policy. Fixed-400 uses nine children,
fixed-150 uses 24, and selective extension uses three cohorts of six starts.
Within each extension cohort, all six get 100 updates, then two receive a
further 300. One extension goes by fresh score and one is a random wildcard.

The first generation starts from the three strongest eligible parents in the
frozen wave-four archive of job 30082565. The ordered overlapping starts are
paired across methods. After each generation each arm selects its three best
eligible parents from the frozen and newly scored molecules in its own archive.
The 13 original molecules are excluded from reported novel yield. Fresh scores
are cached by exact graph and pose across generations. Unused hard-score
allowance expires each generation. Each generation records its parents, their
replayable latents, novel yield, best and tenth-best fresh losses, and elapsed
time. The generation files capture partial progress if an arm stops early.

## Policy choice and dispatch

A CPU job depends on the nine fixed-150, fixed-400 and selective-extension
jobs from the [three-seed allocation pilot](restart-boltz-search-methods.md).
It verifies all nine completed results and their launch receipts. The
[selection rule](../../../src/latentmol/design/boltz_policy_selection.py)
compares all three policies at the shortest observed GPU time within each
seed, including setup. Reaching ten novel valid molecules takes precedence.
Among panels reaching ten, lower tenth-best loss takes priority, then time to
ten and best loss. Below ten, larger yield takes priority, then lower best loss.
Each seed has equal weight in the aggregate ranks. The best fixed-length
control advances with the strongest remaining policy. Tied fixed controls
favor fixed-150 and tied challengers favor selective extension. Identical
outcomes on all three seeds produce an inconclusive decision.

When the decision selects two policies, the CPU job submits those policies
on all three new seeds and writes the six GPU job IDs. The control directory
contains `gate-decision.json`, `gate-outcome.json` and per-arm submission
receipts. A missing or failed prerequisite results in no GPU submission.

At 21:26 CET on 17 November, LSF job **30099978** was pending on its nine
pilot dependencies. All nine pilots were queued and the controller had
submitted zero follow-up GPU jobs. The local replay check reproduced the
shared starts on all three new seeds. [CPU submission receipt](../../../outputs/boltz-restart-brd4-generations-20261117a/gate-submission.json)
and [paired starting-state check](../../../outputs/boltz-brd4-generation-setup-20261117a/seed-replay.json).

From the repository root:

```powershell
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py stage --name boltz-restart-brd4-generations-20261117a --config restart-boltz-generations.json --queue gpuv100 --max-updates 10800 --max-fresh-attempts 144 --hours 24
python experiments/latent-design/protenix-vae/restart-operations/arm_boltz_generations.py --name boltz-restart-brd4-generations-20261117a --pilot boltz-restart-brd4-methods-20261117a
```

For the completed follow-up, pass the six result files and the two selected
policy names with `--generation-policies` to the [comparison script](restart-operations/compare_boltz_methods.py).
It reports the per-generation panel and compares time to ten and tenth-best
loss at common GPU time and at the final update budget.
