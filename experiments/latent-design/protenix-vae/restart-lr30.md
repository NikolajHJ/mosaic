# What changes when the restart search uses LR 30?

7 November 2026. Repeat both LR 70 experiments with a learning rate of 30.
Use the corresponding sealed source archive for each experiment. The
scientific configs differ from the original embedded configs only in
`design.learning_rate`. The staged archive comparison verified all other
files byte for byte apart from the submission plan.

The [standalone config](restart-lr30.json) runs two cycles and up to 10,000
updates on a 32 GB V100, matching the active LR 70 replacement. The
[six-arm config](restart-lr30-six-cycle.json) runs six cycles and up to
46,000 updates per arm on L40S. All runs use 1,000 updates per branch and
the corresponding LR 70 seeds, fan-out, chemistry weights, fresh scoring
budget, step caps, and selection policies.

The step cap acts after learning-rate scaling. At a shared state, both
rates can apply the same displacement when both hit the cap. Compare
actual displacement and cap hits alongside validity and fresh loss.

The [submitter](restart-operations/submit_campaign.py) stages each config
with `--source-receipt` set to its LR 70 submission receipt. The
[standalone records](../../../outputs/protenix-restart-lr30-20261107a/)
and [six-arm records](../../../outputs/protenix-restart-lr30-six-20261107a/)
hold the staged archives, exact commands, and DTU receipts. The combined
new output allowance fit the measured storage capacity.

DTU accepted all seven jobs between 23:46 and 23:47 CET. All were pending
at the 23:47 scheduler check.

| Arm | LR 30 job | Matched LR 70 job |
| --- | ---: | ---: |
| Standalone screened | 29920268 | 29920256 |
| Combined | 29920269 | 29916515 |
| Cap 0.5 | 29920270 | 29916522 |
| Repair only | 29920271 | 29916530 |
| Unchanged parent | 29920272 | 29916537 |
| Relaxed shortlist | 29920273 | 29916544 |
| Top-scoring parents | 29920274 | 29916552 |

## Queue move on 8 November 2026

The six pending comparison jobs moved from L40S to 32 GB V100 with their original scientific settings and output roots. The replacements have a 45-minute startup allowance. DTU confirmed the originals exited and all six replacements were pending at 16:10 CET while dispatch was paused. See the [queue move receipts](../../../outputs/protenix-restart-l40s-move-20261108a/release.json).

| Arm | Active LR 30 job |
| --- | ---: |
| Combined | 29926092 |
| Cap 0.5 | 29926093 |
| Repair only | 29926094 |
| Unchanged parent | 29926095 |
| Relaxed shortlist | 29926096 |
| Top-scoring parents | 29926097 |
