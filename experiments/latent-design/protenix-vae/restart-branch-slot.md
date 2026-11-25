# Which use of a restart branch slot produces better valid molecules?

The [frontier pilot](restart-frontier-slot.json) compares recency-first
selection with one guaranteed child for the best eligible replayable
parent. The two jobs share the same founders, seed, L40S queue, four
waves, and 250 updates per child. Each has up to 3,000 updates and 72
fresh attempts. Both log exact duplicate starts and try up to four
refills per slot. Compare the actual completed updates and scoring
counts, including any slots that could not be filled.

The [near-valid replay](restart-near-valid-replay.json) pairs the actual
invalid starts in waves 2–5 of job 29943712 with the exact valid starts
they displaced. The two arms run on the same GPU at the original
LR 60, cap 0.25, and zero update noise, with 500 updates per child.
The job matches up to six new fresh attempts per arm and pair. It
scores repair witnesses and exact starting molecules in a separate
baseline batch. The source waves had already completed when selected.

The [graph-edit pilot](restart-graph-edit.json) compares a screened
latent restart to valid atom substitutions and bond reattachments
from the same parent. The two seed jobs use six distinct archived
parents in total. Each arm has up to eight fresh attempts per parent,
matched within parent after deduplication. The latent arm spends 500
updates per parent. The result records graph proposals, actual
updates, score attempts, successful scores, and elapsed time.

All jobs use sealed releases with two-hour wall limits. DTU confirmed
all five queued at 20:41 CET on 9 November 2026.

| Pilot | Queue | DTU ID |
| --- | --- | ---: |
| Recent frontier | L40S | 29948377 |
| Best eligible slot | L40S | 29948382 |
| Near-valid vs exact replay | L40S | 29948379 |
| Latent vs graph edits A | L40S | 29948378 |
| Latent vs graph edits B | V100 | 29948383 |

Submission receipts and sealed commands are available for the
[frontier](../../../outputs/protenix-restart-frontier-slot-20261109a/),
[near-valid replay](../../../outputs/protenix-restart-near-valid-replay-20261109a/),
and [graph edits](../../../outputs/protenix-restart-graph-edit-20261109a/).
Nine focused tests passed. The resume smoke restored the archive after
an interrupted wave and matched the uninterrupted run's subsequent
parent choices, score requests, and cumulative budgets.

On 10 November 2026, we shelved the gradient-free graph-edit pilot to
focus on gradient-guided hallucination. Panel A (**29948378**) remains a
completed comparison. DTU confirmed that panel B (**29948383**) was
canceled while pending at 14:40 CET and had no recorded start.
