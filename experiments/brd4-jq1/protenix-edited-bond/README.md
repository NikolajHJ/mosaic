# Seed 23 with extra weight on repaired bonds

Does concentrating the existing repair signal on edited bonds help late seed 23
cross into validity while preserving hard Protenix contact?

The paired test starts both arms from the exact saved 49,000-update checkpoint
of job 29890096. Its verified repair removes the bond between slots 8 and 15
and has observed weighted cost 1.33705. Both arms run 1,000 updates to the
original 50,000 endpoint on the same A100. They use the same saved logits,
detached initial repair, model key 17, context, seed and noise state, LR 0.3
with eight halvings, structural repair, chemistry coefficient 30, cooling
from 0.406 to 0.4 and zero late noise. The original schedule and fallback
behavior are preserved. No margin, valence projection or composition term.

The treatment gives 8 times the original weight to bond labels changed by
the current complete verified repair. It adds seven copies of those label
terms with the original full-loss normalization. Atoms, charges, unedited
bonds, valid-source behavior and fallback keep their original weights.
The selected teacher stays fixed during all nine trials. The original
structural builder refreshes the target between steps. An allocated check
on the saved checkpoint verifies gradient scaling on the edited row and
unchanged gradients elsewhere.

Compare all saved raw validity states and validity exits, the bounded
atom-preserving search's observed weighted repair costs and unresolved counts.
Rescore the same global grid of 49,000, 49,100, ..., 50,000 with the frozen
scorer. Report raw-argmax and verified-repaired hard contact separately;
an absent verified repair leaves that score empty. The contact score is
a model proxy.

The shared paired launcher uses one 90-minute A100 allocation, estimated
from 1.52 s/update on the matched late source, plus setup, rescores and
margin. If the other two research GPU jobs are outstanding, the paired test
becomes eligible when either ends, keeping the concurrency cap at two.

For a fresh authorized run tag and its read-only checkpoint packet:

```powershell
py -3 -B experiments/brd4-jq1/protenix-random-chem/scripts/launch_score_transfer_audit.py --inputs INPUTS.json --tag NEW_TAG --kind edited-bond-weight
```

DTU accepted and released A100 job 29899114 on 5 November 2026. At 14:21 CET
it was queued until either the margin pilot or projection repeat ends.
Actual metadata startup and dispatch capacity passed. Allocated CPU job
29899118 verified the 8x gradient on slots 8–15 and unchanged atom and other
bond gradients on the exact saved checkpoint. The GPU outcome is pending.
See [campaign.json](campaign.json) for bound inputs and receipts.
