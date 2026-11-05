# Seed 23 with Stage 2 repair proposals

Can a valence-projected repair proposal improve the saved seed 23 line-search
protocol? Compare with [the original line-search run](../protenix-line-search/README.md),
job 29890096. The new 50,000-update experiment starts from the same byte-bound
sparse random seed 23 logits. Model and context, model key 17, noise seed 230023,
LR 0.3 plus eight halvings, chemistry schedule 1 to 30 over updates 20k–30k,
temperature 0.7 to 0.4, noise 0.4 to zero by 30k, and original repair are inherited
from the verified plan. The margin and composition guidance remain off.

For an invalid source with room to improve on the ordinary repair, make a
detached copy of the bond probabilities. Blend this proposal input equally
with its one-hot argmax to expose hard valence excess, then use the original
Stage 2 projection with atom labels and neutral caps fixed. Stage 2's
conservative sulfur cap is used only in proposal generation. Decode and pass
the result through the existing atom-preserving structural builder and
sanitization/roundtrip checks. Measure the candidate's edit count and weighted
repair cost against the original graph and probabilities. Use it only when it
is valid and strictly cheaper than the ordinary witness, or when no ordinary
complete witness exists. Skip valid sources and ordinary repairs at the one-edit
cost lower bound. Keep the teacher fixed across all nine trials. Projection
never overwrites live logits and never adds a second gradient path. A numerical
projection miss leaves the ordinary repair/fallback in place and is recorded.

Record attempts and selections, observed original and projected repair cost,
strict raw validity, and runtime. Use the retained matrices for a later hard
contact comparison at matched checkpoints. The original soft contact metric
remains distinct from hard contact and affinity.
Candidate search is bounded. Its best witness is an observed upper bound.

The [launcher](../protenix-random-chem/scripts/launch_seed23_projection.py)
authenticates the prior capsule and snapshots a new one. The
[proposal](../protenix-random-chem/scripts/projected_repair.py) uses unmodified
Stage 2 operator bytes from revision 041c89ea5bdab024bf5671d32749a151f9558e48.
The A100 wall limit is 36 hours, priced from about 1.8 s/update on the active
matched A100 run plus 0.2 s/update for proposal overhead, a 25% margin and
setup. A running baseline is an ended dependency so this run waits its turn
behind the older seed 23 arm.

```powershell
py -3 -B experiments/brd4-jq1/protenix-random-chem/scripts/launch_seed23_projection.py --tag NEW_TAG --parent results/hpc/20261104-protenix-line-search-01/source01
```

Plan for 20261105-seed23-valence-proposal-01. A portable Stage 2 suite passed
74 cases after placing pytest temporary files in a writable workspace path.
An allocated projection-and-target smoke runs in the new job. Preparation
did not run a model or modify any existing experiment.

A100 job 29896980 was accepted and released. At 10:37 CET it was pending
on ended(29890096), with zero applied updates. Metadata startup and live
dispatch capacity passed. Allocated CPU job 29896985 verified a valid bond-only
witness and that a tied original cost preserves the ordinary target. See
[campaign.json](campaign.json) for the verified dispatch and test receipts.
