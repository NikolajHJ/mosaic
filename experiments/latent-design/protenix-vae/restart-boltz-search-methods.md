# BRD4 tests of branch allocation, founder screening and local gradient reuse

The question is which change yields ten distinct valid BRD4 molecules fastest,
and which gives the best tenth-ranked molecule at equal GPU time. The
[configuration](restart-boltz-search-methods.json) inherits LR 1, the 0.25
step-norm cap, constant chemistry weight 1, 31 occupied crystal pose slots and
full FP32 Boltz2 DBMol scoring from the [BRD4 trial](restart-boltz-brd4.md).
There is no optimization noise. Each seed panel runs within one GPU family.
Seeds 811211, 813211 and 817211 use V100 32 GB, A40 and A100 80 GB, respectively.
Each job uses one GPU and serial children.

| Comparison | Work per arm and seed | Fresh FP32 allowance |
|---|---:|---:|
| Fixed long children | 6 × 400 updates | 48 |
| Fixed short children | 16 × 150 updates | 48 |
| Selective extension | Two cohorts of six × 100, with two children per cohort receiving another 300 updates | 48 |
| Local gradient reuse | Same 16 starts × 150 as the fixed short control | 48 |
| Random founders vs CPU-screened founders | 6 × 150 per arm | 24 per arm |

For archive experiments, a frozen copy of the wave-four checkpoint of job
30082565 supplies the three best eligible parents. The 13 molecules already
in that archive are excluded from promotion rewards and reported yield. All
four archive arms use the same ordered sequence of latent perturbations within
a seed and consume the first 6, 12 or 16 starts according to policy. Each
promotion cohort gives an extension to the child with the best freshly scored
new molecule and one other randomly chosen child. If no branch scores a new
molecule, both choices are random. Continuations retain the step-100 latent
and atom mask. Each policy records completed-segment milestones, actual hard
calls, and molecules ranked by their common full FP32 score. These are local
branch-allocation pilots on one pocket and parent panel.

The two founder arms test the first generation from a random VAE prior. They
share one unconditional draw. Screening decodes 512 seeded CPU draws and
checks up to 128 invalid draws for a repair witness. It prefers valid founders,
reserves a slot for a verified repair within two edits when available, and
fills empty slots with the closest larger verified repairs before using raw
chemistry diagnostics. The generated starting latent and its graph are retained even if
eligibility was assessed using a repair witness. The screening receipt records
raw validity, witness distance, selected seed and elapsed time.

The reuse arm can take one update using the latest full derivative and one
using its linear approximation through the current decoder. The current
chemistry gradient is computed for both. A large change in a graph probability
row triggers an early full refresh. Exact checks at refreshes and endpoints
compare actual and predicted loss changes, adjusting the allowed radius for
future steps. The upper and lower radii are 0.25 and 0.04 in per-row total
variation. Logs include full and proxy calls, cosine between refreshed gradients,
agreement, and hard-score forwards. There is no rollback.

Use the shared submitter from the repository root:

```powershell
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py stage --config restart-boltz-search-methods.json --name boltz-restart-brd4-methods-20261117a --max-updates 2400 --max-fresh-attempts 48 --hours 9
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py submit --name boltz-restart-brd4-methods-20261117a
```

For completed results, pass the result.json files to
[compare_boltz_methods.py](restart-operations/compare_boltz_methods.py).
It checks that paired starts, GPU models, update budgets, code and scoring
allowances match. It compares each seed at the shortest observed elapsed time
of its paired arms, including setup and founder screening, and separately at
the final update budget. It uses the latest available scoring milestone at
that time. Time to ten is the first completed assessment reaching ten distinct
scored molecules. Lower counts remain visible when an arm yields fewer than ten.

Submitted 17 November 2026. The scheduler listed all 18 jobs as pending at
20:55 CET. Across the three panels, the planned total is 34,200 updates and
at most 720 fresh hard scores. Each job has a nine-hour scheduler limit.
[Job IDs and scheduler status](../../../outputs/boltz-restart-brd4-methods-20261117a/first-scheduler-status.json).

The CPU preflight replayed the actual wave-four decoder checkpoint. All three
source parents decoded to their recorded graphs. For each seed, 16 distinct
child starts were available and their first six exactly matched an independent
regeneration of the six-start arm. Across the three 512-draw founder screens,
none of the 1,536 raw draws was valid. Of the 15 selected replacements, ten had
a verified repair within two edits and five within three. The time to screen
one seed on the local CPU was 34–56 seconds. These results concern starting
conditions. GPU outcomes are pending.
[Screening receipts](../../../outputs/boltz-screen-dry-20261117b).

18 November, 08:39 CET. The first completed pilot, fixed-400 on the A40
(30099854, seed 813211), completed 2,400 updates in 3 h 47 min including
setup. It freshly scored 17 distinct new valid molecules in 17 attempts.
The first child found the best loss (−2.7914). By 1,200 updates it had
11 scored molecules, and the tenth-best loss improved from −0.4363 then
to −0.5469 at the end. The V100 fixed-400 run (30099848) had finished four
of six children and scored 23 distinct new molecules. The other 16 pilots
were pending. Neither seed had a completed alternative allocation policy.
[Completed A40 result](../../../outputs/boltz-restart-brd4-methods-20261117a/fixed400-b-result.json).

18 November, 10:09 CET. V100 fixed-400 (30099848, seed 811211) finished at
09:47 CET. Its six children used 2,400 updates in 3 h 30 min including setup.
All 35 fresh attempts scored distinct new valid molecules. At 1,600 updates
it had 23, and the final two children added 12. Best loss held at −2.843,
while the tenth-best improved from −0.618 to −0.985. Both fixed-150 and
selective extension were pending on all three seeds. Seven of the nine
allocation pilots remained pending. [Completed V100 result](../../../outputs/boltz-restart-brd4-methods-20261117a/fixed400-a-result.json).

## Fixed-75 comparison

Do 32 short children yield ten new valid molecules faster, or improve the
tenth-best fresh loss at equal GPU time? Each seed gets 32 × 75 updates
(2,400 total) and up to 48 fresh FP32 scores. The first 16 starts are paired
with fixed-150 and the first six with fixed-400. The [fixed-75 configuration](restart-boltz-fixed75.json)
uses the original wave-four parents, LR 1, zero optimization noise, and the
same GPU family per seed. The sealed pilot code is reused with one additional
entry in its fixed-length policy table. This is a separate single-generation
test on the original allocation seeds.

```powershell
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py stage --name boltz-restart-brd4-fixed75-20261118a --config restart-boltz-fixed75.json --source-receipt outputs/boltz-restart-brd4-methods-20261117a/fixed400-a-submission.json --max-updates 2400 --max-fresh-attempts 48 --hours 9
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py submit --name boltz-restart-brd4-fixed75-20261118a
```

Pass the completed fixed-75, fixed-150 and fixed-400 results to
`restart-operations/compare_boltz_methods.py` with
`--fixed75-bundle outputs/boltz-restart-brd4-fixed75-20261118a/source.tar.gz`.
The fixed-length comparison checks each shared prefix, both source digests,
the actual GPU model and both budget caps before reporting each seed.

At 11:38 CET on 18 November, the three jobs were pending: **30104381**
(seed 811211, V100 32 GB), **30104382** (813211, A40), and **30104383**
(817211, A100 80 GB). The fixed-150 V100 control 30099849 had started.
[Submission status](../../../outputs/boltz-restart-brd4-fixed75-20261118a/first-scheduler-status.json).

CPU job 30104395 replayed the sealed GPU inputs on the cluster. All three
seeds gave 32 distinct latent/mask pairs, with the first 16 exactly matching
an independent 16-start replay. All available live control hashes matched
(six V100 fixed-400, sixteen V100 fixed-150, six A40 fixed-400). The Windows
replay differed from the V100 hashes, so the receipt uses the cluster
environment. A preliminary CPU checker exited before running Python due to
command quoting. The GPU submissions were unaffected. All three were pending
at 11:43 CET. [Paired starts and scheduler check](../../../outputs/boltz-brd4-fixed75-check-20261118a/starts.json).

## Completed pilot receipts, 19 November 2026

Eight pilot results were available. In the A40 seed, 16 children × 150 updates yielded 40 distinct freshly scored molecules, versus 17 for six × 400 updates. Final tenth-best losses were −0.633 and −0.547 over 228 and 227 minutes respectively. On the V100 seed, fixed-150 and fixed-400 yielded 32 and 35 at their final 2,400 updates, with the same tenth-best loss to three decimals (−0.985). V100 fixed-150 and gradient reuse ran on the same reported device model. Reuse made zero proxy gradient calls, took 244 minutes versus 218 for fixed-150, and gave the same yield of 32. These are BRD4 Boltz measurements on the 31-slot allocation pilot.

The V100 fixed-150 run used a PCIe card; the fixed-400 and selective-extension runs on that seed used SXM2 cards. The independent comparison script rejects the mixed model names. The sealed generation-selection code checks the same equality and will reject this seed when the outstanding pilot dependencies finish. For the wall-time ranking, repeat the one differing arm on matched hardware or predeclare an update-budget analysis and label the timing comparison separately. [Completed native receipts and device identities](../../../outputs/restart-unreported-20261119/boltz-allocation-snapshot.json) · [Generation selector](../../../src/latentmol/design/boltz_policy_selection.py)

### Hardware-matched timing repeat

Job **30111011** repeats fixed-150 on seed 811211 using the original sealed source and scientific configuration, 2,400 updates and at most 48 fresh scores. DTU assigned the job to the V100 queue with an explicit list of three hosts advertising only V100 SXM2 32 GB cards. The first host-list attempt was rejected before any job was submitted because two SXM2 hosts were outside the queue. The accepted list is the intersection of advertised devices and queue members. The PCIe observation keeps its original job and output directory.

Dependent CPU job **30111012** verifies all 16 initial latent/mask hashes against the prior fixed-150 run and ranks the nine pilots using their common elapsed time within each seed. It requires the repeat and the two other seed-A arms to report SXM2. Its output is a decision for review at `/zhome/95/e/206121/dbmol-control/boltz-restart-brd4-timing-sxm2-20261119b/timing-decision.json`. The original sealed callback still references its original jobs. Use the new receipt to choose the longer comparison. [Timing and analysis submissions](../../../outputs/boltz-restart-brd4-timing-sxm2-20261119b/submissions.json) · [Timing replay](restart-operations/repeat_boltz_timing.py)
## First fixed-75 result, 19 November

V100 PCIe job 30104381 finished at 05:25 CET. Its 16 overlapping starting hashes matched the fixed-150 PCIe control (30099849) on seed 811211. Both used 2,400 updates and an allowance of 48 fresh FP32 scores. Fixed-75 reached ten newly scored distinct molecules in 48.2 minutes (525 updates), versus 55.9 minutes (600 updates) for fixed-150, including setup. The final counts were 48 and 32, and tenth-best losses were −1.13670 and −0.98509. Both found the same best molecule at −2.84251. At the shorter total elapsed time (214.37 minutes), the completed scoring milestones showed 48 versus 29 molecules. In a retrospective comparison of the first 32 actual fresh calls in recorded child and candidate order, both had 32 distinct molecules and identical best and tenth-best losses. Fixed-75 benefited from finding more candidates to spend its remaining allowance on. At 10:56 CET the A40 and A100 fixed-75 jobs were still pending. [75-update result](../../../outputs/boltz-restart-brd4-fixed75-20261118a/fixed75-a-result.json) and [150-update control](../../../outputs/boltz-restart-brd4-methods-20261117a/fixed150-a-result.json).
