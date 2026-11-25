# BRD4 Boltz2 DBMol long restart trial

16 November 2026. Question: does the existing gradient restart/archive search
produce sustained improvements on BRD4 with the current full Boltz2 DBMol
objective? The seven-protein campaign remains deferred.

[Configuration](restart-boltz-brd4.json): LR 1, 400 updates per child,
3 parents x 2 children, six distinct latent bootstrap starts, constant chemistry
and valence weights 1, step-norm cap 0.25, latent restart proposals and no update
noise. Eight bounded retries refill exact duplicate starts. Parent selection
uses the best fresh-scored distinct valid archive molecules. The final export
keeps up to 20 distinct valid molecules, with the best ten marked primary.

Six nominal waves cost 14,400 updates and at most 216 fresh scores. A 20-wave
ceiling permits up to 48,000 updates / 720 fresh scores if the measured pace
allows. The existing completed-wave forecast decides whether another full wave
fits a 79,200-second working budget. This is a soft boundary check, not a precise
22-hour duration guarantee. Each completed wave saves a continuation checkpoint.
The scheduler allocation is 24 hours on one 32 GB V100, one serial GPU worker,
FP32, no TF32/autocast, and full activation checkpointing. V100 has no native
BF16 acceleration; concurrent children are not enabled for this trial.

The new native crystal context uses 3MXF chain A / JQ1 author chain A residue 1,
all 31 reference heavy atoms, all 127 protein residues and the original 19-residue
pocket. Earlier Protenix searches used 25 occupied slots. Coordinates are fixed;
there is no diffusion or pose optimization. Generic DBMol contact selection uses
the specified pocket; the older Mosaic pipeline had two named contact anchors.
These scores therefore are not a directly matched continuation of those older
experiments.

Allocated CPU preparation job 728493 completed in 30.45 seconds. Exact native
seed-feature parity passed and all pocket residues survived. The native JQ1 CCD
reference has an explicit aromatic-H codec round-trip warning; native features
were retained. Generated candidates still require connected, sanitized,
radical-free molecules. This CPU evidence does not establish native gradient
or design quality. [Preparation receipt](../../../outputs/boltz-brd4-setup-20261116a/summary.json).

Submission uses the shared runner, not a copied controller:

```powershell
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py stage --name boltz-restart-brd4-v100-20261116a --config restart-boltz-brd4.json --queue gpuv100 --hours 24 --max-updates 48000
python experiments/latent-design/protenix-vae/restart-operations/submit_campaign.py submit --name boltz-restart-brd4-v100-20261116a
```

Submitted at 17:26:57 CET as **30082565** on `gpuv100`; at 17:27 CET it was
**PEND**, waiting for scheduling. The scheduler confirms `gpu32gb`, 30 GiB GPU
memory, four CPU slots and a 24-hour limit. No start ETA is available.
Source archive SHA256:
`3de4f820e910a41af41fe7d635563792d774f4ea5348a01853fcd71746632e47`.
[Submission receipt](../../../outputs/boltz-restart-brd4-v100-20261116a/brd4-jq1-submission.json),
[sealed launch plan](../../../outputs/boltz-restart-brd4-v100-20261116a/gpu-plan.json).

Validation: 24 focused CPU tests passed, covering native-scorer graph/cache
guards and launch inputs/resources/budgets. The first test invocation hit a
Windows global temporary-directory permission error; rerunning with a fresh
workspace temporary directory passed. No new native GPU evidence is claimed.
Existing jobs and sealed inputs were preserved. Output will be written to
`/work3/s234818/latentmol/runs/boltz-restart-brd4-v100-20261116a-brd4-jq1`.

18 November, 04:20 CET. Job 30082565 exited successfully after the time
budget stopped the search before wave eight. Seven waves used 12,800 updates
and 56 successful fresh scores. The best fresh loss improved from −2.598 in
wave four to −3.105 in wave six. Wave seven held that best and improved the
top-three mean to −2.933. The final 20 distinct valid molecules included a
tenth-ranked loss of −1.464 and a twentieth-ranked loss of −0.580.
[Wave results](../../../outputs/boltz-restart-brd4-v100-20261116a/brd4-jq1-result.json)
and [ranked molecules](../../../outputs/boltz-restart-brd4-v100-20261116a/brd4-jq1-selected-molecules.json).

A retrospective read of the 32 completed branch logs found 63 distinct valid
SMILES. Counting a molecule if it occurred by a given step on any observed
branch, 57 occurred by step 100, 60 by step 150, and all 63 by step 212.
Of the 20 exported molecules, 19 had their selected state by step 50 and
the other at step 53. Their shared founder was bootstrap 01; 18 passed
through cycle 04, branch 03. The winning graph appeared at the initial
state of a wave-six child. These counts reuse the realized branches and
cannot predict which parents a shorter run would choose.

The exported molecules' audited SA ranged from 6.25 to 8.45 (median 7.83).
The winner scored 7.95 and RDKit perceived six three-membered rings in it.
The BRD4 reference (+)-JQ1 scored 3.33 with the same implementation.
[Reference and winning structure](../../../outputs/boltz-restart-brd4-v100-20261116a/brd4-reference-and-winner.png).

## Functional groups in the saved BRD4 molecules

19 November. The question was whether the search could produce common binding groups and whether a graph-level loss could supply a gradient. Among the 63 distinct raw-valid SMILES in the 32 completed branches, RDKit perceived 2 molecules with a carbonyl (`[CX3]=[OX1]`), 0 with an aromatic ring, and 61 with a neutral hydroxyl attached to carbon. In the 20 exported molecules these counts were 0, 0 and 20. Among the 49 successfully docked 31-heavy-atom comparison molecules, 33 had a carbonyl and 43 had an aromatic ring. Two baseline molecules lacking both features also outscored the entire export on Vina. These are related molecules from one productive founder. [Comparison data](<C:/Users/nikol/OneDrive/Dokumenter/GitHub/mosaic/results/brd4-baselines-30082565/comparison/BRD4 benchmark against run 30082565 - interface metrics.json>).

A local CPU replay decoded all 20 exported latents at temperature 1 with their saved occupancy masks and recovered every recorded SMILES. For each ordered pair of occupied slots, the carbonyl proxy was the product of the carbon, double-bond and oxygen probabilities. Its maximum per molecule had median 0.032. The soft triangle count used the trace of the cubed symmetric bond-presence matrix divided by six (zero diagonal). Both proxies had nonzero finite latent gradients on all 20 replays. Median gradient norms were 0.21 for the strongest carbonyl and 5.61 for the triangle count. This took 20 decoder evaluations, zero optimizer updates and zero new binding scores. Formation of discrete motifs and changes in docking quality require a separate test.
