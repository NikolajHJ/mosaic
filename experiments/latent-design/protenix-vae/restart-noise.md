# When does restart noise change the path while retaining the parent?

8 November 2026. In the completed standalone runs, the median Jaccard
overlap of novel valid molecules between siblings was 0.181 at LR 30
and 0.017 at LR 70, across nine pairs per run. For each child, the median
fraction of distinct valid non-parent molecules also seen on its saved
source path was 36.7% and 3.5%, respectively. None of the nine LR 70 sibling
pairs shared a non-parent molecule in updates 97–1,000. The jobs selected
different parents and differed in their valid coverage, so this is a
description of these trajectories.

In two completed LR 70 six-cycle arms, the median sibling Jaccard
overlap was 0.280 with relaxed-only shortlisting and 0.286 with unchanged
parents and independently seeded update noise. For branches from different
parents in the same cycle it was 0.030 and 0.000. Late-window nearest-neighbor
fingerprint similarities were also higher for siblings in both arms.
Exact SMILES differences and differences between structural families are
separate readouts.

## Matched tests

The [config](restart-noise.json) uses 12 archived, fresh-scored parents from
three completed source cohorts. It holds LR at 70, the displacement cap
at 0.25, temperature at 1, and both repair and valence excess weights at 1.
It reuses the epoch-45 decoder and BRD4 Protenix pocket. The same normalized
random direction is used at each raw latent radius for a given parent and
repeat. Most conditions have three repeats. The short tests use two
untouched repeats to measure how closely exact starts replay on the same GPU.

| Panel | Conditions | Updates per job |
| --- | --- | ---: |
| Latent dose A and B | Six parents per job. Radii 0.03, 0.10, 0.25, 1 and 3. Untouched, screened kicks, independent Gaussian starts, and update noise alone. 64 updates per branch. | 9,984 |
| Graph dose A and B | The same 12 parents across two jobs. Sparse and dense protected mixing at 25% and 100% of the safe per-row budget. Untouched, screened, independent, and legacy all-row 50/50 controls. 72 updates per branch. | 9,936 |
| Persistence A, B and C | One distinct parent per job. Radii 0.25 and 1, both high graph doses, screened, independent, and one untouched trajectory. 500 updates per branch. These reuse the short-panel starts for the matching parent and arm. | 9,500 |

Sparse mixing covers three connected atom slots and their existing internal
bonds. Dense protected mixing covers the occupied graph and limits the
extra mass given to bonds on rows whose parent label is no bond. The
protected starts preserve the parent's hard labels at update zero. Both
graph offsets fade by update 25. The legacy all-row control applies
50/50 mixing even to absent bonds. The independent Gaussian baseline uses
the same occupied slots and records invalid starts. All arms fix 25
occupied slots, so this pilot measures no-bond protection at fixed atom count.

The update-noise-only arm keeps the exact parent at update zero. Its
perturbation begins at 5% of the binding step norm and fades to zero by
update 100. Fresh scores use the same independent scoring protocol with
up to three shortlist slots per short branch and six per long branch.

Readouts compare valid non-parent SMILES overlap, aligned agreement and
revisits, first divergence and later rejoin, late fingerprint similarity,
parent retention, endpoint identity, validity, and fresh scores. Aligned
molecule matches count only jointly valid states and retain the denominator.
Slot-aligned hard graph labels also measure exact replay through invalid
states. The historical source paths contain 24 updates and may use different
optimizer settings. Untouched branches provide a same-length continuation
control under the new settings. Each job runs all its conditions on one GPU.

The allocated entry point is `python -m latentmol.design.protenix_restart_study`.
The [submission directory](../../../outputs/protenix-restart-noise-20261108a/)
holds the sealed source, config, and exact commands. Per-job
`path-analysis.json` accompanies the branch trajectories and fresh scores.

DTU accepted all seven jobs at 13:44 CET. All were pending at 13:45 CET.
The jobs request six hours each. A100 and V100 carry the replicated short
panels. The three long panels are split over A100, V100, and A40. L40S
already had a backlog of the earlier restart jobs.

| Panel | GPU queue | DTU job |
| --- | --- | ---: |
| Latent A | gpua100 | 29924369 |
| Latent B | gpuv100 | 29924373 |
| Graph A | gpua100 | 29924374 |
| Graph B | gpuv100 | 29924375 |
| Persistence A | gpua100 | 29924376 |
| Persistence B | gpuv100 | 29924377 |
| Persistence C | gpua40 | 29924379 |

The source archive SHA-256 is
`7cabd5ef4664fbd7c2ea600a0e702f80d88a4417e4b05f0983116cb35c53cb68`.
Twenty-seven focused tests passed. A CPU smoke with the authenticated
decoder, one real archived parent, and mock scorers exercised nine
branches and the path readout. The two untouched smoke paths matched
exactly in latent and raw labels, even where valid-molecule agreement
had no denominator. Native comparisons are pending.

## First completed panels on 8 November 2026

At 15:46 CET, Latent A and Graph A had completed successfully on A100.
Latent A finished 156 branches and 9,984 updates. Graph A finished 138
branches and 9,936 updates. In each panel, all six pairs of untouched
continuations agreed on every saved hard graph.

In Latent A, radius 0.03 preserved the exact starting SMILES in all 18
branches. All 18 sibling pairs ended at different valid molecules after
64 updates. Mean endpoint-to-parent Morgan similarity was 0.177 at
radius 0.03 and 0.124 for independent starts. The radius 1.0 arm had
the largest mean best fresh-loss gain, 0.209, across the six parents.

In Graph A, all four sparse and protected dense arms preserved the
starting hard graph. No sibling pair ended at the same molecule among
the 14–18 jointly valid endpoint pairs per arm. The sparse low arm
had 95.5% valid observations. The protected arms reached at most 31
hard bonds in the first 25 updates, versus 54 for the all-row control.
The all-row control still had the largest mean best fresh-loss gain
in this panel, 0.228, versus 0.220 for screened and 0.219 for dense high.

Latent B was running with 82 of 156 branches complete. Graph B and
all three persistence panels were pending while DTU had their queues
inactive. These two short panels show early divergence. Persistence
at 500 updates is still unmeasured.

[Saved results and path readouts](../../../outputs/protenix-restart-noise-20261108a/readout-20261108-1546/).

## Replicate and persistence results on 8 November 2026

At 22:05 CET, Latent B and Persistence A and C had finished. Graph B had
saved all 9,936 optimization updates and was fresh-scoring its shortlist.
Persistence B remained queued on V100.

Across both latent panels, radius 0.03 kept the exact parent SMILES in 35 of
36 starts. None of 30 jointly valid sibling endpoint pairs matched at update
64, though eight pairs temporarily rejoined. Late sibling Morgan nearest
similarity was 0.255 and 0.223, versus 0.135 and 0.142 for independent starts.
Untouched controls replayed all saved hard labels in both panels. The best
fresh-loss gain among latent radii was at radius 1 in panel A and 0.25 in
panel B. Independent starts led the fresh-loss gain in panel B.

At 500 updates, none of 23 jointly valid parent-seeded sibling endpoint
pairs matched across the two completed persistence parents. Late structural
similarity was often close to independent starts. Sparse-high kept the parent
at all six starts and had 98.5% and 77.8% valid observations, versus 79.8% and
62.2% for dense-high. Dense-high had no jointly valid endpoint pair in panel C.

[Snapshot and path readouts](../../../outputs/protenix-restart-status-20261108-2205/).
