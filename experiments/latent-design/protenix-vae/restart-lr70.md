# Do smaller updates and screened restarts avoid dense invalid graphs?

7 November 2026. The attached graph matches constant 1, cycle 3, branch 03,
update 1 in the [saved tree](../../../outputs/protenix-restart-report-20261107/protenix-restart-report.html?view=trees).
The parent is cycle 2, branch 01, update 498. The branch starts at the parent's
valid 27-bond hard graph, jumps to 235 of 300 possible bonds on update 1, and
returns to the 27-bond parent on update 2. The saved relaxed loss fell from
−0.07914 to −0.11065 during the jump.

The all-row perturbation puts half the probability on alternatives to the
argmax even when the argmax is no bond. In completed panel 29914683, it raised
the mean expected bond count from 28.6 to 160.6 at initialization. Twenty-one of
24 all-row branches produced at least one hard graph with 100 or more bonds.
All 213 such observations were at updates 1–18 of the 25-update offset fade.
The sparse and screened latent policies stayed near 29 expected bonds at
initialization and never exceeded 40 hard bonds in the panel. The screened
kicks started from a new valid hard molecule in 14 of 24 branches. The other
10 changed the latent while keeping the hard molecule.

Every sampled all-row frame with 100 or more hard bonds had zero chemistry
loss. When the bounded repair search produces no witness, the objective has no
repair gradient for that state. The branch in the screenshot eventually
supplied all three cycle 3 elites at updates 79, 80 and 104. Those molecules
passed sanitization and connectivity but had 8, 6 and 4 three-membered rings
in RDKit's ring set. The archive uses canonical SMILES deduplication and fresh
contact loss. Across 80 mixed shortlists with at least two fresh scores in
panel 29914683, the best relaxed-ranked molecule was also the fresh winner
only 13 times.

## New pilot

[Config](restart-lr70.json) for the existing allocated
`python -m latentmol.design.protenix_restarts --arm screened` entry point.
Use seed 17, the same frozen epoch-45 VAE and BRD4 Protenix pocket, 25 occupied
slots, constant chemistry weight 1, temperature 1, and independent fresh FILE
scoring with conformer seed 1201. The first cycle has one 1,000-update
trajectory. The second cycle starts three independent 1,000-update branches
per scored elite, up to nine branches and 10,000 updates overall.

Use LR 70 and a latent displacement cap of 0.25. Constant 1 hit its 0.5 cap
on 9,496 of its first 9,500 updates. Replaying the step-size arithmetic at
the same observed gradients, LR 70 with cap 0.5 would still clip 9,491 of them.
The cap change tests smaller applied steps. This is a combined follow-up
that changes LR, cap, branch length and restart policy together.

Restart from the exact archived latent, screening four seeded directions at
each radius 0.03, 0.10 and 0.25. Prefer connected valid molecules one to three
label edits away. If every kick is invalid, keep the valid parent and record
the fallback. There is no graph logit offset in this arm. Record actual
displacement, cap hits, validity, start bond mass, and independent fresh
scores. The archive still ranks distinct molecules by fresh loss.

Twenty focused local tests passed, including the invalid-kick fallback.
An eight-update CPU smoke with the authenticated decoder and mock scorers
exercised both cycles, produced three valid restarts from one scored elite,
and verified the displacement cap and frozen decoder identity.

DTU job **29915688** was submitted at 15:47 CET to `gpul40s` and was pending
at the first check. Two recent panels had started on L40S while V100 was at
the per-user GPU limit. The job requests one exclusive GPU and a six-hour
wall limit. Input hash and output capacity checks passed. The allocated
native run and fresh scores are pending.

The [submission receipt](../../../outputs/protenix-restart-lr70-20261107a/submission.json)
records the exact command. The source archive SHA-256 is
`f5dcdcff5438f8b611f9fb20dc4b9b7df25558f6dfa675d0c9fae624f35ae23e`.

## V100 replacement on 7 November 2026

DTU job **29920256** replaces the original L40S submission **29915688**.
It requests an exclusive V100 with at least 30 GiB of GPU memory and uses
the same source snapshot, scientific settings, output path, and six-hour
wall limit. The scheduler confirmed that the original exited before
launch. After release, the replacement was pending for a physical GPU
reservation at 23:42 CET.
[Requeue receipt](../../../outputs/protenix-restart-lr70-20261107a/v100-requeue-receipt.json).
