# Can the restart search deliver six distinct high-scoring molecules?

7 November 2026. The [pilot config](restart-diversity-lr100.json) uses
LR 100, 500 updates per branch, and five cycles. Run one bootstrap branch
and one screened child per selected parent in the four restart cycles.
Three parents per cycle give at most 13 branches, 6,500 updates, and 208
fresh scoring slots. Use the frozen epoch-45 VAE, prepared BRD4 Protenix
pocket, seed 17, 25 occupied slots, latent step cap 0.25, and a mixed
fresh-score shortlist. Keep the repair and valence excess weights at 1
throughout. That valence weight was calibrated on the archived states in
the [six-cycle comparison](restart-lr70-six-cycle.md).

The export keeps the best fresh score, then walks down the fresh-score
ranking. Admit each candidate only when its Morgan fingerprint has
Tanimoto similarity at most 0.60 to every molecule already selected.
Fingerprints use radius 2, 2,048 bits, and chirality. The final
`diversity.json` compares up to six selected molecules with the six
score-only leaders from the same cumulative pool. It records panel size,
maximum pairwise similarity, mean fresh loss, molecule identities, source
steps, and chemistry flags. The threshold is fixed. The score-ordered
selection reports a smaller panel if it finds fewer than six. During
optimization, the existing diversity-aware policy selects restart parents.

A replay on earlier cumulative pools from constant 1, constant 10, and the
late ramp filled all six slots at 0.60. The mean fresh-loss cost relative to
the score-only panel was 0.000–0.002 across those pools. The 0.65 threshold
left those three score-only panels unchanged.

The allocated entry point is `python -m latentmol.design.protenix_restarts`
with `--arm diversity`. DTU accepted job **29918056** on `gpuv100` at
19:40 CET with a six-hour wall limit. It was pending at the first check.
The [submission receipt](../../../outputs/protenix-restart-diversity-lr100-20261107a/diversity-submission.json)
records the exact command and source hash. Input and storage checks passed.
Twenty-one focused tests passed. A mock five-cycle smoke exercised all
13 branches and exported both panels from the cumulative archive.
Native Protenix results are pending.
