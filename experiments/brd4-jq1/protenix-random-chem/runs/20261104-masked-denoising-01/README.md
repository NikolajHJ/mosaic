# Masked denoising from saved Mosaic checkpoints

This read-only benchmark compares how many valid, distinct, high-contact molecules three proposal methods find near saved states. It runs on DTU V100 job **29890650**. Source, input, startup, storage and held release proofs are in [campaign.json](campaign.json). The first post-release scheduler observation was pending on 4 November 2026 at 13:33 CET.

## Matched protocol

- Use the saved sparse seed 23 invalid 50k endpoint, the sparse seed 41 valid 50k endpoint and the exact repaired starting snapshot from the projection comparison. Preserve all saved FP32 probability arrays, the neutral 25-atom schema and the frozen Protenix Tiny pocket scorer with key 17.
- Draw 64 independent categorical graphs per checkpoint. Each draw is a proposal in the independent method and the common input to one bounded structural repair and one masked DeFoG completion. A rejected or duplicate proposal consumes its attempt. The structural arm uses the existing builder and preserves atom identities, charges and occupied slots.
- The frozen GuacaMol DeFoG worker opens implicated or uncertain sites and their neighboring bonds, including a possible bridge for disconnected graphs. It applies native corruption and eight forward evaluations from flow time 0.55 to 1.0, with context at the corresponding corruption level and exact preservation outside the final mask. Terminal atom draws use Mosaic's six elements. This is approximate conditional sampling with an unconditional checkpoint. It does not train the generator or feed its output back into a trajectory.
- Score at most 16 distinct valid connected molecules per method per checkpoint. Take the first canonical representative in the fixed attempt order before consulting scores. Keep empty scoring slots unused and report actual calls, alongside the three separate checkpoint reference forwards. Maximum total is 147 Protenix forwards, plus 1,536 DeFoG forwards.
- Retain each proposal's hard graph, validity, canonical identity, edit counts, sulfur count, mask where applicable and runtime. Report valid and distinct yields, hard-call counts, median and best hard contact, and elapsed time. The scorer measures frozen model contact on discrete graphs. The design compares proposal methods on these checkpoints and performs zero optimization updates.

The remote output is `/work3/s234818/mosaic/runs/20261104-masked-denoising-01/payload/result.json`. Verify terminal LSF status, all nine cells, per-cell budgets, frozen model receipts and input preservation before interpreting the results.
