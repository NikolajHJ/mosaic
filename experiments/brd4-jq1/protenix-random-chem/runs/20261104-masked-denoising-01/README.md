# Masked denoising from saved Mosaic checkpoints

DTU V100 job **29890650** finished successfully on 4 November 2026 at 18:54 CET. The [terminal audit](terminal-verification.json) checked all nine cells, 576 attempts, the hard-scoring shortlists, the same-job LSF receipt, and preservation of the frozen models and saved inputs. [Dispatch evidence](campaign.json)

## Matched protocol

- Use the saved sparse seed 23 invalid 50k endpoint, the sparse seed 41 valid 50k endpoint and the exact repaired starting snapshot from the projection comparison. Preserve all saved FP32 probability arrays, the neutral 25-atom schema and the frozen Protenix Tiny pocket scorer with key 17.
- Draw 64 independent categorical graphs per checkpoint. Each draw is a proposal in the independent method and the common input to one bounded structural repair and one masked DeFoG completion. A rejected or duplicate proposal consumes its attempt. The structural arm uses the existing builder and preserves atom identities, charges and occupied slots.
- The frozen GuacaMol DeFoG worker opens implicated or uncertain sites and their neighboring bonds, including a possible bridge for disconnected graphs. It applies native corruption and eight forward evaluations from flow time 0.55 to 1.0, with context at the corresponding corruption level and exact preservation outside the final mask. Terminal atom draws use Mosaic's six elements. This is approximate conditional sampling with an unconditional checkpoint. It does not train the generator or feed its output back into a trajectory.
- Score at most 16 distinct valid connected molecules per method per checkpoint. Take the first canonical representative in the fixed attempt order before consulting scores. Keep empty scoring slots unused and report actual calls, alongside the three separate checkpoint reference forwards. Maximum total is 147 Protenix forwards, plus 1,536 DeFoG forwards.
- Retain each proposal's hard graph, validity, canonical identity, edit counts, sulfur count, mask where applicable and runtime. Report valid and distinct yields, hard-call counts, median and best hard contact, and elapsed time. The scorer measures frozen model contact on discrete graphs. The design compares proposal methods on these checkpoints and performs zero optimization updates.

## Results

Each cell had 64 attempts. Contact scores are from the subset of distinct valid molecules selected within the scoring cap.

| Checkpoint | Method | Valid | Distinct valid | Scored | Median contact | Best contact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sparse seed 23 | Independent | 1 | 1 | 1 | 0.243669 | 0.243669 |
| Sparse seed 23 | Structural | 64 | 4 | 4 | 0.218192 | 0.268890 |
| Sparse seed 23 | Masked DeFoG | 0 | 0 | 0 | — | — |
| Sparse seed 41 | Independent | 5 | 2 | 2 | 0.209066 | 0.215035 |
| Sparse seed 41 | Structural | 64 | 16 | 16 | 0.203493 | 0.219310 |
| Sparse seed 41 | Masked DeFoG | 2 | 2 | 2 | 0.205412 | 0.207727 |
| Shared repaired | Independent | 4 | 2 | 2 | 0.345818 | 0.390706 |
| Shared repaired | Structural | 64 | 46 | 16 | 0.319495 | 0.439214 |
| Shared repaired | Masked DeFoG | 0 | 0 | 0 | — | — |

Proposal generation took 0.047 s for independent sampling, 4.296 s for structural repair and 81.428 s for masked DeFoG across 192 attempts each. Hard scoring added 0.398 s, 2.807 s and 0.165 s respectively. LSF measured 138 s for the whole allocation, including setup and checkpoint reference forwards. The frozen DeFoG CPU worker recorded all 1,536 forwards.

Structural repair found the highest observed contact score at each checkpoint and a wider set of valid candidates, with more usable scoring calls under the shared cap. Its scored medians were lower than those of independent sampling, which yielded few valid graphs. Of the 192 masked DeFoG outputs, 181 were disconnected, including outputs that also failed sanitization. The paired input to DeFoG was the sampled graph, so even a valid checkpoint could supply a disconnected starting draw. Editing a patch of a known-valid hard molecule would test a different use of the generator.
