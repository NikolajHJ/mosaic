# Does DeFoG recognize repairs and guide the current graph?

## Controlled bond corruptions

The probe takes the first two distinct valid shortlist molecules from each of the six archived projected DeFoG and projected repair trajectories on seeds 17, 23 and 42. Parent identities are deduplicated across trajectories. Each parent supplies an untouched control, a single bridge deletion, and a single-to-double bond change that causes a recorded valence error if an eligible edit exists. Candidate edges are considered in a seeded order. Inputs use 2% uniform probability mass and the campaign's six-element representation.

The frozen GuacaMol checkpoint makes one query per input at time 0.8. We measure probability gain on the original bond, the rank of the best validated single-edit repair among all bond-category substitutions, and the hard graph after interpolating toward the teacher by 0.1, 0.25, 0.5 or 0.75. The full teacher argmax is also recorded. Ranks use the gradient's predicted gain in the proposed bond's margin over the current winner. RDKit checks all single substitutions, with atoms fixed for repair ranking. Interpolation allows atom and bond changes, matching the two guided heads. The clean parent controls measure how much the same intervention changes an already valid graph.

DTU CPU job 30364221 runs [the probe](scripts/probe_defog_repair_direction.py) against the archived 18 November shortlists and the frozen campaign assets. Initial job 30364205 stopped during archive reading because the shortlist stores `hard_X` and `hard_E_upper`. The corrected reader uses those keys. Both attempts have separate scheduler logs.

Job 30364221 completed with 36 queries and 36 denoiser calls. The frozen state digest matched at the end. All 12 parents supplied both corruption types. [Raw probabilities, ranks and readouts](../../../results/hpc/20261125-defog-mechanism-01/repair-results.json)

| Measurement | Bridge deleted | Valence violation |
|---|---:|---:|
| Original bond choice gained probability | 10/12 | 12/12 |
| Median rank of the original bond choice (out of 1,200 edits) | 132 | 5 |
| Median rank of the best verified one-edit repair | 28.5 | 3 |
| Best verified repair in top 10 | 1/12 | 12/12 |
| Valid connected after 25% move toward target | 0/12 | 0/12 |
| Valid connected after 75% move toward target | 0/12 | 2/12 |
| Valid connected at the teacher argmax | 0/12 | 1/12 |

All 12 untouched controls remained valid after the 25% and 50% moves. At 75%, eight stayed valid. All clean graphs come from the same BRD4 archive. The corruption and probability mixing are controlled interventions on those graphs. A high repair rank here measures the local direction of the frozen loss. Actual hard validity also depends on margin size and changes to the other atom and bond decisions. The interpolation probe supplies a separate check of that joint effect without Protenix or Stage 2 operators.

## Matched optimization controls

The six new trajectories use the original three fragmented starts from the [18 November comparison](repair-pair-20261118.md). The plan copies the archived live-teacher run configuration, including Protenix key policy, LR 60, 500 updates, weight 0.5, model time 0.8, connected projection with the archived solver recovery, temperature 1, and 0.0001 uniform mass. Each seed's controls share an allocation on the GPU family used for its archived live baseline.

- **Frozen initial prediction.** Query the initial graph once and keep the teacher's prediction fixed throughout the run.
- **Shuffled prediction.** Query the live graph, then permute complete prediction rows within each current winning category separately for atoms and bonds. Randomness is fixed by seed and update. Each head's MSE weight is rescaled to match the live teacher's tangent-gradient norm on that same graph. The run records per-head scales, shuffled-row counts and any degenerate norm match. This retains category-level information and tests whether assigning predictions to their original rows matters.

The primary outcomes are the best valid hard Protenix contact, distinct valid identities and endpoint validity. Compare each seed against its archived live trajectory and inspect retained settings and initial tensors. Projection-only results from the broader sweep provide context, with a different probability interior policy. The live baseline is an archived execution on the same GPU family, so run-to-run variation remains relevant.

| Seed | Matched GPU family | Job |
|---:|---|---:|
| 17 | L40S | 30364222 |
| 23 | V100 | 30364225 |
| 42 | A40 | 30364226 |

Generate the six-arm plan with [plan_defog_mechanism.py](scripts/plan_defog_mechanism.py) using the original campaign's `plan.json`. The GPU wrappers call the standard `stage3_campaign.py run-bundle` entry point with the new plan. The sources derive from the archived 18 November source snapshot with the teacher-control change in the runner. Results are written under `/work3/s234818/mosaic/runs/20261125-defog-mechanism-01`. LSF job 30364247 waits for the GPU allocations to end and runs [the matched results collector](scripts/summarize_defog_mechanism.py).

Both three-update mock controls completed in CPU allocation 30364252. Its post-run check attempted to open compressed gradients, while mock mode saves plain JSONL. The saved-output check of the correct file verified the 1 and 3 teacher calls and a maximum error of about `1.2e-9` in matching the per-head tangent gradient norm. This checks the control's execution and calibration on the fixture.
