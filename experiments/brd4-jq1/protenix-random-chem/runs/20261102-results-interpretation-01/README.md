# Structural repair and noise: interpretation of completed results

Question: do the new structural and gradient-noise results improve validity and
contact, and should the next budget buy longer runs, combinations or seeds?

Read-only allocated CPU29872685 completed in173 seconds. It summarized the
immutable edition0016 trajectories and generated128 diagnostic endpoint draws
per selected endpoint with the original categorical-sampling function, saved
FP32 graphs and common seed convention. No optimization trajectory, model
evaluation or new optimization seed was run. Source code and raw result are
under `results/hpc/20261102-chemistry-interpretation-01`; the actual command is
recorded in submission/LSF evidence. See [results](results.json) and
[verification](verification.json). The two existing sampled endpoints reproduce
their original NPZ hashes exactly.

| Comparison | Update | Contact loss, lower is better | Valid connected samples |
| --- | ---: | ---: | ---: |
| Existing repair, sparse seed17, coefficient1 | 20,000 | -0.8737822771 | 0/128 |
| Structural repair, same sparse start/coefficient | 20,000 | -0.8441933393 | 0/128 |
| Structural repair, additional10k | 30,000 | -0.8545953035 | 0/128 |
| Dense original Protenix + delayed chemistry | 50,000 | -0.2077595890 | 0/128 |
| Same dense initialization/schedule + noise0.4 | 50,000 | -0.5235603452 | 0/128 |

Compare within the sparse and dense pairs. Their initial bond distributions
differ. Hardware/continuation histories also differ; these are exploratory
observations, without a multi-seed efficacy or same-device speedup claim.
Contact is the saved relaxed fixed-pose objective; no sampled/hard molecule was
rescored and no experimental binding/activity was measured.

Structural discovery is a strong positive mechanism result: every30,001 recorded
state had a full valid repair target (route1). Nevertheless, the live argmax was
never valid connected. From25k through30k the recorded failure remained a
two-bond fluorine and the accepted target required three bond edits. Late median
weighted bond-repair/native-gradient norm ratio was0.05979; the endpoint ratio
is similarly small. The full-target NLL is averaged over350 decisions for N25,
including neutral charge slots, while partial repair averages edited bonds.
On full-target routes, current local/cut fallback risks are masked off. Thus a
small full NLL or zero fallback component does not certify a valid argmax.
Weak/diluted guidance for the remaining edits is a leading hypothesis, not a
proved directional-gradient diagnosis.

Noise improved the dense run's final contact score, especially after30k, but
neither dense run produced a valid connected state in its recorded trajectory
or a valid connected endpoint sample. There were0/2501 valid retained graphs in
each. Noise is promising for exploration, without a demonstrated validity gain.

The all-in combinations ended with20 and24 fragments for seeds17/23, contact
-0.14970/-0.10922, despite sanitized fragments. The delayed20k controls only
reached chemistry coefficient0.04643, since their ramp still ends at50k; they
are not strong-chemistry endpoint tests. Stronger old coefficient10 alone also
failed, but that arm never found a full valid teacher and used fallbacks.

Recommended next comparison, not launched: keep sparse initialization seed17,
structural discovery, composition0, detached step-fixed targets, T1 and the
existing optimizer. Test constant chemistry10 against the existing structural1
baseline over30k; then compare an otherwise identical arm with binding-scaled
noise annealed from0.4 to0 by20k, leaving10k for noiseless settling. Reuse the
already-submitted seed23/chemistry-first arms as additional evidence before
adding seeds. If stronger structural guidance still leaves the same few bad
bonds, prioritize an edited-bond-focused NLL/bounded target margin over another
longer unchanged run or a large hyperparameter grid.

Annealing noise has precedent in neural-network optimization
([Neelakantan et al.](https://arxiv.org/pdf/1511.06807)); benefit for these
molecules and this gradient-relative rule is an untested hypothesis.
