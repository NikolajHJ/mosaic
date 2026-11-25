# Why Stage 3 switched from sampled endpoints to a one-query target

**Question.** When the molecule and model time stay fixed, how much does the target change on repeat evaluation? We first measured the target changes in the actual Boltz/DBMol trajectory. We then compared both ways of querying the same frozen GuacaMol checkpoint on the same inputs, independently of either scorer.

## What happened in the early Boltz run

The 500-update guided arm drew four candidates from the decoder every five accepted updates. It inserted each supported hard candidate at late model time (about 0.8), ran the remaining 447 native transitions on the 1,000-step distorted grid, and used the first eligible output as a fixed categorical likelihood target for the decoder. A gradient through the decoder moved the VAE latent variable. In the later Protenix runs the optimized variables were the graph probabilities themselves.

The verified offline trajectory contains 501 ordered states. At steps 0, 5, …, 495 the run selected **100 distinct slot-aligned target graphs**. The target changed in **99/99** adjacent refresh comparisons. Across those same pairs, the input argmax on the occupied atom, charge and unordered bond rows was identical **81/99** times. The median input change, averaged over rows as total variation, was 0.00000335. Targets stayed fixed between refreshes. This measures change in the actual supervised labels. A slot-aligned target identity does not imply a distinct canonical SMILES. The decoder distribution, candidate draw, sampler draw and eligibility filter all contributed to this experiment. The endpoint evaluation counted zero valid raw molecules among its argmax and eight categorical draws. [Historical methods and pilot outcomes](https://github.com/NikolajHJ/mosaic/blob/e1b57fc2cab68e17a39c366aa3fbb8c21a0ff2a9/docs/defog_status_2026-11-19/agent_reference.md) · [Refresh analysis](scripts/summarize_defog_refreshes.py) · [Per-refresh measurements](../../../results/hpc/20261125-defog-target-repeatability-01/boltz-refreshes.json)

## Fixed-input checkpoint test, 25 November 2026

We reused all five neutral fixed-slot controls from the 23 October locality probe, including the deliberately invalid carbon star. Each control started as one-hot input with its original atom order. The saved CPU model time was 0.8001909852 (native grid index 553). We used direct insertion without added corruption, the checkpoint's sampling parameters, and 447 native transitions through the final categorical draw. For each graph, eight draws ran in one batch with a recorded random seed, with no postselection or retries. We separately repeated the one-query clean prediction eight times on the very same input and time, changing the random seed. It used the same denoiser with the hard auxiliary features and soft core used in Stage 3. On a one-hot input these feature policies agree. The model state hash was unchanged. DTU CPU job 30363657 completed in 171 seconds. [Test implementation](scripts/compare_defog_target_repeats.py) · [Scheduler output](../../../results/hpc/20261125-defog-target-repeatability-01/job.log) · [Raw labels, probabilities, readouts and pair measurements](../../../results/hpc/20261125-defog-target-repeatability-01/results.json)

| Input | Differing sampled pairs / 28 | Labels differing between two samples, median (range) | Valid connected samples / 8 | One-query argmax |
|---|---:|---:|---:|---|
| Hexane, 6 atoms | 28 | 11.5 (6–15) | 8 | Hexane |
| Pyridine, 6 atoms | 28 | 12 (7–15) | 8 | Pyridine |
| Acetamide, 4 atoms | 28 | 7 (4–9) | 7 | Acetamide |
| Historical 13-atom control | 28 | 30 (26–34) | 8 | Sanitized, two components |
| Overvalent carbon star, 6 atoms | 28 | 10 (6–15) | 8 | Failed sanitization |

We counted one decision per atom and per unordered bond, including the no-bond category. Eight draws yield 28 overlapping pairs per control. All **140/140** sampled pairs differed, by a pooled median of **11** labels. Even after requiring a sanitizable, connected output, the 39 qualifying samples had distinct canonical SMILES within each control. In contrast, the maximum change between repeated single-query categorical distributions was exactly zero for each control. The two procedures have different outputs, so we compare their argmax labels on a common scale and report probability repeatability separately. This small, purposively chosen panel measures same-input sampling variability. It does not estimate a population validity rate or a difference in Protenix optimization performance.

### Does drawing a categorical answer explain all of the variation?

A post hoc calculation on the saved probabilities separates variation from the final categorical draw from variation across complete continuations. For each fixed input, the expected number of differing labels between two independent draws from its one-query predictions is the sum over rows of `1 - sum(p_category ** 2)`. We compared this expectation with the observed mean of the 28 within-input pairs of full continuations. In the table's input order the expected one-query counts were **1.43, 0.57, 0.86, 3.57, 1.61**. The observed full-continuation means were **10.89, 11.89, 6.86, 30.18, 10.36**. Atom slots are aligned and each unordered bond counts once. The one-query means are analytic expectations, while the full-continuation means come from eight draws per input. The difference concerns label variability on these controls. It does not measure the quality of a subsequent Protenix update. The calculation uses only `soft_atom_probabilities`, `soft_bond_probabilities` and `suffix_pair_edits` in the linked results JSON.

## Why we used the clean prediction

At its first forward pass the denoiser already produces clean endpoint probabilities for every atom and bond. The native flow uses those probabilities to construct rates and then sample the next state. The Stage 3 teacher takes the probabilities directly. The reported projection experiments minimize the atom- and bond-group mean squared distances from the current relaxed graph to those probabilities, with the prediction detached for one update. Repeating the query on a new graph gives a new target. The optimized representation is the graph's atom and bond probabilities. Earlier one-query logit experiments used KL divergence, and a separate attached-gradient KL variant passed CPU checks without a completed comparative chemistry trajectory. [One-query implementation and benchmark context](../../../models/defog/BENCHMARKS.md) · [Stage 3 methods](https://github.com/NikolajHJ/mosaic/blob/e1b57fc2cab68e17a39c366aa3fbb8c21a0ff2a9/docs/defog_status_2026-11-19/agent_reference.md)

The October perturbation test asked a different question. The adapted teacher completed 400/400 queries. Ordinary perturbations of size up to 0.01 produced a largest per-row output total variation of 0.005573. Across a pyridine argmax tie, input row total variation of 0.000200 produced output row variation of 0.498548. Total variation is half the sum of the absolute category-probability differences, from zero to one. The checkpoint's auxiliary features use a hard graph and can change abruptly at such a tie. The subsequent 1,000-update Protenix pilot kept the same valid C25 chain in 1,001 readouts. Repeated predictions are therefore reproducible, while locality, chemical repair and exploration remain separate empirical questions. The later nine-pair BRD4 experiment compared one-query guidance plus projection to projection alone. The historical Boltz run cannot serve as a matched full-rollout Protenix control. [Archived locality probe implementation](https://github.com/NikolajHJ/mosaic/blob/880d5d941a4c53abdf1374f9aa84a97df399a644/experiments/learned-prior/defog-soft-teacher/probe.py) · [Stage 3 results](../../../docs/stage3_defog_report_draft.pdf)

## Reproduction

The historical graph summaries are in the verified export `DBMol_exploratory/outputs/hpc-results/source120a004-boltz-defog.html` (SHA-256 `d155a1082be35aad8c1e53f5af159d2ae39b97c1138044633e2b36adb24c6080`). The original DBMol protocol is documented in `docs/37_defog_repair_guidance.md` and `docs/60_long_boltz_defog_comparison.md`. The detailed October locality and 1,000-update reports are in the Mosaic research checkout under `experiments/learned-prior/{defog-soft-teacher,protenix-defog-1000}/RESULTS.md`.

From the experiment's scripts directory the two analysis commands are as follows. Use the native DeFoG environment on an allocated CPU for the second command and a fresh output directory. The historical helper and control definitions are read from the existing archive.

```sh
python summarize_defog_refreshes.py /path/to/source120a004-boltz-defog.html boltz-refreshes.json
/work3/s234818/latentmol/env/defog311/bin/python compare_defog_target_repeats.py \
  --probe-release /work3/s234818/mosaic/releases/20261023-defog-soft-teacher-01 \
  --assets /work3/s234818/latentmol/cache/defog-official-guacamol --output data
```

The new GPU-independent run and artifacts are under `/work3/s234818/mosaic/runs/20261125-defog-target-repeatability-01`. The downloaded result SHA-256 is `81b9193a281edd0fd5cb0030ebe2d209714e98bad0acf24046402b91c3085ed7`. The source revision was `365bda9affadd5c2307014a0532ddaa244399441` and checkpoint SHA-256 was `bc97d52ccf33a20d991f98349a1d37662273ab1c3e3f9777b938f9f144bc61f5`.
