# DeFoG guidance: benchmark evidence

Collected 12 November 2026. These are exploratory **historical pipeline results**,
not fresh optimization runs using this package. The package extracts the same
one-query, detached probability-MSE guidance pattern without importing those
experiment runners. Machine-readable counts and original record paths are in
[benchmarks.json](benchmarks.json). The research branch's
[experiment note](https://github.com/NikolajHJ/mosaic/blob/codex/mosaic-experiments/experiments/learned-prior/protenix-defog-composition-30000/stage2-pair.md)
contains the campaign context. Raw results live in the research checkout and HPC
archives; they are not bundled in the package.

## Interpretation

Most comparisons use one BRD4 target, 25 occupied atom slots, seed 17, a frozen
Protenix-Tiny contact objective, and Stage 2 connected projection with readout
valence caps and a fluidity floor. The low-learning-rate starts use the same
carbon-chain graph mixed 50% with uniform probabilities. All categories share
the same graph slots. Protenix consumes bond existence; projection and DeFoG see
the full bond-type probabilities. The GuacaMol teacher uses source revision
`365bda9affadd5c2307014a0532ddaa244399441`, checkpoint SHA-256
`bc97d52ccf33a20d991f98349a1d37662273ab1c3e3f9777b938f9f144bc61f5`,
and `hard_aux_soft_core` features.

Each saved readout is a raw argmax graph, with no fragment removal or repair.
“Valid connected” means RDKit sanitization passes and there is exactly one
fragment. Counts include the initial state. Adjacent trajectory states are
correlated observations, not independent molecular samples. Distinct counts are
canonical SMILES within each arm and do not establish chemical diversity beyond
that measure.

Protenix contact scores are in `[0,1]`, higher is better; the optimizer minimizes
their negation. They are model proxies, not binding measurements. “Gap” below is
mean relaxed contact minus hard contact, equivalent to hard loss minus soft loss.
Best valid hard scores and soft scores must be kept separate. A larger gap can
accompany improved hard scores and chemistry.

## Light guidance at learning rate 0.3

The teacher time was fixed near 0.8. Each guided state received one frozen clean
prediction. These arms minimized native loss plus the weighted sum of mean atom
and unordered-bond squared distances to that prediction.

| Updates | Weight | Valid connected | Distinct valid SMILES | Best valid hard contact | Gap | Valid connected endpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 500 | 0 | 475 / 501 | 17 | 0.310847 | 0.017368 | No |
| 500 | 0.0625 | 501 / 501 | 10 | 0.329073 | 0.034311 | Yes |
| 500 | 0.125 | 501 / 501 | 10 | 0.290305 | 0.048315 | Yes |
| 500 | 0.25 | 501 / 501 | 8 | 0.278176 | 0.054111 | Yes |
| 500 | 0.5 | 501 / 501 | 8 | 0.229311 | 0.074832 | Yes |
| 2,000 | 0 | 512 / 2,001 | 23 | 0.310847 | 0.035030 | No |
| 2,000 | 0.0625 | 1,998 / 2,001 | 68 | 0.384597 | 0.134927 | Yes |
| 2,000 | 0.125 | 2,000 / 2,001 | 71 | 0.358074 | 0.170681 | Yes |

Weight 0.0625 improved the best valid hard score in these matched runs. At 500
updates every guided arm retained validity, but stronger guidance reduced the
number of distinct molecules and best hard contact. At 2,000 updates light
guidance improved validity and distinct counts while increasing the soft/hard
gap. This motivates the package's starting weight of 0.0625; it is not a
universally tuned setting.

Sources: `stage2-protenix-defog-weights-20261105a/comparison.json` and
`stage2-protenix-defog-2000-20261106a/comparison.json`, under research `results/hpc/`.
The 500-update sweep reused the completed weight-0 and weight-0.5 pair records.

## Cyclic guidance depends strongly on the update size

Both 10,000-update arms used a raised-cosine weight from 0 to 0.5, with a period
of 1,000 updates, fixed teacher time, and the shared chain start and projection.

| Atom/bond learning rate | Valid connected | Distinct valid SMILES | Best valid hard contact | Endpoint |
| ---: | ---: | ---: | ---: | --- |
| 0.3 | 9,958 / 10,001 | 278 | 0.378666 | Valid and connected |
| 150 | 39 / 10,001 | 39 | 0.260659 | Sanitized, 11 fragments |

The LR 0.3 arm ran for about 177 minutes on L40S, including teacher calls and
pipeline overhead. Its 10,001 teacher queries/forwards and frozen-state checks
passed. Extending the trajectory did not beat the 2,000-update fixed-weight
arm's best hard score. The LR 150 arm shows that a connected projection mode
and a teacher prior do not ensure connectivity after large updates.

Sources: `stage2-protenix-defog-cyclic-10000-20261106a/comparison.json` and
`stage2-protenix-defog-cyclic-lr150-10000-20261106a/comparison.json`.

## Nine matched starts at learning rate 70

This completed comparison used seeds 17, 23, and 42 crossed with sparse-random,
dense-random, and sparse-disconnected starts: nine matched pairs, 500 updates
per arm. The teacher weight was constant at 0.0625. Hardware
was matched within pairs: A10 for seeds 17/23 and A100 for seed 42.

| Metric | Projection | Projection + DeFoG |
| --- | ---: | ---: |
| Valid connected readouts | 31 / 4,509 | 64 / 4,509 |
| Valid connected endpoints | 0 / 9 | 0 / 9 |
| Mean best valid hard contact | 0.222907 | 0.267698 |
| Endpoint fragments | 8–10 | 9–13 |

Guidance had the higher best valid hard score in seven of nine pairs. Every
guided valid connected state occurred by update 20. The late trajectories
fragmented despite sanitized endpoints and passed frozen checks. This is
evidence of an early search benefit, not sustained validity or reliable
endpoint generation. All 18 arms completed, and teacher query/forward and
native frozen checks passed. Guided arms took about 413–500 seconds each,
versus 113–206 seconds for their controls; these timings include model and
pipeline costs and are not an isolated teacher throughput benchmark.

Sources: `stage2-protenix-defog-lr70-starts-500-20261111a/` files
`aggregate-20261112a.json`, `recap-20261112a.json`, and
`terminal-evidence-20261112a.json`.

## Boltz-2 counterexample

A separate 500-update BRD4 pair used the same intended 25-slot graph, projection,
and frozen fixed-pose Boltz-2 objective within the pair. DeFoG weight was 0.5.
The Boltz context differs from the Protenix context, and its affinity loss has
different units: lower is better.

| Metric | Projection | Projection + DeFoG |
| --- | ---: | ---: |
| Valid connected readouts | 400 / 501 | 188 / 501 |
| Best valid hard Boltz loss | −1.391950 | −0.823589 |
| Valid connected endpoint | Yes | No |

The guided endpoint had two fragments and failed sanitization. Native and
teacher frozen checks passed. The result does not support an across-model
claim that this teacher improves chemistry or the native objective.
Source: `stage2-defog-pair-20261104a/comparison.json`.

## Separate formulation: 30,000 logit updates with KL and composition JS

These two earlier arms used a sparse seed-17 start, bounded logit updates,
a KL teacher term, a composition-JS term, and a linear teacher-time ramp from
0.2 to 0.95. They **do not use this package's probability-MSE formulation**.
The second arm added valence-only projection with decoded-element caps, with
no component bridging.

| Metric | Sparse + JS | Same setup + valence projection |
| --- | ---: | ---: |
| Sanitized raw states | 10,611 / 30,001 | 22,656 / 30,001 |
| Sanitized and connected states | 10,155 / 30,001 | 10,036 / 30,001 |
| Distinct valid connected SMILES | 3 | 3 |
| Endpoint | Nitrogen valence failure | Sanitized, 3 fragments |
| Elapsed seconds | 7,382 | 7,601 |

Projection increased sanitization without improving connected-valid coverage
or distinct counts. Both recorded late windows had zero connected-valid
coverage. Both teacher checks passed with 30,001 queries and forwards. An
initial finite backward pass was recorded; this is not full derivative
qualification. No result here establishes an improvement in molecular binding.

Sources: `20261104-defog-sparse-js-30000-01/completed-sparse-js-summary.json`
and `20261106-defog-sparse-js-valence-30000-01/completed-summary-20261112.json`.

## Validation of the extracted package

Package checks are separate from the historical optimization results. The
model-free suite exercises graph validation, stopped targets and live student
gradients, schedules, composition with native losses, transport failures, and
Mosaic's preparation hook. The runnable `--mock` example exercises a complete
toy update/projection loop. It does not test a pretrained teacher or native
structure scorer.

The final local suite passed 57 checks, with one Torch-dependent module skipped
in the local environment. Actual Torch student-loss gradients were checked in
the allocated native validation below. The suite covers all five Stage 2
operator modes, hardening composition, actual toy optimization-loop scoring,
and a real subprocess import-isolation check.

The [native validation record](native-validation.json) passed in DTU job
30017019 on 12 November 2026. It compared the extracted worker with the historical
adapter on three seed-17, 25-atom starts (sparse-random, dense-random, and
sparse-disconnected), each at teacher times 0.2 and 0.7999949998333278. Maximum
target error was exactly 0.0 in all six comparisons in the tested environment.
The new worker made seven queries/forwards including a repeated input; the
historical adapter made six. Both retained the same model state digest, and
repeatability passed. Analytic JAX student-gradient checks and Torch/JAX student
loss/gradient parity passed with `rtol=1e-6, atol=1e-7`.

That native smoke used source archive SHA256
`4dfa7c836e84a316ea7d17bd9a61fb462f15e953c656a5d36492da8b25e4073a`.
Later transport diagnostics and interpreter-isolation changes have model-free
regression coverage; native prediction and student-loss calculations are unchanged.

The worker used Torch 2.5.1+cu121 on CPU with two threads. Individual prediction
calls took 0.06845–0.07617 seconds for these six graphs, excluding startup.
The allocated smoke used four CPU slots and 8 GB, took 53 seconds overall,
and reported 1,171 MB maximum memory. These observations are not a general
throughput benchmark. They establish extraction parity on six inputs and an
attached student loss; they do not qualify derivatives of a native structure
scorer or reproduce an optimization trajectory. No fresh GPU optimization
trajectory has been run with the extracted package as part of packaging.

An additional [installed-wheel check](wheel-validation.json), job 30017889,
passed on 12 November. The coordinator used Python 3.12.11 and NumPy 2.5.3;
the teacher used its own NumPy 1.26.4. A deliberately failing NumPy module in
the coordinator's installation was not imported by the teacher. One native
prediction and the frozen-state check passed. This checks the dependency
boundary used by non-editable installs, independently of the source-tree
validation above; it is not an additional chemical-quality benchmark.

Distribution checks on 12 November also passed: `uv build --offline --wheel
models/defog` produced a wheel that installed into a fresh Windows Python 3.12
environment outside the checkout. With `PYTHONPATH` removed and Python isolated
mode enabled, the package imported from the environment's `site-packages`
without eagerly loading JAX, Torch, or Mosaic. After installing the JAX extra
from cached dependencies (JAX 0.12.0.1, NumPy 2.5.4), the public mock example,
copied outside the checkout, completed ten updates and ten teacher queries.
Both module and console-script CLI help commands passed. The wheel includes
the worker entry point, upstream notice, and MIT license. These distribution
checks exercise installation and the toy path, not a native checkpoint query.

The package was subsequently moved to the repository-root `stage3/` folder.
After relocation, `python -m pytest stage3/tests` passed 57 checks with one
optional Torch module skipped. A wheel built from `stage3/` imported from an
isolated installation outside the checkout and completed the ten-update mock
example with ten teacher queries. All 45 local documentation links checked
across the package and updated guides resolved. This was a layout and
installation check; no new native model or hallucination run was performed.

The package now lives next to Stage 2 under `src/mosaic/small_molecule/Stage3/`.
Build it from the repository root with
`uv build --offline --wheel src/mosaic/small_molecule/Stage3`.
