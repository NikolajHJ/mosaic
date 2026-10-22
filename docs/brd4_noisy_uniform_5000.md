# BRD4-BD1 noisy-uniform control, 5000 updates

> **Scope and access note (22 October 2026):** this document contains an
> experiment protocol and dated operator evidence. Concrete account paths,
> environments, jobs and source hashes describe that operator/attempt; they
> are not shared-account defaults or a fresh status/acceptance claim. Use
> [your own HPC setup](HPC_GUIDE.md). Keep the scientific gates and historical
> outcomes below intact when preparing a separately validated candidate.
> Canonical working home: [brd4/noisy-uniform-5000](../experiments/brd4/noisy-uniform-5000/README.md).


Requested 2026-10-16 as a third study alongside the unchanged chain-mixture
and exact-uniform studies. CPU readiness passed; native job 29736761 and
dependent report 29736762 were submitted and remain pending. Actual noisy
native gradient checks and the 5000-update trajectory are not yet complete.

## Fixed setup and initialization decision

Use the same frozen Protenix-Tiny checkpoint, full 127-residue BRD4-BD1 chain A
from 3MXF, whole-target native contact loss, fixed six neutral occupied slots,
seed 17, highest precision, 0.03 initial step and 5000 accepted updates.
Preserve the prior derivative/null/parity/frozen-model/descent gates and the
ten-halving backtracking policy. No affinity or chemistry loss is added.

The user suggested `0.5 * uniform + 0.5 * random noise (mean 0, variance 1)`.
**Pivotal implementation decision:** first project the Gaussian component
with Mosaic's existing row-wise Euclidean simplex projection, then mix it
50/50 with the uniform baseline:

`X0 = 0.5 * U_X + 0.5 * simplex(Z_X)`

`E0_upper = 0.5 * U_E + 0.5 * simplex(Z_E)`

`U_X` has C/N/O/S/F/Cl each 1/6. `U_E` has NO_BOND=0.5 and
SINGLE/DOUBLE/TRIPLE/AROMATIC each 0.125, from P(exists)=3/6.
`Z_X` and `Z_E` are independent standard-normal draws per row/category from
the two `jax.random.split(jax.random.key(17))` subkeys, atom first and bond
second. Every unordered pair is sampled once; both directed native entries
reuse that pair. Self-bonds remain absent. No logits, clipping/renormalization,
extra jitter or repeated search for a favorable seed is used.

Gaussian draws are not probabilities, so projection is necessary. Static
review found that projecting the combined sum can introduce zero entries
incompatible with the existing central finite-difference/null probes. Mixing
AFTER projection keeps at least 1/12 in each atom category, 1/4 in NO_BOND
and 1/16 in each present-bond category. These exceed the unchanged 0.01
largest central epsilon and 0.005 bond-order null move, without modifying
native checks or searching for a favorable seed. The realized rows do not
retain zero-mean/unit-variance Gaussian moments, equal category weights,
P(bond)=0.5 or expected degree 2.5. The random component may have zeros;
the mixed graph is interior. This is one
exploratory seed, not evidence of replicated success or chemical validity.
Noise is symmetric across all five categorical bond states, so it does not
preserve the baseline existence prior: both symmetry and bond density change.
The hard preparation scaffold and fixed native slot/name metadata are
unchanged; all optimized X/E entries are replaced by this recipe.

## Shared implementation, output and execution gates

The initialization recipe is explicit in
`examples/configs/brd4_noisy_uniform_5000.json`. It reuses the shared graph
contract and projector without changing `src/` or either earlier immutable
release. Exact-uniform and hard-chain mixture behavior remain covered by
regression tests. Tests additionally check the formula against an independent
NumPy simplex projection, seeded reproducibility, changed seeds, independence
from scaffold labels, absent self-bonds and rejected invalid settings.
Numerical tests run only on allocated DTU CPU workers.

Authenticate the previous passed BRD4 pilot/context, source 02 and resource
assessment. Before GPU submission, require the selected CPU suite and source,
configuration, fixture and aggregate-quota gates to pass. Reserve the old full
three-case artifact bound for this singleton control AND each of the two
outstanding studies: `4 + 3 * (7.324157069437206 - 4)` = about 13.9725 GiB.
Use an early 14 GiB home-space gate. Check it again at native launch.
No existing job is cancelled or changed.

All native derivative/null/parity/descent gates run on the actual noisy graph
at GPU entry before accepting updates. Stop on a failed check; do not change
the seed, add tolerances or repair a graph to force success. The renderer
labels the mixed projected-Gaussian start and shows all exact saved probabilities
and raw categorical graphs, including invalid or disconnected structures.
Readable atom labels and probability-dependent atom/bond sizes are retained.
The native model still sees bond existence, not bond order. A chemically valid
hard molecule is an observation to assess, not a promised outcome.

## Immutable source and readiness submission

Source 01: 261 files; archive 38,980,076 bytes, SHA-256
`4d36c3d8d0bb945553eca3e9adf5c53fce754bdf832c896c09866030e51cde21`;
manifest `ea4ceab78cb7605fdd715cb79bebacefc03ec8c8a5947834fa58540d5028c02e`.
Git reference is `43fdcb17e06127ef6b5eec2a4c378d36aeec739a` with dirty source
bytes bound by the manifest. Immutable remote source:
`/work3/s234818/mosaic/releases/20261016-brd4-noisy-uniform-5000-01/source`.
Archive upload SHA and launcher Bash syntax checks passed. Local Ruff and
bundled Node `--check` passed (static checks only, no local numerical work).

CPU readiness **29736756 submitted**, hpc two slots/4GB per slot, 15-minute
bound. Run:
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-noisy-uniform-readiness-01`.
It requires the selected small-molecule CPU suite and unchanged-source,
config/fixture/reference identities, with two outstanding studies counted in
addition to this new control and an early 14 GiB free-home gate.
Submission alone is not a passed test or native gradient result.

## Actual readiness outcome and queued work

CPU **29736756 DONE**, worker n-62-30-25: **167 passed, three native-model
items deselected, two existing dtype warnings, 50.02s**. This includes all
13 uniform/noisy-control checks. Source/reference/config/fixture bindings
and aggregate storage passed. Required free home space is
**13.972471208311617 GiB**; the conservative home gate reported 25 GiB.
Config SHA `c74107f4720b148ac3b19968fa32bd5dd4f093824cae5527345ec24c93625c71`;
assessment SHA `4e61063a54e64f671a9c4ce97b009b0fa5f22e659401235c3d4eb4d14a441185`.
Local evidence is under `results/hpc/20261016-brd4-noisy-uniform-5000/`.

Native **29736761 submitted**, one exclusive A10, four CPU slots/32GB total
host memory request, 16-hour bound, dependency `done(29736756)`. Run:
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-noisy-uniform-5000-01`.
It rechecks immutable source/config/pilot/assessment and quota, then loads the
passed pilot's frozen native context with explicit initialization and
update-count replacement policies. No pilot endpoint is used as the start.
Every actual noisy-graph native gate remains mandatory before updates.

Report **29736762 submitted**, one CPU slot/4GB, 90-minute bound, dependency
`ended(29736761)`. Output:
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-noisy-uniform-5000-report-01/payload/report/`.
The renderer requires a complete passed study. The final snapshot recorded
both new jobs and the original/exact-uniform pairs as PEND. Earlier jobs
were read only. No trajectory, hard-molecule result, rendered report,
downloaded final result or browser QA is claimed yet.
