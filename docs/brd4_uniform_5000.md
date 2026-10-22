# BRD4-BD1 uniform-start control, 5000 updates

> **Scope and access note (22 October 2026):** this document contains an
> experiment protocol and dated operator evidence. Concrete account paths,
> environments, jobs and source hashes describe that operator/attempt; they
> are not shared-account defaults or a fresh status/acceptance claim. Use
> [your own HPC setup](HPC_GUIDE.md). Keep the scientific gates and historical
> outcomes below intact when preparing a separately validated candidate.
> Canonical working home: [brd4/uniform-5000](../experiments/brd4/uniform-5000/README.md).


Requested 2026-10-16. CPU readiness and resource gates passed; uniform native
job **29736735 submitted** on A10, with report **29736736** attached. Native
uniform checks/optimization remain pending. The original study is unchanged.

Change only graph initialization for a separate control of the same frozen
BRD4-BD1/3MXF whole-target contact experiment. Keep six occupied neutral
slots, seed 17, initial step 0.03, 5000 accepted-update request, highest FP32,
one recycle, geometry off, ten halving trials, and every derivative/null/
parity/frozen/descent check. No affinity, chemistry or repair term is added.

Every atom starts at exactly 1/6 probability for C/N/O/S/F/Cl. For every
unordered pair, `P(exists)=3/N=0.5`, `P(NO_BOND)=0.5`, and SINGLE/DOUBLE/
TRIPLE/AROMATIC each receive 0.125. The diagonal stays NO_BOND. Expected
degree is `(N-1)*3/N=2.5`, not exactly three. Use the shared Mosaic initializer
with zero atom/bond noise and equal element weights; add no hidden jitter.
All rows of each type are identical. One control suffices: three copies
with identical initialization, frozen context and RNG would be duplicates.

Pivotal context choice: authenticate and reuse the passed BRD4 pilot's exact
prepared context/model/key. The six-carbon hard fixture is only the native
slot/feature preparation scaffold. Every optimized element and edge is
replaced by the new uniform probabilities; no hard chain edge is retained.
Native fixed slot/atom-name metadata stays identical to the baseline, so
uniform graph probabilities do not assert a permutation-symmetric model.

Initial raw argmax has six carbon labels and no bonds: ties among elements
choose the first declared category, and NO_BOND is initially largest.
Positive bond-order categories start tied, and the existence-only model
cannot distinguish their order. Preserve these limitations and show every
raw output, including invalid/disconnected graphs, without repair/filtering.
The report must label the uniform start correctly and show probabilities;
the context scaffold must not be described as the actual initial graph.

Pivotal execution choice: reuse the already passed native BRD4 target/model/
shape pilot and its full conservative three-case artifact bound for this
single control. CPU readiness verifies the new initializer and explicit
initialization-only replay policy, unchanged `src/` bytes and all reference
hashes. Reserve the full bound twice on home for the original pending study
plus this control, with the same 4 GiB reserve. This requires about 10.65 GiB;
the early CPU gate asks for 11 GiB. No existing job or evidence is altered.
The actual uniform graph must pass all native initial derivative/null/parity
checks before any updates are accepted. A failed gate stays failed; no noise,
tolerance change or favorable replacement initialization is introduced.

The original study remains native job 29736629 and report 29736630. New
CPU/GPU/report IDs, immutable identities and actual outcomes belong in the
experiment ledger. A requested 5000 updates is not a completed trajectory
or evidence that a chemically meaningful hard molecule has emerged.

## Actual readiness and submission

CPU **29736713 DONE**, allocated hpc two slots, n-62-31-16: **158 passed**,
three native model items deselected, two existing warnings, 48.91s. Exact
uniform rows, zero hidden seed noise, old mixture regression and explicit
replay restrictions passed. The native `src/` byte map matches the measured
BRD4 source 02. Resource assessment passed with **10.648314139 GiB** required
for both outstanding studies versus 25 GiB conservatively available on home.
This readiness does not claim native gradient validation of the uniform graph.

Uniform source 01 manifest:
`cefff19ac9b68d4b546427565b1036fc625c0cde42f22a5157116c930d8252fd`;
archive SHA `058c9dd851e33657897ed3f8c442e94cde6f3c4697c38518a98eab4810dd098c`.
Final config SHA
`898d4bda519ebe53ce40881a5bf0ff66c53f81788ab1679111f990a201161836`;
readiness assessment SHA
`58aa2f68a88a13fa5d21d6d504808850a51e864651f4bff2e82cea793fca2fa1`.

Native **29736735**, gpua10, one exclusive GPU, four CPU slots/32GB total
host request, 16-hour bound, dependency `done(29736713)`. Run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-uniform-5000-01`.
The launcher verifies source/config/pilot/assessment and actual quota again,
then explicitly authorizes initialization replacement plus steps 3→5000
when loading the authenticated old pilot context. It starts from the uniform
graph, not the pilot endpoint. Native derivative/null/parity/descent and
frozen identity checks remain mandatory.

Report **29736736**, allocated CPU one slot/4GB, 90-minute bound, dependency
`ended(29736735)`; output
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-uniform-5000-report-01/payload/report/`.
It requires a complete passed study, labels the uniform start and actual
0.03 rate, and renders saved raw graphs without model reruns. Native updates,
final report, download verification and browser QA remain pending. All
actual receipts and failures belong in the ledger.
