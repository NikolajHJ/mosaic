# Section 4.2 diffuse initialization, six slots, BRD4 5000 updates

Requested 2026-10-16. Read the complete section 4.2 across PDF pages 3-4,
including its unordered-pair/noise/projection rules. Source PDF SHA-256:
`c9785f45dcd34fbc3feaebcc0bc57905dc606387f76308bb6364eadb40cbeda0`.
The user subsequently explicitly requested keeping the current atom count.
**Retain N=6**; do not prepare a 25-slot context. The proposed 25-slot default
in the PDF is superseded for this experiment only. Submission/outcome receipts
below distinguish implementation and actual completed native checks.

## Pivotal decisions and resolved recipe

Use one canonical diffuse case, not new seeds or a parameter sweep. Preserve
the exact six-slot BRD4 fixture, full deposited 127-residue sequence, frozen
Protenix-Tiny checkpoint, neutral charge, geometry-off native context, seed 17,
highest precision, whole-target native contact loss, 0.03 initial step and
5000 accepted updates. Keep all native gradient/null/parity/frozen/descent
checks and backtracking tolerances unchanged. No chemistry or affinity loss.

Section 4.2 specifies broad carbon-heavy atom probabilities and small
independent probability-space perturbations, without exact carbon/noise
coefficients. Choose and seal these values before testing:

- Atom prior: C=0.75; N/O/S/F/Cl each 0.05.
- Nominal expected degree d0=2.25, the midpoint of the proposed 2.0-2.5 range.
- Pair existence p0=d0/(N-1)=2.25/5=0.45. Baseline NO_BOND=0.55;
  SINGLE/DOUBLE/TRIPLE/AROMATIC each 0.1125. Total baseline bond mass=6.75.
- Independent Gaussian noise sigmaX=sigmaE=0.005, added to each prior row
  BEFORE row-wise Euclidean simplex projection. Draw independent subkeys for
  X and upper-triangle E from seed 17; no clipping, logits, noise-first convex
  mixture, hidden jitter or favorable-seed search.
- One vector per unordered pair; mirror on native input construction and
  keep diagonal NO_BOND. Charge and occupancy remain fixed.

This uses the existing shared `initialize_graph`; the missing part was exposing
its recipe in the actual Protenix experiment/config/report path. The native
graph/model library stays unchanged. Small noise breaks equal-row symmetry
without imposing strong hard-chain edges. Every pair starts from the same
existence prior regardless of its position in the native preparation scaffold.
The native importer still needs that scaffold for fixed atom slots/names;
all optimized X/E entries are overwritten and geometry remains disabled.
Fixed native metadata means this is not a claim of complete permutation symmetry.

The 0.08-0.10 pair probabilities in the PDF follow from N=25, not an independent
constant to copy into a six-slot run. Using d0/(N-1) retains the requested
degree scale with N=6. Finite-sample noise/projection can change the realized
degree/bond mass; saved graph probabilities are authoritative. This does not
establish size-scaling behavior at N=25, chemical validity or Stage 1 acceptance.
Initial raw readout is expected to be carbon-labelled, disconnected atoms;
readouts remain exactly as output, with no repairs or fragment filtering.

## Readiness and execution

Tests compare the actual recipe with an independent NumPy simplex reference,
check seed reproducibility and independent rows, six-slot occupancy, symmetric
bonds/absent self-bonds, baseline degree scale, rejected parameters and replay
permissions. The declared seed must permit the unchanged 0.01 central probes
and 0.005 bond-order null move. A failed native check remains failed.
All numerical checks stay on allocated DTU CPU/GPU workers; no local model
or numerical test is authorized by this preparation.

Authenticate the old passed BRD4 pilot/context and exact unchanged `src/`
byte map. Resource readiness reuses the full original THREE-case bound for
this ONE case plus each of the three outstanding studies, and 3 GiB extra
for the combined comparison report/archive. Required headroom is about
20.2966 GiB including the 4 GiB reserve; early CPU gate asks 21 GiB. The native
launcher rechecks the bound before starting. No existing GPU run changes.

Submit a separate canonical native job after CPU/source/config/quota gates.
Actual native checks on this initialization run at GPU entry before accepting
any update. Replace only the still-pending three-study combined report job
with a new report-only identity waiting on all four native studies, retaining
its intentional cancellation receipt. Keep the three native runs and their
individual report jobs untouched. Update the existing completion heartbeat
to download and verify the final four-study/six-trajectory report. No failed
study may be omitted to force a complete comparison.


Canonical diffuse source 01: 271 files, archive 38,989,767 bytes, SHA
`f6efb3c2c8889e6f792482cd1d666c38631afe9b2bf78bcda5ba942e81f5e4c9`,
manifest `421213580f84a038cde5e47bb4048a659bfa4daee46645deccbb2e7568d7ead4`.
Git reference 48a287cb2743879fa00bff32ab5d339730b46b08 plus manifested dirty
bytes; remote immutable
`/work3/s234818/mosaic/releases/20261016-brd4-canonical-diffuse-01/source`.
Local `results/releases/brd4-canonical-diffuse-20261016-01/`.
Uploaded archive SHA and native/readiness/combined-report Bash syntax checks
passed. Local static Ruff and Node `--check` passed. PDF text extraction and
page-image inspection are document reading, not local numerical validation.

CPU readiness **29736979 submitted**, hpc two slots/4GB per slot (8GB total),
15-minute bound, run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-canonical-diffuse-readiness-01`.
Exact operator `scripts/hpc/receipts/20261016-brd4-canonical-diffuse/submit-readiness-01.sh`;
local campaign `results/hpc/20261016-brd4-canonical-diffuse/`.
It requires the selected small-molecule suite, the unchanged native-library
byte map, initialization-only config/fixture/reference gates, four study
budgets and 3 GiB comparison reserve. Early quota asks 21 GiB. Numerical and
native submission outcomes are pending; this is a submission receipt only.


CPU readiness **29736979 DONE**, worker n-62-30-25: **186 passed, three
native-model items deselected, two existing dtype warnings, 76.67s**. This
includes all 10 canonical recipe/guard tests, old initialization regressions,
report rendering and four-study/six-case recipe mapping. Resource/source,
config/fixture/reference gates passed. Aggregate required free home space
**20.296628277748823 GiB**, conservative availability **25 GiB**. Config SHA
`49e01e601f77e21770d19b4a40df365315e4521318245d14ccec6e7cdbe6e359`;
assessment SHA `4ad7e7f04c740815d248ee405f3979f4ba648f2cdaca872d48d290a387c53821`.
Local unit/assessment/quota/final scheduler receipts were downloaded to the
canonical campaign directory. This establishes CPU readiness only; native
canonical gradient and 5000-update acceptance remain pending.

Canonical native **29736988 submitted**, gpua10, one exclusive A10, four
slots/8GB per slot (32GB total host memory), 16-hour bound, dependency
`done(29736979)`. Run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-canonical-diffuse-5000-01`.
Exact tested source 01; source/pilot/assessment/config/quota are authenticated
again at launch. Reuse the exact previous six-slot frozen context, replacing
only initialization and the previous three-update pilot budget with 5000.
No 25-slot context or new model/target/loss was introduced. Actual native
FD/null/parity/frozen/descent checks run before accepting updates.

Final combined report/evidence **29736989 submitted**, hpc two slots/4GB per
slot (8GB total), 180-minute bound. Dependencies: CPU readiness done and all
four native jobs ended (29736629, 29736735, 29736761, 29736988). Run
`/zhome/95/e/206121/mosaic-runs/20261016-brd4-combined-report-02`;
final output `payload/report/`; archive is its sibling
`20261016-brd4-combined-report-02.tar.gz` with `.tar.receipt.json` receipt.
It requires four complete passed studies, renders all six 5001-state cases
with six distinct colors, and archives all four native runs plus the report.
Exact operator
`scripts/hpc/receipts/20261016-brd4-canonical-diffuse/submit-native-and-comparison-01.sh`.

Superseded combined report **29736911 cancellation requested while PEND**
after the four-study replacement was successfully submitted. Normal owner
`bkill` acknowledged termination; polling confirmed actual **EXIT** before
completion of the operator. No allocation had started and no model work was
cancelled. This intentional report replacement is not a numerical failure.
Receipt `superseded-report-29736911.txt` and final compact scheduler snapshot
are in the canonical local campaign directory. Original three native and
individual-report pairs remain unchanged and PEND; new native/report are
also PEND. No accepted canonical updates or final report/download are claimed.

Updated the existing active hourly heartbeat
`finish-combined-brd4-trajectory-report` (same name/schedule, no duplicate).
It now checks current native/report IDs, downloads/verifies report-02,
checks four source studies and six 5001-state six-slot cases, and ignores the
intentional superseded report cancellation. Browser QA and delivery remain
future work after actual completion; failed native gates are not waived.
