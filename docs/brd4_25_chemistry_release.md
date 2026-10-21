# Approved 25-atom Boltz-2 plus repair submission

21 October 2026. The user accepted the measured repair overhead and explicitly
requested submission. This releases the earlier runtime-review hold. All original
native, derivative, parity, finite-value, frozen-identity, runtime and quota gates
remain required. Temperature sharpening was discussed, not implemented.

The user also explicitly confirmed that gradients must **not** flow through
diffusion back to the molecule. That is the existing implementation: native pose
prediction runs once during preparation; saved coordinates are detached and the
affinity heads receive `coordinates.detach()`. Optimization differentiates the
affinity trunk/input features and existing repair loss with respect to atom/bond
logits, at the fixed pose. Protenix's separate path uses trunk/distogram contact,
without diffusion. These are partial derivatives, not a full denoising gradient.
No model weights, pose, temperature or objective were changed for this approval.

## Scientific configuration retained

Scientific source04 is unchanged: 461 files at
`/work3/s234818/mosaic/releases/20261021-brd4-25-04/source`, manifest
`e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79`.
Its actual scientific CPU347-test pass and all original proofs remain required.
The full127-residue BRD4-BD1 target, 25 occupied neutral slots, 50% chain/uniform
initialization, LR0.03, key17, ten-trial best_primal and 10,000 total updates are
unchanged. Affinity coefficient1; composition0; existing repair coefficient0.001
through applied5000, geometric increase to50 at10000. T1, zero recycles and no
MW correction. Each arm is independent; raw invalid output remains visible.

## Launcher recovery and allocated contracts

Preparation29769178 actually DONE10:30:48Paris. Its old callback29769179 saw a
stale RUN from the scheduler cache immediately after its ended dependency fired,
and stopped before submitting either Boltz validator. The original captured RUN
and failed callback remain preserved. The new controller waits up to seven
queries ten seconds apart for an **actual** final state and LSF completion;
it never promotes a dependency, timeout, missing record or RUN to DONE.

Launcher-only source07 authenticated five files at
`/work3/s234818/mosaic/releases/20261021-brd4-25-release-07/source`, manifest
`80274d8d5f066af3e364f0356182ab0aa124302613800aa8b68fe3ed3cdec964`;
archive10,105bytes SHA
`61d23c4d5f6c7e34b59cdfec60947c1a001c838994fd175d5dc07a5a44c8127b`.
It authenticates source04 scientific CPU and source07 launcher CPU independently,
retains the existing prepared context and executes the original source04 driver.
The new launcher permits the user-approved chemistry arm, accounts for all ready
arms in shared quota plus4GiB reserve, and retains exclusive per-arm submission
markers/unknown-outcome protection. Future long jobs request gpul40s, one exclusive
L40S GPU, four CPU slots,8GiB/slot,23h, without inter-arm dependencies.

Tools CPU29769499 actually DONE10:54:45Paris: **24 passed**, no failures/errors/
skips. Raw XML and all byte proofs were authenticated. Bootstrap29769500 actually
DONE10:55:02 and authenticated actual preparation success before issuing jobs:

| Actual job | Scope | Latest observed status |
| --- | --- | --- |
| 29769502 | Boltz-only native validation | PEND gpul40s |
| 29769503 | Boltz-only readiness callback | PEND hpc |
| 29769504 | Boltz plus repair native validation and both pilots | PEND gpul40s |
| 29769505 | Chemistry readiness callback and conditional submission | PEND hpc |

Pending reason is physical GPU availability. The full10,000-update chemistry job
has **not yet been submitted**; only a genuinely passed readiness/fit/quota may
produce its actual later ID in `long-submissions.json` and
`run-boltz2_chemistry/submission.json`. User approval alone is not a native pass.

The obsolete Protenix callback29769177 was verified PEND/unstarted and intentionally
cancelled, actual EXIT10:50:49, then replaced by29769501. Protenix validation
29769176 itself ran unchanged10:48:00–10:51:47 and EXIT at the original hard-input
parity threshold2e-5 (maximum reported discrepancy~1.564e-4). This remains a failed
Protenix readiness and prevents its long submission. It does not block the two
independent Boltz validations. No threshold was relaxed or failed arm omitted.

Source05 tools were sealed/uploaded but not staged after static detection of an
old callback-directory collision. Source06 staged five files, but Bash rejected
Windows CRLF before any job/test submission. Both releases are preserved; they
are launcher preparation failures, not model failures. Source07 uses LF wrappers,
passed Bash syntax and the allocated24 contracts. Scientific source04 never changed.

## Evidence and follow-up

All56 CPU/controller/XML/source/preparation/cancellation/Protenix-failure artifacts
were safely downloaded and byte-verified at
`results/hpc/20261021-brd4-25/release07/evidence/`; archive153,209bytes SHA
`055c6df3db215eed088da083f07c5ddfbf4473b1527ffc01892b062bed49aded`, manifest
`fe7ffeb2cb3e1e4d35d3fdc9450e8c0cdc3cb4a22458b694bf121f95e64e689d`.
Receipt: `scripts/hpc/receipts/20261021-brd4-25-release-07/campaign.json`.
No local numerical test/model/chemistry work ran. The existing heartbeat follows
actual native results and conditional full chemistry submission, quiet on unchanged
queue state. Historical six-slot studies, failed random gradients, reports/servers
and unrelated user PDF/OneNote changes remain unchanged.
