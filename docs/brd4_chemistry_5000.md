# BRD4 contact plus failure-aware chemistry — 19 October 2026

The user requested integrating the DBMol failure-aware, atom-preserving repair
objective and a fresh **50% chain/uniform initialization**. This is a separate
study; the six completed historical trajectories and their reports stay sealed.

## Decisions

- Keep six occupied, neutral slots; the actual Mosaic element order is
  **C, N, O, S, F, Cl**, with NO_BOND/SINGLE/DOUBLE/TRIPLE/AROMATIC unordered
  bonds. The source's NO_ATOM/charge schema is mapped explicitly. No new atom
  count, protein pocket, learned prior, scorer or model training is introduced.
  The fixed-support adapter has a nontrainable NO_ATOM category with zero
  FP32 mass and a one-category neutral charge law. Complete-target NLL keeps
  **6 + 6 + 15 = 27 decisions**, including six fixed charge decisions with
  zero NLL. Generic inputs retain padding and their full charge vocabulary.
- Reuse authenticated BRD4 3MXF context, frozen Protenix-Tiny, seed 17, highest
  FP32 matmul precision, LR 0.03 and **5000 total graph updates**. Start fresh
  from the same 50% carbon-chain/uniform probability law.
- **Pivotal parameterization decision:** optimize categorical logits, exposing
  their softmax as the one shared relaxed graph. The historical Euclidean
  projected-probability optimizer can create exact zeros; clipping those for
  target NLL would violate the supplied unfloored log-softmax requirement.
  This changes optimization geometry and makes the historical contact-only
  comparison exploratory, rather than an isolated chemistry-loss ablation.
- Use `contact + 0.5 * failure_aware_chemistry + 0.1 * element_JS` at constant
  weights. Internal new-loss weights are 1/1/0.1/0.1/0.1 as supplied. These
  research coefficients are declared, not claimed to be calibrated. The
  empirical element reference comes from the authenticated DBMol training
  split only, conditioned on Mosaic's supported elements; retain discarded
  counts. Fixed occupancy and neutral charge make empty/size/charge guards
  constant in this adapter; the reusable chemistry API retains the full terms.
  VAE reconstruction/KL, real/prior branch mixing and epoch ramps do not apply
  to this single-molecule optimization. No decoder is trained.
- Original bond-only teacher: depth 3, 32 checks/depth, beam 4, regret weight 1,
  detached ranking floor 1e-12. On censored/unsupported rows, start a separate
  depth-6 search from the same raw graph with the same 32/4/1 settings and atom
  substitutions disabled. This is not a nine-edit certificate.
- Freeze target, partial witness, caps, components, composition reference and
  coefficients throughout each state's ten halving trials. Refresh explicitly
  at the next accepted state. Strict actual decrease is attempted first;
  otherwise retain the finite trial with lowest independent-primal combined
  loss, including an increase. Nonfinite/native/identity failures still stop.
- Record fixed-preparation before/after totals separately from the refreshed
  next-state surrogate, whose repair target may differ. Keep contact and each
  chemistry component, reference identities, coverage/timing, fallback/no-op
  counts and raw output graph. Never rank changing surrogate totals as one
  fixed objective or send a repaired target to Protenix.

## Port and validation

Detached search/native conversion helpers are extracted from the supplied
source with a manifest of original file hashes and selected symbols. They use
NumPy/RDKit and no Torch, VAE, Boltz or DeFoG runtime. Live differentiable terms
are implemented in JAX and must pass allocated parity/finite-difference tests.
The exact versioned allowed-valence table is retained; the older expected-
valence-excess objective is excluded.

The source's latest docs/ledger record CPU 29752937: 790 passed, seven CUDA
skips, with native model adoption still pending. Those results do not validate
this JAX port. Required Mosaic checks include full NLL normalization, partial
and fallback routing, exact search constraints/deeper coverage, provenance,
composition statistics, all component gradients, frozen-reference directional
differences, native contact FD/order-null/parity and model/context immutability.
All numerical tests, empirical-reference preprocessing, pilots, optimization
and report rendering remain on allocated DTU HPC.

## Status

- [x] Implement reusable port, detached preparation and logit optimizer.
- [x] Integrate fresh 50% run, recording and interactive report support.
- [x] Author allocated parity, gradient, identity and trajectory fixtures.
- [x] Seal current source/configuration locally and prepare unique HPC launchers.
- [ ] Pass allocated CPU readiness and bounded native GPU pilot.
- [ ] Submit the gated 5000-update run and record actual job IDs.
- [ ] Verify completed evidence and deliver the saved-only report.

Initial access attempt: `ssh -o BatchMode=yes -o ConnectTimeout=15 dtu-hpc`
reached the server but authentication returned **Permission denied (password)**.
No remote command, quota check, upload, numerical test or submission ran.
Continue local editing/static review/packaging; do not retry this unchanged
outage until the user reports restored access. No local numerical validation.

## Module boundaries and evidence

| File under `src/mosaic/small_molecule/chemistry/` | Responsibility |
| --- | --- |
| `state.py` | Batched logits, fixed-support Mosaic adapter and schema ordering |
| `_reference/` | Pinned NumPy/RDKit search/conversion and exact rule table |
| `preparation.py` | Original snapshot, separate deeper search, frozen targets/caps/cuts and provenance |
| `loss.py` | JAX full/partial NLL, local/cut/nonempty risks and aggregate JS |
| `trajectory.py` | Native gradient plus exact softmax pullback, ten-trial selection, refresh and recording |
| `checks.py` | Frozen-reference directional gates and per-probe failure evidence |

`scripts/vendor_repair_reference.py` extracts verbatim source symbols and
records original/extracted hashes. Eleven full original Torch files are captured
under `tests/data/chemistry_oracle/` solely for independent numerical oracles.
Their old auxiliary objectives are not invoked. Production includes only the
versioned table from the old regularization module. No Torch installation is
needed in the resident Tiny environment. Captured working-source bytes identify
the reference; a historical CPU pass does not validate this new port.
Git preserves snapshot bytes across Windows/Linux checkouts; Ruff excludes the
two byte-pinned reference trees from automatic formatting.

`scripts/create_chemistry_oracles.py` authenticates those originals and emits
twelve value/VJP fixtures in the existing allocated Torch environment. The JAX
suite covers all six terms, normalization/padding, extreme-logit gradients,
local risk's absent atom/charge gradient, genuine depth-four coverage, identity
rejection, native softmax pullback, no-op fallbacks and nonfinite failures.
FP32 and promoted FP16 math have separate fixtures. None has run for this port.

The new report entry point is `scripts/render_chemistry_trajectory.py`, with
`scripts/chemistry_report_template.html`. It reads captured probabilities only,
embeds plots/data offline and draws one selected raw frame on demand, with
element/slot labels, probability sizes, heatmaps and playback. The historical
50% scalar contact curve is authenticated by summary SHA; missing historical
chemistry terms/probabilities are never invented. Old molecule playback remains
in the old report. Refreshed surrogates, within-state deltas and raw native/schema
validity are distinct from detached valid/partial witnesses.

## Prepared execution sequence

1. After access is restored, make a fresh compact official quota/queue query.
   Require room for a new source/run and the 8 GiB home readiness/entry reserve;
   do not delete other campaigns to make room.
2. Transfer to a new release directory, authenticate archive/manifest hashes,
   then invoke `scripts/hpc/submit_chemistry_readiness.sh` with release base,
   manifest SHA, archive SHA and a unique campaign tag.
3. The four-slot CPU job runs legacy tests in FP32, original Torch oracles,
   JAX parity/gradient/report fixtures in FP64, and training-only composition
   preprocessing. Assessment binds source/config, XML proofs and reference bytes.
4. Only after actual CPU DONE and assessment authentication, invoke
   `scripts/hpc/submit_chemistry_native.sh` with CPU job ID and assessment SHA.
   One exclusive A10/four CPU slots/32 GB host RAM requests at most 23 h. Native
   contact FD/null gates, a separate invalid internal-F native-input assay
   requiring positive chemistry loss/nonzero chemistry VJP and combined FD,
   and a disposable ten-update combined pilot precede
   identity/parity/finiteness/runtime-fit gates. Reset to the same initialization
   and key for the fresh 5000 updates; pilot updates never become a prefix.
5. A separate CPU report/evidence job waits for the native job to end. Preserve
   failures, label partial data and never waive native identity gates. Confirm
   scheduler/persisted outcomes before download and integrity checks.

Wrappers: `scripts/hpc/brd4_chemistry_readiness.lsf`, `brd4_chemistry.lsf`,
`brd4_chemistry_report.lsf`. Config: `examples/configs/brd4_chemistry_5000.json`.
Every retry needs a new immutable release/run, including engineering fixes.
No new job IDs or numerical outcomes exist yet.

Static review passed: 35 Python AST parses, eight extracted-source and eleven
original-source byte checks, handwritten-code Ruff check/format, JavaScript
syntax and five shell syntax checks. No numerical fixture, model or scientific
renderer ran locally. The test-only oracle snapshot is independent from the
production JAX implementation; its future pass remains an allocated gate.

Local sealed release: `results/releases/brd4-chemistry-20261019-01/`;
350 files, archive 39,159,288 bytes. Source commit reference `933d45d`.
Archive SHA-256 `9d10271ab846d150bee1d47a13ac9892fa699f91a459f593ecd4ec35d08eac3f`;
manifest `1a8ead22c031b6d525f93072b6752598d6cbc00b0d42fdfb1f2cd92804ffc2cb`.
Independent stdlib archive/member verification passed. No transfer occurred.
These post-packaging receipt notes do not modify the sealed source archive.

Planned remote release base:
`/work3/s234818/mosaic/releases/20261019-brd4-chemistry-01`;
campaign tag `20261019-brd4-chemistry-01`. Transfer the readiness bootstrap
operator alongside the archive; authenticate its SHA before executing it,
because it creates/extracts the source directory. Bootstrap SHA-256:
`af8be632de5e3383acfe231e33e771f62bbc8bd55e7f8b4389a4a16309d0f4d1`.
The native submission operator is then available inside the verified source.

## Restored access and allocated validation — 19 October

SSH authentication now succeeds. Official work quota is 89.77/100 GiB;
cached home quota is 10.71/30 G. The A10 queue is Open:Active and permits
24 hours, so the declared one-GPU/four-slot/23-hour request is supported.
The original sealed release authenticated all 350 files remotely, and CPU
readiness 29753139 started on `n-62-30-25` with a passing 19 GiB available
home reserve. It has not passed: its legacy-phase file glob explicitly
included the new chemistry fixtures despite pytest `--ignore` arguments.
Those fixtures require the later oracle/FP64 phase and produced a failure
and skips. The native run remains unsubmitted.

Corrected only the readiness wrapper's file selection: build a Bash array
excluding the two new modules, then execute the unchanged legacy test
selection. The new modules remain mandatory in their separate FP64 phase
with authenticated original-source oracles. No gate or tolerance is relaxed.
Preserve release/run 01 and its actual outcome; use unique release/run 02
for the corrected launcher. All scientific production/test/config bytes
remain unchanged. Local Bash syntax and Git whitespace checks passed.

Job 29753139 ended EXIT: 216 passed, one failed, 35 skipped, three deselected
in 110.23 seconds. The early roundtrip fixture observed absolute FP32 error
1.1920929e-7, two ULPs near probability .5833, above its decimal 1e-7 bound.
Use an explicit `2 * finfo(dtype).eps` absolute roundoff bound (2.3841858e-7
for FP32) in the initialization replay check, with both FP32/FP64 unit
coverage. This checks log/softmax numerical replay, not exact bitwise
initialization; native FD/parity and chemistry-oracle tolerances are unchanged.
Release 02 contains only the launcher fix and was transferred but never
submitted. Release/run 03 includes this bounded roundoff correction and the
same scientific configuration, production loss and optimizer. Preserve both
earlier sealed archives and the failed job. Ruff and whitespace checks passed.

Readiness 29753140 (release 03) passed all 208 legacy tests, then failed in
the oracle generator before new JAX tests or reference preprocessing. The
original Torch source returns detached constant zeros for inactive terms;
`autograd.grad(..., allow_unused=True)` cannot accept a constant output.
The generator now records an explicit zero VJP and `constant_components`
for those terms, retaining normal autodiff for differentiable terms. No
original-source, JAX implementation, scientific setting or tolerance changes.
Preserve this failed job and use release/run 04 for complete revalidation.

Current execution blocker: after sealing release 04, one SSH attempt timed
out and one bounded retry after 30 seconds disconnected with `Connection
reset`. Both failed in the initial metadata-only SSH stage, before the new
archive SCP or any submission command. The remote release-04 directory or
bootstrap may have been created; inspect before reuse. No release-04 source
archive was uploaded and no new test/native/report job was submitted.
Stop repeated connections during this outage. The completed 208-test legacy
pass remains evidence for release 03, not a chemistry-parity or native pass.

Prepared local release 04: `results/releases/brd4-chemistry-20261019-04/`,
350 files, 39,163,735 bytes; archive SHA-256
`3280845270bc0f647d309b5671a2670e0d49a72c5c60f14a334427d479be5983`,
manifest `368f455800678c7aee020f09c1625fda67e6504e84a355f49b975516453531b6`.
These final receipt notes postdate that immutable local archive. All original
source snapshots, JAX loss/optimizer and configuration bytes remain unchanged.

Latest user-requested connection retry (19 October): the SSH server responded,
but authentication failed with `s234818@login1.hpc.dtu.dk: Permission denied
(password)`. The metadata command did not execute. No fresh quota/scheduler
observation, release-04 upload, corrected test run or native/report submission
occurred. Keep the sealed corrected source and passed historical legacy proof;
chemistry readiness remains pending restored SSH authentication. One connection
attempt was made for this follow-up, with no repeated authentication attempts.
