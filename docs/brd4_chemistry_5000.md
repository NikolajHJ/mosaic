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
- [x] Pass allocated CPU readiness: 29753148, 253 selected tests passed.
- [x] Pass the bounded native GPU pilot within allocation 29753152.
- [x] Submit the gated 5000-update run: GPU 29753152, report 29753153.
- [x] Complete native 5000-update study and verify terminal model/context identities.
- [x] Download, authenticate and browser-check the saved-only report.
- [x] Finish full downloaded evidence-archive verification: 15,995/15,995 files matched.

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
FP32 and promoted FP16 math have separate fixtures. Allocated CPU results
are recorded below; they do not establish native model validation.

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
At initial preparation, no new job IDs or numerical outcomes existed.

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

Access interruption before release-04 validation: one SSH attempt timed
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

An earlier user-requested connection retry (19 October): the SSH server responded,
but authentication failed with `s234818@login1.hpc.dtu.dk: Permission denied
(password)`. The metadata command did not execute. No fresh quota/scheduler
observation, release-04 upload, corrected test run or native/report submission
occurred. Keep the sealed corrected source and passed historical legacy proof;
chemistry readiness remains pending restored SSH authentication. One connection
attempt was made for this follow-up, with no repeated authentication attempts.

## Corrected validation passed; study submitted — 19 October

Access is restored. CPU readiness **29753148** completed successfully on
`n-62-30-25` in 170 scheduler seconds: **208 legacy tests passed** (64.97 s),
**45 chemistry/report tests passed** (87.76 s), zero failures/errors/skips.
Three legacy native-model tests were deselected by the existing CPU marker
selection. All twelve original-source value/VJP fixtures generated, and the
new JAX suite passed their values, gradients, directional differences,
normalization, fixed-target identities and report fixtures. This establishes
allocated CPU port/engineering evidence, not native Protenix validation.

The training-only composition export passed for 1,688,539 authenticated
training molecules. It records the full source counts and excludes 54,376 Br
atoms when conditioning onto C/N/O/S/F/Cl; P and I excluded counts are zero.
Reference SHA-256:
`9135e9985dfd6c6d6afd96726ae1d4c1a5edc6a4966127475a1fec888c849bdc`.
CPU assessment SHA-256:
`dd4afb5f403c664997f8c8f9058c0d46c48ea3870800ea9fba79faa948f41d22`.
Source/configuration remain the sealed release-04 identities above.

After actual CPU DONE and assessment authentication, submitted **29753152**
to `gpua10` at 01:55:49 Europe/Paris: one exclusive A10, four CPU slots,
8 GiB/slot (32 GiB total), 23 hours, dependency `done(29753148)`. Its initial
state is **PEND**, waiting for scheduling. Native gradient/null/parity and
invalid-F chemistry checks, frozen model/context identities, the disposable
ten-update pilot and runtime-fit gate must pass before the fresh 5000 updates.
N=6, seed 17, 50% chain/uniform law, LR .03 and all loss weights are unchanged.
Run: `/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-04-native`.

Report/evidence **29753153** is submitted to `hpc`, two CPU slots, one hour,
with dependency `ended(29753152)`. It preserves failed/partial evidence and
cannot waive identity gates. Run:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-04-report`.
Neither native scientific validation, optimization completion nor a new
rendered report is claimed from these submissions.

Downloaded CPU proof bundle: `results/hpc/20261019-brd4-chemistry/readiness-04/`.
Archive 36,663 bytes, SHA-256
`7bf1a097f04338ccbc2c11ce4c6c07e28ef2611481a3afadad604efb067f8869`.
Local stdlib byte/schema checks authenticated 24 files, assessment/source/config,
XML/oracle/reference proofs, CPU success and native/report submission receipts.
No local numerical execution or scientific rendering occurred. Failed readiness
03 was also downloaded/verified and remains visible beside failed readiness 01.

## Native progress — 19 October, 08:36 Europe/Paris

GPU 29753152 started at 07:12:54 Europe/Paris on `n-62-18-5` and is RUN.
At 08:36:46, the saved update log contains **4544 contiguous completed
updates out of 5000**. Native contact checks and the coupled invalid-F
chemistry/combined-gradient probe passed; its chemistry gradient norm is
0.0925925821. The disposable ten-update pilot passed with ten strict
improvements, no fallbacks/no-ops and mean update time 0.95134 seconds.
The source-bound frozen-identity and runtime-fit checks passed before the
requested fresh trajectory began; terminal identities remain pending.

Recent saved timings average 0.95874 s/update over 100 updates and 0.99246
over 500. The remaining 456 updates project to 437–453 seconds, approximately
08:44 Europe/Paris; quote about 08:45 allowing ordinary overhead. This is
a pace-based estimate, not completed-budget evidence. Latest saved contact
loss is -0.3821232319. Report 29753153 remains PEND on the native dependency;
its CPU scheduling/rendering/packaging time is additional. Neither job was
changed or restarted by this status inspection.

## Native complete; report packaging recovery — 19 October

Native **29753152 DONE**, 5000 applied/strict-improving updates, zero fallback
or no-op updates, 5001 saved states. Persisted native and case statuses are
passed; frozen model/context before/after digests match their original pins.
Scheduler runtime is 1:32:04 (driver elapsed 5503.295 s). Final contact loss
is -0.3851259649 and the saved raw argmax readout is connected sanitized
`CCCCCC`. Do not interpret this one hard readout as a validity guarantee for
the relaxed probability law or causal evidence for the chemistry term.

Report **29753153 EXIT** is an engineering packaging failure, not a native
failure. Rendering completed with exit 0 and a complete 5001-state receipt;
HTML SHA `01ff604c69b6a80374a3447c68f4976c6da061b69b8aae9ef72f9cc191aec3cf`
(18,169,151 bytes). Packaging rejected pytest's internal `current` alias under
the explicitly named `pytest-unit-tmp-legacy` root, because the collector
recognized only `pytest-unit-tmp`.

Report-only recovery recognizes exactly the old/legacy/new declared pytest
basetemps, records internal directory aliases without following them, and
continues rejecting escaping, cross-basetemp, file, unknown-root and other
symlinks. Add allocated regression tests for those boundaries. A new sealed
source/run copies the already authenticated report bytes and packages the
complete native/readiness evidence plus the original failed report attempt.
No native rerun, new chemistry evaluation, repair or scientific rendering is
needed. Native summary SHA:
`c110b4e0bcd716679102022ad51d46590a658f8f3f9471b7b91bfbc1166d85fa`.

## Report-only recovery and presentation

Recovery **29753409 DONE** on `n-62-30-25`, 08:59:42–09:01:13 Europe/Paris.
All **13 allocated packaging/report tests passed**, no failures/errors/skips
(5.78 s). Native production/example/config byte maps match source 04 exactly.
Recovery source 05: 351 files; manifest
`14e59d968eb9f342a0da68b9fac31b44a2702bcf564780fa7db199123147b069`;
source archive `8d8e2ebc7b8a34c1b14651763cc25a6725c55fc5375462223f6cad598a9d2d89`.
The recovery authenticated complete terminal native gates and copied the
original report bytes; original render job 29753153 remains recorded.

Evidence archive: 249,546,618 bytes, 15,995 files, SHA-256
`c0309f186da134d48f69bc6fd8594535325934f4904def3aaad377b2452d0c89`.
Remote path:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-05-repackage.tar.gz`.
It retains the original report packaging failure and full native/readiness
evidence. Official packaging quota gate passed with 19 GiB available versus
4.4112 GiB required. Final LSF scheduler records and stdout authenticate native
DONE at 08:44:58, original report EXIT at 08:45:34, and recovery DONE at 09:01:13.

The separately downloaded report's four artifact hashes and input receipt
passed local stdlib checks. Viewer: `http://127.0.0.1:8772/report.html`, backed
by `results/hpc/20261019-brd4-chemistry/report-view-01/report/`. Browser inspection
confirmed the completed-study banner, initial/final/midpoint controls, playback,
readable C1–C6 labels, probability-dependent sizes and exact-probability heatmaps.
The tab is retained as a deliverable at update 5000. No scientific rendering or
chemistry evaluation occurred locally.

Saved evidence shows **zero repair penalty in all 5001 states** and only
`CCCCCC` among all saved hard readouts; all 5001 saved readouts sanitized.
Probabilities changed and contact improved from -0.2778552473 to -0.3851259649,
but repair guidance never activated along this trajectory. This is not an
active-repair-effect demonstration. The separate invalid-F native probe did
establish nonzero chemistry gradient and combined directional agreement.

## Final local evidence verification

The full archive's exact remote path, byte count and SHA matched its receipt.
`verify_hpc_evidence.py` authenticated the archive and completed safe extraction;
its slow serial Windows per-file pass was stopped after extraction. The
versioned post-extraction operator below reauthenticated the archive, all safe
member names and the exact manifest bytes, then checked **15,995/15,995 files**
with eight bounded I/O workers in 83 seconds. No missing files, size/path
differences or checksum mismatches. No scientific work was executed locally.

```powershell
py -3.13 scripts/hpc/receipts/20261019-brd4-chemistry/verify-complete-report-download.py results/hpc/20261019-brd4-chemistry/complete-report-01
```

Manifest SHA-256:
`0ecf15951cb5bf7c61ee6706445f73270a5431700afb9bd510ad61d7537797c1`.
Final byte receipt: `complete-report-01/20261019-brd4-chemistry-05-repackage-download.json`.
`study-verification.json` authenticates all 5001 state/snapshot receipts,
5000 updates, N=6, native/config/reference identities, original input IDs and
the successful 13-test recovery proof. The recovered archive's HTML is byte
identical to the independently verified, browser-checked viewer copy.
Report delivery is complete at `http://127.0.0.1:8772/report.html`; the same
self-contained file also lives under `report-view-01/report/report.html`.
