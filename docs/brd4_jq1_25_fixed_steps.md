# Fresh 25-atom JQ1 runs with fixed steps

21 October 2026. The user explicitly selected BOTH Boltz arms for restart at
fixed LR0.01 and requested no halving for Protenix, whose newly requested nominal
rate is0.3. This supersedes the earlier instruction to leave native jobs29770093
and29770124 unchanged, and supersedes the unstarted LR0.3 backtracking proposal.
Original completed Protenix29770179, all prior results and historical reports
remain immutable. Each replacement starts fresh from the same saved initialization
and targets10000updates; none resumes an old partially completed optimizer.

| Arm | Fixed logit learning rate | Other scientific settings |
| --- | ---: | --- |
| Protenix |0.3|same geometry-off JQ1 pocket contact,key17,N25,50%chain/uniform|
| Boltz-2 |0.01|same detached predicted JQ1-pocket pose,zero recycles,noMWcorrection|
| Boltz-2+chemistry |0.01|same pose; original chemistry.001 through5000, geometricto50at10000|

All have exactly ONE optimization candidate per update: apply z_next=z-lr*g
whenever the authenticated native/combined evaluation is finite and passes parity
and frozen identities. No halving, trial search, clipping, gradient normalization
or optimizer substitution. A loss increase, equality or no-op is logged honestly
and consumes one applied update, without being labelled a search fallback or
strict descent. Best-observed evidence is retained separately; changing chemistry
repair totals are not a single comparable objective. A nonfinite value or failed
native/scientific gate is still a hard failure, never a fabricated full budget.

The explicit fixed_step policy is additive; all original best_primal defaults
and sealed historical studies keep their original behavior. Configuration version
mosaic-brd4-jq1-25-fixed-step-v1 fixes rates by arm and requiresmax_backtracks1.
Actual first/selected rates and the single-attempt count are checked for every
pilot and production update. Protenix's one-step native gradient probe also uses
the requested fixed rate and one candidate; its FD/parity/null/frozen checks stay
unchanged. Finite differences remain.01/.003/.001,15% and two adjacent scales;
those derivative scales are not learning-rate backtracking.

Full allocated CPU regressions plus fixed-step tests cover a finite uphill update,
exact saved displacement, equality/no-op counts, no fallback and nonfinite failure.
Each arm then repeats original native hard parity/rawatom/bondFD/frozen/context/
pocket/no-diffusion checks and a disposable10-update pilot. Chemistry additionally
retains nonzero signal,low/high coupled checks and BOTH disposable pilots. No
new temperature,composition,loss weights or diffusion gradients. All model/
numerical/chemistry work stays allocatedDTUHPC; localAST/byte/schema only.

After all3actualnativeDONE/LSFsuccess/persistedproofpasses, authenticate the sum
of all full artifact projections with fresh finite work3 byte/inode quota and
unchanged4GiBreserve/100000inodeheadroom. GPU entry repeats own projection gate.
Run root `/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-01` keeps new output
off home storage. Same resources: gpul40s,one exclusiveL40S48GB,4CPUslots,
8GiB/slot,2hreadiness/23hproduction; long runs have no inter-arm dependency.
If any check/storage gate fails, preserve evidence and stop rather than silently
changing the declared rate,policy,budget or loss.

## Actual previous-run stops and engineering history

Original Boltz29770093/29770124 were RUN when cancellation was requested. bkill
was issued21October16:13Paris and both actualEXIT/TERM_OWNER were authenticated
at16:14Paris. These are intentional interrupted partial trajectories, not native
instability or completed10000-update results. Original saved prefixes/logs/source
and fixed-pose bytes remain untouched. Original report collectors29770707/08
correctly EXIT because those studies are incomplete; their EXIT is expected for
this user-requested stop. Old final report29770709 is obsolete and will be retired
only after verifying it never started. Stable report edition0006 remains an
immutable historical capture, not a full three-arm completion certificate.

Superseded Protenix LR0.3 source01 CPU29776049 EXIT134passes/one fixture failure
before native execution; callback29776050EXIT correctly blocked dispatch. Its
fixture path assumption is corrected in this new source. No GPU or long job was
submitted for it, and its failed records remain preserved.

Exact release,argv,source/configuration/proof hashes and actual job outcomes:
`scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-01/campaign.json`.

## Actual source01 submission and initial CPU progress

Immutable fixed source01 authenticated585files remotely, manifest
`dc4ac68d1818038ba177241311aecf62072a065ef147356186f1f49eba2d20f3`;
archive39,583,098bytes SHA
`ad7492baed5193abaa632810694c86e2ed29455cb5c6bd136b276cd59eb63e68`.
Actually submitted21October16:20:08Paris: CPU29776083 andafterCPU29776084.
At16:21Paris CPUwasRUN,150contractfixturespassed,broaderregressionsstillrunning;
callbackPEND. No fullCPU/native pass or actual long job is established by this.
Initial finitework3 gatepassed9GiBconservativeavailable/5GiBrequired plusinode
headroom. Actual argv,submissionresponses,source andquotalogsin campaign receipts.

Historical originalBoltz prefixes were byte-hashed through committed states
2407/2208 (updateindices2406/2207). BothactualEXIT/TERM_OWNER afteruserstop.
Oldreportfinal29770709 actualEXIT afterverified-unstarted retirement; oldcollector
EXITs are expected for incomplete user-stopped studies. Historicalreports remain
available; a future report for fixed runs must use new pins/count contracts.

## Source01 CPU failure and new fixture-only source02

CPU29776083 andcallback29776084 actuallyEXIT before any nativeGPU dispatch.
150contractfixturespassed; legacy237passed/twofailed/3deliberatelydeselected.
The uphill toy had no bond derivative and the constant-loss no-op toy had no
atom/bond derivative; mandatory nonzero coupled checks correctly rejected them.
This is test-fixture engineering evidence, not failed native model derivatives.
The new immutable source02 adds a live typed-bond term to the uphill toy so it
passes real derivative gates. The constant-loss bookkeeping unit test explicitly
isolates that gate, with no runtime/production bypass. Production source/rates/
FD/parity/identity/finite/runtime/storage gates remain unchanged. Full allocated
CPU regression and native readiness are required again. Source01/failures remain
sealed. Retryroot `/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-02`;
new receipts `scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-02/campaign.json`.

### Source02 actually submitted

597files authenticated read-only at
`/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-02/source`.
Manifest `1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64`;
archive39,597,594bytes SHA
`c024ac03ddcf6091e35f1a666117194cd70b05024ee4563c6d8c965d75ed5798`.
All productiontrajectory/driver/fixedconfiguration bytes matchsource01; only
fixture corrections anddeclaredoutputroot/sourceevidence differ.
CPU29776103/afterCPU29776104 actuallysubmitted16:24:29Paris. Freshwork3quota
passed8GiBconservativeavailableversus5GiBrequired, with unchanged4GiBreserve and
100000inodeheadroom. At16:26Paris,150contracts and239legacytests passed; native
mapping/chemistry regressions remain pending. No fullCPU/native/longpass claimed.
Source01failureXML/logs downloaded/byte+schema checked in itsfailure-proofs directory.

### Source02 full CPU pass and actual native GPU submissions

CPU29776103 andafterCPU29776104 actuallyDONE/LSFsuccess. All **482 tests passed**:
150contracts,239legacy/bridge (includingfixeduphill/no-op),34native-mapping,
59chemistry; nofailures/errors/skips,3deliberatelydeselectedslow/modellegacychecks.
RawXML,eachproducerproofhash,actualschedulerDONE/hpc/LSFsuccess,source/configuration,
frozenreference/pocket andcertificate authenticated withstdlibbyte/XML/schema
checks only. Twelve compact proof files downloaded under
`results/hpc/20261021-brd4-jq1-25-fixed-02/cpu-proofs/verification.json`.
AssessmentSHA `0e821878a92575c3a02370fc22c2b66df3307accf9d7d1c17917af806eabbad9`;
certificateSHA `a595c100b39816d0f87f5d7d44c7a94c35d9138e6f510a0d284a79f20d5317f2`.

Actuallysubmitted16:28:45â€“47Paris: native Protenix **29776117** /callback29776118,
Boltz-2 **29776119** /callback29776120, chemistry **29776121** /callback29776122.
At16:30:34Paris all6PEND. Actualargv andbjobs resource vectors match gpul40s,
oneexclusiveL40S48GB,4CPU,8GiB/slot,2h; nativejobs have no interarmdependency.
Callbacks waitonlyontheirownnativeended event. Full10,000-update runs have NOT
been submitted; controller still requiresall3fullnativepasses andwholeartifact
projection/freshquota, thenindependent23hlongsubmissions. CPUtests do not certify
nativeGPUgradients or completion. Existingautomation now tracks this new scope.
Code andinitialreceipts pushed634a1e5; no scientific or storagegate weakened.

### 21 October 19:42 Paris â€” all fixed-rate native validations passed; storage blocked launch

All three original GPU readiness assessments and allocated callback decisions now
report passed: Protenix29776117, Boltz-2 29776119 and chemistry29776121. Their
saved actual DONE/gpul40s/LSF-success proofs remain available even where live LSF
records have aged out. Native proof counts are 74/52/128; original fixed rates,
parity/derivative/pilot/runtime/frozen/pocket/no-diffusion gates are unchanged.
Full raw local proof download is pending after a connection reset; the captured
producer assessments, decisions and proof hashes are preserved in the campaign.

No 10,000-update job was submitted. Callback29776122 stopped with exit1 at19:11
after the whole-work3 storage check rejected 8,427,765,472 projected bytes plus
4,294,967,296 reserve against 8,419,460,428 conservative available bytes. This
failed storage receipt stays failed; it is not a native gradient failure.

The user subsequently requested downloading previous results and cleaning storage.
Three existing local historical archives were freshly byte/member verified
(21,093 files), together with noncurrent duplicate source-upload archives. Remote
checksums and dependency exclusions are still required before removing each copy.
Only exact backed-up prior Mosaic files may be pruned; keep current inputs,
proofs, release directories, model caches, environments and other projects.
See `scripts/hpc/receipts/20261021-mosaic-storage-cleanup-01/campaign.json`.
A fallback split-output launcher is prepared locally with AST checks only; no
launcher test, release, new job or automatic adoption is established.

### 21 October, 20:03 Paris â€” verified-backup cleanup completed

Cleanup attempt `attempt-01` completed at `2026-10-21T18:03:02.766380+00:00`.
It removed **21,099 exact backed-up previous Mosaic files**, totaling
**5,459,183,284 logical bytes**. Two changed files were retained and one candidate
was already absent. Both saved unfinished-job inspections found no Mosaic jobs;
the cleanup retained current scientific inputs, proofs, releases, caches,
environments, home artifacts and other projects. No unknown file was removed.

The refreshed storage-pool 6 quota was **86.45 / 100 GiB**, with 296,102 of
20,000,000 chunk files used. This is the recorded post-cleanup snapshot, not a
substitute for the required fresh sum-of-artifacts projection and entry quota
checks. The earlier failed storage decision remains preserved.

The downloaded cleanup audit archive is 4,611,888 bytes, SHA-256
`6db36da0eb9418f475fef04ae3314876db43751fa743ecdd5b0b6225583521e6`.
All 10 manifest files were byte/hash verified locally. The exact completion
receipt, compact audit receipt and manifest are retained in
`scripts/hpc/receipts/20261021-mosaic-storage-cleanup-01/`; full journals and the
plan remain under `results/hpc/20261021-mosaic-storage-cleanup-01/cleanup-audit-02/`.
The campaign now binds the updated local backup index and completion receipt.

The native validation download also completed: **294 files**, archive
157,243,652 bytes, SHA-256
`fa4fd76c68be6b310f6b8f7724e26e3c5c86ae0ef530905d2c94f676c00dd210`.
Its verification receipt is
`results/hpc/20261021-brd4-jq1-25-fixed-02/native-evidence-01/native-evidence-download-02-download.json`.
The earlier empty partial is retained. These local checks establish byte
integrity; they do not repeat numerical validation. All three original native
validations passed, but **no new long-run job IDs are established**.

The all-work3 launcher tagged `20261021-brd4-jq1-25-fixed-storage-01` is prepared
locally with AST checks only. CPU validation and staging remain pending. Its
storage placement does not change the fixed rates or scientific gates; it is
not a submitted or completed 10,000-update run.

### 21 October â€” storage-01 and storage-02 staged, then stopped at preflight

The subsequent storage-01 release was staged read-only with 11 authenticated
files, manifest SHA-256
`9cc079f0b89256cd723b5c6ee87e76f8a93d1b2db3cfdc8d99fc29272f152e7d`.
Its preflight rejected the original pytest `current` symlinks before any CPU
validation or job submission. The failed attempt remains preserved.

Storage-02 was then staged with 11 authenticated files, manifest SHA-256
`c8115e2686b94d78ff1889b9dc8161fdf00356639524853c46d4100fd8638fd9`.
Its archive was 18,560 bytes, SHA-256
`8ed85ddae3e116640513b84e4786e8891315541768bb069d8cbfd0546124accc`.
Preflight again stopped before CPU validation or submission. Byte headroom
passed: 13,468,288,122 available versus 5,368,709,120 required. The newly added
`statvfs` inode minimum was zero: the saved `fhgfs` filesystem counters were
exactly `f_files = f_ffree = f_favail = 0`, while authenticated finite user quota
reported 19,703,870 remaining files. This was a storage-metric rejection, not a
scientific failure or a failed native gradient check.

A new immutable storage-03 candidate is pending. It will record that exact
all-zero counter triplet as an unreported supplemental filesystem metric and
still require the original finite quota plus 100,000-file headroom. Coherent
positive filesystem counters will continue to limit availability by their
minimum; invalid or inconsistent metrics fail. Byte budgets, reserves,
source02 scientific identities and all scientific gates remain unchanged.
Neither failed release is rewritten or relabelled as passed.

The original raw CPU/readiness proofs were independently verified locally;
verification receipt SHA-256
`76821f5448323ee08819a21f394dc78105ca4745f58cd3112db024ce49605d6c`.
That integrity check does not establish a storage-launcher CPU pass. No CPU pass
or new job is established for storage-01, storage-02 or pending storage-03.

### 21 October, 20:16 Paris â€” storage-03 passed; three fresh long jobs queued

Storage-03 sealed and staged 11 files, manifest SHA-256
`9ee7bf1d6bf412acdc37fc83415476dbefdf45db89533571d4a646ee16f33772`;
archive 19,732 bytes, SHA-256
`0473d0908ff6139bc4cbaf98421eab96a33db1f5614db78638500d0bb9684b8e`.
Allocated CPU job **29777704** and callback **29777705** actually completed
DONE/LSF success at 20:15:26 and 20:15:44 Paris. All 65 storage-contract cases
passed, with no failures, errors or skips. The original 597-file science02
source and its 482 CPU / 254 native proof results were adopted unchanged;
these new contract cases do not rerun the native models.

The fresh dispatch storage gate passed at 20:15:42 Paris: full projected
artifacts 8,427,765,472 bytes plus unchanged reserve 4,294,967,296 bytes required
12,722,732,768 bytes, below conservative available **13,468,221,517 bytes**.
Finite quota reported 19,703,830 remaining files against the unchanged
100,000-file reserve. Raw filesystem inode counters remained exactly all zero
and were recorded as `unreported_all_zero`; they supply no independent
filesystem inode proof. Rounded quota/headroom is not a reservation. The exact
receipt is
`results/hpc/20261021-brd4-jq1-25-fixed-storage-03/launcher-evidence-02/quota-dispatch/split-storage-gate.json`.
Each GPU entry still repeats its required storage checks.

The following fresh 10,000-update production jobs were actually submitted at
20:15:43â€“44 Paris, with no inter-arm dependencies:

| Arm | Job | Fixed learning rate | Last observed status |
| --- | --- | ---: | --- |
| Protenix | 29777709 | 0.3 | PEND at 20:16:22 Paris |
| Boltz-2 | 29777710 | 0.01 | PEND at 20:16:22 Paris |
| Boltz-2 plus chemistry | 29777711 | 0.01 | PEND at 20:16:22 Paris |

Each requests `gpul40s`, one exclusive GPU, four CPU slots, 8 GiB per slot and
23 hours. They were waiting for physical GPU availability; submission does not
establish a started or completed trajectory. New outputs use
`/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-03`.

The downloaded launcher-evidence archive contains 33 files, 17,016 bytes,
SHA-256 `bd62f30084ec05c0c0e63063f0090a3c9c4d9d05d43d9f0549317d057d2c4c90`.
Local byte checks completed; independent XML/schema verification is underway.
The original storage rejection and storage-01/02 failures remain preserved.

Latest compact query, **20:35:30 Paris**: Protenix **29777709 is RUN**, its entry
quota gate passed, and 951 states / 950 updates are committed (state index 950,
last update index 949). Its summary is absent, so this is partial progress, not
completion. Boltz-2 **29777710** and chemistry **29777711** remain **PEND**, with
no study outputs or established initial molecule. Independent verification of
the new 65-case XML and launcher proofs is still underway.

Independent standard-library byte/XML/schema verification of all 33 launcher
proofs is complete. Raw XML confirms 65 passes without errors, failures or skips;
actual CPU/callback DONE and LSF success, all original adoption pins, whole-campaign
storage and the three independent submission/resource vectors matched. Receipt:
`results/hpc/20261021-brd4-jq1-25-fixed-storage-03/verification.json`, SHA-256
`e7edc4a4fd07ec1968fba600ce7ef19f691404aefede8fbdd588d0dc198bd90f`.
This verifies launch readiness and submission; it does not certify trajectory
completion.

### 21 October, 20:39:34 Paris - Boltz entry storage failures; Protenix still running

The latest actual query shows Protenix **29777709 RUN**, with a contiguous
**1,576-state / 1,575-update** committed prefix and no summary. This is partial
progress, not completion; its existing job and results remain unchanged.

Boltz-2 **29777710** and chemistry **29777711** actually **EXIT** before any
study or model execution because their whole-campaign entry byte-quota guard
failed:

| Arm | Required bytes including reserve | Conservative available bytes |
| --- | ---: | ---: |
| Boltz-2 | 12,502,649,138 | 12,173,988,090 |
| Boltz-2 plus chemistry | 12,496,820,290 | 12,140,892,177 |

Raw quota used was 87.29 / 87.41 GiB, versus 86.70 GiB at Protenix entry;
the rounded quota-derived available bound decreased by 1 GiB. File quota
retained about 19.7 million available files, and the exact all-zero filesystem
inode counters were correctly recorded as unreported. These are failed storage
entries, not native gradient failures: both original readiness proofs remain
passed, and neither failed Boltz job performed a scientific step.

The shared byte verifier passed the 13-file failure archive, 6,088 bytes,
SHA-256 `7cdbb679cc3acb4e38c8a1a28fa8727a2b934362a95c3104858c6a94da157739`.
Local evidence is under
`results/hpc/20261021-brd4-jq1-25-fixed-storage-03/entry-failures-01/`.
All three original source/outcome records and current Protenix progress are
preserved.

A declared launcher-only split-storage recovery for the **two Boltz arms only**
is under investigation. No new recovery source, CPU pass or job is established.
The home quota snapshot was 18.62 / 30 GiB, last updated at 19:03:58; a bounded
Mosaic-only home inventory is underway, but output fit is not established. No
storage or scientific gate is waived, and Protenix is not restarted.

### 21 October - local split-storage recovery candidate; access and capacity unresolved

The active source04 recovery candidate is packaged locally under
`20261021-brd4-jq1-25-fixed-storage-04`, package **source03**: **12 files**,
archive **25,613 bytes**, SHA-256
`0ae3e8359f5ec7e6b25684746fca90aecc080c1fb2e86c59b71822981fd784a0`,
manifest SHA-256
`75a76dacf8de25f4ebd2d3b3fe114b1af6092064a01edf141650295300670ab3`.
The campaign, source manifest and `local-static-verification.json` are saved in
`scripts/hpc/receipts/20261021-brd4-jq1-25-fixed-storage-04/`.
AST and standard-library byte checks passed for all 12 files against the archive,
manifest, working files and Git staging bytes. Two earlier local packages are
preserved with separate receipts; only mixed line endings in the storage helper
and fixture were normalized, with identical Python ASTs. **No remote staging,
allocated CPU pass, observer or replacement Boltz job is established.**

The declared placement keeps existing Protenix 29777709 unchanged on work3,
places only a new Boltz-2 output on work3, and a new chemistry output in a
separate dedicated home Mosaic directory. Scientific source02, configurations,
readiness and fixed-step settings stay pinned. Each filesystem must independently
fit its combined full remaining assigned projections plus the unchanged 4 GiB
and 100,000-file headroom. The candidate includes a CPU observer for actual
existing-Protenix terminal evidence, gated by allocated contracts and unchanged
adoption identities. All failed storage decisions remain failed.

Home capacity remains **unproven**. The 55-second metadata walk and subsequent
bounded read-only `find` attempt did not complete; partial inventory is not a
resource pass. Saved positive home filesystem inode counters do not establish
byte capacity or a user inode quota. Later SSH attempts failed authentication,
including the bounded 22:15 Paris terminal-evidence attempt (return 255), so no
later Protenix outcome was observed. Its latest verified state remains RUN at
1,575 updates / 1,576 states at 20:39:34 Paris; elapsed time is not completion.
All failed attempts and logs are preserved. No further storage deletion,
scientific execution, local numerical test or report publication occurred.

### 21 October, 22:29 Paris - access restored; source04 CPU validation submitted

The actual read-only query at 22:27:56 Paris found **Protenix 29777709 RUN**,
with **9,656 contiguous committed states / 9,655 updates**, no unfinished JSONL
tail and no summary. It remains unchanged and is not yet complete. The two old
Boltz records have aged out of live scheduler history; their authenticated EXIT
and LSF failure proofs remain preserved, with no study or model execution.

Source04 package source03 was uploaded with exact length/SHA, authenticated for
all **12 files** alongside original source03/science02, checked for shell syntax,
and made read-only. The fresh CPU-only work3 preflight passed: **17,100,677,313
available bytes versus 5,368,709,120 required**, with 19,674,822-file quota
headroom. Raw work3 usage was 82.87 / 100 GiB. This is an updated quota snapshot,
not a reservation or additional cleanup; the complete home/campaign storage gate
is still required. The exact old read-only inventory process inspection found
no active matching Mosaic `find` process before submission.

Actual CPU **29778180** and callback **29778181** were submitted at 22:29:43
Paris and initially observed PEND. CPU requests two hpc slots, 4 GiB per slot,
20 minutes; callback requests one slot, 2 GiB, 20 minutes and only
`ended(29778180)`. Exact actual arguments, IDs and scheduler receipts are in the
source04 campaign and `initial-submissions.json`. Submission does not establish
an allocated CPU pass, a home storage pass, an observer or new Boltz long jobs.
All scientific identities and existing Protenix remain unchanged.

### 21 October, 22:30:44 Paris - allocated recovery contracts passed; completion observer queued

CPU **29778180** actually completed DONE/LSF success. The saved raw XML contains
**85 cases**, zero failures, errors or skips across both recovery modules. The
immutable original-science adoption was captured. Callback **29778181** is RUN
and observer **29778184** is actually submitted/PEND with only
`ended(29777709)`; it observes saved Protenix terminal metadata and does not run
or modify Protenix. No replacement Boltz GPU IDs or complete home/campaign
storage pass are established. The current controller is performing the bounded
complete inventory and fresh per-filesystem budget checks.

A 27-file metadata/CPU proof archive was downloaded and safely byte-verified:
47,750 bytes, SHA-256
`c17196252f310101fb79fa17b1e72a40188149673f2d23d7770635a615656f98`.
Independent local XML/schema verification is underway. Protenix remains RUN in
this actual scheduler snapshot; no terminal scientific completion is inferred.

Independent standard-library byte/XML/schema verification is complete. All
27 metadata files and source archives (12 / 11 / 597 files) matched; raw XML
confirms **37 storage + 48 controller = 85 passes**, no failures, errors or skips,
with raw test-log runtime 3.69 seconds. CPU 29778180 actual DONE/LSF success,
the unchanged original adoption/native proof chain, both old failed Boltz
admissions and Protenix's exact original invocation were authenticated. Receipt:
`results/hpc/20261021-brd4-jq1-25-fixed-storage-04/verification.json`, SHA-256
`dff06db043a51285a90be3d363acf796f51cc8b216d12cc130a5902f31b2919b`.
CPU certificate SHA-256
`b9f7e4a35289b5b6e7ab240b71fe119ed54ca91ef322736152195f83071408b1`;
adoption SHA-256
`58b6b9fc2c55d3f8ebd144f474ab0928145597f9e34ffaa6fa33a68dc08d1cfd`.
This is existing-proof integrity and allocated launcher-contract evidence; it
does not establish a complete storage pass, new Boltz long jobs or Protenix
completion. At 22:33:09 Paris, callback remained RUN and observer PEND.

### 21 October, 22:36:48 Paris - full dispatch storage passed; two replacement Boltz jobs submitted

The saved actual scheduler/artifact query is
`results/hpc/20261021-brd4-jq1-25-fixed-storage-04/user-status-20261021T203646Z/status.json`.
Callback **29778181** completed DONE/LSF success after the complete per-filesystem
dispatch budget passed: work3 **15,874,334,545 available / 6,827,669,155 required
bytes**, home **11,811,160,064 / 9,459,309,120 bytes**. Both required budgets retain
the unchanged 4 GiB reserve. The bounded no-follow home inventory completed with
return code 0 and no recent Mosaic file/link bytes; it was not a partial-inventory
pass. Work3 file headroom is authenticated finite user quota; home's positive
physical filesystem counters are not a reported per-user inode quota. These
checks are observations, not reservations, and each GPU entry must repeat them.

Both independent replacement jobs were actually submitted at **22:34:20-21 Paris**:
**Boltz-2 29778203** is scheduler **RUN**, but has no study, saved state/update or
completed entry gate in this snapshot; **Boltz-2 + chemistry 29778204** is **PEND**,
also with no study or saved steps. Thus both jobs are submitted, but scientific
optimization has not yet been verified to start for either. Their fixed LR remains
**0.01**, one candidate per update and no halving, with unchanged scientific
source02, configurations and passed native readiness. The old failed admission
jobs 29777710/29777711 and their original gates remain failed and preserved.

Existing **Protenix 29777709** is actual scheduler **DONE**, with its summary
present. Allocated observer **29778184** also completed DONE/LSF success and saved
`observed_success`, authenticating the actual same-job scheduler/LSF and terminal
metadata for **10,000 updates / 10,001 states**, including saved summary/case,
JSONL and derivative proof hashes. This does not substitute for the future full
report frame/snapshot completion certificate or imply that all original native
bytes have been downloaded locally. No Protenix restart or further cleanup was
performed. No fixed-step report is published yet; published milestone progress
remains 0/0/0 until the first verified fixed-report edition.

Independent byte/JSON/schema verification of the 28-file compact delivery
evidence passed: archive 21,152 bytes, SHA-256
`ce780f8ffdbeef0b6dacdb2a1dd701caa057537dccfee82b8f318a56ee2129c0`;
verification receipt SHA-256
`f175fa887bfe65aed6bf227b4dea712ecb7e966aa16585ba5982a6ee44746dd6`.
Exact dispatch budgets/reserves, completed inventory/fresh quota, both actual
new submission/resource vectors, callback/Protenix/observer scheduler and LSF
outcomes and compact derivative proof hashes were authenticated. GPU entry
checks and saved Boltz optimization steps remain unverified in this bundle;
the four large Protenix original trajectory files are remote hash references
only, not full local original-byte or report-frame verification.
