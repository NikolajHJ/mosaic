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

Actuallysubmitted16:28:45–47Paris: native Protenix **29776117** /callback29776118,
Boltz-2 **29776119** /callback29776120, chemistry **29776121** /callback29776122.
At16:30:34Paris all6PEND. Actualargv andbjobs resource vectors match gpul40s,
oneexclusiveL40S48GB,4CPU,8GiB/slot,2h; nativejobs have no interarmdependency.
Callbacks waitonlyontheirownnativeended event. Full10,000-update runs have NOT
been submitted; controller still requiresall3fullnativepasses andwholeartifact
projection/freshquota, thenindependent23hlongsubmissions. CPUtests do not certify
nativeGPUgradients or completion. Existingautomation now tracks this new scope.
Code andinitialreceipts pushed634a1e5; no scientific or storagegate weakened.
