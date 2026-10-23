# Fresh DBMol binding-loss pilot, attempt 01

Prepared 23 October 2026 by Codex under the user's explicit request for a new
LR0.0025 / 10,000-update DBMol-loss pilot and gradient-flow checks. No saved run
is changed, resumed or restarted. This is separate from the two native-affinity
30,000-update comparison arms in `brd4-jq1/boltz-lr0025`.

The objective is `r - mean(raw_binary_logit) + 2.5*contact + 2.5*confidence`.
It uses two separately frozen, hash-authenticated native Boltz checkpoints,
the original detached JQ1 pose/MSA/pocket and 25 neutral occupied slots. The
contact head and scalar assembly use declared FP64 arithmetic; graph variables
and raw graph derivatives stay FP32. Binder probability is a separate metric.
No chemistry validity term, decoder, VAE, diffusion update or gradient, clipping,
normalization, line search or temperature change is introduced.

The fresh initialization remains 50% chain/uniform, model key17, temperature1.
Exactly one finite candidate at LR0.0025 is applied per update, including honest
uphill/equal/no-op outcomes. Four serial 2,500-update segments retain exact
saved FP32 logits, probabilities and key at every boundary. This bounds each
allocation without changing the requested 10,000-update objective or step rule.
Every successor requires exact predecessor scientific acceptance. A failed
native/raw FD/parity/frozen/runtime/storage gate stops that chain.

Every component and the total require independent-primal X/E finite differences
at 0.01/0.003/0.001, 15% relative tolerance and two adjacent passing scales,
finite nonzero/typed-bond signal, hard native parity, raw-to-JAX VJP agreement,
live component recomposition, key/logit null controls and frozen before/after.
DBMol's separate AD-only diagnostic contract is not adopted. A ten-update
disposable pilot establishes actual runtime/storage before optimization.

All tests/model work execute only on allocated DTU HPC. Local preparation is
stdlib AST/JSON/byte packaging and evidence verification. GPU allocations use
gpul40s, one exclusive L40S48GB, four CPUs, 8GiB/slot, 2h native /23h segment.
Allocated CPU suites use hpc/four CPUs/4GiB per slot/1h. Callbacks authenticate
same-job scheduler, LSF, raw XML and immutable input/output hashes. Storage
includes existing chemistry, diagnostics/reporting, both Protenix extensions,
both new 30k comparison arms and this pilot, plus 4GiB/100,000-file reserves.
Unknown submission markers forbid blind duplicate launches.

The run owns protocol, driver, controller, wrappers, packaging, submission and
small receipts in this directory. The worker/bridge live in the experiment's
`scripts/`; numerical validators/fixtures in `tests/`; reusable native composite
in `src/mosaic/small_molecule/boltz2/dbmol.py`. Historical flat paths are created
only as authenticated release aliases to retain inherited CPU suites. This is
an account-bound launcher, not a portable setup example.

Planned remote root: `/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/boltz-dbmol-pilot-01`.
Planned immutable source: sibling `boltz-dbmol-pilot-release-01/source`.
Large local artifacts: ignored `results/hpc/20261023-boltz-dbmol-pilot-01/`.
Current status: prepared only; no CPU/native/pilot pass or job ID yet.
Exact package hashes and executed commands will be added after sealing.


Actual source657 staged read-only manifest24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436, archive39676321B SHAec333fd3e0b33d9e5b82e0cd5e4e12c14ffb4f91782c18cd820d4bfab53de1f8. First preflight failed20,597,463bytes short before anyCPU submission; exactfailedgate and6-file independentbyteaudit preserved. Three redundant upload copies createdbythisrequest were removed only after exactlocalbackups/all633+657+633immutablemembers verified;118,913,015bytes recovered, allstudies/releases preserved. Unique metadata-only admission02 reauthenticatedunchangedsource/failedgate/absenceofanypriorCPUorunknownsubmission, wroteafreshgatewithoutoverwritingfailure. CPU29798549/callback29798550 actuallysubmitted10:41:05UTC afterfreshvalidationcapacity19225068665required/19235502439available passed;4GiB/100000filereservesunchanged. Thisisnotfull10kcapacityorgradientsuccess. AllCPU/native/optimizationoutcomespending.


Actual CPU29798549/callback29798550 subsequently EXIT1 beforepytest: CPUentry required19,175,262,065bytes versus18,158,117,863 conservativeavailable. No rawXML, numerical tests/model/native checks or optimization occurred. Independent24-file/source657 byte/schema/scheduler/LSF audit is cpu-entry-failure-verification.json. Both original failedpreflight and passing admission02 gate remain unchanged. A new small launcher-only accounting adoption is being prepared to authenticate original Pro29792670 and Bol29792671 terminal metadata before assigning zero future growth; their existing bytes remain fully stale-charged, Bol scientific failure remains failed, every active/prospective budget and4GiB/100000file reserve retained. No new adoption pass or resubmission is established here.
