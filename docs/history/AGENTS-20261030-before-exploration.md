# Active project instructions

## Common gradient validation policy — required for new work

Read and apply [the common gradient validation policy](docs/GRADIENT_VALIDATION_POLICY.md)
before creating, changing, copying or interpreting a gradient checker. The user
adopted `gradient-validation-policy-v1` for both repositories on 30 October 2026.
This prospective policy supersedes blanket historical demands for nonzero
component gradients, fixed infeasible FD probes or bitwise equality of
recomputed floating outputs in new work. Keep explicit legacy replay contracts
and historical outcomes; never silently reinterpret an old failure as a pass.

Use maintained shared checks and versioned producer/reader contracts. Keep
nonfinite, required-path, mapping/context, frozen-state and material derivative
error failures. Record inconclusive measurements honestly; obtain independent
evidence for qualification rather than weakening tolerances or passing no-ops.
Distinguish valid flat derivatives from detached required paths and computed
numerical variation from immutable-asset mutation. Add positive and negative
regressions, validate numerical code on allocated HPC, and record exact scope.

Adopt fixes only in a fresh authenticated release. Leave running/pending jobs,
callbacks, sealed sources, live states, caches and checkpoints unchanged; this
policy does not authorize a restart, cancellation or scientific-setting change.

Regression baseline, 30 October 2026: all 543 scoped allocated CPU cases passed
(292 DBMol / 251 Mosaic, zero failures/errors/skips). See
[the receipt](outputs/implementation-checks/gradient-policy-20261030.json).
This does not waive new native-model/GPU qualification or authorize adopting
working-tree changes into a live or sealed campaign.

## Default DTU HPC connection for local agents

**Use the installed unattended connector by default for all new local HPC
commands, status checks, submissions and file transfers on Nikolajs_laptop.**
This is a workstation-specific route for Windows user `nikol`, DTU account
`s234818` and `login1.hpc.dtu.dk`; it is not portable account configuration.

```powershell
$dtu = 'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe'
& $dtu exec 'hostname'
& $dtu exec "bash -lc 'bjobs -u s234818'"
```

Read [the current connection guide](docs/HPC_CONNECTION.md) before operating.
`exec` takes one quoted remote-command string; `exec --timeout SECONDS` sets the
command timeout. It preserves stdin/stdout/stderr and the remote exit code.
`download /absolute/remote/file LOCAL_NEW_FILE` and
`upload LOCAL_FILE /absolute/remote/file` use the same pinned login1 host.
Keep login-shell setup such as `bash -lc` when an existing command needs LSF PATH.

The existing SSH key/Windows agent and the saved generic credential
`DTU-HPC:s234818` provide authentication. Do not ask the user to reconnect VPN or
enter a password as the routine first step; try this connector first. Never print
credential contents or put passwords in commands, files, logs or chat. Plain
`ssh`/`scp` and existing scripts do not automatically use this helper: route their
local transport through it explicitly while preserving commands, inputs, receipts,
timeouts and experiment guards. Preserve sealed releases and running jobs.
A failed connection or lost submission response does not establish job failure;
reconcile scheduler/receipt evidence before retrying a submission.

Use the path above, never the superseded AppData path (MSIX-private redirection).
The helper passed fresh logins, scheduler, stream/exit-status and SCP checks;
the user confirmed it in ordinary PowerShell on 29 October 2026. Reboot persistence
is untested. Local usage, evidence and rollback notes are under
`C:/Users/nikol/DTU-HPC/helper-v1/README.txt` and
`C:/Users/nikol/DTU-HPC/helper-v1/checks/summary002-path-repair.json`.
This transport instruction authorizes no new experiment, restart or monitor.

**VAE readonly30 October2026 12:33CET:** source140GPU29849760 RUN/callback
29849761PENDnormal; latestsavedpanel12:26:15 at279256=2224/6596(33.72%)into43.
Raw/strict772/1024(75.39%) vscompleted42's782(76.37%); exact/validrecon256/256,
sulfur1.41499%, latestrepresentationfailures0. All157savedearlypanels/2826
healthchecks passed; no43completion,42latestcheckpoint. Existingloss/1:2/
optimizer/anchors/guards/endpoint45unchanged; failedlocalneighborlossNOTenabled.
ReadonlynewSSHcollection no numericalwork/submissions/runtimechanges. Docs80/
ledger and source140attempt002/snapshot-1793359995796733500/status.json bind
exacttimestampedreceipts, not an exact live-update counter.

**Local neighbor-loss pilot safety-stopped,30 October2026 12:07CET:** dedicated
local environment/source001/147CPUtests and two-updateGPUbenchmark passed;
SO3 regression passed. Mainpilot001 control100 updates completed but neighbor
weight0.1 stopped at10 under the declared10pp prior-validity-drop check.
Compare COMMONupdate10: rawvalid823initial->734control/611neighbor of1024,
strict822->733/611; exactrecon255->254/255 of256. Localvalenceviolations
75->89/36, sanitization119->118/61, but native disconnected82->172/352.
Independent raw-graph traversal confirms actualdisconnected91->188/366 and
totaledge losses335control/1255neighbor; notjustfailure-labelunmasking.
ExpectedS1.403->1.503/1.480%, versuspopulation1.688%; sulfurinflationisnot
theprimaryfailure. Reviewer verified identicalinitialarrays/classifications,
matchedfirst10rows+posterior/priorRNGstreams andexactcenterweighting. Allsaved
lossesfinite; evidenceconsistentwithdownward-closedlocalvalenceeventfavoring
bondremoval/connectivityloss atthisweight+smallbatchregime, notglobalproof.
DO NOT ADOPT. Failedarmmissingterminalfresh/categoricalpanels, failedweights
snapshot andfinalfrozenanchordigest; donotpromotemissingchecks. Frozenlocal
sourceandinputcheckpointbytes unchanged; rawinvalidgraphs retainedexternal.
No additionaltraining orHPCchangesperformed. Docs88/ledger and
outputs/local-neighbor-loss/20261030-attempt001/terminal-summary.json,
pilot-report-reviewed/report.md and reviewer-audit.json bind actualevidence.

**Scoped local neighbor-loss pilot authorized,30 October2026:** the user
explicitly requested an isolated local environment, loss/gradient checks and
a small VAE training comparison, leaving the HPC continuation unchanged. This
overrides HPC-only numerical restrictions solely for this bounded local task.
Dedicated environment/inputs/source/results live under
`C:/Users/nikol/DBMol_Local/neighbor-loss-v1`; existing .venv is untouched.
Checkpoint42/AdamW moments and frozen35occupancy/36valid-target anchors are
authenticated. New local RNGs/smaller batches make this a separate diagnostic,
not an exact HPC continuation. Existing repair/reference/JS/occupancy and actual
quarter-AdamW displacement remain, with only the neighbor arm adding explicit
common-organic-neighbor-v1 weight0.1 (real/prior weights0.25/0.75, branch-global
assessed-center means). Default shared-training path remains unchanged.
Source001 SHA61af99abee308a9f2cc169d28c9a1e668a2d21473b006ff6713ad5d9b4ed8d74
binds448source/test/toml files;147sealed CPUtests passed. Two-update GPU
benchmark002 passed matched streams/frozen anchors; failed benchmark001
CuBLAS startup is preserved. Mainpilot001100updates/arm,16real+32prior,
micro8 and1024fixed/end-heldout panels is in progress, not an endpoint pass.
All invalid hard graphs retained; reports separate chemical/representation
failure, connectivity/size, reconstruction and hard/expected sulfur frequency.
Local diagnostic snapshots are nonpromotable. No HPC job, loss, plan, source
or state may be changed by this pilot. See docs88 for exact receipts/limitations.

**Isolated V100 resume passed,30 October2026 12:15CET:** source150 GPU29853488
DONE; restored rebased case002 copy20 and completed one original-policy update
21 without stall/noop, frozenstate unchanged, CPU source149 clone untouched.
Status passed_one_isolated_resume_update_not_adopted. Fifteen raw assets verified
under source150/terminal-1793358953495143700; docs87. This is not a full segment
or productionadoption. V100 serialcontroller and exclusive cutover still require
implementation and testing; original A10029851407 PEND, VAE29849760 RUN.

**V100 handoff progress,30 October2026 12:10CET:** cache audit GPU29853377 DONE,
all3 savedhardgraph scores within inheritedscalar tolerances. CPU29853481 DONE
9rebase tests passed/zero failures/errors/skips; CPU29853484 DONE rebased isolated
copies of all3 actual checkpoint/cache/best states at1899/20/20, preserving latent,
start,counts and A100 originals. Source149result SHA
a853164c8da83372679cb133772fc8d9ce3e764426b785dfe30119a095578ab3;
clonesnotadopted. One-update production-policy isolated V100 resume pilot
GPU29853488 RUN since12:12 on n-62-20-11, no GPU result as of12:13. Actual V100 serialcontroller and exclusive
cutover remain outstanding; longrun NOT moved. Original A10029851407 PEND;
VAE29849760 RUN unchanged. Docs87 and v100-handoff-progress-1793358450274591500,
v100-real-rebase-terminal-1793358555578355300, v100-resume-pilot-progress-
1793358624488479400 receipts. No productionstate/cache/job modified.

**V100 continuation handoff in preparation,30 October2026 11:30CET:** user
explicitly requested moving the unchanged epoch36 mapping run to V100, leaving
VAE continuation unchanged. Source146 V100 device qualification passed but
A100 baseline score caches are not bitwise portable. Fresh snapshot confirmed
case001/002/003 saved1899/20/20, each one cached hard graph and one best graph;
A100case002 job29851407 PEND, callback29851408 PEND, VAE29849760 RUN.
Independent source147 cache audit archivef9bff6a14b41f27128a5d5970de7e663bb0451e0c8b7d530b7e6f28ce44c1e8c,
manifestec19a9a2711806e8ff12bcba6ccba6811123c54c1a314fcdb37354859dbdb182,
plan28f34414a615aec46d80e0a376a529288c4ff217685429215b3ddbf9ee775ffb
was staged; GPU29853377 submitted to32GBV100, PEND11:37 physicalGPU
unavailable on four hosts, no pass/result. Draft v100_rebase_core.py is AST-only
and not tested/adopted.
Audit authenticates original146 pass/states/caches and scores exact saved graphs
on V100; no originaljob/caches/states/source/VAEmodified. Handoff controller,
rebase/clone, bounded resume and safe cutover remain to be implemented and
verified; run is NOT yet moved. Do not cancel original pending A100 without an
exclusive safe cutover and complete V100 handoff, or duplicate writers if it
starts. See docs87 and v100-migration-preflight-1793356054191164800.json.

**V100 qualification passed,30 October2026 11:24CET:** source146a001 GPU29852732
DONE10:31:25, 2987.10s worker on TeslaV100-SXM2-32GB. All3 native,6 start and
3 saved-resume derivative gates plus freshsame-device productionbindings passed;
allmodels/context unchanged. Peakreserved26.36GiB,headroom5.37GiB >=4GiB;
singlefixed.0025candidate means12.87/13.38/13.38s (not fulltrajectory). A100
scores/gradientdigests not bitwiseequal, crossdevicecache/bindingmigration not
performed; passing qualification is not automaticcontinuationadoption. Original
A100job29851407 PEND, VAE29849760 RUN. 33rawassetsSHA/sizeverified under
source146a001/terminal-1793355904505738800; docs86/summary.json. No source,
scientificsettings, livejob or productionstate changed by these checks.

**V100 qualification running,30 October2026 09:42CET:** user authorized
32GBV100 checks. Independent source146a001 archive68b10f1cc0702ef8df6b34ce5956177a45d51bbcc0de78d570a379a7ce3eb64c,
manifestf288580e454e7d5fbc91fe76260e7f016f15908624f8bb15fe2a0ae63d92399a,
plane1355e4b638c6f6bda2cdc6ba0e62cde9504dfeb5bf251bcaf10ec43b26a5afa.
CPU29852731 DONE38passed/zero failures/errors/skips. GPU29852732 RUN on
TeslaV100-SXM2-32GB, torch2.5.1+cu121 compiledsm70; hardware initialized,
09:44 models loaded/case001 resume eligibility passed; its native gate was
in progress, not yet native/derivative/memory/timing pass. Tests three original starts and
immutable resume states1899/20/20, fresh same-device native/derivative bindings,
4GiBheadroom and one fixed.0025 candidate; no automaticproductionadoption.
Original142 case002 A100job29851407 PEND; VAE29849760 RUN. Their settings,
weights, live states and caches untouched. Docs86 and source146a001 receipts.

**Unchanged long-run first segment finished,30 October2026 09:15CET:**
source142case001 GPU29846090 DONE29Oct23:58:42,callback29846091DONE23:59:35;
fullresult/executionSHA+bhist+terminalproofverified. Completed1899=20pilot+
1879exactplannedadditionalupdates; segment_complete_target_pending, no deadline
stop/stall, zero noops,1899applied/1239improving/660forcedfallback. Allguards
andfullfrozen/contextchecks passed,bestvalidstill0. Latestobs1890strictvalid/
eligible butsameinitialhardidentity/allscores(hardtotal8.34434405); learned
8.34882049->7.80366408(-6.53%) isnotchemical/bindingimprovement. Plannedserial
roundrobin callbacksubmittedcase002GPU29851407PEND23:59:25+callback29851408;
case002still20,maxadditional2628; case003still20. Case001willresumeaftercase2/3,
notfinished10000. No jobs/runtime/settings/continuationchanged byread-onlycheck.
Docs81 and source142/status-20261030-1793348203235586400/integrity.json plus
embedding-long-status-1793348116554671200.json bindexactreceipts.

**Read-only new-SSH status,29 October2026 22:14CET:** source140 completed42
checkpoint5ef851101a5f675edc8f65b31b75c23a5ec460f1c40aa5571829bf8ad6714cfd,
absolute277032,measured39162.440s/10.88h; all19 endchecks/134earlypanels/
2412checks passed. Raw/strict782/1024 (76.37%) vs41's741; exact/validrecon256,
sulfur1.41675%. Pairedgeneration argmax792/811 of1024, rawcategorical2732/2762
of4096 (one firstpanel representationfailure). Callback submitted43GPU29849760,
nowPENDjobslotlimits5hosts; callback29849761normal. No43update, endpoint45/loss/
1:2/guardsunchanged. Long29846090 RUN,case001saved1705 at22:13:34CET,
observed1700eligible/natural+conditionedstrictvalid,hardidentity/lossunchanged,
bestvalidstill0; learnedloss8.34882049->7.80659526 isnotbindinggain.1705applied
includes1124improving/581forcedfallbacks/zeronoops. Cases002/003still20pilot.
Callback29846091normal; independentDeFoGdiagnostic29846730PSUSP. Readonly
collectedno numerical/modelruns/newjobs/settingchanges. Docs80/81/ledger and
outputs/hpc-results/hpc-run-status-20261029-2214.json bind exact receipts.

**Connector verified and neighbor-environment qualification passed, 29 October
2026 21:42 CET:** fresh helper login/stdin/scheduler and a SHA-verified download
passed; the helper uploaded and submitted the already-authorized source145a004
CPU validation as job29851083, now DONE. Exact terminal receipts are in
outputs/hpc-results/source145-neighbor-environment-validation-attempt004/
snapshot-1793306498640264200. All138Torch/26JAX fixtures passed with zero
failures/errors/skips; 10,000 validation molecules (3,337 sulfur-containing)
were retained, frozen40latentVJPnorm0.0665616244 was finite/nonzero with model
unchanged, and six-case Torch/JAX parity passed (max scalar4.44e-16,
rawbondVJP8.33e-17; unchanged1e-9 cap). Status is
passed_validation_not_training_adoption; no active loss/default changed.
Recovered a003 had24JAXpasses/two fixture failures from a genuinely flat C-S
SINGLE-vs-DOUBLE direction. A004 asserts that plateau and tests informative
S-O DOUBLE-vs-TRIPLE instead; all nonzero/FD thresholds unchanged. Production
and driver bytes match a003; byte audit in
outputs/implementation-checks/source145-attempt003-to004-static.json.
Archive9b1e408a4bcb9794980c6685ee7c1e5511794a7d6765c44c279b754f50370068,
manifestc00079b4c0c45dcc5c1e49ff405deeb547f0525db7b9692871f4f00f41161fda.
Earlier failed releases/logs are preserved; docs85/ledger bind exact scope.


**Neighbor-environment loss implemented; partial CPU qualification,29Oct2026:**
User authorized neighbor-aware sulfur/edge cases and gradient/legitimate-molecule
tests, explicitly accepting rare specialized S6C exclusions and a simple rule.
New opt-in common-organic-neighbor-v1 limitsS to4nonF heavyneighbors (F still
consumesvalence), adds frozen-neighbor remainingcapacity, preserves inherited
charge-aware rules and P5C, and uses exact FP64 logeventmass with bond-only
direct gradients, including unsafe tails of validargmax. Native validity/
defaultlosses/currentVAE/Boltz/Mosaic scientificruns remain unchanged. Torch
chemical_environment[_policy].py and Mosaic chemistry/environment[_policy].py
share policySHA8f7cdda339d16d0284b3141cf0511be71c21e49c167c03be74f65f795ffb50d9.
Source145a001 CPU29848286 passed138fixtures/61legitimate motifs/10kVALallretained
(3337S-containing)/actualfrozen40latentVJPnorm0.0665616244,weightsunchanged,
thenEXIT beforeJAXtests dueinheritedPython3.11PYTHONHOME inJAXPython3.12.
A002CPU29848305 repeatedTorchpasses butJAXpytestcollectionEXIT4: missingMosaic
layouthelperinarchive. Preservebothfailedreleases/logs,notnumericalJAXpasses.
A003 archiveab86bb6346b283428b4d7df904a70014653cc2326f69e7fef010f3fea4b747fe,
manifest043a0583faa6276e200a0e8088942a7a487d2eaae30209f03b7c75bfd2eca9ce,
662members/CPU29848323 submitted withper-childPYTHONHOMEunset+layoutmetadata.
Latecollection authdenied255/no remoteoutput; no verifieda003start/JAX/parity
result or adoption. UseraskedfreshSSH; do notguessstatus. Staticattemptaudit
confirmsallnewscientificpolicy/Torch/JAX/tests/updaterbytesidentical across3;
onlylauncher/packaging/docs changed. AllnumericalworkallocatedHPC4slots16GiB
60mwall/55mininternal; noGPU/trainingoroptimizationjobstartedbythisfeature.
Docs85/ledger/source145attemptreceipts bindexactscope and failed evidence.

**Runtime probe completed; VAE/mapping cost small,29 October2026 22:12CET:**
source144a002 GPU29846982 DONE16:07:28--16:27:09 onL40S n-62-13-19;
1175.302s diagnostic, bothoriginal/saved277 exactsame-device primal/latentVJP
passed, splitVJPerror0, allfrozenmodels/context unchanged. Forty-one rawassets
SHA/sizeverified in terminal-20261029; bhist/terminalDONE proof retained even
thoughliveschedulerrecordagedout. VAE+mapforward~.0206s andforward+native
cotangentVJP~.049s; singleLR.0025candidate directDBMol~7.75s,rawlatent~8.02s,
guarded~9.06s; guards/integrity~1s, I/Osmall. Priorproductionaudit8.92candidate
Boltzcalls/update identifiesmajorrepeatedwork. Five repeats/twopoints/onecase,
nottrajectory/convergenceproof. HistoricalDBMol3.52s differscontext/software;
bhist16:11:04power/clocks event meansno cleanhardware/causalabsolutegap claim.
OriginalA100certificatecrossdevicefalse reportedonly; no scientificgateadoption.
Main29846090 RUN latestsavedcase0011703at22:12:26; observation1700 eligible/
strictvalid butsameinitialhardidentity/hardloss8.344344,learned7.806595 isnot
chemicalgain. Others20. VAE next29849760PEND/callback29849761PEND; continuation
unchanged. No jobs/runtime/scientificsettings changed. Docs83 and verified-
collection-1793308361109116200/status-1793308311281196600 bind evidence.

**Read-only probe/long-run status,29 October2026 13:05CET:** source144 L40S
probe29846982 stillPEND physicalGPU availability3hosts; four reservedCPUslots
are not execution. NoGPUsetup/logs/timings/result yet. SavedCPU21pass remains
valid despite aged-outCPU29846941 causing mixed bjobs return255; independent
probe/live query returned0. Source142long29846090 RUN, latestsavedcase001652
at13:04:37, notexactlivecounter; cases002/003 at20. Observation650 eligible,
natural/conditionedstrictvalid, no rejectionreasons; no bindinggain established.
VAE29845931 RUN unchanged. Read-only collections changed no jobs/runtime/
scientificsettings/VPN/SSH. Evidence embedding-runtime-probe/live-status-
1793275506640697500.json and observation650-1793275528912142900.json; docs83.

**Read-only VAE status,29 October2026 12:35CET:** source140snapshot014 confirms
GPU29845931 RUN,callback29845932PENDnormally. Latestpanel12:26:01 atabsolute
273656 is3220/6596(~49%)into42,raw/strict739/1024(72.17%),exact/validrecon
256/256,sulfur1.48285%,zero representationfailuresinlatestpanel. Priorpanel
764/1024(74.61%); fluctuationsarenotcompletedepochresults. All100savedearly
panels/1800checks pass. Epoch41latestcompleted; no42checkpoint/endmetrics.
Loss/ratio/AdamW/RNG/endpoint45/guardsunchanged; read-onlycollection/docs only.
Docs80/ledger and source140attempt002/snapshot014 bind exact saved provenance.

**Read-only live queue,29 October2026 12:21CET:** account has3GPUjobs RUN:
VAE29845931(A100,n-62-18-8),source142long29846090(A100,n-62-18-12),separate
Mosaicprotenix-defog-balanced-3000029846882(L40S,n-62-13-16). Source144runtime
probe29846982 PEND; callbacks29845932/29846091 PENDnormally. VAEsource140
snapshot013 at12:20:33 confirms42stillrunning; latestsavedpanel12:16:12 is
absolute273556=3120/6596(~47%)into42,raw/strict764/1024(74.61%),recon256/256,
sulfur1.65681%;all99panels/1782checks passed. Epoch41latestcompleted.
Embeddinglateststate567saved12:20:54,latestobservation560eligible/valid,
case001hardidentityunchanged; cases002/003remain20. Savedstateisnotlivecounter.
Rawqueue/progressreceipt outputs/hpc-results/live-queue-1793272868751196200.json.
No numericaltests,submission,runtime/settings/jobs/VPN/SSHconfiguration changes.

**Fixed-step runtime probe queued,29 October2026 10:41CET:** user selected
most recent direct Boltz step size: LR0.0025, one candidate, no search. Historical
Mosaic L40S affinity/DBMol timings1.724877/3.515263s per update excluding first10;
different contexts/software, affinity terminal failed remains failed. Source144
attempt002 archive29c17cb7e64d9b78f9c342e309401525e247ab75d5e3613ae613eaedcf848435,
manifest1f7b5c9b19f05e1077f8b1f83ae8632f0981ccc827c2d35824a53b01b9a9beec,
plan3f89c017315e6537b4352c550fdac3e1ee2bfb9cf0549e681b10bade52facfbe.
CPU29846941 DONE21passed/0failures/errors/skips; L40S GPU29846982 PEND physical
GPU availability4hosts. Uses exact frozen36/trainedmapc8d865/cal1c4366/original
142 modules/context and start+saved277; same-device exact primal/gradient gates
required, cross-device A100 certificate differences report-only/no adoption.
No GPU timings or causal VAE overhead measured yet. Only unstarted exact-name
A100 diagnostic29846866 retired after PEND/nooutput/replacementCPU checks;
no production stop. Source142long29846090 and VAE29845931 independently RUN;
settings/state unchanged. Failed first retirement preflight retained (wrong
local CPU status spelling; no mutation). Docs83 and source144a002 status-summary001.

**VAE41 completed/42 running,29 October2026 09:46CET:** source140snapshot012
confirms segment00129838765 DONE04:04:40,actualepoch41 41545.531s(11.54h),
checkpoint690193d3c0d3cd1059dc4cfec7a9ad4d3d50172a577f25dec0851e91ef50e966,
absolute270436. Endfixedraw/strict741/1024(72.36%) vs40's753(73.54%),-1.17pp;
exact/validrecon256/256,sulfur1.52873%;all19endchecks passed. Freshpairedargmax
774->766 and773->757/1024; categorical2676->2566 and2599->2590/4096,also small
declines,notmonotonicimprovement. Generationobservationintegritypassed.
AutomaticGPU29845931 RUNsince07:02:21 n-62-18-8 exact41resume; latestpanel
09:37:50absolute271956=1520/6596(~23%)into42,raw/strict750/1024(73.24%),
recon256/256,sulfur1.45131%. All83savedearlypanels/1494checks passed; callback
29845932PENDnormally; endpoint45/loss/1:2/AdamW/RNG/guardsunchanged.
Independentembeddingpilotpassed/longrunning asnextentry; additional09:47
parentstatus1793263617831468300 confirmsRUN/gates butnotlivecounter; separate
source143pilot-running-status001 binds262savedat09:45:02,notexactlivecount.
Read-onlycollections/docs only,no runtime/settings/jobs/VPN/SSH changed.
Docs80/ledger/fix-resubmit-20261029-status001.json bindcurrentVAEreceipts.

**Source142/143 pilot passed; long optimization running,29 October2026 09:45CET:**
pilot29845020 saved execution06:52:22--07:40:42CET passed; all3 native/6start+
production gates passed; eachcase20acceptednonzeroiterations,zero noops,strict
validbest,frozenstate/contextunchanged. Repairedcallback29845021 advanced;
firstlong GPU29846090 RUN since07:44:15 n-62-18-12,callback29846091PENDnormal.
Latestpersistedcase001state262/10000at09:45:02,latestobservation260at09:44:16;
cases002/003at20pilot while serialcase001runs. Savedcountsnotexactlivecount.
Every10actualrelaxed/hardobservations continue. Latestconditionedcase001valid,
identity/hardscoreunchanged,bestvalidstill0; nobindinggain or completedtarget.
Alloriginalobjective/caps/validity/runtime/quota/stallchecks retained. Original
VAE36/mapc8d865.../cal1c4366... unchanged bypilot; no VAEcontinuationmodification.
Read-onlycollectionchangednocode/settings/jobs/VPN/SSH. Savedpilotproof valid
whenliveschedulerrecordagedout. Docs82/docs81/ledger and source143a002
pilot-running-status001.json bind exactsnapshot/progress/frozenreceipts.

**Source143 handoff repaired; pilot resubmitted,28 October2026 20:58CET:** user
explicitly authorized fix/resubmit using trained mapping then10000/case ifchecks
pass, VAEcontinuationunchanged. Originalfit29839755 outerLSF/bhist DONE16:39:52
nowverified; realagedoutbjobs returns0 with exactnotfound ERROR, unlikeoldassumption.
Newcontrol-only143a002 archive6e851def4a043df774bb259fc5a16c84ca9faada85c4616c99d65b5c6a092d41,
manifest57469430d6e13a8708fb593d5433b454a867cd0109c62a8355dfde3d7107077f,
planfe14de14231cb395bcb0781419c4d2b536772a82f9cde5c44ee097768cb0571c.
CPU29845008 DONE:85passed/zero failureserrorsskips; actualoriginalpilotCLI
pre-modeldispatch passed, authenticCPUtorch/numpyimports but CUDAhidden/cached
false/noinitialization/no model loads/no parent writes/submissions/updates.
Readiness2470071046d191334a8e89bc1de63c203ae8614609c399d83d9f68736de07674.
GPU pilot29845020 and callback29845021 submitted20:57CET, PEND20:58 waiting
processor-affinity units1host; callbackexpectedended dependency. No nativegate/
pilotpass/updateyet. Allfuture callbacks route143a002authhelper, original142
scientificcwd/source/manifest/plan/controls/worker/mapc8d865.../calibration1c4366...
remain exact; no refit or modifiedscientificthreshold. Full10000/case serial
continuation and observations every10 conditionalonunchangednative/derivative/
validity/consistency/progress/runtime/quota checks; stallsstopincomplete.
143a001 CPU29844977 had75fixturepasses but actualpreflightfailedoneagerCPU
Torchimport; failedimmutable release/logs retainedunpromoted, fixedonlyin002v2.
Originalcallbackmismatchcauseunproven(rawnotretained); lateragedoutbugdemonstrated.
Source140 VAE/source141 recovery untouched. Docs82/docs81/ledger and
outputs/hpc-results/source143-embedding-handoff-recovery-attempt002/submission-summary001.json.

**Source142 fit worker passed; pilot handoff stopped, 28 October2026 evening CET:**
fit29839755 saved exact-source execution completed/return0 at16:39:52CET,
all5000 updates, best4800; same-new-dev loss0.05108844->0.01481416(-71.0%),
audit0.01519852,1605/1653rows meet all10 fixed caps. Audit635distinctidentities,
neutralcharge-only/noP/I; no useful-gradient or chemical-improvement result.
VAE/nativebank/oldbridge unchanged. Newbridge c8d86551849c7f7e1de5218b7121cbecacb7f9dbc9ee11a5aea5972187a81551,
calibration1c436618bc6dc26dfb049908b11477afc5f2e7b291132db4577a99631f0c69f8.
Protect active fitted inputs. Callback29839756 EXIT1 at16:39:57 on exact-nameDONE
scheduler check before transition/pilot intent; original raw bjobs response not
saved, cause not established. Original sealed advance-after-fit retry stopped
on scheduler-status-unavailable before submission; no waiver/source/settings
changes. Subsequent readonly terminal-log/bjobs attempts SSHtimeout/reset with
no remote stdout. Savedworker pass is not independently outerLSFDONE proof.
No pilot/optimization/newjobs or source140/source141 change in this check.
Preserve failures; resolve actual scheduler evidence and retain scientificgates
before resuming; do not repeat goodfit merely to bypass handoff. Docs81 and
outputs/hpc-results/source142-epoch36-guarded-embedding-long-attempt001/status-summary002.json.

**Read-only actual VAE training/native recovery passed; long-fit handoff failed,
28 October2026 20:22 CET:** source140attempt002 GPU29838765 RUN since16:20:45
on n-62-18-9; all649 current CUDAfixtures passed105.06s,zero failures/errors/
skips. Exact40 model/AdamW/RNG/frozen36 handoff passed. Actualtrainer created
16:27:12; latestsavedpanel20:14:23 atabsolute265956 is2116/6596 (~32%) into41,
not a completedcheckpoint/livecounter. Raw/strict754/1024 (73.63%) vs40's753
(73.54%); exact/validrecon256/256,sulfur1.42309%. Fixedpanels fluctuate and
precedingpanel had1representationfailure; no epoch41completion. Original40
latestcompleted,serialcallback29838766PENDnormally,endpoint45/loss/1:2/guards
unchanged. Source140snapshot011 bindsactualRUNandhealthreceipts.
Source141recovery29838737 DONE15:33:57--15:47:33CET n-62-18-10; terminal
passed_native_recovery,6/6 ordinarypredictions+identityaudits passed,ownership
probe/coordinatorCUDAfree passed,zero failed/ineligible/unattempted. Valid
relaxed001/003 unchangedfromstarts,freshnativeaffinitydelta0; relaxed002 and
all3learnedterminal/best invalid. No bindinggain/superiority; no optimization
rerun. Use terminalsummary/result,not stale5-complete progress. Collect006/docs78.
Source142fit29839755 DONE16:23:53--16:39:52CET n-62-18-12;5000bridgeupdates,
dev0.0510884->0.0148142,best4800,calibrationpassed,audit1605/1653withincaps,
VAE/bankfrozen. Callback29839756 EXIT1 at16:39:57 exact-nameDONE scheduler
guard BEFOREpilotdispatch. Actualschedulerfieldsnotlogged,specificmismatch
unestablished; no pilot/native/longoptimizationpass. Savedfitpreserved.
Status1793215369272490300/callback-inspection1793215376515856900/docs81 bind
evidence. Read-onlystatuschangedno runtime/settings/jobs/VPN/SSH config.

**Read-only all-three queued status, 28 October2026 13:58 CET:** source140
attempt002 GPU29838765 PEND at13:57:59,12of16CPUslots reserved n-62-18-8,
processor-affinity units unavailable1host,14944eligible/0ineligible seconds.
Scheduler's provisional no-sooner-than28Oct21:19:19CET is not a guaranteed
start; callback29838766 waits normally. Source141 recovery29838737 PEND
at13:57:48,4slotsreserved samehost,affinityunits1host,15220eligible/0ineligible.
Source142 fit29839755/callback29839756 PEND at13:58,4slotsreserved n-62-18-11,
affinityunits1host,8982eligible/0ineligible; callback expected ended dependency.
No newGPUartifacts/ownership/native/fit/calibration/pilot/optimization result
or VAEepoch41. Original40 remains latest. Saved CPU gates2041/59/450 retain
their exact recorded skip/profile/authentication limits; aged-out live CPU
records are not failed GPU jobs. Read-only source140snapshot010/source141
collect005/source142status1793192294889844300 bind evidence. No code/settings/
jobs/VPN/SSH changes. Docs80/78/81 and fix-resubmit-20261028-status003.json.

**Guarded long embedding experiment authorized/CPU passed, 28 October2026 11:29 CET:**
user clarified10,000 iterations per each of the same3 original36 starts (30,000
planned total), observe actual relaxed/hard graphs every10. Independent source142
attempt001 archive d2ac85ad644a94f3b5967537f1c777bb7318faf2a8507038888bc32728a0d805,
manifest89dd0f43752a417cb4a4ab0658a4fba294eba72eb65fd5899036a42d8207c4f2,
plan35adaf248e9011ce81a5bcd165d1897e97901864695a10d56fad70eebe63f70e.
CPU29839742/callback29839743 DONE:450 passed,zero failures/errors/skips,
original36 frozen decode parity and exact starts passed. GPU fit29839755 and
callback29839756 PEND at11:29CET,4slots/exclusive80GB A100/4h,3h50 internal;
reservation n-62-18-12 is not execution. New5000-update warm bridge fit requires
dev improvement; disjoint new dev q99 per-head MSE caps fixed before audit/starts.
Candidates require strict conditioned validity+all caps before learned Boltz.
Same target/pose/occupancy/objective/LR400/clip1/radius3/best_trial_v2/16trials;
common-hard-score best-valid retention, atomic state/RNG/receipt resume,
25consecutive noops =>scientific stall/incomplete, no automatic gate loosening.
All3 native+6 fresh derivative/production gates and20 pilot iterations per case
must pass before conditional round-robin24h/23h serial allocations. Final native
baseline/terminal/best-valid seeds17/23 is separate, max18requests; baseline smoke
counted once. CPU-free coordinators and separate CUDA children retained.
No new fit/calibration/pilot/native pass/latent update established. Source140 VAE
continuation and source141 native recovery remain unchanged. See docs81 and
outputs/hpc-results/source142-epoch36-guarded-embedding-long-attempt001/submission-summary001.json.

**Read-only queued status, 28 October2026 10:39:55 CET:** source140 attempt002
GPU29838765 remains PEND,16CPUslots reserved n-62-18-11,job-slot limits4hosts,
3056eligible/0ineligible seconds at10:39:51; callback29838766 waits expected
ended dependency. Source141 recovery29838737 remains PEND,4CPUslots reserved
samehost,processor-affinity units unavailable on1host,3348eligible/0ineligible
seconds at10:39:55. Neither has GPU artifacts,ownership/native results or
new training/epoch41; original40 remains latest. Prior2041/120CPU-onlyskip
VAE gate/profile/adoption/CLI and59/0skip recovery CPU/auth passes remain
valid. No reliable startETA. New source140 snapshot008 and source141 collect004
bind read-only evidence; no code/settings/jobs/VPN/SSH configuration changed.
See docs80/docs78/ledger and fix-resubmit-20261028-status002.json.

**Process/interface fixes CPU validated and resubmitted, 28 October2026 09:51 CET:**
user authorized fixing/resubmitting both failed paths. Immutable source140
epoch40->45 attempt002 archive375272634d1836625e3d9a2bb8548e93f4cf4ac982ad1853fd0c621af8a344ef,
manifestac6b06edeff9ce6d35413d84890ee3738a68f517230ac06bd6345f6f86ca4d65,
plana794d3f7a9b3bcbcdc357aadb6feade06806a4692b08a65e7c3a0e8e0a9420f6.
Observer/callback normalize str paths; CPU-only worker coordinates separate
baseline/trainer/saved-observer CUDA children, with actual two-child ownership
probes after CUDA pytest exits. Shared cuda_process helper guards context
ownership and cleans owned child sessions. All loss/trainer/runtime bytes and
scientific plan remain identical to failed140a001; original35/36 anchors,
exact40 model/moments/RNG,1:2 and every check/45 endpoint unchanged.
CPU29838695/callback29838696 DONE:2041 passed,120 CUDA-only skips,zero failures/
errors; repair profile/adoption/actual generated CLI passed. GPU29838765 and
serial callback29838766 submitted; snapshot00709:50:38CET both PEND normally,
GPU waits job-slot limits5hosts, one exclusive80GB A100/16slots/24h/23h internal.
No new CUDA/ownership/handoff/generation/training proof or epoch41 yet.

Separate source141 native-recovery attempt002 archive531c4113ba55148ac6e4549bc73cdfc4231f84663727e720f382f61892da9800,
planfaca1ab5af193bd3a4cfc8f035c3e82a072b64528d0adbc24d04308a7c080523.
CPU29838721 DONE:59 passed,zero skips/errors/failures and exact parent CLI
authentication; GPU29838737 submitted6h/5h internal/exclusive80GB A100/4slots,
PEND on processor-affinity units onehost, no GPU artifacts/ownership/native pass.
It reuses completed006 selection/gates/checkpoints and retries exactly6 saved
native requests; no300-update optimization rerun. CUDA-free coordinator,
tiny CUDA child probe and first actual native success gate precede remaining5.
Future full comparisons also use a separate optimization child, but are not
rerun here. Source141a001 CPU52-pass evidence is preserved/unpromoted after
peer review added exact validation-receipt binding in the GPU adapter; no GPU
was submitted for it. Failed old releases and all checks remain intact.
Docs80/docs78/ledger and outputs/hpc-results/fix-resubmit-20261028-summary.json
bind actual receipts. GPU checks are conditional/pending, not reported passed.

**Read-only terminal status, 28 October2026 09:12 CET:** source140 GPU29829875
ran02:44:35--02:49:37 CET on n-62-18-11; all631 mandatory CUDA fixtures passed
(zero errors/failures/skips,96.51s), then startup failed in the additive epoch40
generation observer: CHECKPOINT is str, observe_checkpoint passes it directly
to file_sha256 requiring Path, raising AttributeError ('str' has no attribute
'open'). No training segment/epoch41/update exists; authentic40 remains latest.
Callback29829876 failed on absent segment completion and submitted no successor.
Source140 snapshot009 and docs80 bind evidence; preserve failed release.
Independent embedding attempt006 GPU29829828 successfully completed27Oct
17:44:25--21:48:58 CET: all3 native oracles/6 pointwise start derivative gates
passed, 6x50 updates completed, frozen VAE/bridge/context state unchanged.
Relaxed terminal graphs valid2/3 (both unchanged identities), learned bridge0/3
(sanitization failures). All6 fresh-native requests failed CUDA busy/unavailable;
no binding gain or superiority is confirmed. Full untruncated gate files and
HTML report were downloaded with remote SHA/size checks to terminal-gpu-20261028.
See docs78 and status-1793174972681581400.json. This status check changed no
runtime code, scientific settings or jobs and submitted no new numerical work.

**Verified model offload and retired HOME repository removal, 27 October 2026:**
131 inactive checkpoint payloads / 323 hard-linked aliases were preserved under
`C:/Users/nikol/HPC_Archives/DBMol_exploratory/2026-10-27/cold-models-history006`
with full SHA verification and `restore-map.json`, then removed from HPC runs at
15:35:35 CET: 6,392,482,304 allocated bytes / 5.953463 GiB. The separate retired
HOME `projects/relative-weight-models` was fully archived locally and removed
at 14:42:54 CET, reclaiming a conservative 949,370,880 bytes / 0.884171 GiB.
Both independent absence/integrity checks passed; all 138 protected files
including original35/36, completed40, bridge/contexts and active VAE/embedding
inputs remain unchanged. Only latentmol/Mosaic project roots remained in the
checked locations at 15:39:36 CET. That last verified WORK3 quota was
39.84/100 GiB used, 60.16 free; it is a timestamped observation, not current
usage. Final category reporting hit a timestamp collision, then SSH reset/auth
failure; these do not undo the completed removals. Preserve failed receipts.
The category table is an earlier snapshot adjusted for exact removals, not a
fresh du. No jobs, scientific settings, VPN or SSH configuration changed.
Historical offloaded checkpoints require restoration from the local map before
future replay. Docs50 and `outputs/hpc-cleanup/20261027-history-offload/summary.json`
bind verified backups, exact removals and partial reporting provenance.

**Source140 epoch40->45 CPU passed; GPU queued, 27 October2026 12:18:50 CET:**
immutable attempt001 archive3fd30b055b078a44c247779966e6c353cf6b66279b9740655e70294766c695f8,
manifest514a5c5c521510eb09556861542d8733ba91b13afb3a7491c397fec58bd0ba86,
plan934764491608b7f354bf5d47aec0f72f014a66d845f3c3ff76589d71c2ee3fb8;
655 members/583 scientific. CPU29829798/callback29829799 DONE;2028 fixtures
passed,115 CUDA-only skips,zero failures/errors,full repair profile and
authenticated source135/source138 adoption passed. Exact generated CLI
dispatch passed with epoch40 resume SHA31af05597dbd176b02a8108651db0d6b33bb3a8883f69ecf354a4465ed18787e
and endpoint45. Harness start_epoch36 is preserved origin/reference ancestry,
not the operational resume. GPU29829875/callback29829876 PEND,job-slot limits
on five hosts;one exclusive80GB A100/16slots/24h/23h internal. No current CUDA,
native handoff,fresh generation result or epoch41 training yet;no reliable ETA.
All loss/1:2/anchors/AdamW/RNG/scientific+health+runtime+resume+quota checks
unchanged;fresh observations additive,all epochs saved,serial stop45. Source138
and independent studies preserved. Docs80/snapshot008/submission-summary001.

**Epoch40->45 continuation authorized, 27 October2026:** user requested five
more epochs through absolute45 if checks pass, unchanged loss/safeguards,
every epoch saved and validity/diversity tracked. Prepare a new immutable
source140 campaign from exact completed source138 epoch40 SHA
31af05597dbd176b02a8108651db0d6b33bb3a8883f69ecf354a4465ed18787e;
do not rewrite completed source138. Preserve original35 occupancy/ramp,
original36 valid-target reference, 1:2/fullNLL/alpha0.1/composition/search,
quarter actual AdamW/moments and native/fresh RNG, every existing CPU/profile/
CUDA/native/health/runtime/resume/quota check. Add closed epoch45_extension
metadata and fresh saved-checkpoint observations at40--45: two1024-prior
panels/four categorical draws, raw/strict validity, uniqueness/fingerprint/
scaffold diversity, all invalid graphs retained. No new scientific threshold
or changed training objective. Numerical checks only allocated HPC. Preflight
12:03:39 CET matched all protected hashes; quota46.07/100GiB. Preparation
is not a passed gate or submission/start. See docs80.

**Source138 epoch40 completed; branch finished, 27 October2026 11:44:59 CET:**
read-only snapshot018 confirms GPU29822674 DONE11:34:10/callback29822675
DONE11:34:39, passed segment completion and `completed_bounded_branch`.
Checkpoint40 SHA31af05597dbd176b02a8108651db0d6b33bb3a8883f69ecf354a4465ed18787e,
absolute263840. Measured epoch41,316.285s (~11.48h); completion59,011.392s
is conservative estimate. Final raw/strict validity753/1,024 (73.54%) versus
epoch39's724 (70.70%), +29/+2.83points; original36 baseline641 (62.60%).
Exact/valid recon256/256, sulfur1.4394%; all268 early panels/checks and19
end-epoch checks passed. Endpoint panel has zero representation failures,
but prior occasional failures remain. Repeated fixed-panel monitoring is
not fresh-seed/categorical/full-test confirmation. Original36->40 branch
is complete, no successor or further evaluation submitted; no authorization
beyond40. CUDA599/science/guards unchanged; no runtime/settings/jobs changed
by check. Docs75/snapshot018/training-status007.json.

**Source138 epoch40 running, 27 October 2026 00:17:07 CET:** read-only
snapshot017 confirms GPU29822674 RUN since00:01:55 on n-62-18-8; actual run
created00:02:30 restores exact checkpoint39 SHA
e5d7f99b9eb0fe04cd8b6b800e4bd90b6f1c24a65368216ce384a95061a7f3b0.
Latest saved panel00:16:41 is absolute257356,112/6,596 (~1.70%) intoepoch40,
not an exact live count. Raw validity744/1,024 (72.66%) versus epoch39's724
(70.70%); strict743 (72.56%) with one representation failure. Exact/valid
reconstruction256/256, sulfur1.5048%; all203 saved panels/checks passed.
Early fixed-panel improvement is not a completed checkpoint/full-test result.
Epoch39 latest completed; callback29822675 PEND normally, endpoint40 and
CUDA599/science/guards unchanged. No runtime/settings/jobs changed or new
experiment submitted. Docs75/snapshot017/training-status006.json.

**Source138 epoch39 complete; final epoch40 queued, 26 October 2026 22:39:53 CET:**
read-only snapshot016 confirms segment003 successfully completed at 16:50:10,
checkpoint39 SHA e5d7f99b9eb0fe04cd8b6b800e4bd90b6f1c24a65368216ce384a95061a7f3b0,
absolute update257244. Measured epoch39 42,354.638s (~11.77h); normal guarded
whole-epoch budget stop, not failure. End raw/strict validity724/1,024 (70.70%)
versus baseline641 (62.60%), equal aggregate raw count to epoch38 and below
earlier intermediate763 (74.51%). Do not report the intermediate value as the
completed checkpoint result. Exact/valid recon256/256, sulfur1.4237%; all201
saved early panels/checks and end-epoch checks passed. Auto callback submitted
GPU29822674/callback29822675 for exact checkpoint39 resume through40. Both PEND;
GPU waits for processor units/affinity on one host, four of16 slots reserved
on n-62-18-8. Reservation is not execution; no epoch40 run directory or reliable
ETA. CUDA599 and science/guards unchanged; no runtime/settings/jobs changed by
status check. Docs75/snapshot016/training-status005.json. Fixed-panel counts
fluctuate; no fresh-seed/full-test result or authorization beyond40.

**Embedding comparison attempt006 CPU passed; GPU queued, 27 October 12:11:25 CET:**
CPU 29829824 DONE 12:10:00–12:10:50 on n-62-11-63, full runner 41.6918s,
all 283 fixtures passed/zero failures/errors/skips under explicit CPU-only
mkl_compatible_four_threads_v1, with original36 default decoder bitwise parity
and unchanged frozen state. Three exact parent z/masks/dtypes/records/hard
identities passed; starts.pt SHA 832592de7183b51ac22542ead558bc453006d3d349d942e2933dd1a52fb40ebd.
Final-policy FP64/cast-back maximum probability change 1.7881393432617188e-7;
this is CPU readiness, not a native pass. GPU 29829828 submitted 12:11:13,
one exclusive 80GB A100/four slots/24h, internal 23h; PEND at 12:11:25, new scheduling,
no GPU artifacts/native gates/optimization/ETA. Full fresh gates remain mandatory.
Attempt005's 266/268 inherited exactness failures and kernel baseline controls
remain preserved; no assertion or tolerance was weakened. Independent VAE/epoch45
work is unchanged by this comparison. See docs78 for exact seals/receipts.

**Embedding diagnostic access recovered, 26 October 2026 22:39:48 CET:**
fresh successful snapshot `status-1793050788490890700.json` confirms GPU
29821249 PEND, four CPU slots reserved on n-62-18-8, waiting for physical
GPU availability. No GPU artifacts, logs or native diagnostic result exists;
no reliable ETA. Saved full CPU pass/223 fixtures/original36 parity remain
valid despite aged-out CPU scheduler record. Numerical correction/full
comparison retry still await native evidence; authorization remains active.
Read-only check changed no runtime code, settings, jobs or VAE continuation.
See docs78.

**Source138 epoch38 complete; epoch39 running, 26 October 2026 12:33:25 CET:**
read-only snapshot015 confirms segment002 completed successfully25 October
21:37:32 CET, checkpoint38 SHA
1736ad792f244fda59d0b74dc87867880fe632db0417ddfb0c70d07269fd319d.
Measured epoch38 was42,905.556s (~11.92h); completion's59,011.392s remains
the guarded estimate, not measured time. Automatic successor GPU29817247 RUN
since26 October05:02:28 CET on n-62-18-9, exact checkpoint38 restored;
callback29817248 PEND normally. Last saved panel12:23:50 CET at absolute254756
is +17,300 sinceoriginal36/4,108 of6,596 (~62%) intoepoch39. Latest raw/strict
validity763/1,024 (74.51%) versus baseline641 (62.60%), exact/valid recon256/256,
sulfur1.5057% versus1.5070%. All176 saved early panels/checks passed; preceding
panel254656 had raw737/strict736 and one representation failure, so do not
claim zero failures across all graphs/panels. Epoch38 is latest completed
checkpoint; epoch39 not complete. CUDA599, science/guards and endpoint40
unchanged. No runtime/settings/jobs changed by check. Docs75/snapshot015/
training-status004.json; fixed-panel fluctuations are not full-test guarantees.

**Source138 epoch38 progress, 25 October 2026 16:29:02 CET:** read-only
snapshot014 confirms GPU 29809056 still RUN, callback 29809057 PEND normally.
Latest saved panel at 16:20:42 CET is absolute247556: +10,100 since original36,
3,504/6,596 updates (~53%) into epoch38, not an exact live update count.
Raw/strict fixed-prior validity727/1,024 (71.00%) versus original641 (62.60%);
exact/valid reconstruction256/256, latest sulfur1.4730% versus1.5070% baseline.
All103 saved early-health panels and their checks passed. No epoch38 completion;
epoch37 remains latest completed checkpoint. CUDA599 passes, loss/ratio/AdamW/
RNG/guards and endpoint40 unchanged. Fixed panels fluctuate and are not an
independent/full-test result. No runtime code/settings/jobs changed by check.
See docs75, snapshot014 and training-status003.json.

**Source138 epoch37 completed; epoch38 automatically running, 25 October 2026
11:39:11 CET (DTU-local):** read-only snapshot013 confirms segment001 completed
successfully at 24 October 23:43:24 CEST, latest completed checkpoint epoch37,
SHA eb6ecd25c36cff6fb158e2d24aa1888fdc17e8c8654bb100894d834510839446.
Epoch37 processed 1,688,539 real/3,377,078 fresh-prior points in 6,596 updates;
actual epoch row duration 48,502.236s (~13.47h). Completion.epoch_seconds
59,011.392s is the guarded scheduling estimate, not measured training duration.
The deliberate whole-epoch budget stop passed, and the serial callback queued
GPU29809056/callback29809057. GPU29809056 RUN since25 October09:40:30 CET on
n-62-18-8, exact epoch37 checkpoint restored; branch still ends40. Last saved
panel +7,500 sinceoriginal36/904 intoepoch38 (absolute244956), saved11:32:03 CET,
all18 checks passed; all77 saved early-health panels/checks pass so far.
Epoch37-end and latest fixed raw/strict validity705/1,024 (68.85%) versus
baseline641/1,024 (62.60%); exact/valid reconstruction256/256. Latest sulfur
1.4696% versus1.5070% baseline. Metrics fluctuate; these are fixed monitoring
panels, not independent or full-test guarantees. CUDA599 passes remain valid.
Endpoint/loss/ratio/optimizer/RNG/checks unchanged. No runtime code/settings/
jobs changed by this status check; source136 and independent work preserved.
See docs75, source138 snapshot013 and training-status002.json.

**Source138 CUDA passed and actual training running, 24 October 2026 10:55:46
DTU-local:** read-only snapshot012 supersedes the queued snapshots. GPU
29802716 RUN since 10:11:51 on n-62-18-10; all 599 mandatory current-release
CUDA fixtures passed, zero failures/errors/skips, 76.303s. CUDA gate SHA
adf8b3bbf47bf6b5c3afbec0122929d03cbff3d01357f1a2d772d0ee16f8b504.
Actual training record created 10:13:49, authenticated original36 checkpoint
3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f,
endpoint40. Epoch37 is in progress. Last persisted panel is +300 updates
(absolute237756), saved 10:51:32; all 18 health checks passed at +32/100/200/300.
Fixed raw/strict validity 641 baseline -> 642/648/659/666 of1024; exact/valid
reconstruction remains256/256, zero representation failures in these four
panels, sulfur1.5070% ->1.4665%. These are early actual training panels, not
a finished epoch or an independent-seed replication of the earlier proof.
Original36 is still the latest completed checkpoint; no epoch37 completion.
Callback29802717 remains PEND normally. No runtime code/settings/jobs changed.
See docs75, snapshot012 and training-status001.json.

**Embedding diagnostic completed; precision candidate selected, 27 October 2026:**
GPU 29821249 DONE, ran 26 Oct 23:31:48–23:34:47 CET on n-62-18-12. Exact parent
scores/gradient/direction, cut/full primals and complete captured-input VJP
tensor equality passed. The reproduced total local duality error is 3.477208e-6
in final graph policy versus 4.64198e-10 decoder/5.79268e-15 native adapter.
Final-policy-only FP64 then cast-back fixed-cotangent control reduced error to
5.07137e-10; no native score at its changed primal was run, so this is not a
native correction pass. Select that smallest comparison-only precision candidate;
shared VAE36/default training/decoding, bridge/native models and continuation
remain unchanged by this work. Preserve same three starts/settings/tolerances
and all failed evidence; fresh complete native/derivative/production gates remain
mandatory before optimization. No comparison004 seal/submission or native fix
pass yet. Verified terminal raw files/integrity and exact measurements: docs78.

**Embedding diagnostic access limited, 26 October 2026:** last successful
snapshot status-1793019169106562100.json at 13:52:49 CET records GPU29821249
PEND waiting for ngpus_physical on one host, no GPU artifacts/result. Two later
normal collections failed SSH255/timeouts to login1; verified alternate login2
also timed out. These access failures do not establish GPU failure or a new job
state. Official host-key verification used task-local known_hosts only; global
SSH configuration was unchanged and authentication was not bypassed. Correction/
resubmission authorization remains active: read native diagnostic evidence,
implement the smallest demonstrated correction and rerun full gates. No numerical
fix or comparison004 is selected yet. Preserve all failures and independent VAE
continuation. Exact access receipts and last successful status are linked in docs78.

**Embedding gradient diagnostic CPU passed; GPU queued, 26 October 13:50:06 CET:**
`source139-epoch36-embedding-gradient-diagnostic-attempt001` CPU 29821224 DONE
13:47:52–13:48:43 on n-62-30-23, all 223 fixtures passed, zero failures/errors/skips.
Full stage41.000s; original36 parity/frozen state and all three parent starts
passed. CPU decoder-only probe error1.5692636872219985e-8 was within inherited
bound1.1245987208140518e-6 using deterministic engineering cotangents, not native
Boltz cotangents; this is diagnostic only, not a native gate pass or selected fix.
GPU 29821249 submitted13:49:29, one exclusive80GB A100/four slots/1h, internal55m;
13:50:06 snapshot status-1793019006060069300.json confirms PEND waiting for
ngpus_physical on one host, no reliable ETA or GPU artifacts/result.
Archive bd677171c8890c888191e1280ed62c8b84bb622d6b8e8b0f13cdbe5656ac8740;
manifest 90f744e73a71dc79a9165c2db33353c096d0484bde5398ba7c95386eda547ff9;
plan 517d1086d9eb69c9ea101cb1125f8d87e31ee2341fc6e053b5e989467ac3df7a.
Static comparison preserves protected VAE/native/bridge/production optimizer;
the broad archive also includes unused concurrent Protenix changes. No native
pass, selected fix, optimization or comparison004 submission follows. Independent
VAE continuation and all failures remain unchanged. See docs78 and exact receipts.

**Matched embedding correction/resubmission authorized, 26 October 2026:**
the user requested “fix it and resubmit”. Prepare new immutable
`source139-epoch36-embedding-gradient-diagnostic-attempt001`: allocated CPU
readiness, then one-hour GPU/55-minute internal diagnostic, splitting the full
derivative at decoder logits, relaxed graph and native chemistry. No optimization
runs in this diagnostic. Preserve original36, trained bridge, exact starts,
objective/budgets/tolerances and all failed evidence; no automatic gate weakening
or start removal. No root cause/correction/pass/submission follows from this
authorization. Corrected comparison attempt004 is not yet selected/submitted;
independent VAE continuation remains unchanged. See docs78.

**Matched embedding comparison terminal status, collected 26 October 12:52 CET:**
GPU 29802743 ran 25 October 06:33:02–06:48:42 CET on n-62-18-12, then EXIT1;
saved LSF output establishes the exit despite the aged-out live job record.
All three native-oracle gates passed; cases001/002 passed both relaxed/learned
start gates and exact production bindings. Case003 RELAXED failed independent
forward/reverse AD in the gradient direction: affinity error 2.5204821007445213e-6
exceeded allowed 2.0817236646993208e-6; total 3.4778529073341627e-6 exceeded
2.583432959841326e-6. All three random directions and frozen/context checks
passed. Generic “Learned-start” error is not the path identity: scoring_path is
relaxed; case003 learned never ran. No optimization/fresh-native directory or
updates exist, no molecular comparison/root cause/fix is established. Preserve
all failed evidence and exact terminal gate copies/integrity in docs78. This
status changes no runtime code, settings, jobs or independent VAE continuation.

**Matched embedding comparison status, 24 October 2026 10:54 CEST:** fresh
attempt003 snapshot `status-1792832067824362000.json` confirms GPU 29802743
still PEND, job-slot limits on four hosts, 47,901 eligible seconds/zero
ineligible. No GPU artifacts, native derivative gate or optimization start;
no reliable ETA. Saved CPU full-stage pass, 212 fixture passes and exact
shared starts remain valid. This read-only check changed no runtime code,
settings or jobs and left the independent VAE continuation unchanged. Docs78.

**Source138 read-only status, 24 October 2026 01:30:28 DTU-local:** snapshot010
confirms GPU 29802716 still PEND, job-slot limits on five hosts, 14,365 seconds
eligible/zero ineligible. Callback 29802717 is PEND on ended(29802716).
No current-release CUDA gate/JUnit, GPU log, segment directory, actual setup
or training progress exists; original epoch36 remains latest. No reliable
start ETA. Earlier CPU/callback/targeted scheduler records have aged out;
their saved DONE logs, 1,998 CPU passes/profile/adoption and both startup
dispatch passes remain valid, not new failures. No jobs/settings/code changed.
See docs75 and source138 snapshot010.

**Matched embedding comparison status, 24 October 2026 01:30 CEST:** fresh
read-only attempt003 snapshot `status-1792798218154998400.json` confirms GPU
29802743 remains PEND, job-slot limits on five hosts, 14,051 eligible seconds
and zero ineligible seconds. No GPU artifacts, native derivative result or
optimization start exists; no reliable ETA. CPU scheduler record aged out,
while saved full-stage pass, 212 fixture passes and authenticated shared starts
remain valid. This check changed no runtime code, settings or jobs and left
the independent VAE continuation unchanged. See docs78.

**Source138 final read-only check, 23 October 2026 21:33:51 DTU-local:**
GPU 29802716 PEND due to job-slot limits on five hosts; callback 29802717
PEND on its expected dependency. Existing full CPU/profile/adoption and two
actual-command startup passes remain valid. No current CUDA launch artifact,
training setup or epoch37; original36 remains latest. Snapshot009/docs75.

**VAE CLI fix passed and continuation resubmitted, 23 October 2026 21:31:**
source138 CPU29802660 and callback29802661 DONE. Full CPU gate/profile and
complete source135 adoption passed, with1,998 fixture passes/113 CUDA-only
skips/zero failures/errors. Allocated actual generated-command checks passed
both targeted29802690 (81 fixture passes) and production callback29802661,
with identical argv and zero updates before setup. GPU29802716 submitted
21:31:04, one exclusive80GB A100/16slots/24h, internal23h; serial callback
29802717 PEND on ended(29802716). GPU PEND waiting for scheduling at21:31:21;
no CUDA-launch pass, actual training setup or epoch37 yet. Original36 through
absolute40 only, all loss/1:2/optimizer/native+fresh RNG/checks unchanged.
Source136 remains failed/preserved. See docs75/source138 snapshot008 and
startup-fix-submission-summary001.json; submission is not a training start.

**VAE source138 actual-command startup pass, 23 October 2026 21:26 DTU-local:**
new source138 staged/authenticated648 members/577 scientific members, immutable
plan `183abba4bcd8e185b407ccdb5596a8fd6c37c0f9a82aeaf7c21e28a9cdb219c4`.
Full CPU29802660 RUN since21:23:27 (16 slots), callback29802661 PEND.
Independent two-slot allocated startup job29802690 DONE21:25:50:81 targeted
fixtures passed, no errors/failures/skips, and exact first-segment argv extracted
from sealed Campaign.advance passed the real CLI/plan/readiness dispatch.
Setup was intentionally not entered; updates=0. Deterministic source135 proof
adoption is not a new native result or substitute for full CPU/profile/CUDA.
Callback repeats actual-command check before conditional GPU submission.
No training/GPU job or epoch37 established yet. Preserve failed136 and every
scientific/health/runtime/resume/quota guard; original36->40 scope unchanged.
See docs75 and source138 snapshot003/targeted receipt.

**Scoped VAE CLI startup correction, 23 October 2026:** the user authorized
fixing source136's startup argument rejection, testing the actual generated
launch command, and resubmitting original epoch36 through absolute epoch40.
Prepare a new immutable source138; preserve source136's failed release/jobs and
all source135 evidence. Only align main's readiness applicability with the
existing launcher/downstream guard (failure guidance OR short continuation OR
valid-reference continuation). The source-adoption audit permits exactly that
AST predicate change, not arbitrary main changes. Add real module-entrypoint
argv regression and an allocated actual-plan dispatch preflight before GPU
submission, stopping before model setup/updates. Keep every existing loss,
optimizer/RNG, ratio, proof/health/CUDA/runtime/resume/quota check and endpoint
unchanged. Local work is AST/bytes only; numerical fixtures remain allocated
HPC. Preparation/submission is not a passed gate or training start. See docs75.

**Matched embedding comparison CPU passed; GPU queued, 23 October 21:37:42 DTU-local:**
independent `source138-epoch36-embedding-comparison-attempt003` CPU 29802735 DONE
21:32:09–21:32:49, all 212 fixtures and full original36/shared-start preparation
passed, zero failures/errors/skips. Exact starting samples 0/3/6 have 25/19/19
occupied slots; starts.pt SHA 832592de7183b51ac22542ead558bc453006d3d349d942e2933dd1a52fb40ebd.
All 44 cold-offloaded source056 contexts were restored exactly; all 50 catalog
and original36/bridge/source136 protected hashes passed independent verification.
Attempt003 has unchanged 582-member manifest 4a598d1143f6afc180d18dea618cdc09a859f019a7fc042b9025fcb8fb8487dd
and plan 60f555eb6193b804d8446dc864a57c52662e4c146be424b8f7cc118f6d6ecd6c;
archive bccba5d6a762349b450baf03b24251e12994be598926ce8960f6e5869d55ce9b.
GPU 29802743 submitted 21:36:07: one exclusive 80GB A100/four slots/24h, internal 23h.
At 21:37:42 it is PEND, job-slot limits on five hosts, 96s eligible/0s ineligible,
with no GPU artifacts/native pass/optimization start and no reliable ETA.
Source136/independent VAE continuation jobs/settings/releases remain unchanged
by this study. Attempt001's two fixture Path failures and attempt002's passed
212 fixtures followed by missing-context start-preparation failure remain
preserved. See docs78 and saved attempt003 receipts.

**Matched embedding optimization authorized, 2026-10-23:** the user authorized
testing the trained source137 mapping against the original relaxed-molecule
path with the same frozen original epoch-36 VAE and starting latents. Keep
source136 continuation jobs/settings/releases unchanged. Independent source138
uses the first three distinct audit posterior-mean groups/decoded identities
in authenticated stored order, without score filtering or replacement. Each
pair fixes the same native context, mask and pose, full DBMol objective,
50-update budget, LR 400, gradient cap 1, radius 3 and best_trial_v2/16 trials.
Both start derivative gates and fresh native-oracle checks must pass before
trajectories. Preserve all latent/graph/validity/call/step receipts, then rebuild
all valid baseline/terminal/best identities with ordinary native inputs at
seeds 17/23. Fresh-pose affinity confirmation is separate from fixed-pose full
DBMol scores. No model weights are optimized. No pass or submitted job follows
from this authorization alone. See `docs/78_embedding_latent_comparison.md`.

**Embedding pilot terminal status, 23 October 2026:** saved terminal evidence
confirms GPU 29796624 completed successfully on n-62-18-10, 19:41:35–19:46:30
DTU-local. Native exact-oracle scalar/constrained-gradient parity passed, then
all 1,000 mapping updates completed with original epoch-36 VAE/Boltz frozen.
Audit element/bond-type feature identities were 99.9045%/99.9500% across 193
correlated samples/71 distinct decoded identities; charged categories have no
audit coverage. Bridge SHA256 is
`013834958d52f7aee999ca4c39496e14fc5333d69eb697d47433821f48290343`.
Fit remains trained_unvalidated; FD observations and one finite learned latent
gradient do not validate useful optimization. No optimization ran and source136
continuation was untouched. Preserve sealed attempts and terminal receipts in
docs76; the live job record has aged out, not failed.

**Independent frozen epoch-36 embedding prototype, 2026-10-23:** the user
authorized implementation, supervised mapping training and HPC submission on
`codex/epoch36-boltz-embedding-bridge`, while leaving the existing VAE
continuation unchanged. Use only authenticated original epoch 36, SHA256
`3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f`.
The new heads fit native early chemical projections from final decoder hidden
states; VAE/Boltz stay frozen and desired scores are never training labels.
This is an explicit experimental alternative to the relaxed-graph boundary,
with a common decoded hard identity, fixed occupancy/pose, independent native
oracle validation and diagnostic feature audits. No learned/native gradient,
chemical improvement or optimization approval follows from a supervised fit.
Do not change source136 jobs, settings, checkpoints or its immutable release.
Attempt001 CPU 29796597 passed 128 tests but is not promoted because its launch
manifest binding was corrected in a new release. Attempt002 CPU 29796619
passed 139 tests with zero failures/errors/skips and exact original-VAE output
parity. GPU 29796624 has been submitted for native checks then a bounded
1,000-update mapping pilot; submission is not a native pass or training start.
Preserve both attempts. See `docs/76_epoch36_boltz_embedding_bridge.md`.

The user-directed scope is **graph AE -> matched graph VAE -> frozen Boltz-2
latent optimization**. Read `docs/04_latent_pipeline.md` and the experiment ledger
before continuing. The older small-KIBA-predictor plan is historical context.

**VAE terminal launch outcome, 23 October 2026 21:02 DTU-local:** read-only
status017 confirms GPU 29792164 passed all 586 CUDA fixtures (zero skips,
failures/errors), then exited before actual training. Its CLI rejects the
readiness SHA because its predicate recognizes only `protocol.failure_guidance`,
while the launcher supplies it for `valid_reference_continuation`. Downstream
readiness already handles the marker. No actual setup, update, segment or
epoch37 exists; original epoch36 remains latest. Callback 29792165 exited on
predecessor EXIT and submitted no successor. Preserve this launch failure;
no fix, waiver, source136 rewrite/restart or replacement submission follows
from this status turn. A future correction requires a new immutable release
and targeted actual CLI coverage, with scientific settings/guards unchanged.
Docs75 and the ledger bind terminal evidence. The separate completed bridge
pilot and its limits remain in docs76; no runtime code, setting or job changed.

**Retired-repository cleanup completed, 23 October 2026 12:51 DTU-local:**
the user explicitly authorized deletion of old repositories. Exact inactive
`/work3/s234818/relative-weight-models` was removed, reclaiming 14.463106 GiB
plus its root's 512 bytes. Independent verification passed absence, durable
receipt/journal and all 51 protected checkpoint/source/proof hashes, including
original epoch 36, source136 and source137. Active latentmol/Mosaic and all
jobs/settings were untouched. Work3 quota is 61.04/100 GiB used, 38.96 free
at this observation. Keep intentional old-repo deletion distinct from offload
and concurrent Mosaic output changes. See docs50 and the cleanup ledger entry.

**Latest combined status, 23 October 2026 12:32 DTU-local:** read-only
`status-016.json` confirms VAE GPU 29792164 and callback 29792165 remain PEND,
with all 16 CPU slots reserved on n-62-18-10 and job-slot limits on four hosts.
No CUDA launch artifact, segment directory, training run or progress exists;
original epoch 36 remains latest. Independent bridge GPU 29796624 is also
PEND, all four CPU slots reserved on n-62-18-11, with no GPU logs/artifacts,
native check or fit. Both are eligible resource/core waits, without a reliable
ETA. Earlier CPU passes remain valid despite aged-out scheduler records.
The account has one RUN/14 PEND across projects; the RUN Mosaic L40S job is
separate. This read-only check changed no code, settings or jobs. See docs75/76.

**Latest source136 status, 23 October 2026 10:22 DTU-local:** read-only
`status-015.json` confirms GPU 29792164 still PEND, reporting job-slot
limits on four hosts, with eligible pending time and no dependency failure.
LSF has reserved all 16 CPU slots on n-62-18-10, superseding the prior 12-slot
reservation on another host. The previous provisional 24 October 04:19:11
timestamp is no longer displayed; no reliable new start estimate is established.
Reservation is not execution or a CUDA pass. The account-wide four-queued/
none-running count is the earlier 10:10 snapshot, not a fresh count here.
After-segment 29792165 is PEND. No CUDA launch gate/JUnit or actual training
run/progress exists yet. The completed CPU/callback scheduler entries have
aged out; their saved DONE logs and CPU/profile/adoption passes remain valid
historical evidence, not new failures. No code, settings or jobs changed.

**Scoped validated frozen-reference training authorization, 2026-10-22:** the
user explicitly authorized packaging/submitting the validated source135 loss
continuation from authentic original epoch 36 **through absolute epoch 40**.
This supersedes the historical epoch-39 cap only for the new versioned training
plan; do not edit old short/test policies or source135's validation-only seal.
Actual `source136-valid-reference-epoch40-continuation-attempt001` is sealed and
staged/authenticated: 646 members, 575 scientific members, 12 Python overlays.
Stage/queue returned zero and submitted CPU 29792116 / callback 29792117,
initially submitted_not_passed. At 22 October 19:49:43 DTU-local, CPU 29792116
and callback 29792117 are DONE, full CPU gate/repair profile/additive adoption
passed, with 1,985 unit passes/113 CUDA-only skips, zero failures/errors. Callback 29792117 submitted
GPU training segment 001 job 29792164 and after-segment 29792165; both are PEND.
The GPU waits for eligible resource/core availability on an 80GB host, not a
dependency failure; current saved provenance is source136 status-011.json.
No new CUDA launch checks or actual training start/epoch 37 is established yet.
The profile's inherited hardcoded 1:3 note is not the plan: the profile is a
four-real/32-prior diagnostic cohort; actual training remains 256 real/512 prior,
1:2. Do not modify the raw profile or call it native chemistry health.
Require additive
authenticated adoption of source135's completed native/probe/restoration/2,500
proof and all 26 health panels, new allocated controller contracts and fresh
launch checks. Preserve full unweighted repair, frozen original-36 same-z valid
targets/weight 0.1/all-attempt mean, frozen epoch-35 occupancy/ramp, 1:2 sampling,
quarter actual native AdamW displacement/moments and native/fresh RNG, other
weights/search/data and every scientific/health/runtime/resume/quota guard.
Actual training restores original epoch 36; proof weights stay discarded.
Serial allocations remain at most 24h/23h internal, with fixed early panels
after 32/every 100 updates and authenticated completion/health before any
successor. Stop at epoch 40, not 41 or the global 100 endpoint. No gate waiver
or training-start claim follows from authorization. Storage-cleanup receipts
and breakdown are separate; preserve earlier failures and independent Protenix
work. See `docs/75_frozen_reference_epoch40_continuation.md`.

The explicit new marker is `valid_reference_continuation={version:
"epoch36-valid-reference-training-v1",start_epoch:36,end_epoch:40,purpose:
"validated_reference_training"}`; `valid_reference_adoption` uses
`validated-frozen-reference-adoption-v1` and binds completed source135 proof
artifacts. CPU contracts/repair profile/adoption must pass before serial GPU
segment 1, which requires current-release CUDA/launch contracts with at least
518 cases and zero failures/errors/skips before actual training.
No passed launch gate or training start follows from staging. Preserve exact
source/manifest/plan hashes and operation receipts in docs75, sealed bytes and
all earlier source135/source134/Protenix evidence.

**Scoped frozen-valid-reference investigation, 2026-10-22:** the user authorized
a new source135 candidate from authenticated original epoch 36 after source134's
failure. Replace moving current-valid self targets with same-z strict-valid
targets from a separate frozen original epoch-36 model, using full unweighted NLL,
all-attempt prior mean and weight 0.1. Restore accepted-repair full NLL; remove
source134 target weighting and all current-self/iteration/probe fields. New
`valid_reference_test` binds version `epoch36-frozen-valid-anchor-test-v1`,
reference epoch 36 and the original epoch-36 checkpoint SHA; protocol
`valid_prior_reference_guidance` is `frozen-valid-prior-reference-v1`, weight 0.1.
`valid_reference_probe` uses `valid-reference-probe-v1`, 200 updates and gradient
indices 1/39/96/100/199/200. Keep the epoch-35 occupancy anchor, inherited moments/
native and fresh RNG, quarter step/search/1:2/other weights and all native/health/
runtime/resume/quota checks, probe/restoration/fresh 2,500-update proof and panels
at 32 and every 100 updates.
This is validation-only, no training/promotion or old source134 restart. Novelty constraints,
reference-invalid uncovered rows and incomplete causal attribution remain
explicit in `docs/74_frozen_valid_reference_guidance.md`. Docs73 is independent.

Actual `source135-epoch36-valid-reference-attempt001` is sealed and staged with
640 authenticated members; stage and queue returned zero. CPU 29783433 and
after-CPU 29783434 were submitted and are now DONE. At 22 October 12:27:05
DTU-local, the saved full CPU gate and repair profile are passed, with JUnit
2,030 cases: 1,917 passed, 113 CUDA-only skips, zero failures/errors (200.253s).
The callback submitted GPU 29783863, RUN since 12:26:55 on n-62-18-10, gpua100,
one exclusive 80GB A100/16 slots/24h. At 12:28:28 DTU-local, all 518 mandatory
CUDA fixtures passed without skips/failures/errors (63.999s). At 12:30:33,
the four new native reference counterfactual gates and eight restoration checks
passed, including exact current-self/reference production-step parity and an
invalid-live gradient/FD check. At 12:35:02, all 18 original native gates and
15 supplementary gates passed: matched eight-update raw validity 149→153/256,
exact/valid reconstruction 256/256, projected/conservative epoch estimates
47,613.124s/49,234.053s with runtime gates passed. The separate 1,024-point
baseline is 641/1,024. At 12:38:21, GPU remains RUN with 38/200 disposable probe
updates. Its first 32-update fixed panel passed all 18 health checks: raw/strict
641→642/1,024, exact/valid reconstruction 256/256, sulfur 1.5070→1.4917%.
Reference-covered/live-invalid row
attempts total 1,043 across 38 batches, not unique or paired-panel molecules.
At the later 13:06:10 DTU-local observation, the 200-update disposable probe
passed (1,513.430s), with raw/strict 641→642→648→659/1,024 at 0/32/100/200;
all 18 health checks passed at every panel, reconstruction stayed exact/valid
256/256, representation failures were zero and sulfur decreased to 1.4644%.
All eight restoration checks passed and the frozen epoch-36 anchor was unchanged.
The fresh 2,500-update proof started at 12:57:58 from original epoch 36. At the
13:24:51 observation, its last persisted panel is update 200/2,500, saved at
13:22:27: all 18 health checks passed, raw/strict 659/1,024 versus 641 initially,
exact/valid reconstruction 256/256, sulfur 1.464444%, 659 unique valid and zero
representation failures. This is not an exact live update count or an independent
seed replication of the same 200-update probe prefix. GPU 29783863 remains RUN;
the terminal stability gate is absent/running_not_passed. The full collector
timed out after 120s without a full news002 snapshot; a narrowed read-only check
returned zero and saved status-news002-small.json. The later successful narrow
read at 13:50:15 DTU-local records the last persisted panel 400/2,500, saved
13:45:44: all 18 checks passed, raw/strict 649/1,024 versus 641 initially
(659 at update 200), exact/valid reconstruction 256/256, sulfur 1.4446033%,
649 unique valid and zero representation failures. GPU remains RUN, terminal
stability gate absent/running_not_passed; 400 is not an exact live counter.
Those intermediate snapshots did not establish a finished longer proof,
adoption or training.
The earlier failed readiness sentinel
was intermediate only. Corrected derived summary002 supersedes the
first summary's mixed denominators; authoritative raw status is unchanged.

Terminal evidence collected at 22 October 19:03:36/19:04:58 DTU-local now
supersedes the running snapshots: source135 completed all 2,500 disposable
updates, all 13 final gates and all 26 authenticated panel receipts/18 checks
passed. Stability finished 17:48:28.86 and the GPU wrapper completed returncode
zero at 17:48:30; its live LSF record has aged out, not failed. Raw/strict
validity is 641→665/1,024, with panel ranges raw 631–725/strict 630–725;
update 500 had one representation failure, so do not claim zero across all
panels. Exact/valid reconstruction stayed 256/256 throughout. Categorical
validity is 2,091→2,323/4,096 draws from 1,024 latents (correlated four draws
per latent), unique valid 1,976→2,187; sulfur 1.5070→1.5089%. Frozen original-36
anchor remains unchanged, validation weights are discarded, and original epoch
36 remains the latest training checkpoint. The gate SHA is
218a3c9e0318286f9122017cf9c4507885202ed7981bea61852d0c16886a39b0.
All required scientific validation now passes for a guarded original-36 restart,
but the sealed controller is validation-only: no new training/adoption plan
is prepared or submitted and this status question supplies no launch
authorization. Fresh quota observed 86.50/100 GiB used (13.50 GiB free); any
future launch retains current identity/health/runtime/resume/quota guards and
requires its own authenticated plan. See docs74 terminal evidence and limits.

Independent static
seal audit passed 640 members/570 scientific
members/21 Python overlays, with protected fields unchanged; this is byte/AST/
metadata evidence only. Original epoch-36 checkpoint identity matched
the read-only preflight, which observed 87.69/100 GiB used (12.31 GiB free).
Exact immutable hashes and command/receipt links are in docs74. Preserve the
sealed source/plan, every source134 failure and the separate Protenix entries.

**Scoped bond-weighted target test, 2026-10-22:** the user resumed targeted
chemistry-collapse investigation from original epoch 36 before any training.
The chosen new candidate declares `chemistry_target_weighting={version:
"bond-weighted-target-ce-v1",none_bond_weight:0.1}` for accepted repair and valid
self target CE, normalized by `N+K+E+0.1*NE`. Keep alpha 0.1, quarter actual
AdamW displacement/native moments, search/1:2/reference35/ramp and all other
terms/guards unchanged. Top marker `target_weighting_test={version:
"epoch36-bond-weighting-test-v1"}` is validation-only with the alpha-0.1
self-guidance iteration; no implicit policy or training/promotion is allowed.
After CPU readiness, native checks then a 200-update probe (panels 0/32/100/200;
full component observations 1/39/96/100/199/200), eight-part restoration, then
a fresh 2,500-update proof share the original 24h allocation/23h internal budget.
The exact new `target_weighting_probe` marker uses
`chemistry-target-weighting-probe-v1`; remove the old `self_strength_probe` field.
Metadata/controller changes have static AST validation; new weighted native
gates and the longer scientific proof remain pending allocated GPU validation.
Source134 attempt001 is sealed but its CPU gate failed below; no native
candidate pass is established. This is an unproven hypothesis, not a
waiver or restart of source133. See `docs/72_bond_weighted_target_guidance.md`.

Actual source134 staged/authenticated 629 members and submitted CPU 29780665 /
after-CPU 29780666 on 22 October. The operation returned zero for stage and
queue, with `submitted_not_passed`. Allocated CPU outcome: 1,838 passed,
123 CUDA-only skips, four failed fake-loop fixtures, zero errors. Their setup
lacked failure_readiness.json; the CPU gate failed and both 29780665/29780666
are EXIT. Production numerical fixtures passed, but no GPU/native chemistry
outcome or training occurred. New fixture-only attempt002 is now sealed/staged
with 629 authenticated members. At 22 October 10:40 DTU-local, CPU 29780794 and
callback 29780795 are DONE, full CPU gate/profile passed (1,842 passes/123
CUDA-only skips, zero failures/errors). GPU 29780820 is submitted/PEND, one
exclusive 80GB A100/16 slots/24h; host slot-limit wait, provisional no-sooner
than 14:15:03 DTU-local. Only the mocked-loop fixture changed; the 560-member
static audit confirms production loss/protocol/guards unchanged. CPU profile
cohort of four real/32 prior points is execution parity, not the training ratio or native chemistry
outcome. At 10:43 DTU-local, a fresh saved status shows GPU 29780820 RUN since
10:42:58 on n-62-18-8, superseding the provisional wait estimate. CUDA fixtures
are in progress; no completed CUDA/native/200-probe/2,500-proof or training
outcome yet. Reduced NONE weighting has no fixed-panel validity result yet;
preserve every failed attempt001 record. Local preparation AST-parsed 18 Python overlays and generated
stage/operator/collector Python, with no numerical execution. Exact immutable
source/manifest/plan hashes and command receipts are in docs72. Preserve the
sealed release and source133's failed evidence unchanged.

At 22 October 11:32 DTU-local, actual terminal evidence supersedes the above
running snapshot: GPU 29780820 EXIT at 10:48:23, exit 1. All 453 mandatory CUDA
fixtures (no errors/failures/skips) and 18 original native gates passed, but
the eight-update benchmark failed supplementary raw-prior validity retention:
149/256 to 12/256, with exact/valid reconstruction 256/256. All other 18
supplementary checks passed. Neither the 200-update probe nor the 2,500-update
proof started; no adoption/training, and epoch 36 remains latest. Fresh prior
batches show increased sanitization failures, not representation failure or
sulfur inflation; those draws are not paired fixed-panel observations. Preserve
the sealed source134 attempt002 failure and its native evidence. This status
turn authorizes no new fix, waiver or resubmission. See docs72 terminal outcome
and `outputs/hpc-results/source134-bond-weighted-chemistry-attempt002/status-news002.json`.

**Scoped repair/valid-output loss iteration authorization, 2026-10-21:** the user
authorized targeted revision of the new repair-plus-valid-output guidance to
serve its intended purpose, with a new immutable candidate from authenticated
epoch 36. This supersedes the preceding status-only restriction for future
candidate work; never restart or rewrite failed source131/source132. The initial
candidate chooses alpha 0.1 only for the unchanged normalized full valid-self
term, outside original repair weights/ramp, and retains quarter actual AdamW
displacement with native moments. No head balancing, margin change or alpha-1.0
run is selected. After CPU readiness, one 24h GPU/23h shared internal budget
will cover native checks, a disposable 100-update component-gradient probe
(1/39/96/100; panels 0/32/100), full restoration and a fresh 2,500-update proof
from original epoch 36. Measure self gradients over the full logical batch;
diagnostics must not change training arithmetic. Source133 is sealed and tested;
its longer health proof failed, with no adoption or training. Preserve
1:2 sampling, other weights/search, frozen reference 35/original ramp, inherited
AdamW/native and fresh RNG, early fixed panels at 32/every 100, the 2,500-update
disposable proof and all scientific/health/runtime/resume/quota gates. No training
follows without an explicit new scope and all required checks; restore original
epoch 36, retain serial 24h/23h bounds and the absolute-39 short-branch cap.
No source/job/pass follows from this authorization. See
`docs/71_valid_output_guidance_iteration.md` and the experiment ledger.

Actual source133 authenticated 620 members and submitted CPU 29770494/after-CPU
29770495. Full CPU gate/repair profile passed (1,701 unit passes/93 CUDA-only
skips), as did 282 mandatory CUDA fixtures, native readiness, the 100-update
probe and all eight restoration checks. The 21 October 19:44 DTU-local saved
outcome shows the fresh longer proof FAILED at 200/2,500 updates: raw/strict
validity 641→670→643→591/1,024 at 0/32/100/200; exact/valid reconstruction
256/256 throughout. Only the two prior-retention guards failed; other 16 health
checks passed. Persisted LSF records show GPU 29770577 exit 1 at 13:51:34; all
three current scheduler records have aged out. No adoption/training, new fix,
waiver or resubmission follows from this status check; epoch 36 remains latest.
Preparation performed local standard-library AST/byte/hash packaging only;
all numerical checks remain on allocated HPC. Exact source/plan hashes,
commands and receipts are in docs71. This is a validation-only independent test;
the sealed controller forbids training/automatic adoption.

**Scoped collapse investigation and independent weak-test authorization, 2026-10-21:**
the user authorized investigating/correcting the mechanism of source131's
100-update validity collapse, rerunning required checks from authenticated
epoch 36, and testing weak valid-prior self guidance **independently of baseline
success**. This supersedes the prior baseline-pass dependency only for a new
immutable independent test plan; do not rewrite source131's stopped decision or
restart its failed jobs. The weak term remains +0.01 times all-prior-mean
normalized strict-valid self-target NLL, independent of repair weights/ramp;
no weak result automatically selects a training objective. Use targeted
mechanism diagnostics, with no assumed optimizer/loss fix or gate waiver.
Preserve original epoch-36 model/AdamW/native and fresh RNG, reference 35/ramp,
search/1:2/other scientific settings, early fixed panels and all native/health/
retention/runtime/resume/quota checks, including the 2,500 disposable updates.
Any corrected continuation restores original epoch 36 and remains serial,
24h/23h bounded and capped at absolute epoch 39. No new source/job/pass is
established by authorization alone. See `docs/70_epoch36_collapse_mechanism.md`
and the experiment ledger.

Actual source132 staged/authenticated 612 members and submitted CPU 29768881/
after-CPU 29768882. Both are DONE and full CPU gate/profile
passed (1,648 unit passes/84 CUDA-only skips). Mechanism GPU 29768917 and callback
29768918 are DONE; all 100 historical receipt matches and eight restoration
checks passed, alongside 60 mandatory CUDA fixture passes without skips/errors.
Its explicit quarter-AdamW-displacement candidate retains moments
and is not gradient clipping or a confirmed root cause/fix; the independent
weak arm uses the same factor/start. At 11:19 DTU-local, baseline GPU 29769067
and independent weak GPU 29769420 are EXIT: both passed native readiness and
187 CUDA fixtures without failures/errors/skips, then failed the raw/strict
validity-retention guards at 200/2,500 updates. Baseline callback 29769068 DONE
correctly queued the independent weak test; after-weak 29769421 EXIT stopped
because baseline had failed. Raw fixed-panel validity was 641/1,024 initially,
then 587 baseline/616 weak; exact and valid reconstruction remained 256/256.
Neither arm passed, and training was not submitted. Epoch 36 remains latest;
this status update authorizes no fix, waiver or resubmission. Exact identities,
commands and terminal evidence are in docs70 and the ledger.

**Scoped three-epoch baseline and separate weak self-guidance test, 2026-10-20:**
the user authorized a new continuation from authenticated epoch 36 using the
original successful source121 chemistry loss, only after required checks pass.
This branch stops at **absolute epoch 39** (three more epochs), within the global
100-epoch maximum; it must not automatically continue to epoch 40. Preserve native
AdamW moments, native/fresh-prior RNG, frozen reference 35/original ramp, 1:2
sampling, other weights/search, serial 24h allocations/23h internal cap, and
all scientific/health/runtime/resume/quota gates. Keep the 2,500-update disposable
proof and fixed 1,024-prior/256-reconstruction panels after 32 and every 100
updates. Any assertion specific to an absent fallback objective must have explicit
versioned applicability; do not relabel missing checks as passes. Discard proof
state and restore original epoch 36 before actual continuation. Separately test
weak detached self-target NLL for strict-valid prior molecules (alpha 0.01,
full-target normalization and all-prior mean), added independently of repair
branch weights/ramp without new repair-witness counts, with argmax/sample validity
and diversity evaluation. This exploratory test must not enter or select the
clean baseline objective automatically. Both arms require separate 2,500-update
proofs from original epoch 36. Serial order is baseline proof, weak-test proof,
then conditional baseline training after weak-stage terminal authentication;
weak failure neither blocks an independently passing baseline nor waives its gates.
Preserve source127/source128 failures and sealed source129 diagnostics. See
`docs/69_short_baseline_and_valid_self_guidance.md` and the experiment ledger.

Actual immutable source130 authenticated 603 members and submitted CPU 29764146/
after-CPU 29764147. At 19:46 DTU-local both are EXIT after two new controller
API-return assertions failed (1,523 passed/79 skipped/two failures, no errors).
No native GPU or training was submitted. Preserve source130's failed records.
A short-plan-only return-value correction is sealed as new source131, with one
controller overlay and no objective/protocol/test changes. Its staging verified
603 members and submitted CPU 29764158/after-CPU 29764159; full CPU gate/profile
passed (1,525 unit passes/79 CUDA-only skips). Saved 21 October evidence shows
GPU 29764169 passed 85 CUDA fixtures and all 18 native/11 supplementary gates,
then **failed the 2,500-update original-loss proof at update 100**. Fixed
1,024-prior raw validity was 641→674→357 at 0/32/100; strict validity ended
354. Only raw-start/strict-teacher retention failed, with exact/valid held-out
reconstruction 256/256. Original loss also fails under these continued settings;
the mechanism is unproven. The weak test and training did not start, epoch 36
remains latest, and current LSF records have aged out. Preserve this failure;
no code change, new test/submission or waiver is authorized by the status turn.
Exact source/plan/log/panel/gate evidence is in docs69.

**Scoped validity/runtime recovery authorization, 2026-10-20:** the user authorized
investigating and correcting both source128's short-panel validity loss and its
runtime rejection, rerunning required checks from authenticated epoch 36, and
conditionally continuing only after every gate passes. Use a new immutable
candidate; never restart source127/source128 jobs or rewrite their evidence.
Prefer targeted objective-component and inherited-AdamW-step diagnostics over a
broad parameter grid. Any new optimizer-step control must be explicitly versioned
and retain parent moments and native/fresh RNG; its design and validation are not
yet established. Preserve 1:2 sampling, other loss weights, bond-search semantics,
reference 35/ramp, 100 total epochs, serial continuation, 24h allocations/23h
internal cap and all scientific, retention, sulfur, runtime, resume and quota
gates without waiver. The existing early fixed-panel/2,500-update proof and
disposable-state restoration requirements remain mandatory. No new source/job,
pass or restart follows from this authorization. See
`docs/68_validity_runtime_recovery.md` and the experiment ledger.

Actual source129 diagnostic release staged/authenticated 588 members on
20 October 2026 and submitted CPU 29763813/after-CPU 29763814. At 19:08 DTU-local
both CPU records have aged out, retaining their earlier DONE and CPU gate/profile
pass (1,325 passes/70 CUDA-only skips, no failures/errors). A combined scheduler
return 255 from these missing records is not GPU failure. Diagnostic GPU 29763843
remains PEND with no result files, requesting
one 80GB A100, 16 CPU slots and 60 minutes. No GPU result or training is
established. Its versioned
classifier pool is execution work, not a chosen validity fix or optimizer-step
policy. No diagnostic outcome may promote training or replace the required
longer proof. Source/plan/command/receipt identities are recorded in docs68.
The latest status-news001 snapshot is linked there. Startup/shared-clock follow-up
has static AST/hash review only and remains unsealed; no source130 or chosen
validity correction exists. Charge startup once in its epoch projection while
retaining actual startup in elapsed allocation time and conservative training
headroom; no existing epoch-fit/elapsed-allocation gate is loosened.

**Scoped loss-scaling correction and guarded restart authorization, 2026-10-20:**
the user authorized a new versioned partial-loss normalization, early fixed-panel
health checks, and separate chemical/representation failure diagnostics, followed
by allocated DTU HPC validation from authenticated epoch 36. This supersedes the
stop only for a **new immutable plan whose required checks pass**; source127 and
its stopped jobs must never be restarted or rewritten. Preserve epoch-36 model,
AdamW, native/fresh-prior RNG, frozen epoch-35 reference and original ramp, 1:2
real/prior sampling, other loss weights, search semantics, 100-total-epoch limit
and serial continuation. Preserve all native/parity/finite-value, retention,
sulfur, runtime, resume and fresh-quota checks, with allocations at most 24h and
23h internal runtime. Validate the changed `repair-failure-guidance-v2` objective,
including a targeted 2,500-update proof spanning the previously observed roughly
1,500–2,300-update deterioration window; the old eight-update result is insufficient.
Check fixed 1,024-prior/256-reconstruction panels after 32 updates and at every
multiple of 100 thereafter. Retain separate raw-valid and strict-teacher checks,
including maximum 0.02 strict-validity drop from classified epoch 36. Discard all test
updates and restore epoch 36 before any conditional fresh continuation. Record
raw chemical validity separately from schema/canonical roundtrip acceptance and
bounded-search outcomes. Exact immutable source, sealed commands, jobs and results
remain to be recorded; authorization is not a pass or a training start. See
`docs/67_guarded_failure_guidance_restart.md` and the experiment ledger.

Actual source128 release staging authenticated 580 members on 20 October 2026.
The full CPU gate and repair profile passed (1,252 unit passes/66 CUDA-only
skips). At the 17:19 DTU-local status check, persisted evidence shows GPU
29760576 failed `complete_epoch_fits` and callback 29760577 stopped on that
gate; they are no longer queued. CUDA/native-gradient checks passed, but the
eight-update raw-valid panel also fell 149 to 142/256, beyond supplementary
retention limits; that supplementary gate was not assembled after the original
runtime rejection. No classified early baseline, 2,500-update stability proof
or training started. Preserve source127's stop and every failed record; no
waiver or restart follows from this status. Source/plan/evidence links are in docs67.

**Historical stop-and-investigate instruction, 2026-10-20:** the user explicitly
requested stopping source127 training 29758907 and callback 29758908 to
investigate deteriorating saved prior-batch teacher-valid counts. Both exact
jobs are confirmed EXIT at 20 October 00:36 DTU-local after successful user
cancellation; `TERM_OWNER`/exit 130/KeyboardInterrupt are expected intentional
termination evidence, not a spontaneous model/software failure. This supersedes automatic continuation for this branch during the
investigation. Preserve all logs, partial epoch-37 data, checkpoint/optimizer/
RNG lineage and prior failures. Epoch 36 is still the latest completed
checkpoint; no epoch-37 fixed-panel evaluation exists. Finite logged losses/
gradients do not establish scientific health. Record actual stop outcomes
separately from the instruction; no automatic resubmission is authorized by
the earlier continuation policy while this explicit stop remains in effect.

**Scoped chemistry runtime-policy amendment, 2026-10-19:** the user explicitly
authorized starting the current bond6 VAE chemistry continuation despite its
source126 runtime-headroom rejection. For this adoption only, use a new
immutable, declared runtime allowance rather than treating the historical
`1.5 * T + 1800 < 82800` decision as a pass. Preserve source126's failed
benchmark/selection/promotion records. The resolved top-level immutable plan
field is `runtime_policy_amendment={version:
"chemistry-epoch-fit-amendment-v1",factor:1.25,reserve_seconds:1800}`;
scientific protocol/configuration bytes remain unchanged. Declare both mean
and slowest-update runtime estimates. The saved conservative estimate is
61,914.5513s; `1.25*T+1800` requires 79,193.1891s, within the unchanged
82,800s internal cap with 3,606.8109s margin. These are derived estimates,
not a completed training epoch or retroactive native pass. Source127 additive
metadata adoption is sealed. Its first upload failed password authentication
before remote control creation/upload; preserve that receipt. After restored
SSH, 569 members authenticated and CPU 29758886/after-CPU 29758887 were
submitted and completed. Derived evidence-only adoption and allocated CPU
contracts passed (1,168 tests passed/65 CUDA-only skips, no failures/errors).
Training 29758907 ran from 19 October 2026 18:37:56 DTU-local, restoring
epoch 36; it and callback 29758908 were user-stopped at 20 October 00:36.
Latest saved progress is partial epoch 37, not an epoch completion or
fixed-panel health result. Epoch 36 remains the latest completed checkpoint;
no automatic continuation is permitted during the explicit stop/investigation.
Retain the authenticated epoch-36 model/AdamW/native and fresh-prior RNG,
original frozen epoch-35 reference/ramp, 1:2 sampling, loss/weights/search,
100-total-epoch maximum, strict serial continuation and all scientific,
health, identity, native-gradient, retention, finite-value, memory, resume and
fresh-quota checks. Allocations remain at most 24h with 23h internal runtime;
unfinished work remains partial and cannot fabricate an epoch completion.
Reusing completed native evidence requires an additive authenticated adoption
record and unchanged scientific bindings; do not rewrite the failed result
or silently relax another gate. This scoped amendment is authorization, not
a submitted job, passed new policy or training start. See
`docs/66_failure_guidance_execution_speed.md` and the experiment ledger.

**Future latent-optimization step policy, 2026-10-18:** the user authorized
`best_trial_v2` for future optimization runs: attempt the full requested update
budget rather than ending on an ordinary lack of descent. At each state, try
the full proposed step first; apply it immediately when the actual scalar loss
does not increase, including equality/no-op. Strict improvement uses
`first_improvement`; other full-step nonincrease uses `full_step_nonincrease`
and is neither strict descent nor forced fallback. Otherwise halve progressively, using up to 16 sizes
`1, 1/2, ..., 2^-15`, and accept the first strict improvement. If none improves,
apply the lowest-loss finite candidate even when its loss is equal or worse;
use the earliest trial to break ties. Equal/no-op finite candidates may consume
an update and must be reported honestly. Context, mask, repaired target and
coefficient remain fixed throughout that state's trials. Loss need not decrease
monotonically. For future v2 runs this replaces the historical native
Armijo/monotone-decrease acceptance requirement; native source/input/gradient,
finite-value and trust-radius requirements remain unchanged. Distinguish
applied, improving, forced-fallback and no-op counts,
retain best-observed state evidence with its objective/context identity, and do
not rank changing repair totals as one comparable objective. All-nonfinite
trials, failed native gates, unexpected exceptions and resource exhaustion
remain genuine failures/partial runs, not fabricated full-budget completion.
Keep explicit `armijo_v1` compatibility and preserve every sealed historical
run, including source120's 225-update control stall and its guided protocol;
this policy does not authorize rewriting, restarting or rerunning them. See
`docs/62_full_budget_best_trial_optimization.md` and the experiment ledger.
Final source122 attempt003 CPU job 29749480 passed 423 tests across 16 modules
on 18 October 2026, with no failures/errors/skips, including the equality
amendment. The original failed transfer and attempt002's intermediate 411-pass
result remain preserved. This is
optimizer/compatibility/report fixture evidence, not native v2 Boltz-model
validation, a new molecular optimization or a change to existing HPC jobs.

**Presentation preference, 2026-10-16:** provide a standard offline interactive
HTML report for molecular optimization trajectories, with loss overlays,
initialization/run selection, playback, molecule snapshots and saved probability
heatmaps. Raw JSON/JSONL, manifests and scientific run records remain authoritative.
Render saved artifacts only: do not rerun models, resample, repair or invent
missing probabilities for a report. Keep invalid graphs and failed/partial runs
visible, distinguish repaired targets from the current shared molecule, and
separate fixed-context losses from fresh native predictions. Local standard-library
rendering and browser presentation checks are presentation work, not numerical
model validation. Report-generation failures must not change scientific outcomes;
future capture hooks do not alter sealed HPC releases. See
`docs/59_interactive_optimization_reports.md`.

**Scoped long optimization comparison, 2026-10-16:** the user authorized one
sequential GPU comparison of 500 requested updates each for Boltz-only and
Boltz+DeFoG from the same authenticated initial `z`, frozen validated decoder,
protein context and fixed 22-slot initial mask. Preserve the DBMol objective,
pointwise-native v4 checks and actual native Armijo/trust acceptance. Keep the
guided coefficient's original initial 25-update ramp, then hold its endpoint
weight; do not stretch it across 500 updates. Retain checkpoints
0/25/50/100/250/500 and honest saved-only interactive reports, including invalid,
partial and failed data. Authenticate the new source/configuration/resources
and record actual receipts/outcomes; authorization is not submission or a pass.
The separate VAE chemistry continuation and all historical failures remain
unchanged. A verified Armijo stall may end an arm early and allow the independent
other arm only after all prefix/native/descent/unchanged-point/terminal checks;
it is not convergence or a full-500 certificate. Failed native gates, exceptions
and exhausted resources stop the job, and later checkpoints remain `not_reached`.
See `docs/60_long_boltz_defog_comparison.md`.

Actual initial submission for that scope is source114 job **29741066**, recorded
16 October 2026 in the ledger with immutable source/bootstrap bindings. Its
initial state is queued, not checked or passed; preserve the single-allocation
and fail-closed execution contract when reporting or continuing this run.

**Scoped chemistry speed selection, 2026-10-17:** the user authorized native
benchmarks of CPU/GPU overlap, worker count and microbatch size for the current
joint VAE chemistry continuation. Prefer a fully passing 1:3 candidate below
ten hours/real epoch; otherwise evaluate a declared 1:2 family and automatically
select its fastest fully passing member even if still above ten hours. All
scientific/health/runtime gates remain hard requirements. Retain 0.25/0.75 repair
branch means, prior-mean occupancy anchoring and original population-JS mixture
weighting (prior count weight 3/ratio); fewer prior samples are an explicitly
versioned protocol change, not bitwise-equivalent training. Preserve the epoch 35
parent, native optimizer/RNG lineage, 100-total-epoch maximum, strict quota and
serial continuation; no candidate benchmark update may become a checkpoint.
Retire competing pending callbacks with actual receipts before new automatic
promotion. Authorization and selection are not submissions, passed native
checks or a training start. See `docs/61_chemistry_execution_selection.md`.

Updated user preference, 2026-09-28: the existing AE/VAE training is the completed
historical foundation. Use frozen, swappable external pretrained models for
future chemistry-prior, activity, synthesis and scoring branches. Do not train
new in-house priors/scorers or learned reward surrogates. Small supervised
input bridges/representation decoders are allowed only to reproduce a frozen
model's native input or early embedding on paired hard molecules/context, with
held-out fidelity checks, and must be frozen before optimizing molecular `z`.
Prefer a bridge from the shared graph; a bridge also using `z` is allowed only
with hard-input/native-feature parity and cycle/identity consistency with that
same graph. Maintain one shared canonical molecule across branches; declare
feature loss, fixed/variable occupancy, gradient scope and unsupported modes.
Do not independently decode a different chemical identity for each scorer.
See `docs/23_frozen_model_plugin_contract.md` for the proposed boundaries/gates.

**Scoped user-directed extension, 2026-10-07:** the user explicitly authorized
matched decoder-only fine-tuning of the existing VAE using rule-based,
probability-aware chemical guidance, with random-latent generation evaluation.
Preserve the original checkpoint; compare original, matched reconstruction-
plus-shared-reference-anchors control and chemical-regularized decoder, with
the encoder frozen in all trained arms. This does not authorize new activity scorers, learned
reward surrogates or replacement external models. All numerical work stays on
DTU HPC, and any resulting decoder must be frozen before molecular latent
optimization. Bounded valid-repair witnesses must not be called globally
nearest-valid distances. See `docs/39_decoder_chemical_distance_training.md`.

**Scoped user-directed reduction, 2026-10-08:** for the current 100-epoch
dense decoder continuation, run only the seed-17 chemical decoder and benchmark
it against the untouched original epoch-89 VAE, instead of submitting new
matched controls or a second training seed. The original had no additional
unconditional fresh-prior training points; posterior sampling and VAE KL
regularization still apply. Historical coefficient-zero controls included
fresh-prior reference anchors and are not this zero-additional-prior baseline.
Treat the single-candidate comparison as exploratory: extra fine-tuning,
chemical loss and the frozen-encoder regime are confounded, and it does not
isolate the causal effect of random points or provide training-seed replication.
Preserve the original, frozen encoder, HPC-only numerical work and existing
external-model boundaries. See `docs/45_dense_chemistry_100_epoch_continuation.md`.

**Scoped user-directed extension, 2026-10-11:** the user authorized local
implementation and preparation of a new large-corpus triangle graph VAE,
with encoder and decoder trained jointly on supported curated real molecules
under a planned 100-epoch maximum budget. Its objective is reconstruction
plus VAE KL, with no additional unconditional prior training points or
chemistry/repair loss. Preserve the original epoch-89 VAE and all historical
decoder-only studies; their frozen-encoder restrictions remain applicable
to those studies, not this new joint-training scope. Protect original
held-out chemical identities from the new training corpus, report supported
counts and treat historical architecture/data comparisons as confounded.
HPC is unavailable for this implementation turn: no connection, upload,
submission or local numerical tests. Editing, static review and immutable
packaging are authorized; all preprocessing, numerical validation, training
and evaluation remain later DTU HPC work. This does not authorize learned
scorers or reward surrogates; freeze a resulting decoder before latent
optimization. See `docs/52_large_corpus_triangle_vae.md`.

**Scoped HPC execution authorization, 2026-10-11:** the user subsequently
reported restored access and authorized all runs needed for the first
large-corpus triangle-VAE suite. This supersedes the preceding
implementation-turn outage restriction on HPC connection, transfer and
submission. Authenticate immutable source/data/reference bindings and pass
the staged readiness gates before full training; the exact split-count
shortage must be resolved with reported supported counts and a new sealed
configuration, not silent holdout reduction. All numerical work remains on
allocated DTU HPC. Authorization is not a submitted job or a passed gate;
record actual receipts and outcomes in the ledger. Preserve the already
submitted source091 DeFoG pilot and all historical experiment evidence.

**Scoped automatic-continuation authorization, 2026-10-12:** the user
authorized continuing the same large-corpus triangle VAE toward 100 epochs
automatically across serial allocations while checks pass, and reporting
all runs. Preserve the authenticated source098 model/data/config/checkpoint
and RNG/optimizer continuation identities. Authenticate completed readiness,
each segment's persisted completion and health, source/data consistency and
available quota before the next submission; stop on failed checks instead
of approving or replacing a failed result. Run the reserved final evaluation
after the 100-epoch endpoint. This extends the earlier first-segment-only
launcher scope; it does not authorize changed model/loss/data settings,
concurrent training segments or a waiver of historical Boltz/DeFoG gates.
All numerical work remains allocated DTU HPC, and every controller/segment
submission and outcome belongs in the experiment ledger. Authorization alone
does not establish a queued controller, completed epoch or final evaluation.

**Scoped chemistry-continuation authorization, 2026-10-13:** the user
authorized implementing atom-preserving repair loss and a weak empirical
composition regularizer, benchmarking them on allocated DTU HPC, then
starting a **1:3 real-to-fresh-unconditional-prior continuation** from an
authenticated current triangle-VAE checkpoint only if checks pass. The user
selected **100 total epochs**, including the parent-completed epochs, not
100 additional chemistry epochs. This is a new, explicitly changed-objective
joint encoder/decoder branch; intentionally carry the authenticated parent
AdamW moments and RNG lineage and persist a separate new fresh-prior RNG.
Preserve source098's unchanged reconstruction-plus-KL campaign and
its final-evaluation obligations, the original epoch-89 VAE, and every
historical decoder-only study. Initial repair witnesses retain atom identity,
formal charge and occupied-slot identity and change bonds only; failed or
already-valid targets contribute zero in the all-attempt chemistry mean.
Accepted targets use full atom/occupied-charge/unordered-bond categorical
NLL, so unchanged identity labels also receive gradients.
Composition is a declared population prior, not a chemistry guarantee.
Do not include the historical maximum-allowed-valence proxy in this phase.
Record exact parameter/optimizer/RNG scope, coefficient normalization,
source/config/data/checkpoint bindings, native coverage/runtime/retention
gates and all outcomes. Stop on failed checks rather than silently removing
random points or loosening validation. Bounded witnesses are not globally
nearest-valid distances. Freeze the resulting decoder before latent
optimization; no learned scorer, separate branch identity or Boltz/DeFoG
gradient-gate waiver is authorized. See
`docs/53_atom_preserving_triangle_continuation.md`.

**Scoped baseline-stop and chemistry-parent update, 2026-10-13:** the user
authorized stopping the original real-only source098 triangle-VAE run and
its automatic continuation, then using its **most recent completed,
authenticated checkpoint** as the parent of the 1:3 chemistry phase toward
**100 total epochs**. This supersedes the earlier obligation to continue
that real-only baseline to epoch 100 and run its endpoint final evaluation;
it does not mark that baseline complete or create an epoch-100 control.
Preserve all source098 checkpoints, source/data/configuration identities,
original epoch-89 VAE and historical evidence. Record requested and actual
callback/benchmark/training cancellations separately; intentional baseline
interruption is not model instability or a failed scientific health check.
Do not silently repin an already sealed chemistry plan: copy and authenticate
the stopped baseline's latest completed epoch, optimizer and RNG lineage,
then use a new immutable plan/run identity. Retain the native chemistry
readiness/coverage/runtime/composition/retention/quota gates and serial
automatic continuation while checks pass. Final chemistry evaluation remains
gated by its authenticated absolute epoch-100 endpoint. Authorization alone
does not establish a stopped job, new checkpoint or new submission. See
`docs/53_atom_preserving_triangle_continuation.md` and the experiment ledger.

**Scoped allocation preference and staged-execution authorization, 2026-10-17:**
the user requested allocations no longer than24 hours and fewer CPUs for the
chemistry speed benchmark. This supersedes the earlier single48h-allocation
long-comparison plan and the proposed72h VAE segments. The500-update Boltz-only
and Boltz+DeFoG arms may run in separate serial allocations, followed by a
separate evaluation/report allocation, authenticating each completed stage
before the next. Split only between independent arms/stages; this does not
authorize an unauthenticated mid-arm optimizer resume, treating a partial
trajectory as a full-budget pass, or bypassing gradient/descent/terminal checks.
Preserve the same frozen decoder, initial latent/context, objective, first25-step
repair ramp and500-update maximum. The reduced chemistry benchmark uses12 search
workers plus four main Torch threads within16 CPU slots, deferring20/28-worker
variants; retain all native/parity/coverage/retention/runtime/quota checks and
the declared1:3-first/1:2-fallback policy. A shorter walltime does not waive the
complete-epoch-fit gate or authorize changed loss/ratio semantics. Preserve
earlier immutable plans and actual cancellation/submission receipts; preference
alone does not establish retired jobs, new allocations or passed checks. See
`docs/60_long_boltz_defog_comparison.md`,
`docs/61_chemistry_execution_selection.md` and the experiment ledger.

- Run every numerical test, preprocessing experiment, training run, and evaluation
  on DTU HPC. Local editing, static review, packaging and file transfers are allowed.
  **Temporary user override, 2026-10-01:** while DTU HPC is inaccessible,
  light local smoke tests of the current graded-quality v2 code are allowed.
  Record their exact command, environment, scope and outcome in the experiment
  ledger, and do not present them as HPC validation. Full-data preprocessing,
  training, panel evaluation and other substantive numerical experiments
  remain on DTU HPC once access resumes. Do not keep retrying or submit HPC
  jobs during the present access outage.
  **Additional user override, 2026-10-01:** quick local smoke/unit checks
  are also allowed for the current config-driven named frozen-model branch
  and weighted-gradient plumbing work. Record exact commands, environment,
  scope and outcomes in the ledger. The user explicitly requested **no HPC
  testing or submission for this implementation task**. Full model runs,
  training and substantive multi-model experiments remain HPC work for a
  later authorized turn; local checks establish only engineering behavior.
- Maintain a separate reviewer focused on modularity/reuse and a separate owner
  focused on inline documentation, pipeline documentation and experiment history.
- AE and VAE share the graph schema, encoder, decoder, data split and evaluation.
  Document any differences in sampling, loss, initialization or compute budget.
- Keep chemistry conversion outside the differentiable model path. Scorers consume
  the shared relaxed-graph contract; do not hide discrete conversions in adapters.
- Every submitted job and completed/failed experiment belongs in
  `docs/05_experiment_ledger.md`, with source/config/data identity and artifact paths.
  Retain failures. A submitted job is not a passed test or a completed experiment.
- Use immutable source releases and unique run directories on HPC. Source hashes
  must include uncommitted code; a git commit alone is insufficient.
- Preserve and report representation limitations and rejected-data counts. Do not
  claim chemical validity, binding, synthesis feasibility or selectivity beyond
  what the actual evaluation establishes.

Standing permission to overwrite the user's HPC data is recorded in
`docs/03_hpc_execution.md` and `HPC_info/HPC_HELP.tex`.
