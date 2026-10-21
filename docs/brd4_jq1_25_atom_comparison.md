# JQ1-pocket 25-slot, 10,000-update comparison

21 October 2026. The user explicitly authorized stopping the old25-slot runs,
implementing the JQ1 pocket for fresh Protenix, Boltz-2 and Boltz-2+chemistry
arms, testing on allocated DTU HPC and submitting each only after its gates
pass. No diffusion gradient may reach the optimized molecule. This supersedes
the previous instruction to preserve those two live arms unchanged, for exactly
jobs29769525/29769540. All historical evidence remains preserved.

## Actual retirement and queue selection

Both exact old jobs were queried before cancellation. `bkill` was issued for
29769525 and29769540, with actual EXIT confirmed21October11:15:33/11:15:34Paris.
Original source04 readiness passes remain passes; these long trajectories are
intentionally interrupted partial runs, not completed10,000-update results or
spontaneous numerical failures. Original artifacts remain in their old run
directories. Stop/queue evidence is kept remotely at
`/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-stop-01/`.

The user then requested the GPU queue with least waiting. Live `bqueues -w`,
`bhosts -gpu -w`, queue limits and `bqueues -u s234818 -w` were inspected. H100's
lighter queue is restricted and absent from this account's eligible list.
Among eligible long GPU queues, L40S had the smallest pending-slot backlog
(110 versus148A10,160A40,1495A100,2735V100 in the captured snapshot). Those are
LSF slot counts, not job counts or a promised waiting time. Interactive queues
do not provide the long allocation needed. Choose **gpul40s for all three arms**
and use matching hardware for readiness/pilots and optimization: one exclusive
48GiB L40S, four CPU slots,8GiB/slot; two-hour native readiness/preparation and
23-hour long allocations. Preserve the raw queue snapshot; no restricted queue
access or exact start-time prediction is implied.

## Pocket, gradient scope and comparison

Use the existing experimentally measured1.60A BRD4-BD1/JQ1 structure
[3MXF](https://www.rcsb.org/structure/3MXF), pinned PDB SHA
`534dcb8ea9fa29956249b5133b22187c3867363b16a6b98dd8219c18c5d3250f`.
On allocated CPU, derive the protein heavy-atom neighborhood within6A of the
single crystallographic31-heavy-atom JQ1. Solvents/ions are excluded. Alternate
locations are chosen deterministically by occupancy, then blank/A preference.
Bind every selected author residue to its supplied127-residue sequence position
and amino-acid identity. Store the full definition and distances; do not claim a
globally optimal pocket radius or JQ1-like chemistry for generated graphs.

| Arm | New declared pocket treatment | Optimized native term |
| --- | --- | --- |
| Protenix | Full target context; only selected native protein tokens eligible in contact reduction | Negative mean ligand best-pocket contact probability, coefficient1,8A conservative native bins |
| Boltz-2 | Native pocket constraints during fresh pose preparation; passed placement audit; detached fixed coordinates afterward | Native quantitative affinity ensemble, coefficient1 |
| Boltz-2+chemistry | Identical prepared pocket context and initialization to Boltz-only | Same affinity plus existing repair schedule |

Protenix retains its contact algebra with a restricted partner set; it does not
require every pocket residue to contact the ligand and introduces no new mean
coverage/anchor loss. Whole-target historical `NativeContactLoss` calls remain
supported with no pocket argument. Map native residue metadata explicitly rather
than treating author140 as a token index. The earlier unjitted-hard versus jitted-
attached Protenix parity failure is preserved. This new version compiles the
independent original-hard-feature geometry-off reference in the same full
trunk/head scope, checks every hard/injected native feature exactly, and retains
the original2e-5 output tolerance. This is an implementation correction proposal
until the allocated native parity gate actually passes.

Boltz uses native one-based sequence anchors40/99 (authorTrp81/Asn140),6A pocket
constraints with `force:true` and native inference-time potentials only during
fresh carbon-chain pose generation. Native protein/crop/atom maps must retain
all selected pocket identities. At the actual prepared coordinates, both anchor
minimum distances must be <=6A and at least50% of ligand heavy atoms must lie
within6A of the selected pocket's protein atoms. This is a declared placement
gate, not a binding/ligand-RMSD certificate. Failed pose bytes and audits are
saved and must not be promoted to readiness.

Native structure generation happens once outside optimization. Saved coordinates
are detached, affinity heads receive `coordinates.detach()`, and production
scoring uses the input/trunk/affinity path only. No diffusion/coordinate/VAE/latent
optimization or trained surrogate is introduced. The affinity output remains a
whole-complex native scalar at the chosen fixed pose, not a separate per-pocket
affinity predictor. JQ1 supplies a site reference, not a transplanted ligand.

All arms keep25occupied neutral slots/300pairs,50%chain/uniform initialization,
LR0.03,key17,T1,ten-trialbest_primal,10,000total updates,composition0. Boltz
retains zero recycles/noMWcorrection; binder probability is separate. Chemistry
is the existing objective,0.001 through applied5000 then geometric to50at10000;
fixed target/coefficient per trial set. Raw invalid/disconnected graphs stay
visible. No temperature or continuous-chemistry prototype was adopted.

## Validation and conditional submission

All numerical extraction/tests/model/chemistry/evaluation stay allocated DTU HPC.
Local work is static editing, AST/JSON/LF inspection, packaging/transfers and
stdlib byte/XML/schema verification. CPU reruns legacy/bridge, mapping, unchanged
chemistry oracle and controller suites plus new pocket selection/mapping/loss,
pose placement and no-diffusion gradient tests. CPU/controller authentication
requires actual scheduler DONE, actual LSF success and exact raw proof hashes.

Each native arm must pass original hard parity, independent-primal raw atom/bond
FD at.01/.003/.001,15% and two-adjacent criteria, primal/AD parity, finite/frozen
checks and ten-update pilot. Protenix retains positive-bond-order null checks;
Boltz retains typed-bond contrast. Chemistry also requires nonzero invalid-graph
signal, low/high-weight coupled FD and both disposable ten-update pilots.
Pocket definition/context/config/source bindings and no-diffusion scope are new
additional requirements. All original thresholds remain unchanged.

Keep runtime gate `1.5*slowest_pilot_mean*10000+1800<82800` and whole-campaign
fresh conservative quota, summing passed arms' projections plus4GiB reserve.
No partial/native failure can become a completed study. Per-arm independent
callbacks submit only genuinely passed arms; exact IDs/resources must be read
from actual receipts. Failed/missing/aged-out outcomes never become DONE. LSF
whitespace wrapping is normalized with original captures retained; bounded
cache-lag retries require the eventual actual final state. Exclusive attempt
markers prevent blind duplicate submissions after unknown outcomes.

Planned fresh source/run identity `20261021-brd4-jq1-25-01`; this plan alone is
neither a sealed release, submitted GPU nor a passed numerical test. Actual
immutable hashes, jobs and outcomes are added below and in the ledger/receipt.

## Authenticated source01 CPU result and actual native submissions

Source01 is actually sealed at
`/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-01/source`,488files,
manifest`2503c2382b429ec6b79d7a256b2a2fccd09f7739a609e97a876d604d00dc063c`;
archive39,462,299bytes SHA
`5e4bf8601f1b09e244a7763dc6c6d8af5ddc76bfd3490830993d7cfd0b2c0906`.
Source01 includes the uncommitted implementation bytes, not merely its Git ref.

CPU29769654 actually DONE11:30:22Paris and after-CPU29769655 actually DONE
11:30:38. Full CPU gate **419 passed**, no failures/errors/skips; three existing
slow/model checks deliberately deselected. Groups93controller/pocket contracts,
233legacy/JAX,34Torch mapping/config/no-diffusion,59unchanged chemistry. Raw XML,
actual scheduler/LSF and all43proof files downloaded and byte/schema-verified at
`results/hpc/20261021-brd4-jq1-25/cpu01/`. Assessment SHA
`e342bda2bd510fff0aba33192f10dfe4d81ccca53e1b83562c83089ab54095e1`;
archive29,621bytes SHA
`12ffe963f2f5e68989bd80eac6f90aee2e08f77746246d5b6a4631fca2c5a18a`.

Allocated extraction produced19pocket residues, author numbers
81,82,83,85,86,87,88,92,93,94,97,135,136,139,140,144,145,146,149.
Definition SHA`9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43`.
This is actual extraction, not a native predicted-pose validation.

Actual native jobs submitted11:30:38Paris: Protenix validation29769668 and
callback29769669; fresh Boltz pocket-pose preparation29769670 and callback
29769671. Latest observed PEND; GPU jobs request the declared gpul40s resources.
No passed native model gate or long submission is established by this CPU pass.
The controller will submit the two separate Boltz validators only after actual
pose preparation/placement passes, then independent long arms after each actual
native/runtime/storage pass. Current receipt:
`scripts/hpc/receipts/20261021-brd4-jq1-25-01/campaign.json`.


## Protenix-only engineering retry and interactive-node check

Source01 native Protenix29769668 EXIT at12:24:37Paris. Hard native output parity
at2e-5, actual pocket mapping and frozen identities passed, but the checker
raised `Best-primal fallback requires independent primal evaluation` before its
one-step update. The driver had omitted `primal_every_step=True`; the general
checker correctly refused best_primal with its defaultFalse. Preserve the failed
readiness and every raw FD/null artifact; this is not a full native pass.

The correction supplies independent candidate forwards explicitly through the
shared driver helper, preserving epsilon/tolerance/direction/null/finite guards.
A regression exercises this actual driver helper on all25slots/300pairs and
checks both raw directions, null invariance and saved candidate primal bytes.
A new independent Protenix-only campaign redoes allocated CPU and full native
validation. Its arm-scope contracts prevent duplicate Boltz preparation or
submission; source01's Boltz preparation29769670 actually DONE and placement
passed. Two Boltz validators and callbacks were actually submitted afterward.
No pocket long arm has yet been established by these results.

User suggested interactive nodes for short work. Site gpua100i/gpuv100i/
gpuvoltasxm2i are interactive-only. A direct bsub with20min walltime was rejected
by the site hook before submission (no jobID). Official `a100sh` then allocated
gpua100i job29770026 immediately on n-62-12-19. GPUs40GiB: index0 used1295MiB,
index1 used33463MiB at100%utilization; neither was free. No model/test was run,
and the session was closed. Do not bypass shared-node policies or treat quick
interactive shell allocation as free GPU availability. Existing running batch
work stays intact. Long10000-update runs remain batch; interactive allocations
are candidates for bounded debugging only when a suitable GPU is actually free.

New retry planned source20261021-brd4-jq1-25-protenix-02; no new test pass or
submission follows from the correction or this plan alone.


## 21 October 2026 — corrected CPU427 passed; first pocket long job actually queued

Source02 CPU29770038/afterCPU29770039 actuallyDONE/passed427:100controller,
234legacy/JAX,34native mapping/no-diffusion,59unchanged chemistry; zero failures,
errors or skips,3deliberate legacy slow/model deselections. Actualscheduler/LSF,
rawXML and all38proof files locally byte/schema-verified under
results/hpc/20261021-brd4-jq1-25/cpu02/verification.json. Assessment
0bcf6532e8642f51e871f57ff537efaf0ba3504b94f5fcc42ffeaf70dbca9573;
archive28572bytesSHAbfd3762940b5eb3a1522a3a65df7b885ad4da988444155db26ee761a6584a407.
Protenix-only native29770090 and callback29770091 actually submitted.
Callback alone verifiedunstarted and bstop applied; actualPSUSP confirmed.
This reversible submission hold permits fresh whole-three-arm storage review
across BOTH source01/source02 roots before bresume; native remains unchanged.
Include all passed current projections, already-submitted arms and4GiB reserve,
subtract both roots' actual bytes conservatively. This is neither scientific
failure nor a user-requested pause. The existing monitor tracks release after
actual fullnative/identity/runtime/proof and aggregatequota gates.

Source01 Boltz native29770007 and callback29770008 actuallyDONE/passed.
Hard parity/rawatom+typedbondFD/primalAD/finite/frozen/pocket gates and10-update
pilot authenticated; mean1.761389260s/update, conservative28220.838900s<82800,
storageprojection1804441824bytes. Model395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16;
contexttensorfd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4.
Assessment41901b3ad903c5d14b0344a1876564ee2481ace6a3adc8be5a33245214b4f387,
52nativeproofs rehashed. Actual independent10000-update long29770093 submitted
12:35:37Paris, lastPENDgpul40s; bjobs and receipt match1exclusiveL40S48GB,
4slots,8GiB/slot,23h,nointerarmdependency. Submission is not trajectory completion.
Chemistry native29770009 nowRUN, callback29770010PEND; no chemistry long yet.

All168compactproof files downloaded/byte-verified under
results/hpc/20261021-brd4-jq1-25/boltz-ready02/verification.json; archive736630bytes
SHA18e1d6dcede92b79b43ccfc0a828ce1ec480d819f5eb976a2fb4955a470db072,
manifest70c1b16ef5515bfd1cc16aae7105e3ac839c1bb7e689f54588672080538f7e63.
Firstcapture stopped on aged-out liveCPU records; partialfolder preserved.
Correctedcapture authenticates exact pinned prior actualscheduler/LSF bytes and
labels persisted outcomes, retaining live-not-found responses; no fakeDONE or
forbiddenaccounting query. Historicalnativefailures remain failed.


## 21 October 2026 — both Boltz pocket native arms passed and submitted

Chemistry native29770009/callback29770010 actuallyDONE; all native/typedFD/
parity/pocket/finite/frozen plus nonzero invalidassay, low/highcoupledchecks
and both10steppilots passed. Assessment17ce0869f0a9888292e9017e03d407bf80d979a9cbfbed1367ae0785eb532394,
128nativeproofs rehashed. Pilot means2.048396136/2.270196313s/update, original
conservative projection35852.944692s<82800; projected artifacts5187761824bytes.
Independent10000-update chemistry long29770124 actuallysubmitted andPEND;
receipt and actualbjobs matchgpul40s/1exclusiveL40S/4CPU/8GiBslot/23h/nointerarm
dependency. Boltz-only29770093 nowRUN. CorrectedProtenix native29770090RUN;
its callback29770091 remains intentionallyPSUSP pendingfullnative andsharedquota.

All313proof files safely downloaded/byte/schema-verified under
results/hpc/20261021-brd4-jq1-25/boltz-both03/verification.json; archive2176935bytes
SHAb5d176a1f5396982309e5c7a11bdb352134e18a8190a20dc8e4d66a4aacc896a,
manifest1fe0710926a33b5b9d7d8e3c0c2bc1fc1f53bd12cd810eb512861c642dc51533.
Fresh stdlib-only shared-quota auditor authenticates all3actualnative/scheduler/
LSF/source/config/CPU/proofs, adds all3projections+4GiB and subtracts BOTH root
byte totals. It does not resumejobs. SHA
f84e64b64869fe4979cdb5f0e1c31de2a97eb265f12f6f0f1ee169fdc8f529e1;
preparedfor later execution only after ProtenixactuallyDONE/passed. LocalAST
review only; no local numericaltests or scientific execution.
