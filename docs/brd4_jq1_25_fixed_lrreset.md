# Fresh lower-rate JQ1 runs with chemistry endpoint 100

> **Scope and access note (22 October 2026):** this document contains an
> experiment protocol and dated operator evidence. Concrete account paths,
> environments, jobs and source hashes describe that operator/attempt; they
> are not shared-account defaults or a fresh status/acceptance claim. Use
> [your own HPC setup](HPC_GUIDE.md). Keep the scientific gates and historical
> outcomes below intact when preparing a separately validated candidate.
> Canonical working home: [brd4-jq1/fixed-lrreset](../experiments/brd4-jq1/fixed-lrreset/README.md).


22 October 2026. The user requested dividing every higher-rate learning rate
by ten and fresh submission of all three runs. After clarification, the user
selected chemistry coefficients 0.001 to 100. This is an explicit stronger
late chemistry coefficient, not a normalization or an equal-gradient claim.

| Arm | Fixed logit LR | Fresh updates | Chemistry coefficient |
| --- | ---: | ---: | --- |
| Protenix | 0.3 | 10000 | absent |
| Boltz-2 | 0.005 | 10000 | absent |
| Boltz-2 + chem_valid_loss | 0.005 | 10000 | 0.001 through applied update 5000, geometric to 100 at 10000 |

The version is `mosaic-brd4-jq1-25-fixed-step-lrreset-v1`. Each update has exactly
one finite authenticated full-rate candidate, `max_backtracks=1`, fixed_step;
finite increases, equality and no-ops consume updates and are reported honestly.
No halving, backtracking, clipping, gradient normalization or optimizer change.
N25 occupied neutral slots, chain/uniform 50%, key17, T1, JQ1 pocket, original
contact/affinity objectives, loss/search settings and reference stay fixed.
The Boltz pose/context is detached, zero recycles, no MW correction; no gradient
through diffusion, no VAE/decoder/DeFoG/temperature/model change.

Preserve all original hard parity, raw atom/bond FD at .01/.003/.001 with 15%
and two adjacent scales, nonzero/null/independent primal/finite/frozen/runtime
gates. Full allocated CPU regressions include the historical source610 suites
and new rate/coefficient/controller/storage fixtures. All three new-rate native
readiness checks are mandatory. Chemistry must pass its nonzero assay, coupled
low **0.001** and high **100** derivative checks, and both ten-update pilots at
the declared LR. Readiness checks and production route those exact endpoints.
Old-rate passes and fine-scale diagnostics cannot certify these new settings.
The Protenix original initialization NPZ/key equality remains mandatory.

All three higher-rate predecessors are finished: Pro29779177 and Bol29779178
saved 10000 updates but failed mandatory terminal derivatives; chemistry29779179
passed. Their original allocated full-frame collectors authenticate terminal
history. No cancellation or resume is needed, and no old failure is relabeled.
Separate diagnostics29780821/29780823 and their captures/final presentation
remain independent, unchanged existing jobs. They are not new optimization runs.

New immutable tag `20261022-brd4-jq1-25-fixed-lrreset-01` uses a new scientific
release and controller. To keep existing sealed analysis observers accounting
for every new work3 byte, the new run root is nested at
`/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-01`,
the release at sibling `restart-lrreset-release-01/source`, and upload in the
containing analysis root. New Pro/Bol output parents are run-protenix/run-boltz2
under the new root; chemistry is under dedicated home Mosaic TAG/run-boltz2_chemistry.
No historical input or output is moved or rewritten.

Full CPU and all three actual new native DONE/LSF/proof passes precede fresh
whole two-filesystem storage and three independent long submissions. No long
inter-arm dependency. Native/long resources are gpul40s, one exclusive L40S48GB,
4CPU,8GiB/slot,2h/23h. Each GPU entry reauthenticates all source/config/readiness/
CPU pins and full remaining filesystem budgets before model/driver invocation.
Submission-exclusive markers prohibit blind duplication after unknown outcomes.

New storage retains each filesystem's own 4GiB/100000-file headroom, conservative
pilot projections, all finite quota pools, completed no-follow home recent-write
inventory before fresh quota (max4h, Copenhagen/no future date), disk minimum and
coherent positive physical home inodes. Work3 all-zero physical inode tuple is
unreported supplemental evidence only. Account containing roots once, including
all old analysis and new nested outputs/source/upload. Add 2GiB report plus
512MiB diagnostic allowance to every work3 gate for the existing independent
analysis. No cleanup, gate reduction or reserve borrowing is authorized here.

All model/numerical/chemistry tests and scientific rendering run on allocated
DTU HPC. Local work is standard-library static AST/JSON/byte/hash packaging and
evidence/publication checks only. A failed gate preserves evidence and blocks
dispatch; do not change the declared LR, chemistry schedule or FD scales.

Future saved-only reports need a new immutable contract pinning the actual new
jobs/source/config/readiness/CPU. Preserve 8778/8777/8776 historical reports and
all editions. Render every saved FP32 frame in bounded100-frame chunks, with
allocated scalar/simplex/argmax/hash verification and publication/browser QA.
Contact, affinity and binder probability remain separate; changing chemistry
totals are not one fixed comparable objective. Invalid graphs remain visible;
graph validity or fixed-pose scores do not prove binding/activity. Final success
requires each original new summary/case, 10000 contiguous updates/10001 states,
JSONL equality, initial/coupled/terminal derivatives/parity/frozen/readiness pins,
and chemistry terminal coefficient100. Scheduler DONE alone is insufficient.

Actual staging/jobs/gates/proofs belong in receipts/TAG/campaign.json. Local
preparation is not a submitted job or a passed new native/scientific check.

## Attempt01 failed CPU; fixture-only attempt02

Source01 sealed622files, manifest47263fedea42e593455b43d4b75c1c874670bb53d934dd93a531aec439abf403;
CPU29781350 passed388contracts and failed3new fixture expectations; callback
29781351 EXIT correctly blocked all native/long submission. Legacy/mapping/
chemistry groups were not reached. The three fixtures still expected old0.05,
injected now-valid Pro0.3 as an invalid rate, and assumed a global tag in a nested
work3 leaf. These are fixture errors; preserve original raw XML/LSF/source.

Attempt02 is a new immutable tag20261022-brd4-jq1-25-fixed-lrreset-02 with sibling
restart-lrreset-02 and restart-lrreset-release-02/source under the same analysis
root, and a fresh dedicated home TAG. Only those three fixture expectations,
declared root identities and this history change; scientific config/driver/
contract/library bytes and every gate remain identical to attempt01. It repeats
the full allocated CPU suite. No pass or GPU/long ID follows from preparation.

## Actual CPU, verified-backup cleanup and native submissions

Attempt02 CPU29781363 actually DONE/hpc/LSFsuccess with723raw passing cases
(391contracts,239legacy,34mapping,59chemistry), no failures/errors/skips and
3 deliberately deselected legacy cases. CPU certificate
6e02392176f71238b1af36564c796615758ba82af7715ebe487bd48030a5a5d5 is bound to
source622 manifest832f833dd486c2ff305f7e3eb33fc775c87810a2e12a8bc597c5df744699c5d3.
Callback29781364 correctly EXIT before any GPU submission because native-dispatch
storage required10200547328bytes against7836349422available. That failed gate
remains unchanged, SHA2c47b1e90fe4f7a4c2d259f95e370dea38dbef9c392c4becd7a186ebf31d8bc8.

The user's existing verified-backup cleanup authorization was applied only to
inactive older work3 copies. Complete local backup1620686314bytes SHA
1f6b91446dba1866ff7c6e732a1ce2457e60c59661943fdbd2eb6cf870a305f4 and all34847
original members passed independent streaming byte/size/path/hash verification
before removal. Exactly34847 matching remote files/1540133784logicalbytes were
retired, with no changed/absent files. Current higher-rate trajectories/diagnostic
inputs, all home files, source releases/models/caches/envs and other projects
remain untouched. The older lower-rate Pro/Bol study copies, fixed report duplicate
payload and named duplicate archives remain recoverable from the full local backup.
Plan, two unfinished-job inventories and complete durable per-file journal are
preserved and independently audited. Initial slow transfers/extraction and the
first PATH-only cleanup failure were retained; no partial was mislabeled passed.

A metadata-only operator then reauthenticated original CPU723/source622 and all
finished predecessors, took the same dispatch lock, checked absence of actual or
unknown native/long outcomes, and called the original tested submit APIs after
a fresh unchanged full group-storage pass:10200547328required versus10286294816
available with4GiB/100000files and the full old-analysis allowance. The failed
certificate callback was not rerun, no failed receipt overwritten, and no source,
scientific configuration, quota policy or gate was changed. Resource-adoption
SHA434a42093cb466c32eb7fe144eb952f462c72e75014c0a5347fd95b1370ce8d4.

| New-rate arm | Actual native validation | Own-ended callback |
| --- | --- | --- |
| Protenix0.3 |29783703|29783705|
| Boltz0.005 |29783707|29783708|
| Boltz+chemistry0.005, coefficient0.001→100 |29783710|29783711|

All six were actual PEND at10:27:52UTC/12:27:52Paris. No new native readiness pass
or10000-update long ID is established. Existing independent endpoint diagnostics
29780821/23, captures22/24 and final presentation29780830 remained PEND unchanged.
Read ROOT/native-submissions.json and exact own submissions; never duplicate.
The three original source622 callbacks will require all new native passes and
fresh full work3/home gates before independent long submissions. Current storage
admission is not a full-run gate or a reservation.

Native/cleanup delivery7332680bytes SHAf6dadc17eb468f87a46cbb177f545682267b365c6963f6aec3a25210a8e8ecf6,
41files safely extracted/hashed. Independent complete submission/resource/source/
CPU/backup/journal audit passed4c7979a05a635d5f7f19feec407827eb8e8af43ecc3ccdaef0740918d69f8361.
Local archive/report/proof integrity checks are not numerical/model validation.


## 22 October 19:03 Paris: native allocations failed storage admission

All three original validation jobs29783703/29783707/29783710 reached an L40S at19:01-19:02 and exited1 before any model/gradient invocation. Their conservative byte budgets were short by8607251/8617739/8628214 bytes respectively; file headroom remained adequate. Own callbacks recorded failed decisions and no long-submissions file exists. These are resource-admission failures; CPU723/source622 remain unchanged and no native gradient result was evaluated. No retry, cleanup, policy change or submission occurred in this status turn. Exact gates, own-job LSF and decision hashes were safely downloaded/verified in the [dated run record](../experiments/brd4-jq1/fixed-lrreset/runs/20261022-brd4-jq1-25-fixed-lrreset-02/status-20261022-1903.md).


## 22 October: authorized retry03 after verified storage recovery

The user explicitly requested enough space and resubmission of all three lower-rate arms with identical settings, plus a separate Protenix diagnostic correction. Removed1212 exact duplicated report frames/curves/archives,852317966 logicalbytes, only after both complete local report archives and all636/647 memberfiles were reverified. Original native studies, proofs/configs, releases, home/caches/models/envs and other projects remain unchanged. Source622 scientific library145files/configs/driver/contract byte-identical; new launcher/storage roots and conservative separate diagnostic charges were allocated-retested. CPU29792104/afterCPU29792105 DONE/full724passes. Actual new native/callbacks Pro29792127/28, Bol29792129/30, chemistry29792131/32 allPEND19:36Paris; no readiness pass or10000-update long ID established yet. Full native/whole two-filesystem gates remain mandatory. [Current retry record](../experiments/brd4-jq1/fixed-lrreset/runs/20261022-brd4-jq1-25-fixed-lrreset-03/README.md).


## 22 October20:10Paris: both Boltz readiness checks passed; Pro initialization blocked

Native29792129/29792131 actualDONE/LSF and full readinesspassed at newLR0.005, including chemistry low0.001/high100 coupled checks and both ten-update pilots. All52/128 declared proof hashes matched. Pro29792127 actualEXIT1 before raw gradients: exact comparison to original initialization failed150/150 atom entries, reported maximum absolute difference5.9604645e-08. This is an initialization identity failure, not established gradient disagreement; the exact gate was preserved. Pro callbackEXIT and chemistry callbackEXIT block global dispatch despite both passed Boltz readiness decisions. No long-submissions or dispatch marker exists, so none of the three fresh10000-update runs has started. Corrected separate Prodiagnostic29792158/capture29792159 remainPEND. [Dated evidence and verification](../experiments/brd4-jq1/fixed-lrreset/runs/20261022-brd4-jq1-25-fixed-lrreset-03/status-20261022-2010.md). No scientific action or new job in this status turn.
