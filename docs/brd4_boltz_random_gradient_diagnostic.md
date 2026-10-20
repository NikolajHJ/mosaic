# BRD4 Boltz random-initialization gradient diagnostic

**Current status, 20 October 2026, 17:25 Europe/Paris:** replacement diagnostic
**29763693 PEND** was actually submitted at **17:25:41**, with resource
arguments and scheduler state authenticated. Launcher05 is authenticated;
CPU 29763691 DONE/passed 20 in 0.95 seconds (14 proof contracts plus six
unchanged diagnostic contracts), with all 13 downloaded proof files verified.

Original diagnostic 29759213 remains EXIT1 before model execution because
its old live CPU lookup aged out. Its failure evidence, source03 reserve
failure and source04 actual 19-test pass are preserved below. Scientific
source02, initialization, graph/context and all original FD gates remain
unchanged. Random's native derivative failure remains unresolved: no new
GPU diagnostic result and no long random optimization is claimed. Mix50
and its verified report remain unchanged.

Independent mix50 run 29759212 is now scheduler DONE with actual LSF successful
completion at 08:35:34. Completion metadata authenticates 10,000 contiguous
updates and 10,001 contiguous saved states, matching JSONL/case/summary,
six slots, passed initial/coupled/terminal gates and frozen identities.
The 21-file compact evidence bundle, including snapshots 0/5000/10000, is
locally verified; all snapshot bytes and a full rendered report are outside
that compact verification scope. This does not certify random.

## Preserved native failure

Source02 readiness 29759069 ended EXIT1 at 20:09:29. Mix50 passed its complete
native gates; random passed hard/native integer parity and all three atom
finite-difference checks. Random's typed-bond analytic derivative was
9.497333526611328:

| Epsilon | Independent central difference | Original gate |
| --- | ---: | --- |
| 0.01 | 7.599136233329773 | Failed |
| 0.003 | 12.296060721079508 | Failed |
| 0.001 | 9.71454381942749 | Passed |

The unchanged requirement is two adjacent passing scales at 15% relative
tolerance. One of three does not qualify. The failed parent and random
summary remain authoritative; arm-specific mix50 adoption does not certify
random. Random summary SHA
`bc8c12ea3df2a7455da6cadd84b69f8c7b9bf97ed33947c467d8aba867c87ad0`.
All 171 parent evidence files are verified at
`results/hpc/20261019-brd4-boltz-affinity/native-fd-failure-01/verification.json`.
No concrete mapping, transport or backward bug has been established.
Nonlinearity and FP32 effects are hypotheses, not reasons to waive the gate.

## Exact replay and diagnostic comparisons

The diagnostic uses the exact failed probability bytes from
`GraphLogits(saved_logits).graph()`. Raw sampled X/E is not interchangeable
with this logit-roundtripped graph. Keep six occupied neutral slots, all
context fields, frozen model and original configuration unchanged.

The direction is fixed at unordered bond row 2, aromatic category 4 to NONE
category 0. Retain all original central-difference scales 0.01/0.003/0.001,
15% tolerance, finite-value checks, independent primals and the requirement
for two adjacent passing scales. Save every raw value and boolean; do not
select another graph, direction or subset to manufacture a pass.

At the same fixed point, repeat no-grad forward, AD and grad-enabled forward
to distinguish repeatability from mode-specific discrepancies. A declared
checkpoint-disabled control keeps its distinct execution mode and outputs.
Additional 0.0003/0.0001 scales characterize local behavior only. Neither
these scales nor the control replace historical validation, certify random
or select a production configuration.

Authenticate the saved graph and these prepared native identities before
execution, and verify frozen identities afterward:

- Model digest `395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16`.
- Context tensor digest `495002ffa2a52cc85b58ef34424622763dfc4ca556139a98297e8789ae287aa5`.
- Context file SHA `254d5b7f11605fb41db6f5abf7015f9cc856a19dfe1d4b506ccaea9007200770`.

Persist forward modes, repeats, gradients, probes, timings, source/asset/
configuration hashes and actual scheduler outcomes. All numerical tests and
native calculations remain on allocated DTU HPC. Local work is static
editing, packaging and byte/schema verification.

## Authenticated tooling and actual submission

Tools01 omitted the unchanged CPU-proof verifier: CPU 29759210 EXIT retained
13 FileNotFoundError arm-fixture failures and six passing diagnostic tests.
That packaging failure, source and logs remain preserved in the ledger.
Corrected external tools02 added only the verifier; its prior seven files
and source02 scientific bytes were unchanged. All eight files authenticated
at `/work3/s234818/mosaic/releases/20261019-brd4-boltz-independent-02/source`.
Manifest SHA
`848cb213d91be09513439f719ece8e43ec55991bd473f1bc233f3d80eb204c1b`;
archive 18,486 bytes, SHA
`7d5516aae2fa6bb0a8fee2625258fe86d32049d207f2424698f7f7bba8adaa2d`.

CPU 29759211 DONE/passed 19 contracts, 13 adoption and six diagnostic contracts,
zero failures/errors/skips; scheduler/LSF success authenticated. Assessment
SHA `bc76e33927104e88a9e019b706d0ed4c6e1fc8f3bd4460264acad13ca7594448`.
The companion mix50 adoption rehashed 147 proofs; its pins are in
[the campaign plan](brd4_boltz_affinity_10000.md). These are engineering
contracts and evidence adoption, not a native random-gradient pass.

Diagnostic 29759213 requests one exclusive L40S in gpul40s, four CPU slots,
8 GiB per slot and 1h. Output:
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-boltz-random-diagnostic-01/`.
Its 20:41:42 PEND reason was unavailable ngpus_physical reservation on three
hosts, with no initial or inter-arm dependency. Actual submission records:
`results/hpc/20261019-brd4-boltz-affinity/independent-02/gpu-submissions.json`.
The 30-file CPU/adoption/submission proof bundle passed safe extraction,
every-file length/hash and XML checks at
`results/hpc/20261019-brd4-boltz-affinity/independent-02/proofs-01/verification.json`.
Archive 126,516 bytes, SHA
`414dfa385545b8f7e3dc751fbbd4bd964f3e7926ad6feb48f23fafa9de164546`;
manifest SHA
`8e645c895fb51178ff96e25a732dbe077b52281bff91e4f4528644612f00462b`.

Random remains unapproved for long optimization until an actual resolution
passes validation. No diagnostic result or tolerance/model/initialization
change has been selected. Mix50's separate authorized execution does not
depend on diagnosing random and does not relabel the failed parent passed.

At 20:42:41 Europe/Paris, mix 29759212 remained RUN and its repeated initial
source02 raw native FD report passed; this is not trajectory completion and
does not change random's failed gate. The automation API confirmed existing
heartbeat `finish-combined-brd4-trajectory-report` ACTIVE, no duplicate. It
follows the unchanged mix run and diagnostic, stays quiet on unchanged state
or outages, and requires all original gates after any concrete bug fix.
A genuine unresolved curvature blocker pauses further work; no tolerance
or production-protocol change is selected.

## Live scheduler lookup expired; launcher-only retry preparation

The 17:15:21 successful connection authenticated diagnostic 29759213's actual
09:52:19 start/09:52:22 EXIT1 on n-62-13-16. Its assertion queried the old
CPU 29759211 live rather than accepting the persisted authenticated actual
scheduler-final/LSF-success records. It failed before loading the model or
running any diagnostic mode, so no checkpoint-control result or repeated
finite-difference evidence exists from this job.

The user explicitly authorized a launcher-only fix using those persisted
actual final outcome records and their hashes. This does not authorize
fabricating a DONE response, weakening source/assessment/proof checks or
relabeling the failed diagnostic successful. Preserve the original release,
EXIT1 and logs; prepare a new immutable tooling retry with allocated tests.
Scientific source02, initialization, fixed graph/context and every original
finite-difference scale/tolerance/gate remain unchanged. Verified failure
hashes, new launcher identities and actual CPU submission are recorded below;
the CPU result and GPU retry remain pending. No new numerical pass is claimed.

## Verified launch failure evidence and launcher03 CPU submission

The preserved original LSF EXIT1 footer and logs authenticate the failed
diagnostic outcome. `bhist` and `bacct` were permission-denied, so no
scheduler accounting history is fabricated. Evidence capture01 stopped on
the bhist PermissionError and remains preserved. Unique capture02 records
the unavailable accounting tools and actual LSF outcome instead.

Failure archive
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-diagnostic-launch-failure-02.tar.gz`,
5,604 bytes, SHA
`c4b265ab600840b0b0a754e5ac10a8c8c5f37c647f82821a1523a281045b4323`;
manifest SHA
`14f8c555993de01165254d6b9bab2fc5b2f8a2384c5923ffdfa18e2dbf75121b`.
Downloaded, safely extracted and all 11 files byte-verified at
`results/hpc/20261020-brd4-boltz-diagnostic-launch-failure-02/`.
This verifies engineering-failure evidence, not a native diagnostic pass.

New launcher03 authenticated all seven files at
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-diagnostic-launch-03/source`.
Manifest SHA
`dceadf75e4a0ae84e41cd325c5205c4f4d8d54ee43ba735f660bc4e1ca38f2c2`;
archive 13,945 bytes, SHA
`8cf264153647df1149f79e6cf5cc05933b2c509c4ae8515dc3947392b9adfe17`.
Two independent static owners reviewed the proof helper/wrappers without a
blocking finding. Diagnostic implementation bytes remain identical to the
original; scientific source02 and every native scale/tolerance/gate remain
unchanged.

Actual CPU 29763673 is submitted to hpc, one slot, 2 GiB, 15 minutes. Expected
coverage is 19 contracts: 13 new persisted-proof and six unchanged diagnostic
contracts. No outcome/pass is yet claimed. GPU retry remains unsubmitted
until actual CPU completion and authenticated persisted proofs pass.

## Launcher03 CPU reserve configuration failure

CPU 29763673 actually EXIT1 at 17:20:21 Europe/Paris before pytest or native
diagnostic/model execution. `zhome_quota_preflight` rejected a requested
1 GiB storage reserve because its minimum is 4 GiB. The gate was not waived;
this parameter rejection does not establish inadequate available capacity.
No 19-test pass or numerical failure is claimed. Source03 and failed job/logs
remain preserved. Preparing a unique launcher04 changing only the CPU
wrapper reserve to 4 GiB, followed by a new allocated CPU run. New hashes,
submission IDs and outcomes remain pending. Original diagnostic/science02
and all native gates remain unchanged; no GPU retry or long random run.

Launcher04 subsequently authenticated all seven files at
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-diagnostic-launch-04/source`.
Six files match 03 exactly; the 2,925-byte CPU wrapper differs only by the
single reserve character 1-to-4. Manifest SHA
`2a35a34f8db8c5682139f341ec08e08da1a049d8da1cf0423ef9e5f3b4ec2754`;
archive 13,992 bytes, SHA
`90483713ead73297b9481b1aced3978a84b1a20f1c7c78341c02704d13772457`.
Actual CPU 29763686 submitted hpc/one slot/2GiB/15min; output
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-diagnostic-launch-04-cpu/`.
Its 19-contract outcome remains pending. Source03 CPU's actual scheduler-final
EXIT1 is captured, not erased. No retry GPU is submitted.

CPU04 subsequently **29763686 DONE at17:23:10 Europe/Paris**,19 tests passed
in 1.13seconds, fresh 4 GiB reserve and actual LSF success authenticated. This
result remains a valid 19-test engineering pass. Before GPU certification,
the actual bjobs response wrapped status as `D\nONE`; the literal comparison
would reject that genuine DONE token. No failed GPU certification or native
run is inferred. Preparing source05 to normalize whitespace within status
and queue fields, then require the same exact `DONE`/`hpc` values. This is
parser robustness, not accepting a different state or fabricating success.
One additional contract brings planned source05 coverage to 20, unexecuted.
New pins/jobs/results are pending; no GPU retry submitted yet.

## Launcher05 authenticated; allocated CPU 20 passed

All seven source05 files authenticated at
`/work3/s234818/mosaic/releases/20261020-brd4-boltz-diagnostic-launch-05/source`.
Manifest SHA
`e14c41ada91a0a1750776d57affbb33edb782f6d05cacd3519fa543b338960be`;
archive 14,124 bytes, SHA
`bd70f72ed4814e2627371b202db5927c5b1034c4b879676f82fa3522bf5c721d`.
Actual CPU 29763691 DONE/passed 20 in 0.95 seconds, zero failures/errors/skips;
scheduler, LSF, assessment and XML authenticated. Quota recorded 15 GiB
conservative free. Assessment SHA
`36857a6fc0dccef932ffe26aac9b742a5fe1291aec35884a7aa7bb6ca38a8702`.

All 13 CPU05 proof files are locally verified at
`results/hpc/20261020-brd4-boltz-diagnostic-launch-05/cpu-proofs/verification.json`.
Proof archive 6,510 bytes, SHA
`e9c03d3200c97de550a89e3f8be97fc3ee0f1eb264d3e2aafd9c32d279088fd7`;
manifest SHA
`604c444d05e899296ecc95f7d766e90bbc760c394070150a1f99805369de0530`.
The earlier CPU04's 13 proof files were also verified: archive 6,465 bytes SHA
`a9296679b4d0fb703a127dfdab2cde3c77d8e1f56e7e838dd4138527b529cb1e`,
manifest SHA
`fe27760bafd9b965a0c00eb281bb14a9fe149903fc7caa634a41746e7ecab8f9`.

These are launcher/proof engineering results, not random native FD passes.
Original diagnostic/scientific bytes and all gates remain unchanged.
Replacement GPU diagnostic **29763693** actually submitted at 17:25:41
Europe/Paris on 20 October 2026, initial PEND waiting for scheduling. Verified
arguments/state: gpul40s, four CPU slots, 8 GiB per slot, one exclusive L40S GPU,
60 minutes, no inter-arm dependency. Fresh conservative home free 15 GiB.
Output parent:
`/zhome/95/e/206121/mosaic-runs/20261020-brd4-boltz-random-diagnostic-02/`.
Actual local submission record:
`results/hpc/20261020-brd4-boltz-diagnostic-launch-05/gpu-submission/submission.json`.
Submission is not a start, result or pass. No long random run is submitted.
