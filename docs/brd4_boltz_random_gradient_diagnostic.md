# BRD4 Boltz random-initialization gradient diagnostic

19 October 2026. User authorized investigating random gradients separately
while starting the independently passed 50% mix. As of 00:35 on 20 October 2026, Europe/Paris,
**diagnostic 29759213 PEND/gpul40s** waits for a physical GPU. It was submitted
at 20:40:18 with no dependency on mix50 long run 29759212, which started at
20:41:13. The random derivative failure remains unresolved; no long random
optimization is submitted and no production setting has changed.

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
