# 500-update carbon-chain experiment

Requested and executed 2026-10-16. Run two fresh carbon-chain trajectories
with exactly 500 accepted updates each, from the same 15%/30% relaxed neutral
decane starts as the completed [100-update study](carbon_chain_100.md).
This is 500 total updates per start, not 500 additional updates after its endpoint.

Pivotal comparison choice: reuse the authenticated completed study's prepared
protein/ligand context and frozen model. Rebuild the same hard fixture and
relaxation, and start at update zero. The new explicit
`--allow-update-count-change` replay policy accepts only a strictly longer
integer `checks.steps`; every other configuration field, including precision,
RNG, relaxation strengths, pocket, step size and tolerances must match exactly.
Previous summary SHA-256:
`81ae15d61983690fe8a260360ba856e27c17b0ea90be0da8492f251ed6240d51`.
The usual fixture/model/context hash checks remain required. This preserves
the input identity rather than generating a different native conformer.

The new [configuration](../examples/configs/carbon_chain_500.json) retains
highest FP32 matmul, seed 17, the full synthetic 20-residue protein context,
selected F5/G6/H7/I8/K9 region, and the conservative complete-bin `<8 Å`
contact objective. Keep all initial derivative/null/parity checks. Every
accepted update must decrease both AD-evaluated and separately measured
primal loss by more than `1e-7`, with initial eta 0.01 and ten halving trials.
Do not silently reduce the requested count or relax gates if a check fails.

Save all 501 states per case, raw gradients, rejected proposals, detached
unrepaired hard readouts and source/config/environment/model/context identities.
Chemical readouts never guide updates. Fixed occupancy/charge and existence-only
native bonds still apply; these synthetic contact scores do not establish binding.

The 100-update study contains about 289 MB of raw study files. Budget roughly
five times this data plus an evidence archive and analysis outputs, conservatively
reserving 4 GiB for new artifacts while retaining the earlier 4 GiB headroom.
Require **8 GiB and 10000 files free** at allocated GPU preflight. Current
metadata quota is 90.20/100 GiB, with sufficient file headroom. No cleanup is
planned. Abort before numerical work if quota verification/headroom fails.

Numerical tests and native optimization run on an allocated GPU node; unit
checks use its CPU to preserve the fresh-process GPU boundary. After successful
model completion, render on a separate allocated CPU job. The report derives
the update count from configuration, keeps bold readable atom labels, uses
fixed probability scales for atom circle/bond sizes, and provides a 0..500
slider/playback plus molecule/probability/loss views. Preserve the 100-update
report and give the 500-update study its own immutable source, run directory
and local result directory. Actual receipts and outcomes belong in
[the ledger](experiment_ledger.md).

## Additional 50% initialization

The user subsequently requested a 50% uniform mixture, also for 500 updates.
The original two-case run continues unchanged. Submit a separate immutable
one-case run in a serial allocation after it, reusing the same authenticated
100-update context. The new explicit `--allow-mixing-values-change` must be
combined with longer-budget replay; only the mixture list and step count may
then differ from the prior 100-update configuration. All other input and
derivative/descent/identity gates remain required. Preserve both original runs.

The full three-case storage estimate is about 2.2 GB raw studies plus an
approximately similar evidence archive and overhead, fitting the inspected
9.8 GiB free quota while retaining at least 4 GiB. Each numerical allocation
retains the stricter 8 GiB preflight; stop rather than clean up or loosen gates
if that conservative requirement is unmet at its actual start.

The comparison report can read both completed studies directly, without
duplicating raw trajectory files. It requires identical fixture/hard graph,
model and context digests, precision profile and all non-mixture configuration
fields, rejects failed/incomplete or duplicated mixture cases, and retains
both native summary/config/job identities. The combined display configuration
has no native configuration SHA; real hashes are listed per source study.
Use distinct teal/orange/purple curves, three summary cards and probability
heatmaps for all cases. Neither the original 100-update report nor its data
is overwritten.

Pivotal storage follow-up: the original two-case job passed, but the first
50% allocation exited **before testing** because free quota was about 30 MiB
below the deliberately conservative 8 GiB gate. Preserve that failed preflight.
To recover space without changing the gate, remove only the two new redundant
compressed upload archives for the 500-update releases after verifying their
SHA-256 both locally and remotely. Their authenticated local archive copies,
manifests, read-only unpacked HPC source and every experiment artifact remain
intact; these transfers are recoverable byte-for-byte. No other cleanup is
authorized by this operational choice. Recheck quota and retry in a new run
directory only after it exceeds the same 8 GiB threshold. The earlier queued
comparison report is intentionally cancelled and replaced with a new identity.

## Completed three-case result

GPU jobs **29735858 DONE** (15%/30%) and **29735926 DONE** (50%) completed
500 accepted updates per start: **1500 updates and 1503 saved states** total.
Selected unit checks passed: 134 on the original source and 141 on the added
case/report source. Initial native derivative/null/parity gates, every accepted
AD/primal decrease, and frozen model/context identity checks passed.
Comparison rendering **29735927 DONE** used both saved studies directly;
all common input/identity guards passed. No model was rerun for the report.

| Uniform mixture | Primal loss, initial → final | Distinct slot-labelled hard graphs | Final fragments |
| --- | --- | --- | --- |
| 15% | 1.844622493 → 0.508229554 | 4 | 3 |
| 30% | 1.609976292 → 0.553653359 | 3 | 2 |
| 50% | 1.591693997 → 0.668214560 | 2 | 1 |

Every one of the 501 readouts per start sanitized. In all three final hard
graphs slot 3 changes from C to S. The 15% final readout is
`C.CCCCCCC.CS`, the 30% readout is `CC.CCCCCCCS`, and the 50% readout is
`CCCCCCCSCC`. Sanitization and lower contact loss do not imply a connected
ligand: the first two starts remove two and one chain bonds, respectively.
No connectivity/chemistry constraint, repair or fragment removal was added.
This remains a synthetic protein/context gradient experiment, without a
biological binding claim or full Stage 0 A–G acceptance.

The [downloaded offline report](../results/hpc/20261016-carbon-chain-500/report/report.html)
keeps the earlier probability-sized circles/bonds and fixed bold element/slot
labels, now including S3. All three mixtures have a 0..500 slider, playback,
snapshot gallery, loss/change plots and probability heatmaps. Browser QA
verified the 50% endpoint (loss 0.6682146, connected `CCCCCCCSCC`), readable
element labels, the third heatmap and no JavaScript errors. The 100-update
report remains separate. Source/job/quota/download receipts and the preserved
storage-stopped attempt are in [the ledger](experiment_ledger.md).

Evidence packaging **29735961 DONE** and local download verification completed:
the 2,020,603,152-byte archive matches SHA-256
`dc1e1d948c99d487f5c06897964037cd2130a983f003ad4fc1643d996bad14ba`,
and all **4619 declared files** pass size/hash checks. All ten local report
files match the verified archive byte-for-byte. Archive, extracted evidence,
download receipt and report-copy receipt are retained in
`results/hpc/20261016-carbon-chain-500/`. Local verification used only stdlib
file integrity checks; all numerical evaluation/rendering ran on allocated HPC.
Final recorded HPC quota is **94.31/100 GiB**, leaving about 5.69 GiB;
266757/20000000 files are used. No additional cleanup was performed.

Protein clarification: the frozen context contains the complete synthetic
20-residue sequence `ACDEFGHIKLMNPQRSTVWY`. The selected contact region is
one-based residues **F5, G6, H7, I8, K9** (zero-based ordinals 4 through 8).
There is no named biological target or experimentally established binding site.
