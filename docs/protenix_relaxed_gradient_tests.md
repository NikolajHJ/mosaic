# Protenix gradients into a relaxed molecular graph

> **Scope and access note (22 October 2026):** this document contains an
> experiment protocol and dated operator evidence. Concrete account paths,
> environments, jobs and source hashes describe that operator/attempt; they
> are not shared-account defaults or a fresh status/acceptance claim. Use
> [your own HPC setup](HPC_GUIDE.md). Keep the scientific gates and historical
> outcomes below intact when preparing a separately validated candidate.
> Canonical working home: [graph-validation/relaxed-gradients](../experiments/graph-validation/relaxed-gradients/README.md).


Completed: 2026-10-16. Status: **focused study passed on allocated DTU HPC; source-04 job `29735139` DONE**. All earlier failed jobs remain retained. The user confirmed DTU HPC access and authorized these tests on an allocated node. Full Stage 0 A–G acceptance remains pending.

Both relaxed initializations produced finite nonzero atom and edge-existence gradients, passed all three finite-difference scales for both leaves, and completed three declining AD-evaluated projected-step losses at `eta_X=eta_E=0.01`. An independent standalone primal forward verified the initial-to-final loss decrease; intermediate standalone-primal losses were not separately evaluated. Standalone primal endpoint losses were:

| Uniform mixing | Atom tangent derivative | Edge tangent derivative | Initial primal loss | Final primal loss |
| --- | --- | --- | --- | --- |
| `0.15` | `0.1835573018` | `0.1163473874` | `1.7413299084` | `1.7370867729` |
| `0.30` | `0.4851007015` | `0.5728260279` | `1.6747362614` | `1.6332819462` |

Actual checks: **126 ordinary unit checks passed in 59.66 s; seven A10 feature checks passed in 25.55 s; one native pytest item covering both graph cases passed in 207.07 s**. Model and prepared-context array digests were unchanged before/after. Bond-order-null gradient, standalone-primal scalar and output differences were exactly zero in both cases. Results are downloaded to `results/hpc/20261016-relaxed-gradients/20261016-gradient-evidence-04/`; the full summary is under `runs/20261016-relaxed-gradients-04/payload/native-results/contact-gradients/summary.json`. These are engineering results on a synthetic protein fixture, not evidence of binding or chemical validity.

This is a bounded engineering test of the [DBMol idea](https://arxiv.org/html/2607.19237v1): keep a pretrained structural model frozen, represent a molecule continuously, differentiate a model-derived contact objective into that representation, and take small projected graph updates. It uses Mosaic's existing direct graph representation rather than a learned molecular decoder. It does not reproduce DBMol's full affinity/confidence, denoising, charge, chemical-prior or synthesis objectives, and it does not claim binding or chemical validity.

## Shared molecule and frozen model

The input is the existing [Stage 0 fixture](../tests/data/small_molecule_stage0/fixture.json): acetanilide, `CC(=O)Nc1ccccc1`, with ten occupied heavy-atom slots. The conditioning protein is the 20-residue engineering sequence `ACDEFGHIKLMNPQRSTVWY`, with one query MSA row and no template/search. The fixed five-token pocket selects zero-based ordinals `[4,5,6,7,8]` in the prepared protein-token map, then records the corresponding actual native token indices. This sequence and the selected pocket subset are engineering fixtures, not a validated binding target or a known pocket.

Only two probability arrays are differentiated: `X[10,6]` over `C,N,O,S,F,Cl`, and `E_upper[45,5]` over `NO_BOND,SINGLE,DOUBLE,TRIPLE,AROMATIC`. Every row is on its simplex; the two test cases mix initial hard labels with the uniform distribution at weights `0.15` and `0.30` to permit two-sided simplex-tangent perturbations. Both use seed `17`, the same prepared context and a fixed RNG key for every finite-difference/update comparison. All ten slots remain occupied and their formal charges remain zero. Explicit native atom/token maps determine where these arrays enter the model; no ligand-first indexing is assumed.

The frozen model is native `protenix_tiny_default_v0.5.0`, using the audited Protenij revision `97cf8172f61829681582f06de7052485bc94eae4` and pinned assets from `nickrb/protenij` revision `81a257b61a82a17c41a7c20638460c04c29d97ad`. The [native contract](stage_0_native_contract.md) records the complete file/feature boundaries. Model parameters, protein features, maps, occupancy, conformer bytes and reference bookkeeping remain fixed. Ligand geometry is disabled at each audited geometry consumer while element/name/charge conditioning and occupancy remain active. Only one final trunk recycle and the native distogram are used; no diffusion or confidence head is required.

Element probabilities enter native floating element features. Bonds enter as floating edge-existence probabilities `B_ij = 1 - E_ij[NO_BOND]`. The four present-bond categories therefore have identical native effects: a `SINGLE`-to-`DOUBLE` contrast must be invisible to this objective. A nonzero edge-existence derivative establishes a differentiable topology channel, not bond-order-specific or chemically informed guidance.

## Contact objective

The new reduction follows the pocket/anchor contact terms in [DBMol, section 3.2, equations 3–6](https://arxiv.org/html/2607.19237v1#S3.SS2). It uses explicitly mapped ligand tokens `L` and a declared nonempty protein-pocket subset `P`. For each ligand-pocket token pair, define

```text
C_ij = sum_k softmax(distogram_logits_ij)[k] * 1[upper_edge_k < cutoff]
D_ij = -log(C_ij + epsilon)
loss = (mean_j_in_P min_i_in_L D_ij + min_i_in_L,j_in_P D_ij) / 2
```

The first term rewards contact coverage of the declared pocket; the second rewards the best ligand-pocket contact. All reductions are over the native distogram and explicit maps. The contact probability uses the existing conservative whole-bin policy: a cutoff-straddling bin is excluded, and an infinite terminal upper edge is never counted as a finite-distance contact. Authored defaults are an `8.0` Å cutoff and `epsilon=1e-8`; these and the exact pocket selection are retained in the resolved configuration and output receipt. This score is an optimization probe; it is not calibrated affinity.

The hard minimum is piecewise differentiable. Directional finite differences can cross an active-pair change; raw objective values, directional derivatives and the full epsilon sweep must be retained so a mismatch is visible. Predeclared tolerances must not be relaxed after seeing the result.

The update also has a declared implementation difference: this study reuses Mosaic's Euclidean simplex projection for each atom/bond row; DBMol equation 8 clips and renormalizes its relaxed categorical updates. Neither operation establishes chemical validity. There is no learned chemical projection in this test.

## Numerical checks and acceptance

All numerical checks run on allocated DTU HPC. No project numerical computation runs on the Windows laptop. Source editing, static inspection, immutable packaging and receipt integrity checks may run locally.

| Check | Required evidence and interpretation |
| --- | --- |
| Frozen path and inputs | Authenticate consumed native source files, all five model/data assets, actual graph/context bytes, runtime versions, device and model weights. Record fixed seed/RNG policy. |
| Real forward/backward | Finite loss, native output, raw `X` and `E_upper` gradients, and finite nonzero simplex-tangent components in each branch. Model/context remain fixed. |
| Atom directional derivative | Central finite differences along an admissible atom-simplex tangent, compared with autodiff at every predeclared epsilon. Preserve all values and tolerance decisions. |
| Existence directional derivative | Central finite differences transfer mass between `NO_BOND` and a present-bond category, with all perturbed simplex rows feasible. Compare to autodiff using the same epsilon sweep. |
| Bond-order limitation | A tangent exchanging only present-bond categories has zero native effect within declared tolerance; this is a representation boundary, not an optimization failure. |
| Small projected optimization | Use the actual frozen-model objective and strict raw gradients, project atom/bond rows separately, re-evaluate candidate graph states, and retain a bounded loss trace. Report observed descent and any rejected candidate/projection/nonfinite failure. |
| Immutability and failure retention | Recheck model/context identity and the starting input after evaluation. Preserve failed outputs, commands and source/config identities; missing allocation/assets/quota is `tests_not_run`, not a pass. |

The study is narrower than full Stage 0 A–G acceptance. Passing it would demonstrate this contact objective's gradient and bounded graph-update behavior in the recorded fixture/environment. It would not by itself establish all-mode geometry fidelity, checkpoint replay, native-protein regression, chemical validity or a useful ligand-design method. A hard molecule readout, if emitted, is detached and cannot repair or replace the optimized graph inside differentiation.

## Execution and receipts

Use the dedicated resident Tiny environment/assets rather than a full unrelated-model installation. Before submission, bind an immutable source release including uncommitted source bytes, record its manifest hash, establish current quota headroom and use a fresh unique run directory. At least the existing Stage 0 requirement of 4 GiB and 10,000 files of headroom must be met, together with actual run-filesystem space. If quota is insufficient or unverifiable, retain the preflight receipt and leave numerical checks unrun. The historical 2026-10-10 quota is not current headroom.

Storage decision for this bounded study: a fresh BeeGFS path-to-pool lookup was denied because `connAuthFile` was unavailable, with no bypass/configuration change. Instead, inspect every work-pool quota and use the most restrictive finite headroom across all of them, independent of path mapping. The 2026-10-16 inspection reports exactly one finite work pool, pool 6, with `11.51 GiB` free and `241,957/20,000,000` files used; pools 1–5 report unlimited quotas. This conservative bound covers a path in any currently queried work pool; the historical same-path pool-6 receipt only corroborates placement. The job captures fresh complete work-quota output and checks pool 6 against the same 4 GiB/10,000-file threshold plus actual filesystem availability. Quota may lag usage and is not a reservation. No newly authenticated path-to-pool mapping or cleanup is claimed.

Exact CLI/configuration, release identities, scheduler resources/job ID, logs, model/context hashes, test results and downloaded artifact identities belong in [the ledger](experiment_ledger.md). A submitted job is not a completed result. The run should retain resolved settings, raw gradient arrays, derivative sweeps, projected-update attempts, immutability checks and a compact machine-readable summary. No training, alternate molecular identities, learned scorer, hidden discrete conversion or unrequested cleanup is part of this study.

The [focused CLI](../examples/protenix_relaxed_gradients.py) reads [its configuration](../examples/configs/protenix_relaxed_gradients.json). Run it from the authenticated immutable release inside the allocated GPU job, with a fresh output directory:

```bash
export PYTHONPATH="$PWD/src:$PWD/tests"
export HF_HUB_OFFLINE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
/work3/s234818/mosaic/envs/protenix-tiny/bin/python \
  examples/protenix_relaxed_gradients.py \
  --cache-dir /work3/s234818/mosaic/cache/mosaic/protenix \
  --config examples/configs/protenix_relaxed_gradients.json \
  --output /work3/s234818/mosaic/runs/UNIQUE_GRADIENT_RUN/payload/study
```

This is the intended command shape, not an execution receipt. The CLI refuses a missing allocation/GPU or an existing output directory. Each case uses the predeclared central-difference scales `[0.01,0.003,0.001]`, requires a nonzero tangent above `1e-7`, and must achieve three measured loss-decreasing projected steps. Finite differences must have the same sign as autodiff and relative error at most `0.15`, using `max(abs(FD),abs(autodiff))` as denominator, in addition to the combined absolute-plus-relative check (`atol=0.002`, `rtol=0.15`); two adjacent epsilon scales must pass. This prevents an unresolved zero finite difference from passing under the absolute tolerance alone. The initial atom and edge step sizes are `0.01`; bounded backtracking halves them for at most ten trials per step, accepting only an actual loss decrease greater than `1e-7`. Every attempted candidate and both sides of each finite difference are retained. The resolved configuration and checker report define the derivative tolerance decision; their exact final source/config identities must be sealed before execution.

First-attempt receipt: CPU setup job `29735006` completed on `n-62-31-22`, adding only the hash-locked pytest extension to the existing dedicated Tiny environment. GPU job `29735009` was submitted to `gpua10` for one exclusive GPU, four slots on one host, `8GB` memory per slot and one hour, with dependency `done(29735006)`. It initially waited for a physical GPU: both A10 GPUs were busy at that inspection, and reserved CPU slots did not establish GPU availability. A request to change the queue with `bmod -q gpua100 29735009` was rejected as administrator-only. A later state-guarded plan to cancel and replace the still-pending job through ordinary submission to `gpul40s` found the original job already `RUN` and refused the transition; no `bkill`, replacement `bsub` or replacement job occurred. The original A10 job subsequently **EXITed 1** because selected unit tests reported **119 passed, five failed, three deselected in 101.69 seconds**. The fail-fast launcher did not start the native study.

The first attempt exposed one real differentiable feature-boundary bug: A10 TF32 multiplication in the dense element expansion rounded `0.05` to approximately `0.04998779`. The correction replaces that sparse one-hot multiplication with indexed scatter, preserving the exact graph probabilities and their differentiable path. Four other failures concerned a JAX test's unsupported list indexing, parent/child exclusive-GPU contention in a unit resume test, float32 residuals in an exact boundary expectation, and an invalid Equinox wrapper-container object-identity assertion. Corrections use supported indexing, run the unit phase on CPU within the allocated HPC host, use a true overshooting projection fixture, and check shared parameter-array leaves. No numerical tolerance or model setting is weakened. Full failure logs remain retained; fixes require a new immutable release/run and an actual rerun before acceptance.

First-attempt source is the read-only release `/work3/s234818/mosaic/releases/20261016-relaxed-gradients-01/source`; run directory is `/work3/s234818/mosaic/runs/20261016-relaxed-gradients-01`. Local retained failure log: `results/hpc/20261016-relaxed-gradients/unit-01.log`; remote `payload/unit.xml` retains per-test results.

Corrected submission: source release `/work3/s234818/mosaic/releases/20261016-relaxed-gradients-02/source` is separately sealed and read-only; fresh run `/work3/s234818/mosaic/runs/20261016-relaxed-gradients-02` and GPU job `29735041` use the same one-exclusive-A10, four-core/32 GB-host, one-hour budget. The full unit phase actually passed: **124 passed, three deselected, two warnings in 55.77 seconds** on CPU within the allocated DTU host. The independent A10 feature phase actually passed: **seven passed in 23.85 seconds**, including exact relaxed-element preservation and raw boundary gradients. These checks verify the scatter correction on the GPU; they are not the full pretrained-model derivative result. The warnings concern integer-to-boolean scatter in existing contract-test fixtures.

The real native pytest item failed after **135.22 seconds**; job `29735041` exited `1`. Case 0 had finite initial contact loss `1.741985321`, atom-gradient norm `0.335869393`, and selected atom-simplex autodiff derivative `0.1824017093`. Its finite-difference results were:

| Epsilon | Finite difference | Relative error | Gate |
| --- | --- | --- | --- |
| `0.01` | `0.1733541489` | `4.96%` | pass |
| `0.003` | `0.0755389531` | `58.59%` | fail |
| `0.001` | `0.7123947144` | `74.40%` | fail |

No two adjacent scales agreed, so this is a **failed derivative check**. The study stopped before the edge-direction probe, projected updates or second mixing case. Case 0's report, initial gradient arrays and all six perturbed forwards remain under `payload/native-results/contact-gradients/case-0/`. Nonzero backpropagation is observed; numerical agreement and useful descent have not passed.

Precision decision: the reviewed pinned native/JAX paths use default matrix-product precision rather than an explicit high-precision override. Reduced-precision rounding is a hypothesis for the finite-difference disagreement, not a confirmed explanation. The new immutable profile explicitly sets `matmul_precision="highest"` before model evaluation and records the effective JAX option, x64 state and XLA flags. JAX exposes default matrix-product precision separately from array dtype; see [the official precision documentation](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html). Model weights, graph storage, objective, epsilon sweep, tolerance gates and descent threshold remain unchanged.

The matched precision run reuses the exact authenticated source-02 prepared context, rather than generating a new conformer and changing the inputs. CLI options `--prepared-study` and `--prepared-summary-sha256` bind the prior study summary, context byte hashes, fixture and model identities. Every configuration field except `matmul_precision` must match the prior run, and before-run model/context resident-array digests must equal the prior initial digests. Any mismatch stops the replay. The hard initialization graph is regenerated from the authenticated fixture outside differentiation and checked against the existing identity/map contract; actual optimized inputs remain the same graph mixes. The old failed precision profile remains a failure.

Immutable source 03 is `/work3/s234818/mosaic/releases/20261016-relaxed-gradients-03/source`; fresh run `/work3/s234818/mosaic/runs/20261016-relaxed-gradients-03`; actual job **`29735092` submitted** to `gpua10` with the unchanged one-exclusive-GPU, four-slot/32 GB-host, one-hour resources. The native test wrapper leaves JAX initialization to its child so the child alone owns the exclusive GPU. The separate allocated-CPU unit and allocated-GPU feature phases remain ahead of native validation. Exact submission is retained in ignored `results/hpc/20261016-relaxed-gradients/submit-highest-03.sh`.

The source-03 matched CLI invocation adds these arguments to the earlier command shape:

```bash
--prepared-study /work3/s234818/mosaic/runs/20261016-relaxed-gradients-02/payload/native-results/contact-gradients \
--prepared-summary-sha256 663835f89a54ba0e69e202f25062a7af12d5c4c42b4f795206652396fffe3577
```

[The ledger](experiment_ledger.md) retains all release hashes, commands, raw failures and the new profile decision. Actual job `29735092` subsequently exited `1`: selected CPU units passed in `57.45 s`, seven GPU feature checks passed in `23.86 s`, and the native item failed after `191.84 s`. The matched highest profile passed both case-0 directional finite-difference sweeps:

| Leaf | Autodiff directional derivative | Epsilon `0.01` FD (relative error) | Epsilon `0.003` FD (relative error) | Epsilon `0.001` FD (relative error) |
| --- | --- | --- | --- | --- |
| `X` | `0.1835568994` | `0.1834988594` (`0.0316%`) | `0.1834829648` (`0.0403%`) | `0.1833438873` (`0.1160%`) |
| `E_upper` | `0.1163478047` | `0.1169145107` (`0.4847%`) | `0.1162290573` (`0.1021%`) | `0.1159310341` (`0.3582%`) |

Raw gradient norms were `0.3360345716` for atoms and `0.2888619756` for bonds. Present-bond-only gradient contrasts were exactly zero, consistent with the existence-only native map. This establishes observed nonzero graph derivatives and numerical agreement in case 0 under the declared highest profile; it does not retroactively validate default-profile attempt 02.

The next output guard measured a maximum distogram difference of `0.000224113`, exceeding its unchanged `2e-5` threshold. The whole attempt remains **failed**, and no projected update or second case executed. Review identified a comparison confound: this guard compared the null graph's standalone primal forward against auxiliary outputs produced during the autodiff evaluation, which can follow differently compiled floating-point paths. The corrected control compares unperturbed and null graphs through the same standalone primal path, retains the same threshold, saves that baseline, and separately records the autodiff-versus-primal discrepancy as a diagnostic. This isolates the graph perturbation without hiding compilation-path differences. Its new submission is recorded below; the failed source-03 guard is not relabelled as a pass.

Matched-primal correction and final result: source `/work3/s234818/mosaic/releases/20261016-relaxed-gradients-04/source`, fresh run `/work3/s234818/mosaic/runs/20261016-relaxed-gradients-04`, actual GPU job **`29735139` DONE**, with the same A10/resources/highest profile and source-02 authenticated context. Source 04 keeps the `2e-5` same-primal null-output limit, adds `2e-5` scalar AD/primal parity, records the cross-path logit difference as a diagnostic, and requires final standalone-primal loss to improve by more than `1e-7` as well as the three actual evaluated candidate declines. All **126** selected unit checks, seven GPU feature checks and the two-case native item passed. The failed source-03 evidence is downloaded and verified locally (34 declared files; archive SHA-256 `675c497ecdbc3fd5fd55839e5220de70b263e263f27e84c9c3bcae8670607f5c`).

The first two failed attempts are downloaded locally under ignored `results/hpc/20261016-relaxed-gradients/`: archives/extracted directories `20261016-gradient-evidence-01` (setup and first failure) and `20261016-gradient-evidence-02` (second failure, including raw gradients/probes/context). Standard-library safe extraction and manifest checks verified 21 and 26 files respectively; sibling `*-download.json` records the archive hashes. These local operations inspect byte integrity only. The source-02 summary SHA-256 `663835f89a54ba0e69e202f25062a7af12d5c4c42b4f795206652396fffe3577` is the independent binding used for the completed exact-context precision comparison.

The final numerical profile was `jax_default_matmul_precision="highest"`, `jax_enable_x64=false`, empty `XLA_FLAGS`, with exact prepared-context reuse. Execution host `n-62-18-5` used NVIDIA A10, driver `615.71.09`, 23,028 MiB. Scalar AD/primal discrepancies were approximately `1.19e-7` and `7.15e-7`, within `2e-5`. The separately recorded maximum logit discrepancies were approximately `1.88e-4` and `1.09e-4`; the matched primal null probes had zero difference. These diagnostics remain visible rather than being conflated with a graph perturbation.

Final evidence archive `20261016-gradient-evidence-04.tar.gz` is **26,362,938 bytes**, SHA-256 `7ed2474ad86107a41895bcc79531ddff3d31609a8a9466182e8f71ca76536fb2`, with **63 declared files verified** after safe extraction. Final work-pool quota inspection was `89.01/100 GiB`, `250,966/20,000,000` files: no storage issue occurred. Exact release/configuration identities and all failed-attempt history remain in the ledger. Reproduce this matched study on an allocated GPU using the authenticated source-04 CLI/configuration, a fresh output directory and the `--prepared-study`/`--prepared-summary-sha256` arguments above; preserve the stated numerical profile and tolerance gates. The full two-mode A–G/resume/protein-regression suite remains a separate pending task.
