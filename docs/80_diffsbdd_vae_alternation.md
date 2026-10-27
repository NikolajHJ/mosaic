# Independent DiffSBDD / frozen epoch-39 VAE study

## Actual status, 27 October 2026

The user authorized implementation, mapping training and alternating-loop tests
on DTU HPC, proceeding only after required checks, in a separate DBMol branch.
The managed worktree is
`C:\Users\nikol\.codex\worktrees\diffsbdd-vae-alternation\DBMol_exploratory`,
branch **`codex/diffsbdd-vae-alternation-hpc`**. The original DBMol checkout and
Mosaic checkout were used as read-only references. No existing job, environment,
runtime, model weight, training setting or checkpoint was changed by this study.

**Storage admission failed before staging or submission.** The read-only receipt
at 11:00:54 UTC is [storage-admission-001.json](evidence/diffsbdd-v1/storage-admission-001.json).
It authenticates the active Mosaic whole-campaign forecast rather than reducing
its reserves or using an old DBMol free-space observation.

| Initial admission item | Bytes |
| --- | ---: |
| Conservative available capacity | 50,748,885,086 |
| Preserved other-campaign forecasts and reserves | 49,684,462,417 |
| New initial artifact cap | 2,147,483,648 |
| Additional unreserved headroom required | 10,737,418,240 |
| Total required | 62,569,364,305 |
| **Shortfall** | **11,820,479,219 (~11.01 GiB)** |

The fresh raw work3 quota lower bound was 57,885,421,731 bytes free. Even that
larger raw figure does not fit the total requirement. The conservative available
figure additionally preserves the active campaign's authenticated capacity bound.
These are budget observations, not newly created filesystem reservations.

No numerical fixtures, model loads, feature gates, fit updates, GPU sampling,
optimization or fresh-pose predictions have run for this experiment. New runtime
provisioning also remains pending; its measured footprint may require more than
the initial allowance. This is an implementation with a failed prerequisite,
not a trained mapping or a passed numerical experiment.

The completed local seal contains **312 authenticated source/asset members**,
29,291,600 unpacked bytes and a 22,298,467-byte archive. Exact
[seal](evidence/diffsbdd-v1/local-seal-001.json) and
[member inventory](evidence/diffsbdd-v1/source-manifest-001.json):

- Source manifest: `0138868abae498c7cb57a82b610fe35ddadbd45188fb9cc3164f108f0fa3f108`.
- Protocol: `1ea848311d287ccc4e82d2414737bce8aed3fe9a109e4cab97259a766d7a41ff`.
- Archive: `d63b914a414de3d764717ae38e98fea84a193d19b24e376bf12edb54bc4477de`.

The real `stage` command then repeated admission at **11:31:14 UTC** and
returned nonzero with `blocked_storage_before_any_remote_write`. The conservative
shortfall was unchanged; fresh raw work3 availability was 57,799,522,385 bytes.
The [terminal staging receipt](evidence/diffsbdd-v1/staging-blocked-001.json)
records `remote_writes=false`, `jobs_submitted=false` and the exact local seal.
This is a failed transport prerequisite, not an HPC numerical experiment.

## Frozen inputs and scientific scope

The [read-only asset receipt](evidence/diffsbdd-v1/read-only-assets-001.json)
verifies completed epoch 39 exactly:

`/work3/s234818/latentmol/runs/source138-vae-cli-startup-fix-attempt001/segments/003/20261026T040252Z_chemistry-train_40cdfc66/epoch_0039.pt`

SHA256 `e5d7f99b9eb0fe04cd8b6b800e4bd90b6f1c24a65368216ce384a95061a7f3b0`.
The independent VAE continuation has since completed epoch 40 according to its
separate saved status. This study deliberately retains the plan's epoch-39 pin.
It does not reuse the original36 learned Boltz bridge or its fit certificates.

The public DiffSBDD source is commit
`5d0d38d16c8932a0339fd2ce3f67ade98bbdff27`. Source hashes use raw Git blobs,
including the MIT license. Public conditional CrossDocked weights were downloaded
**locally as bytes only** from the Zenodo record and authenticated against the
published MD5/size. SHA256 is
`07f86764bf569aafbc40a9c15fc02de8e2550437dd0f17f657eab3abe66c372c`,
17,861,341 bytes. No checkpoint was unpickled or loaded locally.
[Exact public receipt](evidence/diffsbdd-v1/public-assets-001.json).

The [versioned protocol](../configs/diffsbdd_vae_alternation_v1.json) freezes all
source/checkpoint/data identities, data grouping, source limits, time levels,
training schedule, head architecture, audit thresholds, pocket, inner step
policy, transport limits, seeds/arms and resource caps. It uses a new query-only
MSA matching the exact crystallographic receptor sequence; that is a declared
engineering assay, not an evolutionary-MSA equivalence claim.

## Implemented boundaries

The [experiment README](../experiments/diffsbdd-vae-alternation/README.md) and
module docstrings are the code entry points. The original proposed plan is kept
there as historical material; DBMol owns all new executable code.

- `bridge.py`, `paired_data.py`, `training.py`: six-category constrained head,
  explicit native vocabulary/normalization, train-only statistics, five views,
  source/decoded/scaffold union before split, protected VAE holdouts, compact
  exact categorical storage, development selection and untouched audit.
- `native.py`, `native_checks.py`, `reference.py`: strict original checkpoint
  restore, state and Gaussian ownership, native centering/global frame,
  no-op correction, exact same-runtime replay and separate original Lightning
  versus combined-runtime qualification. No schedule tensor is discarded.
- `codec.py`, `pocket.py`, `transport.py`: complete canonical slot/native atom
  maps, ordinary VAE reconstruction before fixed occupancy, exact biological
  crystal pocket, native builder receipts and explicit T1/T2 pose transport.
- `scorer_context.py`, `scorer.py`, `guidance.py`: new official native features
  for each supplied hard graph and crystal-frame pose. Native reference
  conformers remain distinct from bound coordinates. Existing scorer/native
  AD validators are reused unchanged; no old carbon/JQ1 context is transplanted.
- `audit.py`, `alternation.py`, `runs.py`: held-out native denoiser comparisons,
  bounded latent descent, residual-preserving state correction, six matched
  arms and immediate opportunity/terminal/noise evidence. S starts after the
  native terminal draw. H0's zero-update path consumes no scorer gradients or
  additional Gaussian noise.
- `scripts/diffsbdd_campaign.py` and `scripts/hpc/diffsbdd_operator.py`: separate
  immutable namespace, byte inventories, fresh storage admission, explicit
  serial stage submission, exact predecessor passes, one numerical worker,
  internal process-group timeout and preserved failures/ambiguous intents.

The native sampler is detached. A derivative through DiffSBDD dynamics is not
used by the algorithm and is not a release prerequisite. Native output/boundary
replay remains mandatory; every optimizing Boltz start retains the full native
component/total independent forward/reverse AD and additional resolved FD checks.
The optional native-denoiser derivative diagnostic cannot replace those checks.

Native XYZ serialization rounds coordinates to three decimal places before
OpenBabel bond perception. That transformation is recorded separately from the
full-precision pose used by the scorer and native state; it is not hidden by
loosening the sampler replay tolerance.

## Validation and promotion rules

Only local source editing, AST parsing, JSON/hash checks, public byte transfer
and read-only HPC metadata probes are permitted so far. Authored fixtures are
not passed fixtures. The allocated stage sequence is CPU contracts, native
replay, fresh scorer/context gates, mapping fit, denoiser/T2 audit, smoke and
dense pilot; independent fresh native pose confirmation is separate. No stage
may infer a predecessor's success from submission, elapsed time or missing files.

[Final static inspection](evidence/diffsbdd-v1/static-inspection-002.json) parsed
35 Python files and counted 133 authored test-function definitions. It imported
no numerical module, executed zero fixtures, and loaded no model. The earlier
static-inspection001 is retained as intermediate source evidence. Neither
receipt is a substitute for allocated numerical/native validation.

The fit performs three disposable train-only calibration updates and complete
development timing before its actual first update. It must fit the full
10,000-update/development/audit forecast at factor 1.5 plus 120s. Calibration
weights/moments are discarded, actual initialization and row RNG are preserved,
and a failed forecast leaves actual fit updates at zero. These implementation
contracts themselves remain numerically unverified.

All updating admissions require complete hard identity, supported neutral
25-atom chemistry, same pocket/frame, frozen weights and fresh native scorer
certificates. T2 release additionally needs actual bond-changing native
recovery with a nonworsening hard native objective. A soft-only change is logged
as such. A feature fit, an inconclusive FD probe or zero effective transfers
cannot be promoted into a useful-optimization claim.

Every stage must fit a measured forecast with factor 1.5 and the sealed resource
budget. Exhausted time/storage or failed gates stop the campaign, retain all
attempts, and require a new immutable attempt for corrections. No seed, start,
held-out identity or inherited tolerance is silently replaced.

## Resumption boundary

First obtain a fresh passing capacity receipt preserving all active campaign
forecasts and the additional 10 GiB headroom, with the measured new-runtime
footprint included. No cleanup or reserve reduction is authorized by this study.
Then provision and authenticate the two new runtimes in allocated resources,
stage the sealed source, run CPU/native/context checks and advance only through
passed stages. The [HPC notes](../experiments/diffsbdd-vae-alternation/hpc/README.md)
keep runtime admission distinct from model/sampler qualification.
