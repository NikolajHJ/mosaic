# Structural audit of the frozen selected candidates

Prepared 28 September 2026 for **immutable source019**. CPU job **29505852
completed successfully** on queue hpc after **four tests passed**, with two
import warnings. All parsing, standardization, descriptors and tests ran on DTU
HPC. Audit run `20260928T085606Z_selected-molecule-chemical-audit_3d54051c`
records `completed`.

Local archive: `outputs/source-releases/latentmol-source-019-chemical-audit.tar.gz`.
HPC archive: `/work3/s234818/latentmol/releases/source-019-chemical-audit.tar.gz`.
SHA256: `db046fc49f904df8868db24f09e6814831b1103bbd04496f268c5270dd5ec608`.

The audit asks what structures the fixed candidate pools actually contain,
which submitted identities collapse under native affinity preprocessing, and
which structural features merit further investigation. It does not measure
chemical stability, synthetic accessibility, route recovery or binding.

## Observed structural and identity results

| Diagnostic | Observed count |
|---|---:|
| Distinct submitted input strings | 9 |
| Distinct available native standardized identities | 7 |
| Submitted identities changed by standardization | 3 |
| Native standardization failures | 0 |
| Selected endpoint records across models/arms | 10 |
| Endpoint records still different from both standardized controls | 6 |
| Endpoint records collapsed to standardized start or baseline | 4 |
| Endpoint identities unknown because preprocessing failed | 0 |
| Submitted structures with RDKit catalog alerts | 5 |
| Submitted structures with an oxygen-containing three-membered ring | 2 |

The ten endpoint records preserve arm/model membership and can include the same
molecule more than once. Four purported input-graph changes disappear when the
native endpoint is compared with its standardized controls. Retain those attempts
and costs, but do not count them as distinct native molecular changes. The six
remaining records establish identity differences only, not useful chemistry or
improved binding.

Five alert-bearing inputs and two oxygen-ring motifs are screening observations.
They neither establish instability or synthesis failure nor validate the remaining
structures. The native VAE prediction job **29505759 is still ongoing** at this
audit update; its frozen selection can be screened without claiming native
prediction completion.

Evidence: [summary](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage2/20260928T085606Z_selected-molecule-chemical-audit_3d54051c/summary.json>),
[per-molecule descriptors and alerts](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage2/20260928T085606Z_selected-molecule-chemical-audit_3d54051c/molecules.json>) and
[endpoint/control identity comparisons](<C:/Users/nikol/Documents/Codex/2026-09-26/okay-so-i-want-you-to/outputs/hpc-results/stage2/20260928T085606Z_selected-molecule-chemical-audit_3d54051c/endpoint_identity_audit.json>).
The extracted run sits beside
`outputs/hpc-results/stage2/chemical-audit-source019-results-20260928.tar.gz`;
matching HPC/local SHA256 is
`29bb52544285331783ae04ae94da4084c0eea0cf80bba8ad42c56aaf0252aec9`.
The remote archive is under `/work3/s234818/latentmol/exports/`.
An initial attempt to package into nonexistent `results/` failed and was
recovered using `exports/`; the audit itself did not fail.

## Inputs and scope

The script reads both source018 `frozen_selection.json` files:

- AE parent: `20260928T082423Z_ae-native-fresh-pose_972c2d83`.
- VAE parent: `20260928T083844Z_vae-native-fresh-pose_253bb4de`.

The pool includes encoded baselines, selected endpoints and selected endpoints'
perturbed starts. It is therefore not a list of independent optimized hits.
The union of submitted SMILES strings is deduplicated across models for the
diagnostic, while preserving model, arm, role, repeat and start-index membership.
An identity can be a baseline, a start and/or an endpoint in different records.

The reader checks that each selection file belongs to a native-run name matching
the declared AE/VAE mode, and that its baseline and pool are present. It records
both selection-file hashes. The audit uses the already frozen pools and needs no
new native model predictions; native prediction completion is not itself a
prerequisite for examining those structures. This does not imply an ongoing
native job has completed.

## Code responsibilities and execution gate

| File | Responsibility |
|---|---|
| `src/latentmol/boltz/chemical_audit.py` | Parse submitted structures, call pinned native standardization, compute descriptors/alerts, preserve roles and write results |
| `tests/test_boltz_chemical_audit.py` | Regression coverage for oxygen-containing three-membered rings, neutral/charged identity collapse and end-to-end result/role writing |
| `scripts/hpc/boltz_chemical_audit.sh` | Require an LSF allocation, run the audit regressions, then invoke the CPU diagnostic in the pinned Boltz environment |

The installed Boltz Git identity must match commit
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`; version text alone is insufficient.
The environment snapshot records RDKit **2024.9.6** and ChEMBL structure pipeline
**1.2.2**. Preserve actual runtime versions and source identity with the run;
this planned document is not evidence that the new tests have passed.

## Submitted and effective native identities

Parse each submitted SMILES with RDKit. Preserve the exact input string, its
stereo-free canonical form and parse-failure status. Call the **actual installed
`boltz.data.parse.schema.standardize`** function used by native affinity
preprocessing, then parse and canonicalize its output. Record the standardized
identity, whether identity changed, and any captured standardization failure.

This uses the pinned pipeline rather than an independently approximated charge
normalizer. Canonical output uses `isomericSmiles=False`, consistent with the
current topology-only project. It cannot distinguish stereochemical identities
in a future stereochemistry-aware study without a revised protocol.

The prior HPC diagnostic established that this particular charged/neutral pair
has the same effective native affinity input:

```text
NC(Cc1ccc(O)cc1)C(=O)O    -> NC(Cc1ccc(O)cc1)C(=O)O
NC(Cc1ccc(O)cc1)C(=O)[O-] -> NC(Cc1ccc(O)cc1)C(=O)O
```

Its exact command/output is retained in [the native report](13_native_rescore.md).
The planned audit extends the identity check to every frozen selection; it does
not assume all charged molecules behave that way. Submitted prediction attempts
remain in their historical denominators even when input identities collapse.

## Structural measurements

All descriptors and alerts below are computed on the **submitted parsed
molecule**, not silently on its standardized replacement. The standardized
column explains what native affinity receives; this release does not supply a
second descriptor/alert panel for that replacement.

| Recorded field | Operation and interpretation |
|---|---|
| Heavy atoms / net formal charge | RDKit heavy-atom count and sum of atom formal charges |
| Molecular weight | `Descriptors.MolWt` on the submitted graph |
| cLogP | `Crippen.MolLogP`, a calculated structural descriptor |
| TPSA | `rdMolDescriptors.CalcTPSA`, a topology-based descriptor |
| QED | `QED.qed`, a composite structural score, not a probability of useful chemistry |
| Ring sizes | Sizes of RDKit-perceived atom rings |
| Oxygen-containing three-membered rings | Count perceived rings of size three containing at least one oxygen atom |
| Filter alerts | Matching descriptions from PAINS_A, PAINS_B, PAINS_C and BRENK |

Alert descriptions are pooled, deduplicated and sorted. A report must not invent
catalog-specific attribution beyond the returned descriptions. Ring counts use
RDKit's perceived ring representation, not enumeration of every graph cycle.
The three-membered oxygen-ring check is an explicit structural motif diagnostic;
the presence of that motif alone does not quantify strain, reaction rate or
solution stability.

High QED or absence of alerts does not establish stability, synthesis, absence
of assay interference, affinity or safety. An alert is likewise a reason for
further evaluation rather than an experimentally established defect. These
diagnostics cannot rescue a weak native score or validate a candidate by
themselves.

## Failures, artifacts and reporting

Keep unparsable inputs as failure rows instead of silently discarding them.
Captured standardization failures retain the submitted structure's descriptors
and a failure message, with standardized identity unavailable. Do not count a
missing standardized identity as a successful identity change. Any uncaught
runtime failure must be recorded as a failed experiment; row-level recovery
must not be assumed to cover arbitrary exceptions.

The result artifacts include `molecules.json`, `endpoint_identity_audit.json`,
`summary.json`, `metrics.jsonl` and the experiment manifest. Per-molecule records preserve baseline and separate
`endpoint` / `perturbed_start` memberships. Summary fields cover submitted
identities, available standardized identities, standardization failures and
changes, inputs with alerts, and inputs with oxygen-containing three-membered
rings. The endpoint audit compares each selected endpoint's standardized identity
against both its own standardized start and encoded baseline, retaining unknown
identity when preprocessing fails. Parse failures remain explicit in the molecule
rows and input count.

Report raw descriptors and exact alert names alongside molecular identity and
role. Do not merge genuinely different structures merely because descriptors
match, or inflate discoveries by counting duplicated roles as different
molecules. A standardized-identity count applies only to available standardized
outputs and must be presented with failures.

Pre-release static review caught a missing per-row SMILES assignment in role
mapping and a summary count that could attempt to sum a missing standardized
identity flag. Both were corrected before packaging. Endpoint and perturbed-start
memberships are now separate; captured standardization failures are retained,
and identity-change counts include only explicit true values. Regression
coverage was added and the source019 HPC suite passed four tests, with two import
warnings. This verifies the exercised implementation paths, not chemical utility.

The immutable release and job are recorded in
[the experiment ledger](05_experiment_ledger.md), including the actual test and
audit outcomes and separate packaging recovery. Preserve these facts independently
of the still-running VAE native prediction job.
