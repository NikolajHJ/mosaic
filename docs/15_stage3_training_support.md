# Stage 3 preflight: train-only structural support

Updated 28 September 2026. The verified preflight is **source021 / job 29505886,
DONE**, with three passed tests and completed run
`20260928T091242Z_stage3-training-support-preflight_ceb9b559`.
Preliminary **source020 / job 29505877** also completed, with two passed tests;
its original results remain available. Both requested two CPU slots, 8 GB per
slot and one hour in queue `hpc`.
Neither performs denoiser training, a model-gradient experiment or a new native
binding prediction.

Local archive: `outputs/source-releases/latentmol-source-020-training-support.tar.gz`.
HPC archive: `/work3/s234818/latentmol/releases/source-020-training-support.tar.gz`,
with the extracted release beside it. Source SHA256:
`1cfe1121a618ed6badeac538054ee25157c66ae05492872341c9f2766ba7d9f4`.
Actual tests, run identities, failures and results belong in
[the experiment ledger](05_experiment_ledger.md).

## Question and scope

The completed [chemical audit](14_selected_chemical_audit.md) identified nine
submitted structures, including baselines, selected endpoints and perturbed
starts. This preflight asks how closely those structures resemble the accepted
training data and how often three explicit motifs occur in that data. It also
compares each submitted form with its native-standardized identity.

These candidates were already selected after optimization and score inspection.
They are a **diagnostic challenge set**, not independent validation or an unbiased
sample of chemistry. Motif frequencies and nearest neighbors can help identify
representation gaps worth testing; they cannot validate a molecular prior, its
gradient, chemical stability, synthesis or binding. The broader Stage 2 promotion
decision remains in [the native report](13_native_rescore.md).

## Inputs, identity checks and code boundaries

| Component | Responsibility |
|---|---|
| `src/latentmol/boltz/training_support.py` | Read frozen structures, restrict references to training, compute motifs/neighbors, preserve provenance and write diagnostics |
| `tests/test_boltz_training_support.py` | Motif and neighbor-ranking regression contracts, executed on HPC |
| `scripts/hpc/stage3_training_support.sh` | Require an LSF allocation, run selected tests, then invoke the CPU preflight |

Inputs are the prepared `data/zinc215k-seed17-v1` dataset/manifest and completed
chemical-audit run `20260928T085606Z_selected-molecule-chemical-audit_3d54051c`.
The source021 reader verifies the NPZ file hash, actual split digest and
graph-schema fingerprint against its manifest, requires the expected completed
chemical-audit parent, hashes its
`molecules.json`, and records source/data/split/parent identities and RDKit version.

Reference identities are selected strictly by **training split code 0**. The
manifest count must match the loaded reference count, expected here to be
**215,000**. Neither validation nor test molecules become references. A failure
to parse a previously accepted training SMILES stops the audit rather than
silently shrinking the reference bank. Thirteen-heavy-atom references are
identified from the accepted graph's nonzero atom slots.

## What the reference distribution means

Training SMILES are the dataset's **raw stereo-free canonical identities** after
the original representation filtering. This preflight does not run ChEMBL/Boltz
standardization over the training bank. The reference is also not a bank of
native Boltz features or a learned latent manifold.

Each frozen candidate is queried twice where available:

1. Its submitted SMILES, matching the structure described by the chemical audit.
2. Its already recorded native-standardized SMILES, matching that audit's effective
   native affinity identity.

Both query roles compare against the same unstandardized training references.
This is deliberately a one-sided comparison. Different results can reflect
charge/identity normalization rather than a change in chemical plausibility.
Repeated candidate rows may map to the same standardized query; preserve the
input-to-standardized relationship instead of treating each as a new molecule.

An unavailable standardized identity remains unavailable. Unparsable query
forms receive a failure record. Source021 canonicalizes each parsed query and
reports exact membership separately against the canonical reference strings;
it is not inferred from fingerprint similarity. This does not standardize the
training bank or equate identities across tautomers and protonation forms.

## Fingerprints and nearest neighbors

Use RDKit Morgan fingerprints with **radius 2, 2,048 bits and chirality disabled**.
For each available query form, report the **top three** Tanimoto neighbors in:

- The complete training reference set.
- Its 13-heavy-atom subset, with that subset's actual denominator reported.

Bulk similarities are computed per query, without constructing a dense
query-by-reference matrix. Ties prefer the earlier reference index, giving a
deterministic ordering for the fixed dataset. Exact matches are not removed:
the purpose includes detecting training identity and close structural support.

**Tanimoto 1 is not proof of exact molecular identity.** Finite fingerprints
can collide or omit distinctions; the separate membership flag answers the
exact-reference question under the dataset's SMILES convention. High similarity
is not likelihood, validity, synthesizability or target activity. A low value
indicates weak support under this representation, not necessarily bad chemistry.

## Motif definitions and denominators

The following boolean features are computed on every parsed training reference
and each queried candidate form. Counts represent molecules with a feature,
not the total number of matching substructures.

| Flag | Exact implemented definition |
|---|---|
| `oxygen_three_ring` | At least one RDKit-perceived three-membered ring containing an oxygen atom |
| `oxygen_three_ring_two_aromatic_atoms` | Such a ring has exactly two nonoxygen atoms, and both have RDKit's aromatic flag |
| `aniline_substructure` | A match to SMARTS `[NX3;H2][c]` |

The second flag is specifically about atoms **inside** the three-membered ring;
it does not mean any oxygen ring adjacent to an aromatic substituent. Requiring
two nonoxygen atoms avoids a vacuous match for an all-oxygen ring. The aniline
SMARTS is a particular substructure definition, not every aromatic amine or
protonation state.

Report motif counts against the full training count and separately against the
13-heavy-atom count. These are observed frequencies in the filtered ZINC-derived
training data, not frequencies in all possible or synthesizable molecules.
Absence from this bank is not a chemical impossibility, and commonness is not
proof of quality. The motifs were motivated by the selected-pool review, so
their descriptive counts are not an independent confirmatory hypothesis test.

## Observed source020 results

The completed audit used **215,000 training references**, including **1,559
with 13 heavy atoms**. These counts are the denominators for the motif table.

| Molecules containing the motif | All training references | 13-heavy-atom references |
|---|---:|---:|
| Oxygen-containing three-membered ring | 111 / 215,000 | 5 / 1,559 |
| That ring with two aromatic nonoxygen atoms | 0 / 215,000 | 0 / 1,559 |
| Defined aniline substructure | 5,158 / 215,000 | 85 / 1,559 |

There were nine submitted candidate rows and nine native-standardized query
roles. Source020 reported **0/9 exact string matches** for each form. Its exact
membership implementation did not explicitly canonicalize query strings, so
this is not yet a canonical molecular-identity exclusion. Some standardized
queries alias one another; nine query roles do not mean nine distinct native
identities. Fingerprint neighbors are a separate measurement.

| Selected query | Highest Tanimoto, all training | Highest Tanimoto, size 13 |
|---|---:|---:|
| AE target oxygen-ring candidate | 0.361702 | 0.325581 |
| VAE para-aniline, native-standardized form | 0.606061 | 0.515152 |

The AE candidate's oxygen ring containing two aromatic atoms has no occurrence
in this training bank. The VAE query has closer fingerprint neighbors and its
aniline motif is represented. These observations describe training support
under the specified fingerprints and motifs. They do not rank chemical quality:
rarity is not a defect, prevalence is not validation, and similarity is not
density or a useful prior gradient. The reference bank remains unstandardized
training chemistry, not the native Boltz feature distribution.

The complete result archive is
`outputs/hpc-results/stage3/stage3-support-source020-results-20260928.tar.gz`,
with matching local/HPC SHA256
`f726bc74e2e19d1d2c313e60346082f13d692bc1d4cee28474d84e098374a329`.
Both run directories were extracted. The audit's `summary.json` and
`candidate_training_support.json` reside under
`outputs/hpc-results/stage3/20260928T090722Z_stage3-training-support-preflight_54e96ff7/`.
Source020 and its original outputs remain immutable.

## Source021 verification rerun — completed

Two static-review concerns motivated a separate release: recompute and verify
the actual split digest, and canonicalize each parsed query before exact
membership comparison. A tiny end-to-end test covers exclusion of validation
and test references, query canonicalization and summary output; the explicit
aromatic-ring flag also has a regression test. These changes tighten identity
and isolation checks without turning the diagnostic into a prior evaluation.

CPU job **29505886 completed successfully**, with **three passed tests**,
including the tiny train-only integration fixture. Run
`20260928T091242Z_stage3-training-support-preflight_ceb9b559` records `completed`.
The actual split digest was verified, and every parsed query was canonicalized.
All nine submitted canonical strings happened to equal their original strings.
The membership results remain **0/9** for submitted forms and **0/9** for
native-standardized roles. Reference counts and all motif counts are unchanged
from source020. These canonical membership results supersede source020's
un-normalized string check; they still apply only to the recorded training bank.

The local immutable source archive is
`outputs/source-releases/latentmol-source-021-training-support-verified.tar.gz`;
the coordinator verified its HPC release copy before execution. Source SHA256:
`894ef59ee6871d4ce872f3a1385ef45571197c6873198b755466fd6906e3dedf`.
The downloaded result archive is
`outputs/hpc-results/stage3/stage3-support-source021-results-20260928.tar.gz`,
with matching local/HPC SHA256
`087e7a2dc4b3059e1494f064b5123a94c486b903381690003add7dd9743a874c`.
The local hash and extracted summary were checked. The verified audit's
`summary.json`, `candidate_training_support.json` and manifest reside under
`outputs/hpc-results/stage3/20260928T091242Z_stage3-training-support-preflight_ceb9b559/`.
Use source021 for subsequent support interpretation while retaining source020
as the preliminary immutable outcome. Neither run demonstrates a useful
denoiser, molecular density estimate or gradient objective.

## Artifacts, interpretation and next decision

Outputs are `candidate_training_support.json`, `summary.json`,
`metrics.jsonl` and the experiment manifest. Per-candidate rows retain model/arm
selection metadata, both query roles, exact-membership flags, motifs and both
top-three neighbor lists. Summary fields include reference/subset denominators,
motif counts and candidate membership counts. Candidate counts are per retained
input row; aliased native-standardized queries do not become independently
supported discoveries.

Use the result to make a small, explicit next feasibility decision: for example,
whether a chemical-prior interface should distinguish the observed structural
failure modes from ordinary training molecules. It does not itself supply a
denoising objective, train a new prior, test differentiability or demonstrate
useful guided optimization. Any subsequent DeFoG-interface or same-schema prior
experiment needs its own immutable protocol, derivative checks and hard-candidate
evaluation. No such experiment is claimed submitted by this document.
