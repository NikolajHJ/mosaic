# Mosaic molecular evaluation | Agent guide

This 19 November 2026 snapshot documents the Mosaic evaluator on the local `codex/molecule-evaluation` branch at commit `bc0f6a9`. Run the commands from that checkout. Source paths refer to its repository root.

Use this guide to evaluate the **saved molecules from a completed experiment**, interpret the outputs and hand them to another researcher. The entry point is `python -m mosaic.small_molecule.evaluation`. It accepts a single molecule, a batch of candidates, or reports from completed evaluations. Ligand generation, optimizer updates and checkpoint selection happen upstream. [Group overview (PDF)](molecular_evaluation_overview.pdf) · Evaluator reference (`docs/molecular_evaluation.md`) · CLI source (`src/mosaic/small_molecule/evaluation/__main__.py`)

**Jump to** [First run](#first-run-and-expected-result) · [Input identity](#input-identity-and-geometry) · [Target and compute](#target-and-compute-options) · [Score dictionary](#score-dictionary-and-denominators) · [Reports and comparison](#reading-sharing-and-comparing-a-report) · [Handoff](#handoff-and-troubleshooting)

## What question is being asked?

A training loss tells us how well a run satisfied its own objective. It can improve while the generator repeats one structure, produces invalid graphs or places the ligand outside the intended pocket. Use chemistry, docking and structure prediction to examine those possibilities separately. When a generator was optimized using Boltz, a fresh Boltz cofold gives another view of its predictions but still shares the model family that guided selection. Vina supplies a separate scoring function if it played no part in generation or selection. Experimental binding measurements would answer a different question.

| Question | Entry point | Unit of analysis |
|---|---|---|
| What happened to a particular designed ligand? | `--smiles` or `--input` containing one record | The named molecule. Pocket claims require a target and an appropriate pose or scorer. |
| How consistent is a generator across the candidates we submitted? | `--input` containing the batch in its original order | All input records for validity, valid molecules for chemistry and diversity, successfully measured molecules for score means. |
| How do two methods compare on this pocket? | Evaluate each cohort with the same protocol, then use `--compare` on the resulting JSON files | One row per cohort, with per-molecule detail and the count behind each score. |
| How do our scores relate to the DBMol paper? | Add `--published-reference dbmol-table1` to an evaluation or comparison | An offline copy of the published aggregate alongside explicitly labelled local results. |

Before starting, record the **input population** in plain language. “First 20 valid draws in sampling order,” “all 56 generated graphs” and “the 20 best historical endpoints” have different meanings. Preserve the total draw count and any filtering, deduplication, repair or ranking that happened before the export. The evaluator only counts the rows it receives. It does not impose a 31-heavy-atom filter or sample a fixed number of candidates.

## First run and expected result

Run shell examples from the Mosaic repository root in an environment with Python, NumPy and RDKit. If Mosaic is already installed, its package is importable directly. For a source checkout, use `export PYTHONPATH="$PWD/src"` in Bash or `$env:PYTHONPATH = "$PWD/src"` in PowerShell. The first example uses only the small input file tracked in Git.

```sh
python -m mosaic.small_molecule.evaluation --name "Guide chemistry example" --input examples/small_molecule/evaluation.smi --detail input_population="all example records" --notes "Check yield and diversity on the supplied seven records."
```

Expect **7 attempted, 5 valid, 4 unique valid, and 10 unordered pairs** for diversity. `CCO` and `OCC` describe the same molecule. `C.C` is disconnected and the final record cannot be parsed. Thus validity is 5/7, uniqueness is 4/5 and unique valid yield is 4/7. The CLI prints the generated paths. Open the HTML file locally in a browser. The default output root is `results/evaluation/`. Give a repeat a fresh name or a different output directory because existing reports are protected. Example input (`examples/small_molecule/evaluation.smi`)

For an experiment name such as `DeFog implementation 4`, preserve the spaces. `--detail` accepts distinct repeated `KEY=VALUE` arguments. Use keys such as `source_run`, `updates`, `learning_rate`, `optimizer`, `generation_seed`, `conditioning`, `selection_rule`, `raw_draws` and `budget` when known. These fields describe the experiment. The CLI's separate `--seed` sets the evaluation seed, which defaults to 42. Never infer an unknown generation setting from an evaluation default.

## Input identity and geometry

| Input | What the evaluator reads |
|---|---|
| `--smiles "C1CCCCC1"` | One molecule. Appropriate for an endpoint or a format check. |
| `.smi`, `.smiles`, `.txt` | One SMILES per nonblank, noncomment line, followed by an optional label. IDs are assigned in input order. |
| `.csv`, `.json`, `.jsonl` | CSV expects `smiles` with optional `id` and `label`. JSON accepts strings, records, a list or an object with a `molecules` list. JSONL accepts one record per line. Assign unique IDs and keep stable labels across exports. |
| `.sdf` or JSON record with `pose_sdf` | SDF retains supplied coordinates, assigns one-based IDs and uses the molecule title as the label. A per-record `pose_sdf` resolves relative to the input file, holds one molecule and is checked against the supplied SMILES when present. |
| Mosaic V0 `.npz` or embedded `readout` | One array pair `X[N,6]` and `E_upper[N(N−1)/2,5]`, or discrete `element_indices` and `bond_upper_indices`. The reader validates shape and category ranges. Probability rows must be finite, nonnegative and sum to one within tolerance. |

The V0 atom order is C, N, O, S, F, Cl. Bond classes are absent, single, double, triple and aromatic. Edges run over lexicographic pairs with `i < j`. The evaluator takes raw argmax on the atom and bond probability rows. Formal charges and stereochemistry are absent from this graph representation. Record whether an upstream projection such as DeFoG or a synthesis projection changed the submitted molecule. The evaluator preserves that input stage. Input reader (`src/mosaic/small_molecule/evaluation/inputs.py`)

A SMILES identifies a molecule but gives no pocket location. For input-pose geometry, supply the target and an SDF in the receptor's coordinate frame, then explicitly declare `--pose-source` or a per-record `pose_source`. An arbitrary conformer can appear far from the pocket even for a good ligand. Docking and native cofolding obtain their own coordinates. Use the prefix on every geometry column when interpreting results.

## Target and compute options

Start from the target JSON example (`examples/small_molecule/evaluation-target.example.json`). Replace the placeholder protein and residue IDs with the structure used for the experiment. Paths to the protein, reference SDF and docking receptor resolve relative to the target JSON. The target records the geometric protein and its pocket independently of the optional docking box and prediction sequence.

- **Geometry** uses a unique list of existing `chain:number[icode]` IDs such as `A:42` or `A:42B`. The first protein model and highest-occupancy alternate per atom are used. Protein heavy atoms provide the coordinate reference. Heteroatom cofactors and waters fall outside this analysis. Add `reference_smiles` for ligand fingerprint similarity or an aligned `reference_sdf` for contact-set overlap.
- **CPU docking** needs the Vina executable, Meeko and its dependencies, a finite box center and three positive box dimensions in Å. Use the same prepared PDBQT receptor and the same geometry frame throughout the comparison. A cleaned, hydrogenated receptor PDB can be converted by Meeko when specified in the target. Receptor protonation and the disposition of waters/ions are preparation choices worth recording.
- **Native prediction** needs one canonical protein sequence, distinct protein and ligand chain IDs, an explicit sequence-position to original residue map when numbering differs, and a cached matching A3M or deliberately declared single-sequence mode. A missing reference position can map to `null`. Every scored pocket residue must appear in the map. Predictors receive the protein and canonical ligand, with no template or pocket restraints. The pocket list is used afterwards to measure the pose.

Start with chemistry-only scoring on a workstation or allocated interactive CPU node for a short check. Vina also runs on CPU. For a substantial batch, size the job using a few representative molecules in the intended environment. The defaults are one ligand at a time and one Vina core per ligand. `--dock-workers 4 --cpu 2` allows up to eight docking cores. Both CPU distribution and wall time depend on the ligands and the allocation. Use a compatible allocated GPU for native predictors. The evaluator launches local subprocesses and leaves scheduling to the operator. HPC connection and allocation guide (`docs/codex_dtu_hpc_integration.md`) · Validation and runtime observations (`experiments/validation/molecule-evaluation.md`)

The following are Bash templates. Set the uppercase variables to actual files. The target must include the block needed by the stage you select. The local-model example contains placeholder paths to existing installations. Native configuration example (`examples/small_molecule/evaluation-local-models.example.json`)

```bash
# Dock a saved cohort in a fixed box.
python -m mosaic.small_molecule.evaluation --name "Cyclic search standard docking" \
  --input "$CANDIDATES" --target "$TARGET" --dock --preset standard --cpu 2 --seed 42

# Evaluate a saved pose that already shares the receptor coordinate frame.
python -m mosaic.small_molecule.evaluation --name "Cyclic search saved pose" \
  --input "$ALIGNED_SDF" --target "$TARGET" --pose-source protenix

# Add native Boltz-2 using installed assets. Add --af3 to request both.
python -m mosaic.small_molecule.evaluation --name "Cyclic search Boltz cofold" \
  --input "$CANDIDATES" --target "$TARGET" --boltz2 --model-config "$MODEL_CONFIG"
```

**Docking protocol.** The ligand keeps its supplied protonation. RDKit starts a new ETKDGv3 conformer with the fixed seed, then runs MMFF for up to 200 iterations if parameters are available. Meeko prepares the ligand. Vina uses exhaustiveness 4 with `quick` or 32 with `standard` and requests 10 modes by default. The evaluator picks the lowest finite returned score for each ligand, checks the recovered ligand identity and saves that pose as SDF. MMFF nonconvergence is recorded and can still yield a docking measurement. `--timeout` defaults to 300 seconds per docking/preparation subprocess. Docking implementation (`src/mosaic/small_molecule/evaluation/docking.py`)

**Native protocol.** Both predictors are off by default and run in their own installed Python environments. AF3 needs the official runner and installed model parameters. Boltz-2 needs the structure and affinity checkpoints and the existing molecule dictionary. Local paths in the model configuration resolve relative to that file. Missing assets are reported during preflight. The native worker disables downloads and protects supplied assets against writes. The standard path starts a subprocess per valid molecule and requested backend. Inference, storage and runtime are separate from the cost of opening the report.

Both backends take one structure sample using the evaluation seed. AF3 defaults to 10 recycles and uses the supplied MSA with its data pipeline disabled. Boltz-2 defaults to three recycles, 200 sampling steps and three affinity samples. One Boltz complex supplies the structural metrics, while the separate affinity ensemble supplies its affinity estimate. The target can specify `msa_mode: single_sequence` and `msa: empty` when that is the intended protocol. Record that choice since the paper used target MSAs. The default native timeout is 3,600 seconds per subprocess. Predictor setup and extraction (`src/mosaic/small_molecule/evaluation/prediction.py`) · Offline worker (`src/mosaic/small_molecule/evaluation/native_worker.py`)

**Previously computed scores.** JSON records can carry `model_metrics` with `af3_iptm`, `af3_ipae`, `boltz_affinity`, `boltz_iptm`, `boltz_ipae`, `protenix_iptm` or `protenix_ipae`. Preserve the calculation method and source artifact in the experiment notes. For example, an AF3 pose in the target frame with `pose_source: af3` plus an imported `af3_iptm` can produce AF3 success. The analogous Boltz-2 path uses `pose_source: boltz2` and `boltz_affinity`. A scalar on its own cannot supply pocket geometry. Imported and newly requested scores for the same backend conflict and should have separate named evaluations. Native AF3 currently computes ipTM, geometry and success. Its iPAE column accepts an externally established AF3 scalar.

## Score dictionary and denominators

Let **A** be the number of submitted records, **V** the number of valid records and **U** the number of distinct canonical isomeric SMILES among those V. A valid molecule must be nonempty, RDKit-sanitized, connected and radical-free, have at least one heavy atom and contain no dummy atoms. Atom-map annotations are cleared for identity and ordinary explicit hydrogens are removed during normalization. Salts are not stripped and failed graphs are not repaired.

| Quantity | Definition and interpretation |
|---|---|
| Validity, uniqueness, unique valid yield ↑ | V/A, U/V, and U/A respectively. When V is zero, uniqueness is unavailable. A duplicate valid molecule remains a member of the input population. |
| Diversity ↑ | Mean of `1 − Tanimoto` over every unordered pair of valid records. Uses 2,048-bit Morgan fingerprints, radius 2 (ECFP4), chirality disabled. Duplicate structures contribute zero-distance pairs. The denominator is V(V−1)/2. With fewer than two valid molecules diversity is undefined. Exact pair work grows quadratically. |
| Molecular descriptors | Heavy-atom count, molecular weight, formal charge, logP, TPSA, H-bond donors/acceptors, rotatable bonds, ring count, QED and Ertl SA. Higher QED and lower SA indicate more favorable descriptor estimates. Interpret weight, charge and polarity in the context of the desired chemistry. |
| Lipinski violations ↓ | Count of molecular weight >500, logP >5, HBD >5, HBA >10. A count of zero only means the molecule meets these four thresholds. |
| Vina ↓ | Lowest returned docking score in kcal/mol for one ligand. The cohort mean averages these best-per-ligand scores among successfully docked molecules. A more negative value is more favorable under Vina. |
| BCov / Cov ↑ | For each named pocket residue, compute its smallest heavy-atom separation to the ligand. Cov is the fraction of pocket residues with that distance ≤3.5 Å. The denominator is the whole declared pocket. |
| Dist ↓ | Mean of the per-residue minimum distances used above, in Å, including all residues beyond the 3.5 Å cutoff. A short distance to one residue can coexist with a large mean distance. |
| Contacts, clash and reference measures | Contact residues are the residues included in Cov. The clash diagnostic counts all receptor–ligand heavy-atom pairs <2.0 Å. Reference fingerprint similarity is ECFP4 Tanimoto. Input/docked contact Jaccard is the intersection divided by the union of geometric contact sets, undefined for an empty union. Jaccard does not distinguish interaction chemistry. |
| ipTM ↑ | Inter-chain predicted TM score. A model-specific confidence measure in [0, 1]. Preserve the backend in the column label. |
| AF3 success ↑ | AF3 ipTM ≥0.6 and at least one intended pocket residue at ≤3.5 Å in that AF3 pose. This is a binary prediction per evaluated molecule. |
| Boltz-2 predicted affinity ↓ and success ↑ | Affinity is the native `affinity_pred_value`, in Boltz's output scale. Success requires **affinity <0** and **Boltz Cov >0.5** in the same prediction. Exactly zero affinity or exactly half the pocket fails. |
| Boltz-2 iPAE ↓ | Mean predicted aligned error (Å) over the submatrix whose rows are valid receptor backbone frames and whose columns are present ligand heavy-atom tokens. All eligible pairs across the whole receptor contribute equally, with no contact cutoff. Lower values mean greater model confidence in the predicted ligand placement. |

The geometry prefixes identify different observations of the same molecule. `pose_*` uses the supplied aligned pose, `dock_*` the Vina winner, `af3_*` an AF3 complex, and `boltz_*` a Boltz-2 complex. For native cofolds the distances are measured directly in that predicted complex. Mosaic calculates native Boltz-2 iPAE from the full PAE matrix and matching token/structure files. Missing or malformed optional PAE can leave iPAE blank while affinity and geometry remain usable. The stored definition and any warning belong with the number. Chemical and geometric calculations (`src/mosaic/small_molecule/evaluation/metrics.py`) · Score and iPAE implementation (`src/mosaic/small_molecule/evaluation/prediction.py`)

Summaries include the number of available valid measurements for each numeric metric, plus mean, median, population standard deviation, min and max in JSON. For binary success the mean is the fraction of evaluated molecules passing. Read `*_evaluated`, `*_failed`, `*_success_count` and `*_success_valid_rate` together. A successful cofold that misses the pocket contributes zero to the success mean. A technical failure contributes to the failure count and has missing scores. The separate success/all-valid rate includes valid molecules without a successful prediction in its denominator. A missing value, measured zero and failed chemistry describe three different outcomes.

## Reading, sharing and comparing a report

| Artifact under the output directory | Use |
|---|---|
| `<name>.html` | Open locally to inspect cohort summaries, molecules, warnings and score definitions. Keep the sibling files when sharing the report. |
| `<name>.csv` | One summary row per experiment, with mean/count fields. Use the unrounded machine-readable values for analysis. |
| `<name>.json` | Source of truth for input identity, versions, target hashes, enabled settings, molecule rows, counts and warnings. Comparisons embed their source runs in `runs`. |
| `<name>.md` | Human-editable experiment description, settings, scores and `Agent notes — editable`. Edits to the note leave the recorded JSON and numbers unchanged. |
| `<name>.files/` | `molecules.csv` for a single evaluation, or `run-001/molecules.csv` etc. for a comparison. `best_ligands.csv` records winners and eligible counts by criterion. Raw scorer outputs and logs live with the original scored runs. |

The best-ligand table selects the minimum Vina and the minimum Boltz-2 affinity **independently** for each cohort. Read the `selected_by` and `scored` columns. All other numbers on a row refer to that selected molecule. A failed or unavailable secondary score displays as a dash. Input order resolves ties. Best-of-20 and best-of-11 reflect different opportunities to obtain an extreme value. When discussing generator performance, give the cohort means and counts alongside selected examples. Report generation (`src/mosaic/small_molecule/evaluation/report.py`)

For a local matched comparison, pass two or more completed single-evaluation JSON files. The following Bash example uses the saved BRD4 docking reports, where those study artifacts are present. It only reads the existing measurements.

```bash
reports=results/brd4-baselines-30082565/reports
python -m mosaic.small_molecule.evaluation --name "BRD4 two-cohort docking replay" \
  --compare "$reports/Run 30082565.json" "$reports/DiffSBDD.json" \
  --published-reference dbmol-table1
```

The comparison checks target/protein/reference/receptor hashes, RDKit/NumPy versions, fingerprints, pose-source settings and sources present in molecule rows, native predictor settings, and recorded docking metadata apart from the machine-specific PDBQT path. Even an edited target JSON with equivalent-looking contents can have a different digest. On a mismatch, identify the field and use a common protocol when a new evaluation is warranted. The comparator cannot verify matched generator conditioning, inclusion criteria, generation budgets, imported metric conventions or independence from the search objective. Document these explicitly. Comparison implementation (`src/mosaic/small_molecule/evaluation/pipeline.py`)

The optional DBMol reference uses a pinned Table 1 transcribed from the paper. DBMol's main result uses 100 valid molecules per case and averages case-level means across seven LIT-PCBA protein/pocket cases. Published CGFlow-ZS covers six. The paper's structural Cov, Dist, success and iPAE use AF3. Its precise iPAE token aggregation is unspecified. In a local row, only corresponding AF3 measurements can fill those structural columns. Docking in the paper used a box centered on a Boltz-predicted ligand. Mosaic's fixed box may differ even at the same exhaustiveness. The original rank is displayed with the published rows. Local BRD4 measurements have their own target and cohort, so the reference provides neither a BRD4 head-to-head nor a new rank against DBMol. [DBMol Table 1 and protocols](https://arxiv.org/html/2607.19237v1) · Pinned table and mapping (`src/mosaic/small_molecule/evaluation/literature.py`)

## Worked example | BRD4 and run 30082565

The study used BRD4 bromodomain 1, PDB 3MXF chain A, a 19-residue pocket defined from JQ1, 31 ligand heavy atoms, and a fixed 24 Å docking cube centered on JQ1. All retained molecules entered standard Vina docking with seed 42. A separate single-sequence native Boltz-2 cofold supplied affinity, iPAE and predicted pocket geometry. The control is deposited JQ1. No BRD4 AF3 result or measured BRD4 DBMol cohort was available. Study record, frozen pocket and methods (`experiments/brd4-jq1/generative-baselines/README.md`)

| Cohort | Scored / submitted | Mean Vina (kcal/mol) ↓ | Mean Boltz iPAE (Å) ↓ | Mean Boltz Cov ↑ | Mean Boltz Dist (Å) ↓ |
|---|---:|---:|---:|---:|---:|
| Run 30082565 | 20 / 20 | −3.597 | 3.516 | 0.253 | 4.654 |
| DiffSBDD | 20 / 20 | −7.123 | 2.857 | 0.245 | 4.541 |
| SynCoGen | 18 / 18 | −8.745 | 2.526 | 0.281 | 4.439 |
| ShEPhERD | 11 / 14 | −7.906 | 1.970 | 0.230 | 4.392 |
| JQ1 control | 1 / 1 | −9.048 | 1.318 | 0.158 | 4.308 |

Run 30082565 supplied 20 historical endpoints selected using its fixed-pose Boltz objective. DiffSBDD received the pocket and retained the first 20 valid size-matched draws. ShEPhERD and SynCoGen received JQ1-derived conditioning. CGFlow-ZS required an unavailable ordered building-block vocabulary and contributed no scored cohort. Vina and Boltz conformer preparation failed for the same three ShEPhERD ligands, while all fourteen qualified for its diversity calculation. A mean calculated over the eleven scored ligands must keep that count.

The two selections for our run illustrate how to read the table. `primary-07` was its Vina winner at −5.600 kcal/mol, with Boltz iPAE 2.785 Å, pocket coverage 4/19 and Dist 4.737 Å. `primary-05` was its Boltz affinity winner at −1.092, with iPAE 2.630 Å, coverage 5/19 and Dist 4.668 Å. More than half of 19 means at least **10 residues**. Zero of 69 successfully cofolded candidates reached the joint success criterion. The JQ1 cofold contacted 3/19 and also missed it. The control illustrates how a stringent pocket definition affects a binary endpoint even for a known binder.

For the interface-metrics refresh, all 70 original PAE matrices (69 designed ligands and JQ1) were available. Each BRD4 prediction contributed 127 valid receptor frames × 31 ligand atoms = 3,937 pairs. The archived PAE and original coordinates supplied the new table without repeating inference. The study-specific join checks candidate order, identifier and canonical chemistry, verifies the success flag, and records each enriched native file's digest. Use the exact `add_boltz_results.py` command under **Reproduce the experiment** in the study record to produce a new named report when the artifacts are present. Study replay command (`experiments/brd4-jq1/generative-baselines/README.md#reproduce-the-experiment`) · Native scoring and archive audit (`experiments/brd4-jq1/generative-baselines/boltz-rescore/experiment.md`) · Study join (`experiments/brd4-jq1/generative-baselines/add_boltz_results.py`)

## Handoff and troubleshooting

An agent note should identify what changed in the generator, which candidates entered evaluation, and what the measurements support. Adapt the following paragraph to the available evidence.

> [Experiment and source run] tested [question] with [initialization, number of updates, LR, optimizer and generation seed]. We submitted [population and selection rule] out of [raw draws or saved endpoints] under [target, conditioning and scorers]. [N] molecules scored. [Main comparison and numbers]. [Missing data, objective overlap or control result that affects interpretation].

| Symptom | Where to look and what to do |
|---|---|
| Duplicate ID or malformed probability graph | Check the exported schema, row order and unique record IDs. Preserve true duplicate molecules for diversity/yield measurements. |
| Missing pocket residue or unexpected distance | Check chain IDs, insertion codes, residue maps and coordinate frames. Verify whether the displayed metric came from the input, a docked pose or a native complex. |
| Existing output name or incomplete directory | Inspect the existing report and any `<name>.files/failure.json` and logs. Use a new name for a corrected evaluation. Preserve editable notes and partial evidence. |
| Docking or native failure | Use the individual row's status/error and corresponding log. Distinguish conformer/preparation failures from valid low-scoring predictions. Retain the failed row when reporting counts. |
| Boltz iPAE blank but other scores present | Inspect `boltz_ipae_error`, saved PAE matrix, token structure and matching model CIF. A missing PAE scalar does not invalidate a separately recorded affinity or geometry. |
| A published AF3 column is blank | Check for actual AF3 measurements, matching pose and source labels. Boltz, Vina and Protenix have separate conventions. Report the gap if AF3 assets or verified exports are unavailable. |

For changes to parsing or scoring, use the existing tests from the repository root. The native prediction tests use local synthetic fixtures. The optional real Vina round-trip requires a local binary (`MOSAIC_TEST_VINA`) and the prepared 1IEP tutorial files (`MOSAIC_TEST_DOCKING_ASSETS`). These tests exercise different parts of the pipeline than a GPU inference on a new target.

```sh
python -m pytest -q tests/small_molecule/test_molecular_evaluation.py tests/small_molecule/test_molecular_prediction.py tests/small_molecule/test_molecular_literature.py
```

When handing off a report, retain the source cohort, target/config files, JSON and sibling artifacts. Quote each score with its model or pose source, eligibility count and selection rule. For example, “mean Vina for 11 of 14 ShEPhERD candidates on BRD4” lets the reader reconstruct exactly which molecules the claim covers.
