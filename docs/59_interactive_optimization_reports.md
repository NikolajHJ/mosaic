# Interactive reports for saved molecular optimization runs

## Current implementation and verification

The standard human-facing view of a molecule-optimization trajectory is an
offline interactive HTML report. Its purpose is to inspect existing results:
the raw JSON/JSONL, source manifests and scientific run records remain the
authoritative evidence. Building a report does not rerun a decoder, scorer,
repair procedure or experiment, and a report cannot certify a failed experiment.

Implementation status, 16 October 2026: the offline HTML assets, standard-library
adapter and CLI are implemented. The complete local presentation suite passed
**38 tests in 1.76 seconds**. The
[saved source109 pilot report](../outputs/reports/source109-boltz-defog-pilot-001/index.html)
contains two runs with 26 frames each and 74 hashed source inputs, without
importing Torch, NumPy, RDKit or Boltz. Browser inspection confirmed navigation,
run/sample selection, playback/pause, previous/next, chart/slider navigation,
overlay selection, visible invalid graphs and explicit missing historical charge/
bond probabilities. There were no console warnings/errors or horizontal overflow
at the default viewport. A
[browser preview](../outputs/report-checks/optimization-report-preview-20261016.png)
and [QA receipt](../outputs/report-checks/optimization-report-browser-qa-20261016.json)
record the presentation. Mobile breakpoints and a full browser matrix were not
tested.
These are saved-artifact presentation checks, not new numerical model tests.
No HPC submission or training change is part of this reporting work; the
already sealed source112/source113 releases remain unchanged.

## Generate a report

Run the CLI from the repository root, supplying saved run directories:

```powershell
python scripts/render_optimization_report.py --run RUN --label LABEL --output NEW_DIR
```

Add repeated `--run` and `--label` arguments to overlay trajectories. Optional
arguments are `--title`, `--require-matched`,
`--allow-initialization-comparison` and `--allow-guidance-comparison`.
The output directory must be new; existing output is never overwritten.

| Output | Purpose |
| --- | --- |
| `index.html` | Self-contained interactive presentation |
| `report.json` | Normalized presentation data with source identities and explicit missing fields |
| `manifest.json` | Consumed input hashes and renderer/output provenance |

The implementation lives in
[`src/latentmol/reporting/optimization.py`](../src/latentmol/reporting/optimization.py)
and [`scripts/render_optimization_report.py`](../scripts/render_optimization_report.py).
It reads saved JSON/JSONL, confines reads to the declared run roots and hashes
every consumed source file. It does not load tensor checkpoints to recover
missing information. A file changing during assembly is an error.

Overlays are descriptive by default. `--require-matched` additionally checks
the saved configurations, native contexts and frozen identities; permitted
initialization or guidance differences must be named by their respective
flags. Allowing a declared difference does not waive the remaining comparison
checks. The derived report identity is separate from each original native
configuration/source identity.

The long-comparison matcher treats only `execution_budget.wall_seconds` as an
operational difference. It retains the full raw configurations; execution-budget
version and tracked native-call cap still require exact matching. This exception
does not authorize ignoring an objective, model, context, sampling or gradient
policy difference. `tests/test_long_report_matching.py` covers the exception,
strict remaining fields and nonmutation. The new source114 package is intended
to include the frozen reporting assets and capture hooks; older sealed112/113
releases remain unchanged.

The actual example was generated with the bundled Python at
`C:/Users/nikol/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`
and the following CLI arguments (shown with `python` for readability). Rerunning
it requires a different new output directory:

```powershell
python scripts/render_optimization_report.py --run outputs/hpc-results/source109a002-pointwise-round-003/runs/source-109-pointwise-native-gradient-validation-attempt002-pilot/boltz_only/optimization/20261015T180703Z_matched-pilot-boltz_only_6b8b22ce --run outputs/hpc-results/source109a002-pointwise-round-003/runs/source-109-pointwise-native-gradient-validation-attempt002-pilot/boltz_defog/optimization/20261015T182844Z_matched-pilot-boltz_defog_126f5654 --label "Boltz-2 only" --label "Boltz-2 + DeFoG" --output outputs/reports/source109-boltz-defog-pilot-001 --title "Boltz-2 and DeFoG: matched optimization pilot" --require-matched --allow-guidance-comparison
```

Its [manifest](../outputs/reports/source109-boltz-defog-pilot-001/manifest.json)
records the input and renderer hashes. The declared guidance comparison passed
the other saved-identity checks; this is not training-seed replication or proof
of a chemistry/binding benefit.

All 74 source hashes and both output hashes were verified unchanged after
browser QA using the same bundled Python and standard-library hashing.

| Artifact | SHA256 |
| --- | --- |
| `manifest.json` | `99955be3d987cbff324f78f469c3aa4c04ff08068a739fd7969ee1398dcf7c04` |
| `index.html` | `5f5355a422b068f61c6c79ccbb2cb757c6d3c03453bbf4426376216b6db77204` |
| `report.json` | `34401f1081d41cd2a829fd6e6024dc5c7c5126a473bf9b5fdd0d6b78ea40fd4e` |

## What the interface must show

The interface supports loss overlays, run/initialization selection, step
selection and playback, molecule snapshots, and atom/charge/bond probability
heatmaps when those probabilities were actually saved. Model labels,
vocabularies, occupancy policies, gradient contracts and run status come from
the artifacts; they must not be hardcoded to a particular scorer or element.

The following records describe different objects and must remain distinguishable:

| Object | Interpretation |
| --- | --- |
| Current relaxed shared graph | The atom, charge and bond distributions actually supplied to the scoring branches |
| Raw decoder occupancy | NO_ATOM mass before fixed-mask conditioning; distinct from the occupied model input |
| Argmax graph | A hard labeling of the saved distributions, with its recorded validity result |
| Sampled candidate | A declared categorical draw, with its own draw identity and validity result |
| DeFoG repaired target | A separate stopped target used for guidance; never substituted into the displayed current graph |
| Fresh native prediction | A separately evaluated hard molecule/pose, only where an actual prediction record exists |

Missing historical probabilities remain visibly unavailable. Neither hard
labels nor a saved latent vector are a license to invent or regenerate them.
Invalid or disconnected hard graphs remain unchanged and visibly invalid;
drawing them does not establish chemical validity. A stable slot layout helps
compare topology across steps but is not a chemical conformation or a predicted
three-dimensional pose. Atom/charge labels must remain readable independently
of any probability-based glyph size or bond width.

Step records represent the saved states and actual updates. Gaps, rejected
updates, incomplete trajectories and failed runs remain visible; playback
does not interpolate missing molecules or imply that a requested endpoint
was reached. Expected occupancy and a hard occupied-atom count are separate
quantities.

Binding components and the scheduled weighted total must be separable.
Target, context and coefficient changes require markers: a guided total at
two different repair weights or target graphs is not a single fixed objective.
Do not report a global improvement percentage across those changes. Frozen-pose
proxy losses and fresh native prediction scores are also distinct measurements.
Validity, bounded repair witnesses and fresh-pose eligibility retain their
original definitions and censoring; none become a nearest-valid distance or
binding claim by being plotted.

## Automatic capture and terminal reporting

Future active-set and named-model optimization paths opt in through
`ExperimentRun(report_kind='molecule_optimization')`. They save the full shared
graph as JSON using the graph already computed for candidate recording; this
adds no model forward pass or random draw. After the terminal scientific
record, successful and failed runs may render
`interactive-report/index.html` within the run directory.

`interactive-report-status.json` is a separate presentation receipt. A renderer
failure must not replace, hide or change the scientific run outcome. New
capture hooks are implementation changes awaiting their own HPC numerical
validation; previously sealed and queued releases do not acquire them
retroactively. Historical reports use only the fields their original releases
saved.

## Offline and input-safety boundary

The HTML must work offline, with embedded assets and no external CDN or model
service. Metadata, errors, SMILES and labels are escaped or inserted as text;
embedded JSON escapes `<` so saved strings cannot close a script element.
Only locally generated, constrained depiction SVG may be inserted as markup.
Saved artifacts are input data, not executable instructions. The renderer
does not deserialize tensors, execute stored code or fetch missing assets.

## Presentation verification history

The first launch with the bundled Python stopped with `No module named pytest`;
no assertions ran. The existing `.venv/Scripts/python.exe` provides Python
3.12.12 (Anaconda) and pytest. Its first launch produced 35 setup errors because
the parent `outputs/report-checks` directory did not exist; no assertions ran.
The launcher created that directory, then ran with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and `PYTHONPATH=src`:

```powershell
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_optimization_report.py --basetemp outputs/report-checks/presentation-initial2-20261016 -k 'not offline_html and not presentation_needs and not terminal_experiment_record' --tb=short
```

Initial result: **35 passed, three deselected in 2.69 seconds**. After the HTML
assets were available, the complete suite ran under the same Python/environment:

```powershell
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_optimization_report.py --basetemp outputs/report-checks/presentation-full-20261016 --tb=short
```

Result: **38 passed in 1.76 seconds**, including the frontend artifact and
terminal-report presentation cases. These verify reporting behavior, not model
forward execution or chemical success. The earlier launch failures remain
presentation infrastructure history, not scientific experiment failures.

Completed browser QA of the actual example confirmed the guided step-25 values
against the saved record: total 49.594416, binding component 5.664286, 22 active
atoms / 21 natural-selection atoms and 22 saved atom-probability rows. Charge/bond probabilities
remain explicitly unavailable where not saved, and the invalid raw schematic
remains visible. Run/sample selection and previous/next controls worked. Playback
advanced from step 0 to 21, then paused; the overlay changed from two series to
one and back; a chart click selected step 12 and the slider's End key selected
step 25. There were no console warnings/errors or horizontal overflow at the
default viewport. The final JavaScript syntax check passed:

```powershell
node --check src/latentmol/reporting/assets/optimization_report.js
```

Future capture hooks remain statically reviewed only, without HPC numerical
validation. These browser observations apply to the saved example, not to
unexecuted future model runs.

## Mosaic reference and license

The design reference is the sibling checkout `../mosaic`, specifically
`scripts/render_protenix_trajectory.py`,
`scripts/templates/protenix_trajectory_report.html`,
`scripts/protenix_trajectory_depiction.py` and
`tests/test_small_molecule_report_studies.py`. Its fixed-slot depictions,
probability presentation and strict comparison checks informed this design;
our report adapts the reference to swappable models, repair targets and partial
or failed saved runs. There is no Mosaic runtime dependency.

The reference renderer SHA256 is
`e76cb44f0e014df1e794fd7866eb4bff975bafa79c5850bb24bddedd3fb9a389`;
the template SHA256 is
`77ab62657ce28c565649247cfb27b33fd12ca694640363ad749ec6be534aa08e`.
Mosaic's `LICENSE` is MIT, copyright 2025 Escalante, SHA256
`27d3b7f20cf20905ab4ce6bbb15e8b2d0e373b330d65f490638c9eda56422ae4`.
Retain its license notice if substantial source or template material is copied.

See the [pipeline](04_latent_pipeline.md) and
[experiment ledger](05_experiment_ledger.md) for scientific run status.
