---
name: mosaic-results
description: Build and maintain the Mosaic experiment viewer from saved trajectories, including step-by-step molecular graphs, probabilities, chemistry correction diagnostics, and a curated run library.
---

# Mosaic results

Use this folder for future interactive Mosaic result visualizations. Keep the viewer reusable; add capabilities to the template and scripts instead of making a separate one-off presentation.

## Build and open

Run from the repository root. No model, GPU, JAX, or new experiment is needed.

```sh
python3 -m venv .venv-visualizations
.venv-visualizations/bin/python -m pip install -r visualizations/requirements.txt
.venv-visualizations/bin/python visualizations/scripts/build.py /absolute/path/to/run-or-study --output results/visualizations
.venv-visualizations/bin/python visualizations/scripts/serve.py results/visualizations --open
```

The builder uses the model-free operators in `src/mosaic/small_molecule/Stage2/operators.py` on `main`; it does not need the experimental `stage2/` checkout.

`build.py` accepts multiple run/study roots and adds them to one library. It exports every committed trace state, without downsampling. It reads original artifacts without changing them. Generated files go under ignored `results/`. `template.html` and `assets/` are the maintained source for the viewer. Rebuild after changing them.

For a standalone quantitative-affinity comparison, use `visualizations/scripts/plot_pair.py CONTROL TREATMENT --out results/figures/pair.png --control-label "Control" --treatment-label "Treatment"`. It exports PNG, SVG and a source-hash receipt. It requires matching initial probability arrays, context, learning rates and objective, and plots saved validity, proposal loss, explicitly scored changed valid readouts and recorded connectivity events. Set a title that describes the actual comparison. Inspect the PNG before sharing it; correlated states are not independent replicates.

For gradient and feasibility comparisons across heads or hardening rules, use `visualizations/scripts/plot_diagnostics.py RUN_A RUN_B [RUN_C ...] --label "A" --label "B" [--label "C" ...] --out results/figures/diagnostics.png`. It requires identical initial probability arrays, context and checkpoint; rates are shown explicitly and may differ. It shows raw graph validity, broad frozen geometry, observed versus gradient/simplex movement and descent agreement, without pooling differently scaled losses. The figure exports PNG/SVG and source hashes. Missing states stay unknown; hardening derivatives are surrogates. Inspect the rendered figure before sharing it.

The audit producers under `testing/` are retained in the `stage2-protocol` research checkout. The plotters here read their saved outputs and do not run new experiments.

To inspect where long bonds enter a relaxed trajectory, first run `testing/stage2_tuning/gradient_audit.py` with the matching geometry, then use `visualizations/scripts/plot_gradient_geometry.py RUN_A RUN_B --label "A" --label "B" --out results/figures/gradient-geometry.png`. It checks initial probabilities/gradients, context, checkpoint and objective. It separates NO_BOND-to-bond events in the gradient/simplex update from later operators and shows near/far gradient RMS normalized per pair. Repeated activations are events, not distinct edges or molecules. Different rates/methods remain explicit; a cross-run plot does not isolate their individual effects. Use an exact recorded-step replay for a local operator counterfactual. Inspect the PNG and retain its source-hash receipt.

For the two overnight tuning studies, use `visualizations/scripts/plot_tuning.py RESULTS_ROOT --out results/figures/tuning.png`. Its frozen/native feasibility gates are preserved locally in the plotter, with source hashes in the figure receipt. It reads collected complete run metrics, verifies identical starting probabilities and proposal model/context protocols, and checks a common rebuilt-native protocol for scored native trials. It exports PNG/SVG, CSV and source hashes. Optional `--state SNAPSHOT.json` uses a separately saved scheduler snapshot; completed but uncollected runs are explicitly listed. `--startup-trials` defaults to the declared six. Fixed recipes, random startup and adaptive proposals stay distinct; open markers fail the study's saved feasibility rules. Missing observed scores are listed instead of plotting optimizer penalty values. The feasible two-objective frontier is separate from the predeclared confirmation-selection order. These are correlated development trials on one starting setup, not a population-validity or binding estimate. Inspect the PNG before sharing it.

For same-state hardening diagnostics, first use `testing/stage2_tuning/hardening_backward_audit.py INHERITED_RUN --out AUDIT.json`, then `visualizations/scripts/plot_hardening_backward.py AUDIT.json --out results/figures/backward.png`. The audit checks the retained transform against the run's manifest/source receipts. It algebraically infers native simplex-tangent gradients under that transform and compares inherited, identity and scalar-compensated local steps at the same forward input. Its Jacobian is tested against the retained JAX VJP on CPU. This is not a new model derivative check or a simulation of alternative trajectories. The figure keeps source validity/geometry visible and omits final-state hypothetical steps from its step plot. Interpret these local ratios alongside full matched continuation results; different points on different trajectories do not isolate a backward-rule effect.

For frozen-pose models, pass `--geometry /path/to/context_audit.json`, repeating it for multiple contexts if needed. Audits are matched by context SHA. A run-local `frozen_geometry.json` is also supported. The viewer marks bonds outside 0.8–2.5 A in red and plots their counts and maximum span. This broad necessary check is separate from sanitization, realistic conformers, clashes, cavity placement, and binding. Never apply a different context's coordinates to a run.

The server binds to `127.0.0.1`, with no external scripts or network services. Keep it local. Stop it with Ctrl-C. An HTML file opened directly cannot provide the run-management API.

## Describe each experiment simply

Put a `visualization.json` beside a source run's `stage2_config.json` before building:

```json
{
  "description": "Goal: Test connectivity (5.3) after valence projection (5.2). Does it reduce fragmentation?\nChanges: Same start, seed, and learning rates as the valence projection (5.2) control. Connectivity (5.3) is on.\nNotes: Compare the number of components and valid steps. A lower model loss alone does not show that the molecule is better."
}
```

Use short, ordinary sentences. Say what the experiment tests, name its actual control when known, list the changed hyperparameters and values, and explain anything needed to read the result. Separate a proposed goal from an observed result. Do not claim a matched control unless the recorded settings and initial state support that. Do not use impressive-sounding language instead of an explanation.

Keep the terminology and numbers from the project protocol: representation invariants (5.1), valence projection (5.2), connectivity (5.3), sparsity (5.4), hardening (5.5), and experiment order (5.6). Use those exact terms when describing a method, then explain it simply if needed. Do not replace the method names with vague alternatives such as "chemistry fix". Experiment A adds 5.2; B adds 5.3; C adds 5.4; D tests 5.5 separately. Name extra controls as extra controls. Check section numbers against the current protocol if it changes.

Descriptions can also be edited in the viewer. Those edits, stars, and deletion state persist in the output library's `curation.json` across rebuilds and server restarts. They take precedence over imported descriptions. Agents can use the local API: `POST /api/description` with `{"id":"RUN_ID","description":"..."}`, JSON content type, and `X-Mosaic-UI: 1`. Read IDs with `GET /api/runs`. Rebuilding the same source paths retains their IDs.

The builder creates a settings-based description when none is supplied. Review it and replace it with the experiment's specific question where that is known. Never fill in missing intent or controls as if they had been recorded.

## Interpret the panels

- The slider indexes saved states and displays their actual update numbers. Molecules and probabilities are the evaluated state at that update. A correction receipt describes the *following* update. Final states have no following correction.
- Argmax drawings retain all atom slots and raw bond types, including invalid and disconnected graphs. RDKit provides 2D layout only; drawing is not evidence of validity. Orange indicates a change from the previous saved state. Fixed atom slots are an alternate graph view.
- Retain argmax indices from the unrounded source arrays. Probability display rounding must not change molecular identity or graph-change navigation. Preserve full geometry precision at warning boundaries. Viewer atom labels start at 1; code slot indices in research JSON start at 0.
- The element-aware stretching diagnostic divides frozen distance by RDKit covalent-radius sum and flags ratios above 1.35. Purple marks this diagnostic, red marks the separate broad 0.8–2.5 Å check; warning colours override orange changes, while thicker bonds still mark changes. Use the generous single-bond reference for all bond types, label its scope, and keep it separate from the study gate and full geometry/strain validation.
- Entropy is the mean categorical entropy in nats. Charge entropy stays unavailable unless charge probabilities were actually saved. This version's ordinary graph optimizes atom and bond probabilities, with no charge category.
- Show all numeric recorded loss/auxiliary terms under their own names. Explicit chemistry corrections are not loss components. Loss and predictive scores must be interpreted using the saved backend settings.
- Validity means saved RDKit sanitization. Connectivity and over-valence counts are separate. Missing readouts remain unknown, never carried forward or changed to zero. The validity chart is an indicator per state, not a sample-validity estimate.
- Per-atom valence shows the reconstructed input and output of the first valence projection after the gradient/simplex step and optional fluidity correction. This is explicitly distinguished from the evaluated molecule. Check `replay_matches_receipt` before interpreting the reconstruction. Later reprojections and bridge retries are retained in the receipt.
- Atom-support alternatives have separate counts and probability-displacement curves. Their first bond-valence replay must use the guarded atom probabilities, from the receipt or the following state when the guard is the only atom-changing correction. Unverified replays are hidden; the chart then shows only the evaluated state's valence and says why.
- Connectivity shows proposed, surviving, and lost bridges. Sparsity shows expected bond mass removed; at an existence threshold of 0.5 or below, normalized affected rows already have NO_BOND as argmax, so do not describe the shrink itself as deleting current graph bonds. Later trajectory effects can follow changed continuous inputs and gradients. Exact changed-pair counts stay in receipts. Do not replace missing operator information with zeros.
- For legacy hardening gap records without `soft_score_source`, treat `soft_score` as unavailable: the older writer copied hardened forward auxiliaries. Separate recomputed `soft_loss` and `hard_loss` remain valid. Export a qualification note and preserve original source files. New staged runs explicitly separate relaxed auxiliary scores from optimization-forward scores.
- Keep configured learning rates, raw gradient norms, raw proposed step sizes, simplex step sizes, and observed next-state changes distinct. The recorded forward hardness is a separate curve. Hardening changing a backward Jacobian is not itself a learning-rate schedule.
- Gradient/movement agreement is the cosine between the negative simplex-tangent gradient and the observed following displacement, including corrections. With hardening (5.5), label it as a surrogate derivative. A correction may oppose model descent to maintain feasibility; agreement alone is not a success criterion.
- Nested native-proposal runs can include a separate native evaluation panel. Keep rebuilt native scores separate from the proposal loss and show both ensemble members. A proposal-step link requires exact saved step and canonical-identity agreement. Only an exact affinity-scored-pose capture qualifies that pose as checked; pre-affinity geometry and first-structure confidence describe a different pose. Native results used for tuning are not held out, and binder-head probabilities are model outputs rather than measured binding probabilities.

## Run management

Star runs worth keeping. Starred runs cannot be deleted until unstarred. Single and bulk deletion move **exported copies** to the library's `.trash/`; the Trash filter restores them. Source experiments, checkpoints, and raw artifacts are never deleted by this viewer. Rebuilds do not bring trashed runs back. Moving to trash clears the list but does not free its disk space. The Trash view also has a separate permanent-delete action with a confirmation; it frees only exported copies and retains deletion markers so a rebuild does not bring them back.

Do not delete actual experiments while testing run management. Use temporary library fixtures. Do not commit generated run data, curation, virtual environments, or output libraries.

## Verify changes

```sh
.venv-visualizations/bin/python -m unittest discover -s visualizations/tests -v
node --check visualizations/assets/viewer.js
```

Open a library made from real saved runs. Check the first/middle/final slider positions, switching runs during playback, atom and bond selection, missing charge information, valence before/after values, and recorded hardening. Check wide and narrow layouts. Exercise star/description persistence and single/bulk trash and restore in a temporary library. Keep any source or export failures visible.

## Cross-scorer comparisons and starting-chain baseline

Use `visualizations/scripts/publish_evaluation.py COMPARISON.json LIBRARY` to add a saved 14-identity native Boltz/GNINA panel to the existing viewer. It preserves the run catalog, descriptions, stars and trash. The server exposes the saved comparison at `/evaluation.json`; restart an older server once after adding this route. Open **Boltz / GNINA comparison** in the header.

The matching analysis is retained in the `stage2-protocol` research checkout in `testing/stereo_matched_evaluation/`; it is not included in this template folder. Include C25 as the starting graph and supplied neutral JQ1 separately. Report molecule-minus-C25 deltas only within each scorer, keep all declared GNINA sampling repeats, and label single-seed native predictions. Check stereochemistry reassigned from the actual 3D scored coordinates, not just SMILES or SDF stereo tags. Keep mismatched scores visible and explicitly unqualified. Protein C-alpha alignment is a rigid display transform only. Do not apply a pre-affinity geometry check to an affinity-scored native pose. Pocket proximity, severe overlaps, bond sanity and stereo remain separate gates; none establishes binding.

Publishing a comparison also adds molecule shortcuts to the shared library. These entries open the saved comparison and selected molecule; they are labeled **molecule result** and contain no invented trajectory states. They can be starred, described, trashed and restored through the existing library API. The Starred view puts molecule shortcuts first. Publishing again preserves their curation and does not restore trashed shortcuts.
