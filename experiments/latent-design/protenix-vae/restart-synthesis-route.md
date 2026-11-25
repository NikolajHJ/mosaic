# Binding-matched route-planning pilot

Do the archived BRD4 candidates with lower audited SA admit more routes to purchasable precursors under the same planner budget?

## Matched panel

Six historical Protenix loss centers (2.3025, 2.34, 2.36, 2.38, 2.40 and 2.44) contribute one distinct molecule from each of four groups. The groups have pre-smoothing SA at most 5, above 5 through 6, above 6 through 8, and above 8 with native RDKit SA at most 5. For each cell, choose the unused candidate with the closest successful archived fresh loss; break ties by SMILES. Exclude the three optimization founders. The largest within-block loss spread is 0.00840. Repeated occurrences retain the best observed loss and its job, arm, wave, branch and step. This focuses the pilot on the binding range shared by the groups.

A fixed seeded shuffle mixes the 24 candidates and two process controls (aspirin and paracetamol). The route worker receives opaque IDs and SMILES only. The held-out key preserves the group and provenance. The controls are excluded from research hit-rate and ranking comparisons.

## Frozen RetroGNN readout

The Molecule.one-trained classifier supplies a ranking from a different planner. It read the same 26 blind SMILES with its checkpoint's Chemprop revision and original hard-graph featurizer. All checkpoint keys loaded exactly, and the model ran in CPU inference mode. The released training metadata mentions uncommitted upstream changes, so exact training-runtime identity remains unknown. We saved both the sigmoid output and its input (logit) because some predictions round to zero or one. Higher logits were chosen as the favorable direction before comparison to the route outputs.

On 12 released Molecule.one examples with source score zero and 12 with source score at least eight, the higher-score example had the higher logit in 142 of 144 cross-group pairs. Aspirin and paracetamol both received high logits. Those examples check score direction. The 24 archived structures drive the independent planner comparison. Only two of the 24 received classifier output above 0.5, both in the audited SA-at-most-five group. The numerical outputs have no calibration to laboratory success.

With the original Chemprop checkout on the Python path, the readout command is:

```sh
python -m latentmol.evaluation.retrognn_readout --panel outputs/synthesis-route-20261117/blind-panel.json --checkpoint outputs/synthesis-retrognn-20261117/model.pt --reference-data outputs/synthesis-retrognn-20261117/m1-upstream.csv --output NEW_RESULT.json
```

[Predictions and reference checks](../../../outputs/synthesis-retrognn-20261117/blind-scores.json) · [Input versions and hashes](../../../outputs/synthesis-retrognn-20261117/sources.json) · [Official models](https://github.com/pchliu/RetroGNN) · [Chemprop revision](https://github.com/chemprop/chemprop/tree/7d173c134b286acd4c039bf5cdb80a642fb66c47).

## Run

Use [synthesis_route_readout.py](restart-operations/synthesis_route_readout.py) with `prepare AUDIT_DIR OUTPUT_DIR` followed by `search PANEL ASSETS OUTPUT --shard INDEX --shards COUNT`. The route pilot used AiZynthFinder 4.4.1, the authors' USPTO expansion policy, ring-breaker policy, USPTO filter and ZINC stock. The same limits apply to each target: 100 iterations, 75 seconds and maximum depth six. Stop on a complete route to stock and exclude the target itself from stock; log actual search time, iterations and per-target seed. Failed execution has a distinct null outcome.

Preparation job 30099894 downloaded all assets and installed the isolated environment. PyArrow 26 required NumPy 2 while the planner requires NumPy 1.x. Compatibility job 30099911 completed the import check with PyArrow 17.0.0 and Dask 2024.8.2. The unstarted array 30099904 was cancelled after its preparation dependency exited. Replacement array 30099916[1-4] runs four independent planner shards on the `hpc` CPU queue. [Blinded inputs](../../../outputs/synthesis-route-20261117/blind-panel.json), [provenance and group key](../../../outputs/synthesis-route-20261117/key.json), [submission receipt](../../../outputs/synthesis-route-20261117/search-submission-r2.json), [public data specification](https://github.com/MolecularAI/aizynthfinder/blob/v4.4.1/aizynthfinder/tools/download_public_data.py).

## Results

All four CPU workers finished with 26 usable readouts. AiZynthFinder found a one-step route for both positive controls in one iteration each. The 24 search molecules had zero complete routes, including zero of six audited SA at most 5. Sixteen exhausted the iteration allowance and eight reached the time bound. The workers used 2,246 search iterations and 1,403.5 seconds of planner search time in total. An unsolved target could still have a route outside this stock, policy or budget.

RetroGNN's frozen Molecule.one hard-graph classifier assigned the audited SA-at-most-5 stratum the most favorable median logit (-9.59, compared with -25.76 for SA 5-6, -113.41 for SA 6-8 and -33.15 for the false-low stratum). The two controls scored higher than all archive molecules. With all 24 study outcomes identical, there is no solved/unsolved pair for evaluating the model's rank ordering within blocks or overall. This pilot gives no basis for adding a RetroGNN gradient term to the search.

[Planner readouts](../../../outputs/synthesis-route-20261117/results.json), [per-input provenance, hard scores and outcomes](../../../outputs/synthesis-route-20261117/joined.json), [ranking analysis](../../../outputs/synthesis-route-20261117/ranking.json), [fixed asset hashes and scheduler receipt](../../../outputs/synthesis-route-20261117/summary.json).
