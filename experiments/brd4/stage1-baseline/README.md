# BRD4 Stage 1 baseline

This experiment asks whether gradient updates can improve Protenix-Tiny's
predicted ligand contacts before adding any chemistry rules. It follows
[protocol v12](../../../docs/Small_Molecule_Hallucination_ProtenixTiny_Project_Protocol_v12.md).
The canonical initialization uses 25 atom slots, 75% carbon probability,
expected degree 2.25, and small independent perturbations. Each update uses
the same model key and projects atom and bond rows back to probabilities.

The model receives the observed BRD4-BD1 sequence from 3MXF chain A. JQ1 and
other nonprotein residues are excluded from its input. Every target residue
can contribute to the contact objective. The geometry-free ligand input has
no valid ligand reference frames, so the PAE prerequisite fails by design.
This implementation records that prerequisite; it does not run the native
hard-ligand PAE trial or claim a PAE result.

- `general.yaml`: protein file and short run settings.
- [Stage settings](../../../configs/stage1.yaml): initialization, optional
  separate step sizes, and how often to score the discrete graph.
- [Pipeline](../../../src/mosaic/small_molecule/Stage1/pipeline.py): design loop
  and saved outputs; `diagnostics.py` checks the soft and discrete graphs.

When authorized to run, use the repository root on an allocated DTU HPC GPU
with the pinned Protenij version and model assets used by Stage 0:

```bash
PYTHONPATH=src python -m mosaic.small_molecule.generate_molecule \
  --general-config experiments/brd4/stage1-baseline/general.yaml \
  --stage-config configs/stage1.yaml
```

Choose a fresh output directory for each attempt. The output contains both
YAMLs, `target.json`, and a manifest with input identities and optimizer and
random key settings. `trace.jsonl` records the loss, gradient sizes, graph
movement, bond mass, saturation, and expected valence at each state.
`hard_readouts.jsonl` records raw argmax choices, RDKit validity, fragments,
valence and the soft score minus the hard score, including at the start and
end. The original probabilities are in `trajectory.npz`. The latest readout
checkpoint stores the graph, state, step sizes, and model key; automatic resume
is not implemented.

Implementation only: no GPU run or Stage 1 acceptance test has been performed.

For the separate affinity variant of this workflow, see [Using Boltz2](boltz2.md).
