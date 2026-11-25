# Molecular VAE training

This separately installable project brings the DBMol graph VAE trainer into Mosaic. It preserves the `latentmol` import paths and PyTorch/RDKit environment. Mosaic's existing VAE inference provider and published model remain in their existing locations.

From this directory, in the VAE training environment:

```sh
python -m pip install -e '.[test]'
python -m latentmol.training.workflow --help
python -m experiments.vae.attention_next_leg --help
python -m pytest -q
```

The initial import includes the 256-dimensional graph VAE, graph matching and ordering, projected chemistry and motif guidance, round-trip supervision, and the wider attention implementation. Its focused tests use synthetic fixtures. The epoch-48 recipe captures the existing campaign. Its archive, dataset and frozen teachers still live on DTU, and its paths are tied to the original submission.

The attention entry performs the one-time graft from the older checkpoint. Resuming the output of that leg requires loading the expanded model directly and restoring its optimizer, RNG, and training cursor. Validate that resume on a disposable output before handing off production. Source for this import was the DBMol exploratory working tree on 25 November 2026, including code beyond commit `0eaca69745c0`.
