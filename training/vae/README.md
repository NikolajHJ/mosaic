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

The attention entry supports the original one-time graft and direct resume from its expanded checkpoint. On an allocated DTU GPU, start from this directory and point the command at the original pinned source and the previous segment. Use a new output directory:

```sh
LATENTMOL_EVALUATION_MODULE=experiments.vae.attention_next_leg \
  bash experiments/vae/run_checkpoint_comparison.sh \
  --config experiments/vae/epoch48-seven-round-generated-roundtrip.json \
  --source-dir "$PINNED_PRE_GRAFT_SOURCE" \
  --resume-expanded-from "$PREVIOUS_SEGMENT" \
  --resume-sha256 "$CHECKPOINT_SHA256" \
  --output "$NEW_SEGMENT"
```

The previous segment's `fork.json` and `latest.json` bind the checkpoint and its effective loss policy. Resume restores the saved graph model, AdamW state, RNG, and data cursor. The source, parent checkpoint, frozen teachers, 60M data view, pinned GPU resolver, and existing Python environment are external DTU assets. The included [bootstrap](scripts/hpc/hpc_launch.py) stages sealed source on the allocated worker. Use a disposable output for the first GPU resume check before scheduling a production successor.

The import comes from the November 25 DBMol exploratory working tree, including code beyond commit `0eaca69745c0`. Earlier source, documentation, and non-VAE experiments are available on the [legacy worktree branch](https://github.com/NikolajHJ/mosaic/tree/dbmol/flow-guided-hallucination-worktree). The [migration record](../../docs/dbmol-migration.md) lists the remaining branches and artifact locations.
