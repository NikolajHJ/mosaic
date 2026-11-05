# DBMol VAE · Epoch 45

**Work in progress.** This checkpoint is part of an experiment on
gradient-guided small-molecule design. The encoder maps a molecular graph
to a 256-dimensional latent distribution. The decoder maps a latent vector
to a relaxed graph of atom, formal-charge, and bond probabilities. The
research question is whether optimizing that vector with a differentiable
scorer can produce valid, useful molecular edits.

## Model

- Trained on MOSES through epoch 45 with a variational reconstruction
  objective and chemistry repair guidance.
- Uses four triangle-refinement rounds and supports up to 50 heavy-atom
  slots, with an absent-atom category.
- Has 2,173,216 encoder and 1,768,644 decoder parameters, for 3,941,860 total.
- Represents 2D connectivity, formal charges, and bond orders. The graph
  schema has no stereochemical labels or 3D coordinates.

## Evaluation

Measured on 10,000 held-out MOSES validation molecules and 10,000
independent standard-normal latent draws. Reconstruction uses the encoder's
posterior mean and argmax decoding. Prior sampling uses the same latent
draws for argmax and one categorical graph sample per vector. Evaluation
was at temperature 1.0 before chemical repair or rejection.

| Metric | Result |
| --- | ---: |
| Valid reconstructions | 9,992/10,000 (99.92%) |
| Exact reconstructions | 9,987/10,000 (99.87%) |
| Valid prior decodes, argmax | 7,541/10,000 (75.41%) |
| Valid prior decodes, categorical | 6,330/10,000 (63.30%) |

Validity requires a connected, sanitizable graph that round-trips through
the representation. It measures graph quality, with binding and
synthesizability left to separate evaluations.

## Open questions

- Does training on MOSES plus ChEMBL improve prior validity and scaffold
  coverage while preserving reconstruction?
- How far can binding-guided updates move from an encoded molecule before
  reaching a distinct valid edit?
- Which temperature schedules yield more valid edits that improve on
  fresh discrete scoring?

## Download and load

[Download `epoch_0045.pt`](epoch_0045.pt). This PyTorch inference export
contains the exact epoch-45 encoder and decoder tensors, model configuration,
and graph schema. Load it with the DBMol Exploratory `latentmol` package.
Training resume requires the original training checkpoint.

```python
from pathlib import Path

from latentmol.evaluation.evaluate import load_model

model, checkpoint = load_model(Path("models/dbmol-vae/epoch_0045.pt"), "cpu")
model.eval()
```

| Artifact | SHA-256 |
| --- | --- |
| Inference export (15,853,952 bytes) | `104e1be32954b98f3ae33dded19cbf6e56326f23a4a89cdd8b9209a324a34da3` |
| Source training checkpoint | `5c4946c15515e70bd01281eea7b74693c18598c4b47f6724fcc9898deb1302be` |
