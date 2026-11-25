# Evaluation

Checkpoint metrics, graded molecular quality and quality-panel evaluation.

Use the generic evaluator for new studies; saved native/fresh-pose studies keep their own recorded contracts.

Modules: `evaluate.py`, `panel.py`, `quality.py`.

`checkpoint_comparison.py` evaluates multiple frozen checkpoints on shared
fresh prior, reconstruction and perturbation panels, preserving raw invalid
graphs and reporting paired differences. It never starts training.

`molecular_pairs.py` selects and matches bounded chemical-neighbor pairs;
`interpolation.py` evaluates frozen paths with canonical and consistently
re-encoded matched inputs. Endpoint reconstruction and latent-order sensitivity
are explicit controls. Saved-only presentation lives in `reporting/interpolation.py`.

`atom_order.py` permutes full input graphs, including padding, and measures
posterior invariance, decoded identity/label stability and reordered accuracy.
It can be requested through `checkpoint_comparison.py`'s `atom_order` config.

Keep reusable calculations here. Experiment setup and results belong in the
[experiment homes](../../../experiments/README.md). Old flat imports forward
to these same modules; edit the implementation here.
