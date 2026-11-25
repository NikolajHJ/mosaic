# Training

Losses, training diagnostics and composed train/evaluate workflow.

The existing epoch/checkpoint trainer remains `latentmol.train`; campaign-specific continuations remain at their existing paths.

Modules: `diagnostics.py`, `losses.py`, `workflow.py`.

`matching.py` aligns known reconstruction targets using a bounded detached
graph match, then uses the existing categorical loss. `invariant_pilot.py`
is the short warm-started invariant-encoder experiment; it never resumes or
overwrites a parent checkpoint. Old ordered encoder defaults remain unchanged.

Keep reusable calculations here. Experiment setup and results belong in the
[experiment homes](../../../experiments/README.md). Old flat imports forward
to these same modules; edit the implementation here.
