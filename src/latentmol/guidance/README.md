# Guidance

Reusable chemistry losses and training adapters.

`exchange.py` adds opt-in mixed atom/charge/bond repair costs and a continuous
penalty against a fixed valid ring/functional-group reference. The latent-design
runner enables it only through `exchange_guidance`; existing defaults retain
bond-only repair. See the [comparison and reference requirements](../../../experiments/chemistry-guidance/atom-exchange.md).
The explicit `edit_cost: one_minus_probability` option adds constant exchange
surcharges to `1-p(replacement)`; see the [expanded tests](../../../experiments/chemistry-guidance/atom-exchange-linear.md).

Neighbor environment, local valence and connectivity losses are opt-in components, not evidence that any particular weight improves training.

Modules: `connectivity.py`, `environment.py`, `environment_policy.py`, `neighbor.py`, `regularization.py`.

`repair_proposals.py` provides opt-in fixed-atom target discovery: conditional
bond samples, component-joining/connected pruning proposals and legacy beam
fallback. It reuses the existing teacher/worker/loss interfaces; defaults for
existing callers and running campaigns are unchanged. See the
[comparison and usage](../../../experiments/chemistry-guidance/README.md#fixed-atom-repair-proposals).

`repair_astar.py` is a separate opt-in probability-aware A* backend with a
remaining-edit lower bound, explicit interruption budgets and the same target
and loss interface. See its [benchmark and assumptions](../../../experiments/chemistry-guidance/README.md#probability-aware-astar).

Keep reusable calculations here. Experiment setup and results belong in the
[experiment homes](../../../experiments/README.md). Old flat imports forward
to these same modules; edit the implementation here.
