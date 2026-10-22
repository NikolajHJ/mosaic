# Examples and legacy experiment entry points

General model notebooks and teaching examples stay here. Experiment drivers have
canonical homes in the [experiment index](../experiments/INDEX.md) or documented
shared tooling; their old Python files here are forwarding entry points.

See the [complete file map](../experiments/MIGRATION.md) before editing one.
The [configs/](configs/README.md) files are byte-identical declarative compatibility
copies; canonical experiment configs are the owners. This preserves callers that
open a saved old path directly. No historical release or run output is moved.
