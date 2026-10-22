# Original project brief versus implemented scope

Source review: 22 October 2026. The [original PDF](project_desc.pdf) is preserved
as a design-stage brief. Its opening statement that Mosaic lacks small-molecule
graph support describes the starting point, not today's repository. This page
maps that distinction without rewriting the proposal or claiming numerical
acceptance beyond the recorded experiments.

| Brief / documentation theme | What the current source establishes |
| --- | --- |
| Add a small-molecule graph branch | `src/mosaic/small_molecule/` now contains graph state, features, native hooks, optimizers, validation and evidence IO. Code existence is separate from a complete native acceptance result. |
| Frozen Protenix-Tiny | The adapter and pinned asset/source checks are implemented in `protenix.py` and companion native modules. The [contract](stage_0_native_contract.md) defines supported feature and geometry scope. |
| Initial fixed-occupancy engineering fixture | The Stage 0 JSON declares 10 atoms, seed 17, geometry off, five updates and two context modes. This is an engineering fixture, not a BRD4 binding result. |
| Stabilize unconstrained optimization, then add chemistry | Later [experiment homes](../experiments/INDEX.md) contain distinct control, chemistry, Boltz and JQ1 protocols. Their initializations, objectives and acceptance policies are not one interchangeable global default. |
| Chemistry feasibility / repair | Rule-based chemistry guidance exists under `chemistry/`. Bounded valid-repair witnesses do not establish a globally nearest-valid molecule. |
| Future learned prior | The audited small-molecule source contains no implemented DeFoG integration. The brief's proposed learned-prior stage is not a completed result. |

The [Stage 0 usage guide](stage_0_usage.md) distinguishes the completed focused
16 October gradient study from the separate full A-G acceptance scope. The
[ledger](experiment_ledger.md) and each attempt's authenticated source/config
records determine what actually ran and passed or failed. A proposal, authored
test, static audit, successful installation or report is not a substitute.

The original brief and later dated documents may contain private input/runtime
paths. They remain evidence of the original setup, not shared storage or public
assets. Use the [current HPC guide](HPC_GUIDE.md) when preparing another account.
