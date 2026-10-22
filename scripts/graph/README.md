# Shared graph tools

Shared graph/control driver, fixture preparation, launch admission and allocation scripts. Used by relaxed-gradients, carbon-chain, known-ligand and BRD4 initialization controls.

These are shared dependencies, not independent scientific experiments. [Experiment inventory](../../experiments/INDEX.md) lists their consumers.

| Canonical file | Role | Legacy compatibility path |
| --- | --- | --- |
| [augment_protenix_hard_controls.py](augment_protenix_hard_controls.py) | Add authenticated hard-control results to an existing allocated-HPC report. | `scripts/augment_protenix_hard_controls.py` |
| [prepare_brd4_fixture.py](prepare_brd4_fixture.py) | Authenticate the complete 3MXF BRD4-BD1 sequence on allocated DTU HPC. | `scripts/prepare_brd4_fixture.py` |
| [prepare_known_ligand_fixture.py](prepare_known_ligand_fixture.py) | Prepare the authenticated 181L sequence/contact fixture on allocated DTU HPC. | `scripts/prepare_known_ligand_fixture.py` |
| [assess_uniform_control.py](assess_uniform_control.py) | Bind one uniform control to the passed BRD4 shape budget on allocated CPU. | `scripts/assess_uniform_control.py` |
| [validate_brd4_launch.py](validate_brd4_launch.py) | Authenticate a ready resource assessment before the long BRD4 allocation. | `scripts/validate_brd4_launch.py` |
| [validate_brd4_parallel.py](validate_brd4_parallel.py) | Allocated scheduling-only readiness; preserve the passed optimization engine. | `scripts/validate_brd4_parallel.py` |
| [augment_protenix_controls.lsf](augment_protenix_controls.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/augment_protenix_controls.lsf` |
| [brd4_contact.lsf](brd4_contact.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_contact.lsf` |
| [brd4_parallel_readiness.lsf](brd4_parallel_readiness.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_parallel_readiness.lsf` |
| [brd4_parallel_report.lsf](brd4_parallel_report.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_parallel_report.lsf` |
| [brd4_readiness.lsf](brd4_readiness.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_readiness.lsf` |
| [brd4_uniform_readiness.lsf](brd4_uniform_readiness.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/brd4_uniform_readiness.lsf` |
| [protenix_relaxed_gradients.py](protenix_relaxed_gradients.py) | Run an explicit frozen Tiny contact-gradient study on allocated DTU HPC. | `examples/protenix_relaxed_gradients.py` |
| [score_protenix_hard_controls.py](score_protenix_hard_controls.py) | Like-for-like frozen hard-graph controls for an authenticated completed study. | `examples/score_protenix_hard_controls.py` |

Use canonical paths for new work. Old Python/shell paths forward to these files; retained declarative files are checked against their canonical owner. All original numerical/allocation/scientific guards apply. No model, numerical test or scheduler command was run for this migration.
