# Shared reporting tools

Generic saved Protenix trajectory renderer, raw-graph depictions, SVG display and localhost serving. Used by graph controls and BRD4 reports. Rendering still requires its existing allocated-HPC conditions.

These are shared dependencies, not independent scientific experiments. [Experiment inventory](../../experiments/INDEX.md) lists their consumers.

| Canonical file | Role | Legacy compatibility path |
| --- | --- | --- |
| [render_protenix_trajectory.py](render_protenix_trajectory.py) | Plot and depict saved relaxed trajectories on allocated DTU HPC only. | `scripts/render_protenix_trajectory.py` |
| [protenix_trajectory_depiction.py](protenix_trajectory_depiction.py) | Probability-weighted raw graph depictions for the detached HPC report. | `scripts/protenix_trajectory_depiction.py` |
| [protenix_trajectory_svg.py](protenix_trajectory_svg.py) | Fast SVG for saved graphs, using the report's fixed probability size rules. | `scripts/protenix_trajectory_svg.py` |
| [serve_protenix_report.py](serve_protenix_report.py) | Serve an existing report on loopback, independently of the launching chat. | `scripts/serve_protenix_report.py` |
| [render_protenix_trajectory.lsf](render_protenix_trajectory.lsf) | Allocated execution/collection launcher; arguments and resource guards remain in source. | `scripts/hpc/render_protenix_trajectory.lsf` |
| [protenix_trajectory_report.html](protenix_trajectory_report.html) | Saved-artifact report template; rendering changes no scientific outcome. | `scripts/templates/protenix_trajectory_report.html` |

Use canonical paths for new work. Old Python/shell paths forward to these files; retained declarative files are checked against their canonical owner. All original numerical/allocation/scientific guards apply. No model, numerical test or scheduler command was run for this migration.
