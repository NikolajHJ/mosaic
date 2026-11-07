# Protenix restart search

- [Chemistry pilot](restarts.md) records the three five-cycle runs and the matched fresh scoring setup.
- [Matched follow-up](restart-followup.md) records the perturbation, step cap, duration, and lineage comparisons.
- [Dense restart audit and LR 70 pilot](restart-lr70.md) traces the dense graphs and records the screened follow-up.
- [Six-cycle LR 70 comparison](restart-lr70-six-cycle.md) tests step cap, screened starts, valence guidance, shortlisting, and parent selection.
- [Interactive report](../../../outputs/protenix-restart-report-20261107/protenix-restart-report.html) shows saved molecules, trajectories, and verified restart trees.
- [Historical submission scripts](restart-operations/) preserve the two operators as used from their recorded output directories.

The shared Protenix runtime comes from the verified pre-restart flow snapshot in `outputs/flow-lr-20261106a/source.tar.gz`. The restart jobs use their own sealed source archives and submission receipts in their recorded output directories.

To rebuild the report from the saved snapshots in this checkout, run this command from the repository root.

```powershell
python experiments/latent-design/protenix-vae/restart-report/build.py
```

The builder reads the playback and tree snapshots in `outputs/protenix-restart-report-20261107`. The report folder also contains two read-only DTU collectors. To capture a later snapshot, use `collect.py --include-raw --output <new-path>` or `collect_trees.py --output <new-path>` after reading the [HPC connection guide](../../../docs/HPC_CONNECTION.md). Each collector creates a new file.

## Completed first trajectory — 3 November 2026 09:30 CET

The preceding 50,000-update Protenix study used chemistry weights 1 and 10 with cooling and changing noise. Its later phases had higher validity. The chemistry pilot reuses these two weights with a fixed temperature and a separate restart perturbation.
