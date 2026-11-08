# Protenix restart search

- [Chemistry pilot](restarts.md) records the three five-cycle runs and the matched fresh scoring setup.
- [Matched follow-up](restart-followup.md) records the perturbation, step cap, duration, and lineage comparisons.
- [Dense restart audit and LR 70 pilot](restart-lr70.md) traces the dense graphs and records the screened follow-up.
- [Six-cycle LR 70 comparison](restart-lr70-six-cycle.md) tests step cap, screened starts, valence guidance, shortlisting, and parent selection.
- [Matched LR 30 repeats](restart-lr30.md) repeat the standalone and six-arm LR 70 experiments with the same settings and archived source.
- [Restart noise and trajectory replay](restart-noise.md) compares perturbation strength, molecule retention, and path divergence.
- [LR 100 diversity pilot](restart-diversity-lr100.md) compares a strict final molecule panel with the score-only panel.
- [Interactive report](../../../outputs/protenix-restart-report-20261108/protenix-restart-report.html) shows saved molecules, trajectories, and verified restart trees for nine selected runs. The [7 November report](../../../outputs/protenix-restart-report-20261107/protenix-restart-report.html) covers the earlier chemistry and matched follow-ups.
- [Historical submission scripts](restart-operations/) preserve the two operators as used from their recorded output directories.

The shared Protenix runtime comes from the verified pre-restart flow snapshot in `outputs/flow-lr-20261106a/source.tar.gz`. The restart jobs use their own sealed source archives and submission receipts in their recorded output directories.

## Rebuild the interactive report

Run from the repository root with the project's Python environment. Use the job IDs already in the snapshots to make a smaller report. Repeat `--job` to include several jobs or use `--exclude-job` to drop specific jobs.

```powershell
python experiments/latent-design/protenix-vae/restart-report/build.py --data-dir outputs/protenix-restart-report-20261108 --job 29920256 --job 29918056 --output outputs/protenix-restart-report-20261108/selected.html
```

To refresh the data or include runs that are absent from the snapshots, use the same job selection for both read-only DTU collectors. Each collector writes a new snapshot file. See the [run registry](restart-report/runs.json) and the [HPC connection guide](../../../docs/HPC_CONNECTION.md).

```powershell
$report = 'experiments/latent-design/protenix-vae/restart-report'
$capture = 'outputs/protenix-restart-report-new'
New-Item -ItemType Directory -Path $capture
python "$report/collect.py" --include-raw --job 29920256 --job 29918056 --output "$capture/playback-snapshot.json"
python "$report/collect_trees.py" --job 29920256 --job 29918056 --output "$capture/tree-snapshot.json"
python "$report/build.py" --data-dir $capture
```

Add a new job to `runs.json` with its output and launch receipt paths, layout, and update budget. Both collectors accept `--manifest` for a separate registry and `--all-runs` for every entry. Without a selection, the checked-in registry uses its `default_jobs` list for the original report.

The tree groups repeated molecular states within each branch. The selected path chart retains every saved loss observation through the exact restart update and includes repeated states. Clicking the chart or moving the slider selects the same update. The CSV export uses that selected path.

## Completed first trajectory — 3 November 2026 09:30 CET

The preceding 50,000-update Protenix study used chemistry weights 1 and 10 with cooling and changing noise. Its later phases had higher validity. The chemistry pilot reuses these two weights with a fixed temperature and a separate restart perturbation.
