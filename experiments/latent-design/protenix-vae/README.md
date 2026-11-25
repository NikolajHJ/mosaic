# Protenix restart search

Start with the [two-page illustrated brief](../../../outputs/adaptive-search-handoff-20261119/adaptive-search.pdf) for the motivation and one real parent–child pair. The [agent research guide](adaptive-search-agent-guide.md) covers the matched benchmarks, synthesis and docking evidence, runnable settings, and next experiments.

For new small-molecule optimization experiments, start with the [direct relaxed-graph baseline](restart-boltz-brd4-direct.md). The [synthesis-aware protocol and Vina pilot](restart-synthesis-standard.md) document the earlier VAE latent experiments.

The [paired BRD4 PharmacoNet pilot](restart-boltz-pharmaconet.md) tests direct graph guidance and measures which pocket interactions survive independent redocking.

- [Eighteen-wave ERK2 search](restart-boltz-erk2-18.md) tests 75-update children in the CQ8 kinase pocket with four parents and two children per parent.
- The epoch-48 VAE searches optimize BRD4 [with Protenix](restart-brd4-e48.md) and ERK2 [with the Boltz2 DBMol loss](restart-boltz-erk2-18.md). Both use checkpoint update 623116.
- The [epoch-48 interactive report](../../../outputs/epoch48-interactive-report/index.html) compares the two M02 searches, the two broad BRD4 searches and the ERK2 snapshot. Select a molecule to open its exact step in the [search trees](../../../outputs/epoch48-interactive-report/trees.html).
- [Three-generation BRD4 comparison](restart-boltz-generations.md) advances the two policies selected by the allocation pilots onto three new paired seeds.
- The audited synthesis experiments cover [fresh archive screening](restart-synthesis-archive-screen.md), [three starting panels](restart-synthesis-founders.md), and [binding-matched route planning and RetroGNN](restart-synthesis-route.md).
- [BRD4 search-method pilots](restart-boltz-search-methods.md) compare child allocation, prior screening and local reuse of Boltz gradients across three matched seed panels.
- [BRD4 Boltz2 DBMol long trial](restart-boltz-brd4.md) uses LR 1, 400-update children and a 22-hour working budget on one 32 GB V100.
- [Seven-target Boltz2 DBMol search](restart-boltz-targets.md) switches the new protein panel to Boltz2, LR 1, and sizes a six-generation baseline from a short native timing pilot.
- [Boltz2 parallel branches and speed probes](restart-boltz-speed.md) tests resident CUDA workers, precision and checkpoint options with matched child starts before the full searches.
- [FP32 versus BF16 design quality](restart-boltz-precision.md) compares six paired 200-update children and ranks all valid outputs under one common FP32 DBMol scorer.

- [Configurable restart runner](restart-search.md) runs long and short cyclic searches from a shared entry point with presets, overrides, and per-wave logs.
- [Chemistry evaluation of long-search winners](restart-evaluation.md) evaluates three selected molecules with Mosaic's CPU pipeline and records their source runs and synthetic-accessibility concerns.
- [Frozen synthesis guidance](restart-synthesis-pilot.md) tests pretrained SynFrag, DFRscore and their combination with matched Protenix budgets; includes a completed short CPU diagnostic.
- [Synthesis guidance follow-ups](restart-synthesis-next.md) compares controlled gradients, phased guidance, GASA, DeFoG and real-molecule latent priors, plus a four-wave simplification/recovery panel.
- [Aligned synthesis priors and search policies](restart-synthesis-followup.md) tests aligned latent banks, attraction strength and locality, shared screening, parent selection and intermediate replay across three seeds.
- [Restart branch-slot pilots](restart-branch-slot.md) compare best-parent reservation, near-valid replay, and direct graph edits.
- [Chemistry pilot](restarts.md) records the three five-cycle runs and the matched fresh scoring setup.
- [Matched follow-up](restart-followup.md) records the perturbation, step cap, duration, and lineage comparisons.
- [Dense restart audit and LR 70 pilot](restart-lr70.md) traces the dense graphs and records the screened follow-up.
- [Six-cycle LR 70 comparison](restart-lr70-six-cycle.md) tests step cap, screened starts, valence guidance, shortlisting, and parent selection.
- [Matched LR 30 repeats](restart-lr30.md) repeat the standalone and six-arm LR 70 experiments with the same settings and archived source.
- [Matched LR 70 step-cap pilots](restart-step-cap.md) compare 0.25 and 0.5 across four seed panels, with short cap-schedule and initialization tests.
- [Restart noise and trajectory replay](restart-noise.md) compares perturbation strength, molecule retention, and path divergence.
- [Adaptive, fixed, and independent restarts](restart-lineage-feedback.md) tests whether updating the elite archive improves search at matched cost.
- [Matched 48,000-update search](restart-matched-search.md) compares adaptive, continuous, and independent optimization at common update and fresh scoring budgets.
- [Matched-method comparison PDF](../../../outputs/restart-matched-report-20261116/search-method-comparison.pdf) shows all three completed corrected seeds, learning curves, final scores, diversity and consumed scoring costs.
- [November 10 follow-ups](restart-followup-20261110.md) test a hybrid branch allocation, 500-update parent kicks, and conformer sensitivity.
- [Single-GPU batch throughput](restart-batch-throughput.md) measures child batching and CPU repair overlap on an A100 80 GB.
- [LR 100 diversity pilot](restart-diversity-lr100.md) compares a strict final molecule panel with the score-only panel.
- [Interactive report](../../../outputs/protenix-restart-report-20261110/protenix-restart-report.html) focuses on seven runs. It shows the matched LR 30 and LR 70 cap-0.5 searches, all three long searches, lineage feedback panel C, and the 500-update tiny-kick panel. The [full November 10 report](../../../outputs/protenix-restart-report-20261110/protenix-restart-report-full.html) retains all 49 completed trajectory runs. The [9 November report](../../../outputs/protenix-restart-report-20261109/protenix-restart-report.html), [8 November report](../../../outputs/protenix-restart-report-20261108/protenix-restart-report.html), and [7 November report](../../../outputs/protenix-restart-report-20261107/protenix-restart-report.html) preserve earlier snapshots.
- [Historical submission scripts](restart-operations/) preserve the two operators as used from their recorded output directories.

The shared Protenix runtime comes from the verified pre-restart flow snapshot in `outputs/flow-lr-20261106a/source.tar.gz`. The restart jobs use their own sealed source archives and submission receipts in their recorded output directories.

## Rebuild the interactive report

To rebuild the epoch-48 report from its saved receipts, run these commands from the repository root. Keep both HTML files together for offline navigation.

```powershell
$reports = 'experiments/latent-design/protenix-vae/restart-report'
$out = 'outputs/epoch48-interactive-report'
python "$reports/collect_epoch48.py"
python "$reports/build_epoch48.py" "$out/overview-data.json" "$out/index.html"
python "$reports/epoch48_trees.py" outputs/protenix-restart-report-e48-20261125 "$out/trees.html"
```

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

Add a new job to `runs.json` with its output and launch receipt paths, layout, and update budget. Both collectors accept `--manifest` for a separate registry and `--all-runs` for every entry. Without a selection, the registry's `default_jobs` list collects the completed trajectory runs for the full report. Use `--timeout 420` on each collector for a full refresh. To rebuild the focused report, use `--data-dir outputs/protenix-restart-report-20261111 --output outputs/protenix-restart-report-20261110/protenix-restart-report.html` with `build.py`. These snapshots contain only the seven selected runs.

The tree groups repeated molecular states within each branch. The selected path chart retains every saved loss observation through the exact restart update and includes repeated states. Clicking the chart or moving the slider selects the same update. The CSV export uses that selected path.

The learning-rate buttons switch between the two completed runs for the
same policy. The cycle chart compares the lowest fresh contact loss
among valid molecules scored in each completed cycle. Select a point to
see both winning molecules and open their saved trajectory frames.
Elite archives shows the saved leaders carried forward across cycles. For
lineage feedback, the cycle chart shows the best newly scored molecule
for each policy in that cycle.

## Completed first trajectory — 3 November 2026 09:30 CET

The preceding 50,000-update Protenix study used chemistry weights 1 and 10 with cooling and changing noise. Its later phases had higher validity. The chemistry pilot reuses these two weights with a fixed temperature and a separate restart perturbation.
