# DBMol records used in the Stage 3 DeFoG study

The files here preserve the historical inputs used by the Mosaic reports. The eight notes in [docs](docs/) are snapshots from the DBMol_exploratory working copy at `0eaca69745c0` on 25 November 2026. They describe the sampled DeFoG repair method, integration pilots, longer Boltz comparison and validity-loss diagnostics. Relative links within those notes refer to the historical DBMol checkout and its output archive.

The [guided configuration](source120/boltz_defog.yaml) and [control configuration](source120/boltz_only.yaml) are the resolved configs from the verified source120 attempt004 readiness record. The [compact refresh archive](source120-guided-refreshes.json.gz) contains the selected target hashes and occupied atom, charge and bond probability rows at the 100 refresh boundaries of the guided 500-update run. It was extracted after verifying the original HTML digest and checking all 501 states for the expected refresh cadence. The original and archived inputs produce byte-identical summaries.

The complete original guided and control HTML exports and their verification records were copied and hash-checked under the Mosaic worktree's `results/archives/dbmol-defog-source120/`. Those generated exports can be opened offline. The compact archive above is sufficient to reproduce the target-refresh analysis from a fresh clone.

The interactive Mosaic report builder reads the source notes from this directory by default. The archived configs describe the historical run. The broader DBMol training code, checkpoint history and active-job handoff are handled separately.
