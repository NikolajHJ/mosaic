# Documentation

For current work start with [workflow commands](workflows.md), the
[package map](../src/latentmol/README.md), and the
[experiment index](../experiments/README.md).

| Reference | Purpose |
| --- | --- |
| [VAE training timeline](vae-training-timeline.html) | Trace the September–November training lineage, datasets, losses, and pilot decisions |
| [DBMol retirement into Mosaic](mosaic-retirement-plan.md) | Proposed repository handoff after the current attention leg, checkpoint parity and archival plan |
| [Molecular graph VAE research handoff](vae-research-handoff.md) | Agent guide to architecture, training history, unsuccessful experiments, evaluation, design and checkpoint recovery |
| [DBMol and Mosaic project review, 18 November 2026](project-review-2026-11-18.md) | Branch coverage, strongest results and the next comparison for the research agents |
| [Exploratory research](EXPLORATORY_RESEARCH_POLICY.md) | Small early-signal runs and proportionate checks |
| [Gradient checks](GRADIENT_VALIDATION_POLICY.md) | Smoke versus detailed numerical evidence |
| [HPC connection](HPC_CONNECTION.md) | Current workstation transport |
| [DBMol Boltz loss](77_dbmol_boltz_loss_implementation.md) | Objective reductions, gradient flow and known limits |
| [Graph/model plugin contract](23_frozen_model_plugin_contract.md) | Intended interface and mapping semantics; dated campaign gates are scoped |
| [Glossary](00_glossary.md) | Terminology |

Numbered `00_`–`89_` documents are dated study/protocol records. Their release/job
claims describe the stated observation, not a live queue. In particular the old
[code reference](06_code_reference.md) and [pipeline narrative](04_latent_pipeline.md)
contain historical implementation states; they are not the current API index.

Use the [experiment ledger](05_experiment_ledger.md) to investigate chronology,
not as required reading before a small change. Earlier README/agent snapshots
are under [history/](history/). Keep receipts, old paths and failed outcomes intact.
