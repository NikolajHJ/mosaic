# Historical HPC submission receipts

These scripts preserve the exact source manifests, allocations, dependencies,
configuration/context hashes and unique run paths actually submitted. They are
execution receipts, not reusable launch commands: their directory creation
intentionally refuses reuse, and rerunning them can submit duplicate work.

Reusable launchers are in the parent directory. Numerical outcomes and retained
failures are documented in `docs/experiment_ledger.md`. Large scientific inputs,
downloaded results, model weights and evidence archives stay outside Git.
