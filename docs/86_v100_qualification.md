# V100 qualification for the frozen epoch-36 mapping

The user requested the checks on a 32 GB V100. This independent source146
campaign authenticates the original source142 scientific release, epoch-36 VAE,
trained mapping and calibration. It pins all three original starts and saved
resume states (case001 update1899, case002/003 update20). It never writes to
or adopts state into the original campaign or VAE continuation.

The allocated GPU stage requires a V100 with at least30 GiB, with a declared
4 GiB reserved-memory headroom. For each case it runs the original native oracle,
relaxed and learned start derivative gates, production bindings and a fresh
learned derivative/production binding at the saved resume point. Original
thresholds remain unchanged. The original A100 certificate is reported as a
cross-device comparison; it is not treated as an exact V100 gate or a migration
waiver. The original hard-score cache is not reused. Full frozen-state checks,
peak memory, device metadata, and two-warmup/five-repeat timings are recorded.
Timing uses one fixed0.0025 candidate with no step-size search; it is not an
optimization trajectory or a claim of molecular improvement. Results cannot
alone authorize migration of live production state or score caches.

Source146 attempt001 archive SHA256
`68b10f1cc0702ef8df6b34ce5956177a45d51bbcc0de78d570a379a7ce3eb64c`,
manifest SHA256
`f288580e454e7d5fbc91fe76260e7f016f15908624f8bb15fe2a0ae63d92399a`,
plan SHA256
`e1355e4b638c6f6bda2cdc6ba0e62cde9504dfeb5bf251bcaf10ec43b26a5afa`.
The independent release was staged and authenticated. CPU readiness job
**29852731** was submitted; a successful allocated zero-skip test proof is
required before the V100 GPU job can be submitted. Submission and queueing are
not native, derivative, memory, timing, or migration passes.

The toolchain is `scripts/embedding_v100_campaign.py`,
`scripts/hpc/embedding_v100_qualification.py`,
`tests/test_embedding_v100_qualification.py`, and
`configs/embedding_v100_qualification.json`. All numerical checks run on
allocated DTU HPC resources; the local connector is used for transport.

## Allocated readiness passed; V100 started, 30 October 2026 09:42 CET

CPU job **29852731 DONE**: all **38 fixtures passed**, zero failures, errors or
skips. GPU job **29852732 RUN** in `gpuv100` using the explicit `gpu32gb`
selection. Its own CPU-result and JUnit hashes are authenticated at launch.
Initial `hardware.json` records **Tesla V100-SXM2-32GB**, CUDA12.1, PyTorch
2.5.1 and compiled `sm_70`; this confirms the selected hardware and runtime
architecture, not a passed native, derivative, memory-headroom or timing test.
The setup/gate/result files were not yet present in this early snapshot.

The original case002 A100 job29851407 was PEND and independent VAE job29849760
was RUN in the same snapshot. No running job, VAE setting, learned mapping,
scientific threshold, production state or hard-score cache was changed.
The exact launch and hardware snapshot is in
the timestamped `outputs/hpc-results/v100-early-*.json` receipt.

At09:44 CET the models were loaded and case001's saved update1899 passed the
unchanged detached eligibility check. Its native oracle gate was in progress
at `central_fd_observations` and had not passed yet. GPU job29852732 remained
RUN. [Progress receipt](../outputs/hpc-results/v100-progress-1793349843967016700.json).

## Qualification passed, collected 30 October 2026 11:24 CET

GPU29852732 completed successfully on V100-SXM2-32GB after49m47s of worker
execution. All three native oracle gates, six original-start derivative gates,
three saved-resume learned derivative gates and their fresh same-device
production bindings passed. Frozen model/context checks passed. Peak reserved
GPU memory was26.36 GiB, leaving5.37 GiB by the declared peak-reservation
calculation, above the required4 GiB. Single fixed-candidate means were12.87,
13.38 and13.38 seconds respectively; these are short timing samples, not a
complete multi-trial optimization trajectory or convergence result.

Original A100 baseline scores and gradient digests were not bitwise equal to
the V100 values. Fresh same-device gates passed, but A100 bindings/hard-score
caches were not migrated and a safe continuation handoff remains separate work.
The original A100 continuation was still queued and the VAE continuation was
running at11:24. No original job, source, scientific settings or states changed.
Thirty-three terminal assets were transferred and verified against remote
SHA256/size. [Terminal proof and results](../outputs/hpc-results/source146-embedding-v100-qualification-attempt001/terminal-1793355904505738800/summary.json).
