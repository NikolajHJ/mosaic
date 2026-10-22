# Working-source migration and compatibility map

22 October 2026. Canonical implementations now live with experiments or documented shared capabilities. This table records the initial migration. The subsequent [nested layout map](folder-layout.json) moves flat script/test adapters under [scripts/compat/](../scripts/compat/README.md) and shared tests under named test subfolders. Historical receipt directories and external releases are not relocated.

All compatibility names are owned by the canonical destination below. Former flat script/test names are emitted only inside new authenticated release archives for existing launchers; they are absent from the working roots. Default pytest discovery uses canonical nested tests directly. Other example/HPC compatibility files remain at their documented locations. The original JSON map and hashes are preserved as migration history; the additive nested map records current adapter/test locations. No adapter is permission to add new flat implementations.

| Legacy path | Canonical owner/file | Compatibility behavior |
| --- | --- | --- |
| `examples/brd4_25_atoms.py` | [scripts/campaign25/brd4_25_atoms.py](../scripts/campaign25/brd4_25_atoms.py) | forwarding entry point |
| `examples/brd4_boltz_affinity.py` | [experiments/brd4/boltz-affinity-10000/scripts/brd4_boltz_affinity.py](brd4/boltz-affinity-10000/scripts/brd4_boltz_affinity.py) | forwarding entry point |
| `examples/brd4_chemistry.py` | [scripts/chemistry/brd4_chemistry.py](../scripts/chemistry/brd4_chemistry.py) | forwarding entry point |
| `examples/configs/brd4_25_boltz2_10000.json` | [experiments/brd4/three-model-25/configs/brd4_25_boltz2_10000.json](brd4/three-model-25/configs/brd4_25_boltz2_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_25_boltz2_chemistry_10000.json` | [experiments/brd4/three-model-25/configs/brd4_25_boltz2_chemistry_10000.json](brd4/three-model-25/configs/brd4_25_boltz2_chemistry_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_25_protenix_10000.json` | [experiments/brd4/three-model-25/configs/brd4_25_protenix_10000.json](brd4/three-model-25/configs/brd4_25_protenix_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_5000.json` | [experiments/brd4/contact-5000/configs/brd4_5000.json](brd4/contact-5000/configs/brd4_5000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_boltz_affinity_mix50_10000.json` | [experiments/brd4/boltz-affinity-10000/configs/brd4_boltz_affinity_mix50_10000.json](brd4/boltz-affinity-10000/configs/brd4_boltz_affinity_mix50_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_boltz_affinity_random_10000.json` | [experiments/brd4/boltz-affinity-10000/configs/brd4_boltz_affinity_random_10000.json](brd4/boltz-affinity-10000/configs/brd4_boltz_affinity_random_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_canonical_diffuse_5000.json` | [experiments/brd4/canonical-diffuse-5000/configs/brd4_canonical_diffuse_5000.json](brd4/canonical-diffuse-5000/configs/brd4_canonical_diffuse_5000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_chemistry_5000.json` | [experiments/brd4/chemistry-5000/configs/brd4_chemistry_5000.json](brd4/chemistry-5000/configs/brd4_chemistry_5000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_chemistry_ramp_mix50_10000.json` | [experiments/brd4/chemistry-ramp-10000/configs/brd4_chemistry_ramp_mix50_10000.json](brd4/chemistry-ramp-10000/configs/brd4_chemistry_ramp_mix50_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_chemistry_ramp_random_10000.json` | [experiments/brd4/chemistry-ramp-10000/configs/brd4_chemistry_ramp_random_10000.json](brd4/chemistry-ramp-10000/configs/brd4_chemistry_ramp_random_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_interim_report_scope.json` | [experiments/reporting/brd4-comparison/configs/brd4_interim_report_scope.json](reporting/brd4-comparison/configs/brd4_interim_report_scope.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_boltz2_10000.json` | [experiments/brd4-jq1/backtracking-25/configs/brd4_jq1_25_boltz2_10000.json](brd4-jq1/backtracking-25/configs/brd4_jq1_25_boltz2_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_boltz2_chemistry_10000.json` | [experiments/brd4-jq1/backtracking-25/configs/brd4_jq1_25_boltz2_chemistry_10000.json](brd4-jq1/backtracking-25/configs/brd4_jq1_25_boltz2_chemistry_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_fixed_boltz2_10000.json` | [experiments/brd4-jq1/fixed-steps/configs/brd4_jq1_25_fixed_boltz2_10000.json](brd4-jq1/fixed-steps/configs/brd4_jq1_25_fixed_boltz2_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_fixed_boltz2_chemistry_10000.json` | [experiments/brd4-jq1/fixed-steps/configs/brd4_jq1_25_fixed_boltz2_chemistry_10000.json](brd4-jq1/fixed-steps/configs/brd4_jq1_25_fixed_boltz2_chemistry_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_fixed_lrboost_boltz2_10000.json` | [experiments/brd4-jq1/fixed-lrboost/configs/brd4_jq1_25_fixed_lrboost_boltz2_10000.json](brd4-jq1/fixed-lrboost/configs/brd4_jq1_25_fixed_lrboost_boltz2_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_fixed_lrboost_boltz2_chemistry_10000.json` | [experiments/brd4-jq1/fixed-lrboost/configs/brd4_jq1_25_fixed_lrboost_boltz2_chemistry_10000.json](brd4-jq1/fixed-lrboost/configs/brd4_jq1_25_fixed_lrboost_boltz2_chemistry_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_fixed_lrboost_protenix_10000.json` | [experiments/brd4-jq1/fixed-lrboost/configs/brd4_jq1_25_fixed_lrboost_protenix_10000.json](brd4-jq1/fixed-lrboost/configs/brd4_jq1_25_fixed_lrboost_protenix_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_fixed_lrreset_boltz2_10000.json` | [experiments/brd4-jq1/fixed-lrreset/configs/brd4_jq1_25_fixed_lrreset_boltz2_10000.json](brd4-jq1/fixed-lrreset/configs/brd4_jq1_25_fixed_lrreset_boltz2_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_fixed_lrreset_boltz2_chemistry_10000.json` | [experiments/brd4-jq1/fixed-lrreset/configs/brd4_jq1_25_fixed_lrreset_boltz2_chemistry_10000.json](brd4-jq1/fixed-lrreset/configs/brd4_jq1_25_fixed_lrreset_boltz2_chemistry_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_fixed_lrreset_protenix_10000.json` | [experiments/brd4-jq1/fixed-lrreset/configs/brd4_jq1_25_fixed_lrreset_protenix_10000.json](brd4-jq1/fixed-lrreset/configs/brd4_jq1_25_fixed_lrreset_protenix_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_fixed_protenix_10000.json` | [experiments/brd4-jq1/fixed-steps/configs/brd4_jq1_25_fixed_protenix_10000.json](brd4-jq1/fixed-steps/configs/brd4_jq1_25_fixed_protenix_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_protenix_10000.json` | [experiments/brd4-jq1/backtracking-25/configs/brd4_jq1_25_protenix_10000.json](brd4-jq1/backtracking-25/configs/brd4_jq1_25_protenix_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_jq1_25_protenix_lr03_10000.json` | [experiments/brd4-jq1/protenix-lr03/configs/brd4_jq1_25_protenix_lr03_10000.json](brd4-jq1/protenix-lr03/configs/brd4_jq1_25_protenix_lr03_10000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_noisy_uniform_5000.json` | [experiments/brd4/noisy-uniform-5000/configs/brd4_noisy_uniform_5000.json](brd4/noisy-uniform-5000/configs/brd4_noisy_uniform_5000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_pilot.json` | [experiments/brd4/contact-5000/configs/brd4_pilot.json](brd4/contact-5000/configs/brd4_pilot.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/brd4_uniform_5000.json` | [experiments/brd4/uniform-5000/configs/brd4_uniform_5000.json](brd4/uniform-5000/configs/brd4_uniform_5000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/carbon_chain_100.json` | [experiments/graph-controls/carbon-chain-100/configs/carbon_chain_100.json](graph-controls/carbon-chain-100/configs/carbon_chain_100.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/carbon_chain_500.json` | [experiments/graph-controls/carbon-chain-500/configs/carbon_chain_500.json](graph-controls/carbon-chain-500/configs/carbon_chain_500.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/carbon_chain_500_mix50.json` | [experiments/graph-controls/carbon-chain-500/configs/carbon_chain_500_mix50.json](graph-controls/carbon-chain-500/configs/carbon_chain_500_mix50.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/known_ligand_1000.json` | [experiments/graph-controls/known-ligand-1000/configs/known_ligand_1000.json](graph-controls/known-ligand-1000/configs/known_ligand_1000.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/known_ligand_pilot.json` | [experiments/graph-controls/known-ligand-1000/configs/known_ligand_pilot.json](graph-controls/known-ligand-1000/configs/known_ligand_pilot.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/protenix_relaxed_gradients.json` | [experiments/graph-validation/relaxed-gradients/configs/protenix_relaxed_gradients.json](graph-validation/relaxed-gradients/configs/protenix_relaxed_gradients.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/configs/small_molecule_stage0.json` | [experiments/graph-validation/stage0/configs/small_molecule_stage0.json](graph-validation/stage0/configs/small_molecule_stage0.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `examples/continue_brd4.py` | [experiments/brd4/best-primal-continuation/scripts/continue_brd4.py](brd4/best-primal-continuation/scripts/continue_brd4.py) | forwarding entry point |
| `examples/hpc_protenix_tiny.py` | [experiments/native-inference/protenix-tiny/scripts/hpc_protenix_tiny.py](native-inference/protenix-tiny/scripts/hpc_protenix_tiny.py) | forwarding entry point |
| `examples/protenix_relaxed_gradients.py` | [scripts/graph/protenix_relaxed_gradients.py](../scripts/graph/protenix_relaxed_gradients.py) | forwarding entry point |
| `examples/score_protenix_hard_controls.py` | [scripts/graph/score_protenix_hard_controls.py](../scripts/graph/score_protenix_hard_controls.py) | forwarding entry point |
| `examples/small_molecule_stage0.py` | [experiments/graph-validation/stage0/scripts/small_molecule_stage0.py](graph-validation/stage0/scripts/small_molecule_stage0.py) | forwarding entry point |
| `scripts/analyze_boltz_balance.py` | [experiments/brd4/boltz-gradient-diagnostic/scripts/analyze_boltz_balance.py](brd4/boltz-gradient-diagnostic/scripts/analyze_boltz_balance.py) | forwarding entry point |
| `scripts/analyze_boltz_gradient_diagnostic.py` | [experiments/brd4/boltz-gradient-diagnostic/scripts/analyze_boltz_gradient_diagnostic.py](brd4/boltz-gradient-diagnostic/scripts/analyze_boltz_gradient_diagnostic.py) | forwarding entry point |
| `scripts/assemble_milestone_report.py` | [experiments/reporting/milestones/scripts/assemble_milestone_report.py](reporting/milestones/scripts/assemble_milestone_report.py) | forwarding entry point |
| `scripts/assess_boltz_arm.py` | [experiments/brd4/boltz-affinity-10000/scripts/assess_boltz_arm.py](brd4/boltz-affinity-10000/scripts/assess_boltz_arm.py) | forwarding entry point |
| `scripts/assess_boltz_cpu.py` | [experiments/brd4/boltz-affinity-10000/scripts/assess_boltz_cpu.py](brd4/boltz-affinity-10000/scripts/assess_boltz_cpu.py) | forwarding entry point |
| `scripts/assess_boltz_native.py` | [experiments/brd4/boltz-affinity-10000/scripts/assess_boltz_native.py](brd4/boltz-affinity-10000/scripts/assess_boltz_native.py) | forwarding entry point |
| `scripts/assess_known_ligand_pilot.py` | [experiments/graph-controls/known-ligand-1000/scripts/assess_known_ligand_pilot.py](graph-controls/known-ligand-1000/scripts/assess_known_ligand_pilot.py) | forwarding entry point |
| `scripts/assess_uniform_control.py` | [scripts/graph/assess_uniform_control.py](../scripts/graph/assess_uniform_control.py) | forwarding entry point |
| `scripts/augment_protenix_hard_controls.py` | [scripts/graph/augment_protenix_hard_controls.py](../scripts/graph/augment_protenix_hard_controls.py) | forwarding entry point |
| `scripts/benchmark25_repair.py` | [experiments/brd4/three-model-25/scripts/benchmark25_repair.py](brd4/three-model-25/scripts/benchmark25_repair.py) | forwarding entry point |
| `scripts/campaign25_contract.py` | [scripts/campaign25/campaign25_contract.py](../scripts/campaign25/campaign25_contract.py) | forwarding entry point |
| `scripts/campaign25_ops.py` | [experiments/brd4/three-model-25/scripts/campaign25_ops.py](brd4/three-model-25/scripts/campaign25_ops.py) | forwarding entry point |
| `scripts/campaign25_pocket_ops.py` | [scripts/campaign25/campaign25_pocket_ops.py](../scripts/campaign25/campaign25_pocket_ops.py) | forwarding entry point |
| `scripts/campaign25_release.py` | [experiments/brd4/three-model-25/scripts/campaign25_release.py](brd4/three-model-25/scripts/campaign25_release.py) | forwarding entry point |
| `scripts/capture_jq1_completion.py` | [experiments/brd4-jq1/backtracking-25/scripts/capture_jq1_completion.py](brd4-jq1/backtracking-25/scripts/capture_jq1_completion.py) | forwarding entry point |
| `scripts/capture_jq1_fixed_completion.py` | [experiments/brd4-jq1/fixed-steps/scripts/capture_jq1_fixed_completion.py](brd4-jq1/fixed-steps/scripts/capture_jq1_fixed_completion.py) | forwarding entry point |
| `scripts/capture_jq1_lrboost_completion.py` | [experiments/brd4-jq1/fixed-lrboost/scripts/capture_jq1_lrboost_completion.py](brd4-jq1/fixed-lrboost/scripts/capture_jq1_lrboost_completion.py) | forwarding entry point |
| `scripts/check_boltz_cpu_proof.py` | [scripts/boltz/check_boltz_cpu_proof.py](../scripts/boltz/check_boltz_cpu_proof.py) | forwarding entry point |
| `scripts/check_boltz_diagnostic_cpu.py` | [experiments/brd4/boltz-gradient-diagnostic/scripts/check_boltz_diagnostic_cpu.py](brd4/boltz-gradient-diagnostic/scripts/check_boltz_diagnostic_cpu.py) | forwarding entry point |
| `scripts/chemistry_compatibility_probe.py` | [experiments/brd4/chemistry-compatibility/scripts/chemistry_compatibility_probe.py](brd4/chemistry-compatibility/scripts/chemistry_compatibility_probe.py) | forwarding entry point |
| `scripts/chemistry_report_template.html` | [scripts/chemistry/chemistry_report_template.html](../scripts/chemistry/chemistry_report_template.html) | byte-identical declarative compatibility copy; canonical owner is new path |
| `scripts/collect_protenix_evidence.py` | [scripts/evidence/collect_protenix_evidence.py](../scripts/evidence/collect_protenix_evidence.py) | forwarding entry point |
| `scripts/create_chemistry_oracles.py` | [scripts/chemistry/create_chemistry_oracles.py](../scripts/chemistry/create_chemistry_oracles.py) | forwarding entry point |
| `scripts/derive_brd4_jq1_pocket.py` | [experiments/brd4-jq1/backtracking-25/scripts/derive_brd4_jq1_pocket.py](brd4-jq1/backtracking-25/scripts/derive_brd4_jq1_pocket.py) | forwarding entry point |
| `scripts/diagnose_boltz_random_gradient.py` | [experiments/brd4/boltz-gradient-diagnostic/scripts/diagnose_boltz_random_gradient.py](brd4/boltz-gradient-diagnostic/scripts/diagnose_boltz_random_gradient.py) | forwarding entry point |
| `scripts/download_hpc_when_complete.py` | [scripts/evidence/download_hpc_when_complete.py](../scripts/evidence/download_hpc_when_complete.py) | forwarding entry point |
| `scripts/fixed25_lrboost_ops.py` | [experiments/brd4-jq1/fixed-lrboost/scripts/fixed25_lrboost_ops.py](brd4-jq1/fixed-lrboost/scripts/fixed25_lrboost_ops.py) | forwarding entry point |
| `scripts/fixed25_lrboost_storage.py` | [experiments/brd4-jq1/fixed-lrboost/scripts/fixed25_lrboost_storage.py](brd4-jq1/fixed-lrboost/scripts/fixed25_lrboost_storage.py) | forwarding entry point |
| `scripts/fixed25_lrreset_ops.py` | [experiments/brd4-jq1/fixed-lrreset/scripts/fixed25_lrreset_ops.py](brd4-jq1/fixed-lrreset/scripts/fixed25_lrreset_ops.py) | forwarding entry point |
| `scripts/fixed25_lrreset_storage.py` | [experiments/brd4-jq1/fixed-lrreset/scripts/fixed25_lrreset_storage.py](brd4-jq1/fixed-lrreset/scripts/fixed25_lrreset_storage.py) | forwarding entry point |
| `scripts/fixed25_ops.py` | [experiments/brd4-jq1/fixed-steps/scripts/fixed25_ops.py](brd4-jq1/fixed-steps/scripts/fixed25_ops.py) | forwarding entry point |
| `scripts/fixed25_recover_ops.py` | [experiments/brd4-jq1/fixed-steps/scripts/fixed25_recover_ops.py](brd4-jq1/fixed-steps/scripts/fixed25_recover_ops.py) | forwarding entry point |
| `scripts/fixed25_recover_storage.py` | [scripts/campaign25/fixed25_recover_storage.py](../scripts/campaign25/fixed25_recover_storage.py) | forwarding entry point |
| `scripts/fixed25_split_ops.py` | [experiments/brd4-jq1/fixed-steps/scripts/fixed25_split_ops.py](brd4-jq1/fixed-steps/scripts/fixed25_split_ops.py) | forwarding entry point |
| `scripts/fixed25_split_storage.py` | [scripts/campaign25/fixed25_split_storage.py](../scripts/campaign25/fixed25_split_storage.py) | forwarding entry point |
| `scripts/fixed25_storage.py` | [scripts/campaign25/fixed25_storage.py](../scripts/campaign25/fixed25_storage.py) | forwarding entry point |
| `scripts/hpc/augment_protenix_controls.lsf` | [scripts/graph/augment_protenix_controls.lsf](../scripts/graph/augment_protenix_controls.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/boltz_node_tmpdir.sh` | [scripts/hpc/boltz_node_tmpdir.sh](../scripts/hpc/boltz_node_tmpdir.sh) | unchanged shared canonical file at its existing path |
| `scripts/hpc/brd4_boltz_cpu.lsf` | [experiments/brd4/boltz-affinity-10000/hpc/brd4_boltz_cpu.lsf](brd4/boltz-affinity-10000/hpc/brd4_boltz_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_diagnostic_certificate_cpu.lsf` | [experiments/brd4/boltz-gradient-diagnostic/hpc/brd4_boltz_diagnostic_certificate_cpu.lsf](brd4/boltz-gradient-diagnostic/hpc/brd4_boltz_diagnostic_certificate_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_diagnostic_cpu.lsf` | [experiments/brd4/boltz-gradient-diagnostic/hpc/brd4_boltz_diagnostic_cpu.lsf](brd4/boltz-gradient-diagnostic/hpc/brd4_boltz_diagnostic_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_gradient_analysis_cpu.lsf` | [experiments/brd4/boltz-gradient-diagnostic/hpc/brd4_boltz_gradient_analysis_cpu.lsf](brd4/boltz-gradient-diagnostic/hpc/brd4_boltz_gradient_analysis_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_independent_cpu.lsf` | [experiments/brd4/boltz-affinity-10000/hpc/brd4_boltz_independent_cpu.lsf](brd4/boltz-affinity-10000/hpc/brd4_boltz_independent_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_launcher_cpu.lsf` | [experiments/brd4/boltz-affinity-10000/hpc/brd4_boltz_launcher_cpu.lsf](brd4/boltz-affinity-10000/hpc/brd4_boltz_launcher_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_random_diagnostic.lsf` | [experiments/brd4/boltz-gradient-diagnostic/hpc/brd4_boltz_random_diagnostic.lsf](brd4/boltz-gradient-diagnostic/hpc/brd4_boltz_random_diagnostic.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_random_diagnostic_retry.lsf` | [experiments/brd4/boltz-gradient-diagnostic/hpc/brd4_boltz_random_diagnostic_retry.lsf](brd4/boltz-gradient-diagnostic/hpc/brd4_boltz_random_diagnostic_retry.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_ready.lsf` | [experiments/brd4/boltz-affinity-10000/hpc/brd4_boltz_ready.lsf](brd4/boltz-affinity-10000/hpc/brd4_boltz_ready.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_report.lsf` | [experiments/brd4/boltz-affinity-10000/hpc/brd4_boltz_report.lsf](brd4/boltz-affinity-10000/hpc/brd4_boltz_report.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_run.lsf` | [experiments/brd4/boltz-affinity-10000/hpc/brd4_boltz_run.lsf](brd4/boltz-affinity-10000/hpc/brd4_boltz_run.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_boltz_submit_queue.lsf` | [experiments/brd4/boltz-affinity-10000/hpc/brd4_boltz_submit_queue.lsf](brd4/boltz-affinity-10000/hpc/brd4_boltz_submit_queue.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_chemistry.lsf` | [experiments/brd4/chemistry-5000/hpc/brd4_chemistry.lsf](brd4/chemistry-5000/hpc/brd4_chemistry.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_chemistry_preview.lsf` | [experiments/brd4/chemistry-5000/hpc/brd4_chemistry_preview.lsf](brd4/chemistry-5000/hpc/brd4_chemistry_preview.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_chemistry_ramp_report.lsf` | [experiments/brd4/chemistry-ramp-10000/hpc/brd4_chemistry_ramp_report.lsf](brd4/chemistry-ramp-10000/hpc/brd4_chemistry_ramp_report.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_chemistry_readiness.lsf` | [experiments/brd4/chemistry-5000/hpc/brd4_chemistry_readiness.lsf](brd4/chemistry-5000/hpc/brd4_chemistry_readiness.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_chemistry_repackage.lsf` | [experiments/brd4/chemistry-5000/hpc/brd4_chemistry_repackage.lsf](brd4/chemistry-5000/hpc/brd4_chemistry_repackage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_chemistry_report.lsf` | [experiments/brd4/chemistry-5000/hpc/brd4_chemistry_report.lsf](brd4/chemistry-5000/hpc/brd4_chemistry_report.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_combined_report.lsf` | [experiments/reporting/brd4-comparison/hpc/brd4_combined_report.lsf](reporting/brd4-comparison/hpc/brd4_combined_report.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_comparison_readiness.lsf` | [experiments/reporting/brd4-comparison/hpc/brd4_comparison_readiness.lsf](reporting/brd4-comparison/hpc/brd4_comparison_readiness.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_contact.lsf` | [scripts/graph/brd4_contact.lsf](../scripts/graph/brd4_contact.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_continuation_readiness.lsf` | [experiments/brd4/best-primal-continuation/hpc/brd4_continuation_readiness.lsf](brd4/best-primal-continuation/hpc/brd4_continuation_readiness.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_continuation_report.lsf` | [experiments/brd4/best-primal-continuation/hpc/brd4_continuation_report.lsf](brd4/best-primal-continuation/hpc/brd4_continuation_report.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_continue.lsf` | [experiments/brd4/best-primal-continuation/hpc/brd4_continue.lsf](brd4/best-primal-continuation/hpc/brd4_continue.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_interim_evidence.lsf` | [experiments/reporting/brd4-comparison/hpc/brd4_interim_evidence.lsf](reporting/brd4-comparison/hpc/brd4_interim_evidence.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_interim_report.lsf` | [experiments/reporting/brd4-comparison/hpc/brd4_interim_report.lsf](reporting/brd4-comparison/hpc/brd4_interim_report.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_parallel_readiness.lsf` | [scripts/graph/brd4_parallel_readiness.lsf](../scripts/graph/brd4_parallel_readiness.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_parallel_report.lsf` | [scripts/graph/brd4_parallel_report.lsf](../scripts/graph/brd4_parallel_report.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_readiness.lsf` | [scripts/graph/brd4_readiness.lsf](../scripts/graph/brd4_readiness.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/brd4_uniform_readiness.lsf` | [scripts/graph/brd4_uniform_readiness.lsf](../scripts/graph/brd4_uniform_readiness.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/campaign25_cpu.lsf` | [experiments/brd4/three-model-25/hpc/campaign25_cpu.lsf](brd4/three-model-25/hpc/campaign25_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/campaign25_pocket_cpu.lsf` | [experiments/brd4-jq1/backtracking-25/hpc/campaign25_pocket_cpu.lsf](brd4-jq1/backtracking-25/hpc/campaign25_pocket_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/campaign25_pocket_stage.lsf` | [experiments/brd4-jq1/backtracking-25/hpc/campaign25_pocket_stage.lsf](brd4-jq1/backtracking-25/hpc/campaign25_pocket_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/campaign25_release_cpu.lsf` | [experiments/brd4/three-model-25/hpc/campaign25_release_cpu.lsf](brd4/three-model-25/hpc/campaign25_release_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/campaign25_release_stage.lsf` | [experiments/brd4/three-model-25/hpc/campaign25_release_stage.lsf](brd4/three-model-25/hpc/campaign25_release_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/campaign25_repair_timing.lsf` | [experiments/brd4/three-model-25/hpc/campaign25_repair_timing.lsf](brd4/three-model-25/hpc/campaign25_repair_timing.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/campaign25_stage.lsf` | [experiments/brd4/three-model-25/hpc/campaign25_stage.lsf](brd4/three-model-25/hpc/campaign25_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/carbon_chain_100.lsf` | [experiments/graph-controls/carbon-chain-100/hpc/carbon_chain_100.lsf](graph-controls/carbon-chain-100/hpc/carbon_chain_100.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/carbon_chain_500.lsf` | [experiments/graph-controls/carbon-chain-500/hpc/carbon_chain_500.lsf](graph-controls/carbon-chain-500/hpc/carbon_chain_500.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/carbon_chain_500_mix50.lsf` | [experiments/graph-controls/carbon-chain-500/hpc/carbon_chain_500_mix50.lsf](graph-controls/carbon-chain-500/hpc/carbon_chain_500_mix50.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/chemistry_compatibility_probe.lsf` | [experiments/brd4/chemistry-compatibility/hpc/chemistry_compatibility_probe.lsf](brd4/chemistry-compatibility/hpc/chemistry_compatibility_probe.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_cpu.lsf` | [experiments/brd4-jq1/fixed-steps/hpc/fixed25_cpu.lsf](brd4-jq1/fixed-steps/hpc/fixed25_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_lrboost_cpu.lsf` | [experiments/brd4-jq1/fixed-lrboost/hpc/fixed25_lrboost_cpu.lsf](brd4-jq1/fixed-lrboost/hpc/fixed25_lrboost_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_lrboost_stage.lsf` | [experiments/brd4-jq1/fixed-lrboost/hpc/fixed25_lrboost_stage.lsf](brd4-jq1/fixed-lrboost/hpc/fixed25_lrboost_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_lrreset_cpu.lsf` | [experiments/brd4-jq1/fixed-lrreset/hpc/fixed25_lrreset_cpu.lsf](brd4-jq1/fixed-lrreset/hpc/fixed25_lrreset_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_lrreset_stage.lsf` | [experiments/brd4-jq1/fixed-lrreset/hpc/fixed25_lrreset_stage.lsf](brd4-jq1/fixed-lrreset/hpc/fixed25_lrreset_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_recover_after_cpu.lsf` | [experiments/brd4-jq1/fixed-steps/hpc/fixed25_recover_after_cpu.lsf](brd4-jq1/fixed-steps/hpc/fixed25_recover_after_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_recover_cpu.lsf` | [experiments/brd4-jq1/fixed-steps/hpc/fixed25_recover_cpu.lsf](brd4-jq1/fixed-steps/hpc/fixed25_recover_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_recover_observe_pro.lsf` | [experiments/brd4-jq1/fixed-steps/hpc/fixed25_recover_observe_pro.lsf](brd4-jq1/fixed-steps/hpc/fixed25_recover_observe_pro.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_recover_stage.lsf` | [experiments/brd4-jq1/fixed-steps/hpc/fixed25_recover_stage.lsf](brd4-jq1/fixed-steps/hpc/fixed25_recover_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_split_after_cpu.lsf` | [experiments/brd4-jq1/fixed-steps/hpc/fixed25_split_after_cpu.lsf](brd4-jq1/fixed-steps/hpc/fixed25_split_after_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_split_cpu.lsf` | [experiments/brd4-jq1/fixed-steps/hpc/fixed25_split_cpu.lsf](brd4-jq1/fixed-steps/hpc/fixed25_split_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_split_stage.lsf` | [experiments/brd4-jq1/fixed-steps/hpc/fixed25_split_stage.lsf](brd4-jq1/fixed-steps/hpc/fixed25_split_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/fixed25_stage.lsf` | [experiments/brd4-jq1/fixed-steps/hpc/fixed25_stage.lsf](brd4-jq1/fixed-steps/hpc/fixed25_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/jq1_25_interim_stage.lsf` | [experiments/brd4-jq1/backtracking-25/hpc/jq1_25_interim_stage.lsf](brd4-jq1/backtracking-25/hpc/jq1_25_interim_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/jq1_25_report_stage.lsf` | [experiments/brd4-jq1/backtracking-25/hpc/jq1_25_report_stage.lsf](brd4-jq1/backtracking-25/hpc/jq1_25_report_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/jq1_fixed_report_stage.lsf` | [experiments/brd4-jq1/fixed-steps/hpc/jq1_fixed_report_stage.lsf](brd4-jq1/fixed-steps/hpc/jq1_fixed_report_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/jq1_lrboost_analysis_stage.lsf` | [experiments/brd4-jq1/fixed-lrboost/hpc/jq1_lrboost_analysis_stage.lsf](brd4-jq1/fixed-lrboost/hpc/jq1_lrboost_analysis_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/known_ligand_1000.lsf` | [experiments/graph-controls/known-ligand-1000/hpc/known_ligand_1000.lsf](graph-controls/known-ligand-1000/hpc/known_ligand_1000.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/known_ligand_assess.lsf` | [experiments/graph-controls/known-ligand-1000/hpc/known_ligand_assess.lsf](graph-controls/known-ligand-1000/hpc/known_ligand_assess.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/known_ligand_collect.lsf` | [experiments/graph-controls/known-ligand-1000/hpc/known_ligand_collect.lsf](graph-controls/known-ligand-1000/hpc/known_ligand_collect.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/known_ligand_controls.lsf` | [experiments/graph-controls/known-ligand-1000/hpc/known_ligand_controls.lsf](graph-controls/known-ligand-1000/hpc/known_ligand_controls.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/known_ligand_readiness.lsf` | [experiments/graph-controls/known-ligand-1000/hpc/known_ligand_readiness.lsf](graph-controls/known-ligand-1000/hpc/known_ligand_readiness.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/lrreset_predecessors.json` | [experiments/brd4-jq1/fixed-lrreset/configs/lrreset_predecessors.json](brd4-jq1/fixed-lrreset/configs/lrreset_predecessors.json) | byte-identical declarative compatibility copy; canonical owner is new path |
| `scripts/hpc/protenix_lr03_cpu.lsf` | [experiments/brd4-jq1/protenix-lr03/hpc/protenix_lr03_cpu.lsf](brd4-jq1/protenix-lr03/hpc/protenix_lr03_cpu.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/protenix_lr03_stage.lsf` | [experiments/brd4-jq1/protenix-lr03/hpc/protenix_lr03_stage.lsf](brd4-jq1/protenix-lr03/hpc/protenix_lr03_stage.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/protenix_relaxed_gradients.lsf` | [experiments/graph-validation/relaxed-gradients/hpc/protenix_relaxed_gradients.lsf](graph-validation/relaxed-gradients/hpc/protenix_relaxed_gradients.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/protenix_tiny.lsf` | [experiments/native-inference/protenix-tiny/hpc/protenix_tiny.lsf](native-inference/protenix-tiny/hpc/protenix_tiny.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/recover_verified_500_archive.py` | [scripts/hpc/recover_verified_500_archive.py](../scripts/hpc/recover_verified_500_archive.py) | unchanged shared canonical file at its existing path |
| `scripts/hpc/render_protenix_trajectory.lsf` | [scripts/reporting/render_protenix_trajectory.lsf](../scripts/reporting/render_protenix_trajectory.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/small_molecule_stage0.lsf` | [experiments/graph-validation/stage0/hpc/small_molecule_stage0.lsf](graph-validation/stage0/hpc/small_molecule_stage0.lsf) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/stage0_quota_preflight.py` | [scripts/hpc/stage0_quota_preflight.py](../scripts/hpc/stage0_quota_preflight.py) | unchanged shared canonical file at its existing path |
| `scripts/hpc/stage_tiny_assets.py` | [experiments/native-inference/protenix-tiny/scripts/stage_tiny_assets.py](native-inference/protenix-tiny/scripts/stage_tiny_assets.py) | forwarding entry point |
| `scripts/hpc/submit_chemistry_native.sh` | [experiments/brd4/chemistry-5000/hpc/submit_chemistry_native.sh](brd4/chemistry-5000/hpc/submit_chemistry_native.sh) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/submit_chemistry_ramp.sh` | [experiments/brd4/chemistry-ramp-10000/hpc/submit_chemistry_ramp.sh](brd4/chemistry-ramp-10000/hpc/submit_chemistry_ramp.sh) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/submit_chemistry_readiness.sh` | [experiments/brd4/chemistry-5000/hpc/submit_chemistry_readiness.sh](brd4/chemistry-5000/hpc/submit_chemistry_readiness.sh) | forwarding shell entry point; BSUB directives retained |
| `scripts/hpc/zhome_quota_preflight.py` | [scripts/hpc/zhome_quota_preflight.py](../scripts/hpc/zhome_quota_preflight.py) | unchanged shared canonical file at its existing path |
| `scripts/jq1_fixed_report_contract.py` | [experiments/brd4-jq1/fixed-steps/scripts/jq1_fixed_report_contract.py](brd4-jq1/fixed-steps/scripts/jq1_fixed_report_contract.py) | forwarding entry point |
| `scripts/jq1_fixed_report_ops.py` | [experiments/brd4-jq1/fixed-steps/scripts/jq1_fixed_report_ops.py](brd4-jq1/fixed-steps/scripts/jq1_fixed_report_ops.py) | forwarding entry point |
| `scripts/jq1_lrboost_report_contract.py` | [experiments/brd4-jq1/fixed-lrboost/scripts/jq1_lrboost_report_contract.py](brd4-jq1/fixed-lrboost/scripts/jq1_lrboost_report_contract.py) | forwarding entry point |
| `scripts/jq1_lrboost_report_ops.py` | [experiments/brd4-jq1/fixed-lrboost/scripts/jq1_lrboost_report_ops.py](brd4-jq1/fixed-lrboost/scripts/jq1_lrboost_report_ops.py) | forwarding entry point |
| `scripts/jq1_report_contract.py` | [experiments/brd4-jq1/backtracking-25/scripts/jq1_report_contract.py](brd4-jq1/backtracking-25/scripts/jq1_report_contract.py) | forwarding entry point |
| `scripts/lrboost_endpoint_diagnostics.py` | [experiments/brd4-jq1/fixed-lrboost/scripts/lrboost_endpoint_diagnostics.py](brd4-jq1/fixed-lrboost/scripts/lrboost_endpoint_diagnostics.py) | forwarding entry point |
| `scripts/package_stage0_release.py` | [scripts/releases/package_stage0_release.py](../scripts/releases/package_stage0_release.py) | forwarding entry point |
| `scripts/prepare_brd4_continuation.py` | [experiments/brd4/best-primal-continuation/scripts/prepare_brd4_continuation.py](brd4/best-primal-continuation/scripts/prepare_brd4_continuation.py) | forwarding entry point |
| `scripts/prepare_brd4_fixture.py` | [scripts/graph/prepare_brd4_fixture.py](../scripts/graph/prepare_brd4_fixture.py) | forwarding entry point |
| `scripts/prepare_chemistry_reference.py` | [scripts/chemistry/prepare_chemistry_reference.py](../scripts/chemistry/prepare_chemistry_reference.py) | forwarding entry point |
| `scripts/prepare_known_ligand_fixture.py` | [scripts/graph/prepare_known_ligand_fixture.py](../scripts/graph/prepare_known_ligand_fixture.py) | forwarding entry point |
| `scripts/protenix_lr03_ops.py` | [experiments/brd4-jq1/protenix-lr03/scripts/protenix_lr03_ops.py](brd4-jq1/protenix-lr03/scripts/protenix_lr03_ops.py) | forwarding entry point |
| `scripts/protenix_lr03_storage.py` | [experiments/brd4-jq1/protenix-lr03/scripts/protenix_lr03_storage.py](brd4-jq1/protenix-lr03/scripts/protenix_lr03_storage.py) | forwarding entry point |
| `scripts/protenix_trajectory_depiction.py` | [scripts/reporting/protenix_trajectory_depiction.py](../scripts/reporting/protenix_trajectory_depiction.py) | forwarding entry point |
| `scripts/protenix_trajectory_svg.py` | [scripts/reporting/protenix_trajectory_svg.py](../scripts/reporting/protenix_trajectory_svg.py) | forwarding entry point |
| `scripts/publish_jq1_fixed_report.py` | [experiments/brd4-jq1/fixed-steps/scripts/publish_jq1_fixed_report.py](brd4-jq1/fixed-steps/scripts/publish_jq1_fixed_report.py) | forwarding entry point |
| `scripts/publish_jq1_lrboost_report.py` | [experiments/brd4-jq1/fixed-lrboost/scripts/publish_jq1_lrboost_report.py](brd4-jq1/fixed-lrboost/scripts/publish_jq1_lrboost_report.py) | forwarding entry point |
| `scripts/publish_jq1_report.py` | [experiments/brd4-jq1/backtracking-25/scripts/publish_jq1_report.py](brd4-jq1/backtracking-25/scripts/publish_jq1_report.py) | forwarding entry point |
| `scripts/render_boltz_affinity.py` | [experiments/brd4/boltz-affinity-10000/scripts/render_boltz_affinity.py](brd4/boltz-affinity-10000/scripts/render_boltz_affinity.py) | forwarding entry point |
| `scripts/render_chemistry_ramp.py` | [experiments/brd4/chemistry-ramp-10000/scripts/render_chemistry_ramp.py](brd4/chemistry-ramp-10000/scripts/render_chemistry_ramp.py) | forwarding entry point |
| `scripts/render_chemistry_trajectory.py` | [scripts/chemistry/render_chemistry_trajectory.py](../scripts/chemistry/render_chemistry_trajectory.py) | forwarding entry point |
| `scripts/render_jq1_25_interim.py` | [experiments/brd4-jq1/backtracking-25/scripts/render_jq1_25_interim.py](brd4-jq1/backtracking-25/scripts/render_jq1_25_interim.py) | forwarding entry point |
| `scripts/render_jq1_25_report.py` | [experiments/brd4-jq1/backtracking-25/scripts/render_jq1_25_report.py](brd4-jq1/backtracking-25/scripts/render_jq1_25_report.py) | forwarding entry point |
| `scripts/render_jq1_fixed_interim.py` | [experiments/brd4-jq1/fixed-steps/scripts/render_jq1_fixed_interim.py](brd4-jq1/fixed-steps/scripts/render_jq1_fixed_interim.py) | forwarding entry point |
| `scripts/render_jq1_fixed_report.py` | [experiments/brd4-jq1/fixed-steps/scripts/render_jq1_fixed_report.py](brd4-jq1/fixed-steps/scripts/render_jq1_fixed_report.py) | forwarding entry point |
| `scripts/render_jq1_lrboost_progress.py` | [experiments/brd4-jq1/fixed-lrboost/scripts/render_jq1_lrboost_progress.py](brd4-jq1/fixed-lrboost/scripts/render_jq1_lrboost_progress.py) | forwarding entry point |
| `scripts/render_jq1_lrboost_report.py` | [experiments/brd4-jq1/fixed-lrboost/scripts/render_jq1_lrboost_report.py](brd4-jq1/fixed-lrboost/scripts/render_jq1_lrboost_report.py) | forwarding entry point |
| `scripts/render_protenix_trajectory.py` | [scripts/reporting/render_protenix_trajectory.py](../scripts/reporting/render_protenix_trajectory.py) | forwarding entry point |
| `scripts/run_chemistry_compatibility_probe.py` | [experiments/brd4/chemistry-compatibility/scripts/run_chemistry_compatibility_probe.py](brd4/chemistry-compatibility/scripts/run_chemistry_compatibility_probe.py) | forwarding entry point |
| `scripts/serve_protenix_report.py` | [scripts/reporting/serve_protenix_report.py](../scripts/reporting/serve_protenix_report.py) | forwarding entry point |
| `scripts/submit_boltz_arms.py` | [experiments/brd4/boltz-affinity-10000/scripts/submit_boltz_arms.py](brd4/boltz-affinity-10000/scripts/submit_boltz_arms.py) | forwarding entry point |
| `scripts/templates/jq1_25_report.html` | [experiments/brd4-jq1/backtracking-25/templates/jq1_25_report.html](brd4-jq1/backtracking-25/templates/jq1_25_report.html) | byte-identical declarative compatibility copy; canonical owner is new path |
| `scripts/templates/jq1_fixed_report.html` | [experiments/brd4-jq1/fixed-steps/templates/jq1_fixed_report.html](brd4-jq1/fixed-steps/templates/jq1_fixed_report.html) | byte-identical declarative compatibility copy; canonical owner is new path |
| `scripts/templates/jq1_lrboost_report.html` | [experiments/brd4-jq1/fixed-lrboost/templates/jq1_lrboost_report.html](brd4-jq1/fixed-lrboost/templates/jq1_lrboost_report.html) | byte-identical declarative compatibility copy; canonical owner is new path |
| `scripts/templates/mosaic_milestones.html` | [experiments/reporting/milestones/templates/mosaic_milestones.html](reporting/milestones/templates/mosaic_milestones.html) | byte-identical declarative compatibility copy; canonical owner is new path |
| `scripts/templates/protenix_trajectory_report.html` | [scripts/reporting/protenix_trajectory_report.html](../scripts/reporting/protenix_trajectory_report.html) | byte-identical declarative compatibility copy; canonical owner is new path |
| `scripts/validate_brd4_launch.py` | [scripts/graph/validate_brd4_launch.py](../scripts/graph/validate_brd4_launch.py) | forwarding entry point |
| `scripts/validate_brd4_parallel.py` | [scripts/graph/validate_brd4_parallel.py](../scripts/graph/validate_brd4_parallel.py) | forwarding entry point |
| `scripts/validate_chemistry_readiness.py` | [scripts/chemistry/validate_chemistry_readiness.py](../scripts/chemistry/validate_chemistry_readiness.py) | forwarding entry point |
| `scripts/vendor_repair_reference.py` | [scripts/chemistry/vendor_repair_reference.py](../scripts/chemistry/vendor_repair_reference.py) | forwarding entry point |
| `scripts/verify_hpc_evidence.py` | [scripts/evidence/verify_hpc_evidence.py](../scripts/evidence/verify_hpc_evidence.py) | forwarding entry point |
| `scripts/verify_stage0_release.py` | [scripts/releases/verify_stage0_release.py](../scripts/releases/verify_stage0_release.py) | forwarding entry point |
| `tests/test_boltz_affinity_report.py` | [experiments/brd4/boltz-affinity-10000/tests/test_boltz_affinity_report.py](brd4/boltz-affinity-10000/tests/test_boltz_affinity_report.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_boltz_arm_readiness.py` | [experiments/brd4/boltz-affinity-10000/tests/test_boltz_arm_readiness.py](brd4/boltz-affinity-10000/tests/test_boltz_arm_readiness.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_boltz_diagnostic_cpu.py` | [experiments/brd4/boltz-gradient-diagnostic/tests/test_boltz_diagnostic_cpu.py](brd4/boltz-gradient-diagnostic/tests/test_boltz_diagnostic_cpu.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_boltz_gradient_diagnostic_contract.py` | [experiments/brd4/boltz-gradient-diagnostic/tests/test_boltz_gradient_diagnostic_contract.py](brd4/boltz-gradient-diagnostic/tests/test_boltz_gradient_diagnostic_contract.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_boltz_hpc_tmpdir.py` | [experiments/brd4/boltz-affinity-10000/tests/test_boltz_hpc_tmpdir.py](brd4/boltz-affinity-10000/tests/test_boltz_hpc_tmpdir.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_boltz_native_mapping.py` | [experiments/brd4/boltz-affinity-10000/tests/test_boltz_native_mapping.py](brd4/boltz-affinity-10000/tests/test_boltz_native_mapping.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_boltz_submission_controller.py` | [experiments/brd4/boltz-affinity-10000/tests/test_boltz_submission_controller.py](brd4/boltz-affinity-10000/tests/test_boltz_submission_controller.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_campaign25_contract.py` | [experiments/brd4/three-model-25/tests/test_campaign25_contract.py](brd4/three-model-25/tests/test_campaign25_contract.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_campaign25_native.py` | [experiments/brd4/three-model-25/tests/test_campaign25_native.py](brd4/three-model-25/tests/test_campaign25_native.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_campaign25_ops.py` | [experiments/brd4/three-model-25/tests/test_campaign25_ops.py](brd4/three-model-25/tests/test_campaign25_ops.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_campaign25_pocket_ops.py` | [experiments/brd4/three-model-25/tests/test_campaign25_pocket_ops.py](brd4/three-model-25/tests/test_campaign25_pocket_ops.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_campaign25_release.py` | [experiments/brd4/three-model-25/tests/test_campaign25_release.py](brd4/three-model-25/tests/test_campaign25_release.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_chemistry_compatibility_probe.py` | [experiments/brd4/chemistry-compatibility/tests/test_chemistry_compatibility_probe.py](brd4/chemistry-compatibility/tests/test_chemistry_compatibility_probe.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_lrboost_contract.py` | [experiments/brd4-jq1/fixed-lrboost/tests/test_fixed25_lrboost_contract.py](brd4-jq1/fixed-lrboost/tests/test_fixed25_lrboost_contract.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_lrboost_ops.py` | [experiments/brd4-jq1/fixed-lrboost/tests/test_fixed25_lrboost_ops.py](brd4-jq1/fixed-lrboost/tests/test_fixed25_lrboost_ops.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_lrboost_storage.py` | [experiments/brd4-jq1/fixed-lrboost/tests/test_fixed25_lrboost_storage.py](brd4-jq1/fixed-lrboost/tests/test_fixed25_lrboost_storage.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_lrreset_contract.py` | [experiments/brd4-jq1/fixed-lrreset/tests/test_fixed25_lrreset_contract.py](brd4-jq1/fixed-lrreset/tests/test_fixed25_lrreset_contract.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_lrreset_ops.py` | [experiments/brd4-jq1/fixed-lrreset/tests/test_fixed25_lrreset_ops.py](brd4-jq1/fixed-lrreset/tests/test_fixed25_lrreset_ops.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_lrreset_storage.py` | [experiments/brd4-jq1/fixed-lrreset/tests/test_fixed25_lrreset_storage.py](brd4-jq1/fixed-lrreset/tests/test_fixed25_lrreset_storage.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_ops.py` | [experiments/brd4-jq1/fixed-steps/tests/test_fixed25_ops.py](brd4-jq1/fixed-steps/tests/test_fixed25_ops.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_recover_ops.py` | [experiments/brd4-jq1/fixed-steps/tests/test_fixed25_recover_ops.py](brd4-jq1/fixed-steps/tests/test_fixed25_recover_ops.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_recover_storage.py` | [experiments/brd4-jq1/fixed-steps/tests/test_fixed25_recover_storage.py](brd4-jq1/fixed-steps/tests/test_fixed25_recover_storage.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_split_ops.py` | [experiments/brd4-jq1/fixed-steps/tests/test_fixed25_split_ops.py](brd4-jq1/fixed-steps/tests/test_fixed25_split_ops.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_split_storage.py` | [experiments/brd4-jq1/fixed-steps/tests/test_fixed25_split_storage.py](brd4-jq1/fixed-steps/tests/test_fixed25_split_storage.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_fixed25_storage.py` | [experiments/brd4-jq1/fixed-steps/tests/test_fixed25_storage.py](brd4-jq1/fixed-steps/tests/test_fixed25_storage.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_jq1_25_interim.py` | [experiments/brd4-jq1/backtracking-25/tests/test_jq1_25_interim.py](brd4-jq1/backtracking-25/tests/test_jq1_25_interim.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_jq1_25_report.py` | [experiments/brd4-jq1/backtracking-25/tests/test_jq1_25_report.py](brd4-jq1/backtracking-25/tests/test_jq1_25_report.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_jq1_fixed_ops.py` | [experiments/brd4-jq1/fixed-steps/tests/test_jq1_fixed_ops.py](brd4-jq1/fixed-steps/tests/test_jq1_fixed_ops.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_jq1_fixed_publish.py` | [experiments/brd4-jq1/fixed-steps/tests/test_jq1_fixed_publish.py](brd4-jq1/fixed-steps/tests/test_jq1_fixed_publish.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_jq1_fixed_render.py` | [experiments/brd4-jq1/fixed-steps/tests/test_jq1_fixed_render.py](brd4-jq1/fixed-steps/tests/test_jq1_fixed_render.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_jq1_fixed_report_contract.py` | [experiments/brd4-jq1/fixed-steps/tests/test_jq1_fixed_report_contract.py](brd4-jq1/fixed-steps/tests/test_jq1_fixed_report_contract.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_jq1_lrboost_ops.py` | [experiments/brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_ops.py](brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_ops.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_jq1_lrboost_publish.py` | [experiments/brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_publish.py](brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_publish.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_jq1_lrboost_render.py` | [experiments/brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_render.py](brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_render.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_jq1_lrboost_report_contract.py` | [experiments/brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_report_contract.py](brd4-jq1/fixed-lrboost/tests/test_jq1_lrboost_report_contract.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_lrboost_endpoint_diagnostics.py` | [experiments/brd4-jq1/fixed-lrboost/tests/test_lrboost_endpoint_diagnostics.py](brd4-jq1/fixed-lrboost/tests/test_lrboost_endpoint_diagnostics.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_protenix_lr03_ops.py` | [experiments/brd4-jq1/protenix-lr03/tests/test_protenix_lr03_ops.py](brd4-jq1/protenix-lr03/tests/test_protenix_lr03_ops.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |
| `tests/test_protenix_lr03_storage.py` | [experiments/brd4-jq1/protenix-lr03/tests/test_protenix_lr03_storage.py](brd4-jq1/protenix-lr03/tests/test_protenix_lr03_storage.py) | forwarding pytest entry point; default tests/ discovery remains single-copy |

Shared library tests with repository-path lookups were adapted in place: `tests/test_small_molecule_brd4_fixture.py`, `tests/test_small_molecule_case_selection.py`, `tests/test_small_molecule_chemistry_ramp_report.py`, `tests/test_small_molecule_chemistry_report.py`, `tests/test_small_molecule_contact_native.py`, `tests/test_small_molecule_diffuse_control.py`, `tests/test_small_molecule_evidence_root.py`, `tests/test_small_molecule_gradient_checks.py`, `tests/test_small_molecule_home_quota.py`, `tests/test_small_molecule_integration.py`, `tests/test_small_molecule_known_ligand_fixture.py`, `tests/test_small_molecule_prepared_study.py`, `tests/test_small_molecule_report_scope.py`, `tests/test_small_molecule_report_studies.py`, `tests/test_small_molecule_report_svg.py`, `tests/test_small_molecule_uniform_control.py`.

Canonical shell launchers resolve each source path using `mosaic_path`; canonical Python path construction uses the same registry. Existing external sealed source trees fall back to their original files. All future source archives must include `experiments/`, `scripts/layout/` and the map; source verification rejects unmanifested experiment inputs.

## Verification and preservation record

The [static migration audit](migration-verification.json) records checks and the
local archive smoke receipt. This is source organization evidence, not a CPU,
GPU, native-model or scientific acceptance certificate. The migrated working
tree still needs the applicable allocated numerical checks before future runs.

- 237 file ownership/path records cover 24 scientific/reporting experiment homes
  and documented shared tooling. There are 52 indexed historical attempts.
- All 84 established script module names resolve without importing those modules;
  122 canonical Python files parse, 147 old/canonical shell paths pass Bash syntax,
  and all 41 declarative compatibility pairs remain byte-identical.
- A synthetic stdlib module verifies legacy globals/dataclass behavior. Source
  authentication rejects changed and unmanifested experiment files; execution
  identity changes when a canonical experiment input changes.
- 144 of 145 existing scientific-library files retain their exact bytes. The
  sole change is the execution-file enumeration tuple in
  `src/mosaic/small_molecule/provenance.py`: it now includes `experiments/`.
  Restoring that one tuple exactly reproduces its original hash. This is
  provenance bookkeeping; no model, objective or optimizer code was changed.
- All 736 receipt files present at the baseline remain at their original paths.
  At the 10:24 UTC static audit, 735 retained exact hashes. One initially empty live operator console log was
  populated by its concurrent existing task; its before/observed hashes are
  recorded separately, and it is not claimed byte-identical. Migration tools
  never wrote inside `scripts/hpc/receipts/`.
  A later observation found six campaign summaries updated by the concurrent
  task as well (729 unchanged, seven changed total). All seven pre-migration
  byte states are additionally preserved under ignored
  `results/static/20261022-script-migration/preserved-receipt-baseline/`;
  the audit records their provenance and before/current hashes. Newer current
  summaries stay in place. These observations do not freeze ongoing updates.
- The Git staging index is unchanged from the pre-migration snapshot. Existing
  uncommitted sources, PDF/OneNote work and unrelated staged changes are retained.
  A concurrent agent inserted a lower-rate campaign status entry in the global
  ledger. A line-by-line comparison permits insertions only and confirms every
  pre-existing history line remains; that newer entry is preserved too.
- A disposable local source archive contains 1,731 authenticated files; every
  member was rehashed and the extracted legacy verifier succeeded. This smoke
  archive was not uploaded or selected as a scientific release.

Detailed local receipts, pre-edit source copies and the disposable smoke source
are under ignored `results/static/20261022-script-migration/`. The archive is
under ignored `results/releases/20261022-script-layout-smoke-01/`. These are new
local static-check artifacts; historical run outputs and published reports
remain in their existing locations. Later documentation additions postdate the
smoke snapshot and do not redefine its hashes.
