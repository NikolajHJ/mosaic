# Shared molecular graph tests

Reusable graph state/features, native bridges, chemistry, optimization, artifact IO and shared pocket contracts. Experiment-only tests remain in their experiment folders.

Use default pytest discovery or select this directory in an allocated DTU HPC
job. Local layout verification parses source only and is not a numerical pass.
Shared fixtures remain under [tests/data](../data/).

`test_chemistry_confidence.py` checks the opt-in bounded-slope, valid-source
margin. Run its qualified capsule through the
[confidence experiment](../../experiments/brd4/chemistry-confidence/README.md);
the bitwise prechange-control cases require the exact archived trajectory path
in `MOSAIC_CONTROL_TRAJECTORY`. This is a CPU implementation check, not a native
binding-model or VAE-training qualification.

## Files

- [test_pocket_boltz.py](test_pocket_boltz.py)
- [test_pocket_definition.py](test_pocket_definition.py)
- [test_small_molecule_boltz_bridge.py](test_small_molecule_boltz_bridge.py)
- [test_small_molecule_brd4_fixture.py](test_small_molecule_brd4_fixture.py)
- [test_small_molecule_campaign25.py](test_small_molecule_campaign25.py)
- [test_small_molecule_case_selection.py](test_small_molecule_case_selection.py)
- [test_small_molecule_chemistry.py](test_small_molecule_chemistry.py)
- [test_small_molecule_chemistry_ramp.py](test_small_molecule_chemistry_ramp.py)
- [test_small_molecule_chemistry_ramp_report.py](test_small_molecule_chemistry_ramp_report.py)
- [test_small_molecule_chemistry_report.py](test_small_molecule_chemistry_report.py)
- [test_small_molecule_compact_artifacts.py](test_small_molecule_compact_artifacts.py)
- [test_small_molecule_contact_native.py](test_small_molecule_contact_native.py)
- [test_small_molecule_continuation.py](test_small_molecule_continuation.py)
- [test_small_molecule_contracts.py](test_small_molecule_contracts.py)
- [test_small_molecule_diffuse_control.py](test_small_molecule_diffuse_control.py)
- [test_small_molecule_evidence_root.py](test_small_molecule_evidence_root.py)
- [test_small_molecule_features.py](test_small_molecule_features.py)
- [test_small_molecule_fixed_step.py](test_small_molecule_fixed_step.py)
- [test_small_molecule_gradient_checks.py](test_small_molecule_gradient_checks.py)
- [test_small_molecule_home_quota.py](test_small_molecule_home_quota.py)
- [test_small_molecule_integration.py](test_small_molecule_integration.py)
- [test_small_molecule_io.py](test_small_molecule_io.py)
- [test_small_molecule_jq1_pocket.py](test_small_molecule_jq1_pocket.py)
- [test_small_molecule_known_ligand_fixture.py](test_small_molecule_known_ligand_fixture.py)
- [test_small_molecule_losses.py](test_small_molecule_losses.py)
- [test_small_molecule_optimizers.py](test_small_molecule_optimizers.py)
- [test_small_molecule_prepared_study.py](test_small_molecule_prepared_study.py)
- [test_small_molecule_protenix.py](test_small_molecule_protenix.py)
- [test_small_molecule_report_scope.py](test_small_molecule_report_scope.py)
- [test_small_molecule_report_studies.py](test_small_molecule_report_studies.py)
- [test_small_molecule_report_svg.py](test_small_molecule_report_svg.py)
- [test_small_molecule_state.py](test_small_molecule_state.py)
- [test_small_molecule_uniform_control.py](test_small_molecule_uniform_control.py)
- [test_small_molecule_validation.py](test_small_molecule_validation.py)

See the [test index](../README.md), [experiment index](../../experiments/INDEX.md)
and [nested layout map](../../experiments/folder-layout.json).
