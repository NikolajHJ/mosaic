"""Independent stdlib byte/schema audit of allocated chemistry/DBMol readiness.

The original numerical decisions are authenticated, never recalculated locally.
No controller, model, scientific array or driver module is imported.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def adjacent(rows, fields):
    assert [row['field'] for row in rows] == fields
    for row in rows:
        probes = row['probes']
        assert [probe['epsilon'] for probe in probes] == [0.01, 0.003, 0.001]
        assert any(probes[i]['agrees'] is True and probes[i + 1]['agrees'] is True for i in [0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['chemistry', 'dbmol'], required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location('metadata_only', Path(__file__).with_name('verify_startup_delivery.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    kind = 'baseline' if args.kind == 'chemistry' else 'dbmol'
    cpu = module.verify(kind, args.bundle, False, True)
    e = module.Evidence(args.bundle); cfg = module.CAMPAIGNS[kind]
    _, files = module.load_seal(cfg['tag'], cfg['archive_sha'], cfg['archive_bytes'], cfg['manifest'], cfg['source_files'])
    e.add_archive(cfg['source'], files)
    arm, prefix, job, callback_job = ('boltz2_chemistry', 'boltz2_chemistry', '29805958', '29805959') if kind == 'baseline' else ('boltz2_dbmol', 'pilot', '29805980', '29805981')
    root = cfg['root']; parent = root + '/' + prefix + '-native-0000'; study = parent + '/payload/study/'
    cert_path = root + '/' + prefix + '-native-0000-certificate.json'; cert = e.json(cert_path)
    assert cert['status'] == 'passed' and cert['start'] == 0 and cert['source_manifest_sha256'] == cfg['manifest']
    cert_paths = e.proofs(cert['proofs'])
    assert study + 'assessment.json' in cert_paths
    ready = e.json(study + 'assessment.json'); proof_paths = e.proofs(ready['proofs'])
    assert study + 'summary.json' in proof_paths
    summary = e.json(study + 'summary.json')
    assert ready['status'] == summary['status'] == 'passed' and summary['mode'] == 'validate' and summary['job_id'] == job
    module.lsf(e.text(parent + '/stdout.' + job + '.log'), job)
    callback = root + '/' + prefix + '-after-native-0000'
    module.lsf(e.text(callback + '/stdout.' + callback_job + '.log'), callback_job)
    assert any(module.scheduler(e.text(p), job, 'gpul40s') == 'DONE' for p in cert_paths if '/scheduler-final-' in p)
    assert ready['source_manifest_sha256'] == summary['source_manifest_sha256'] == cfg['manifest']
    config_path = 'examples/configs/boltz_serial_boltz2_chemistry.json' if kind == 'baseline' else 'examples/configs/dbmol_pilot.json'
    assert ready['config_sha256'] == summary['config_sha256'] == module.digest(files[config_path])
    assert ready['arm'] == arm and ready['n_atoms'] == 25
    context = 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'
    model = '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16' if kind == 'baseline' else '7c90f5dd617310781780a8abe5f225f220f6586d4e2d910e67cbfcb8a4470044'
    for value in [ready, summary, summary['native_identity'], summary['frozen_identity_after']]:
        assert value['model_digest'] == model and value['context_digest'] == context
    assert summary['frozen_identity_after']['status'] == 'passed'
    assert ready['diffusion_gradient'] is summary['diffusion_gradient'] is False
    assert ready['pocket_definition_sha256'] == summary['pocket_definition_sha256'] == module.POCKET
    assert summary['source']['git_revision_reference'] == module.REVISION
    assert summary['native_hard_parity']['status'] == 'passed'
    assert ready['runtime_gate']['status'] == summary['runtime_gate']['status'] == 'passed'
    assert ready['runtime_gate'].get('allocation_seconds', ready['runtime_gate'].get('internal_allocation_seconds')) == 82800
    gradient = summary['native_gradient_checks']; assert gradient['status'] == 'passed'
    pilots = ['early_pilot/report.json', 'high_pilot/report.json'] if kind == 'baseline' else ['early_pilot/report.json']
    for leaf in pilots:
        assert study + leaf in proof_paths
        pilot = e.json(study + leaf)
        assert pilot['status'] == 'passed' and pilot['requested_updates'] == 10 and pilot['counts']['applied'] == 10
        assert pilot['acceptance_policy'] == 'fixed_step' and pilot['nominal_step_size'] == 0.0025 and pilot['counts']['forced_fallback'] == 0
    if kind == 'baseline':
        assert summary['chemistry_enabled'] is True and summary['chemistry_assay']['status'] == 'passed'
        assert summary['optimization_parameters'] == {'acceptance_policy': 'fixed_step', 'max_backtracks': 1, 'step_size': 0.0025}
        assert summary['exact_boundary_replay'] == 'passed' and gradient['typed_bond_contrast_nonzero'] is True
        adjacent(gradient['rows'], ['X', 'E_upper'])
        for leaf, weight in [('low_coupled_checks/report.json', 0.001), ('high_coupled_checks/report.json', 100), ('early_pilot/combined_initial_checks/report.json', 0.001), ('high_pilot/combined_initial_checks/report.json', 100)]:
            assert study + leaf in proof_paths
            coupled = e.json(study + leaf)
            assert coupled['status'] == 'passed' and coupled['repair_weight'] == weight
            adjacent(coupled['directions'], ['atoms', 'bonds'])
        execution = ready['serial_execution']
        assert execution == summary['serial_execution'] and execution['target'] == 30000 and execution['end'] == 10000
        assert len(proof_paths) == 128
    else:
        assert summary['chemistry_enabled'] is False and gradient['all_components_and_total_raw_fd_passed'] is True
        assert gradient['can_recertify_saved_failed_run'] is False and gradient['frozen_unchanged'] is True
        assert gradient['frozen_before'] == gradient['frozen_after']
        assert gradient['key_and_logit_row_shift_nulls'] == 'passed'
        assert gradient['live_primitive_recomposition'] == 'passed_exact_scalar_and_raw_gradients'
        assert gradient['hard_native_parity']['status'] == 'passed'
        assert len(gradient['hard_native_parity']['absolute_differences']) == 19 and set(gradient['hard_native_parity']['absolute_differences'].values()) == {0.0}
        assert set(gradient['components']) == {'dbmol_affinity_loss', 'dbmol_confidence_loss', 'dbmol_contact_loss', 'dbmol_total_loss'}
        for component in gradient['components'].values():
            assert component['status'] == 'passed' and component['epsilons'] == [0.01, 0.003, 0.001]
            assert component['independent_primal'] is component['raw_unclipped_gradients'] is component['typed_bond_contrast_nonzero'] is True
            assert component['production_jax_transport'] == 'passed' and component['relative_tolerance'] == 0.15 and component['required_adjacent_passes'] == 2
            adjacent(component['rows'], ['X', 'E_upper'])
        native = summary['native_identity']
        assert native['physical_trunks_per_forward'] == 2 and native['graph_dtype'] == native['gradient_dtype'] == 'float32' and native['scalar_dtype'] == 'float64'
        assert native['objective_arithmetic'] == 'contact_head_and_assembly_float64_v1' and native['contact_weight'] == native['confidence_weight'] == 2.5
        assert native['affinity_mw_correction'] is native['diffusion_gradient'] is False and native['recycling_steps'] == 0
        assert native['context_file_sha256'] == module.CONTEXT and native['structural_checkpoint_sha256'] == '090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1'
        execution = ready['execution']; assert execution == summary['execution'] and execution['target'] == 10000 and execution['end'] == 2500
        assert len(proof_paths) == 53
    assert execution['start'] == 0 and execution['input_snapshot'] is execution['input_snapshot_sha256'] is None
    result = {'status': 'passed_native_readiness_byte_schema_audit', 'arm': arm, 'native_job': job, 'callback_job': callback_job, 'source_manifest_sha256': cfg['manifest'],
              'readiness_sha256': e.checksum(study + 'assessment.json'), 'native_certificate_sha256': e.checksum(cert_path), 'summary_sha256': e.checksum(study + 'summary.json'),
              'readiness_proof_files_verified': len(proof_paths), 'cpu_cases_verified': cpu['cpu']['cases'], 'raw_epsilon_ladder': [0.01, 0.003, 0.001],
              'reported_adjacent_agreement_authenticated': True, 'hard_parity_frozen_runtime_passes_authenticated': True, 'reported_ten_update_pilots_authenticated': len(pilots),
              'production_optimization_started': False, 'storage_projection': ready['storage_projection'], 'metadata_sha256': module.digest((args.bundle / 'metadata.json').read_bytes()),
              'scope': 'Local stdlib byte/schema/LSF/source proof verification only. No array decoding, derivative recomputation, inference or model execution. Original allocated numerical checks remain authoritative.'}
    with args.output.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
