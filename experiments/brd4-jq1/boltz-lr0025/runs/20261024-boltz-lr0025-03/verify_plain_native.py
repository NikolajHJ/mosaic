"""Independent saved byte/schema audit of the first plain Boltz readiness proof.

No model/array imports or derivative calculations. Numerical gate decisions
remain the original allocated producer's authenticated evidence.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = importlib.util.spec_from_file_location('delivery_metadata', Path(__file__).with_name('verify_startup_delivery.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cpu = module.verify('baseline', args.bundle, False, True)
    evidence = module.Evidence(args.bundle)
    config = module.CAMPAIGNS['baseline']
    manifest, files = module.load_seal(config['tag'], config['archive_sha'], config['archive_bytes'], config['manifest'], 635)
    evidence.add_archive(config['source'], files)
    root = config['root']; parent = root + '/boltz2-native-0000'; callback = root + '/boltz2-after-native-0000'
    certificate_path = root + '/boltz2-native-0000-certificate.json'
    certificate = evidence.json(certificate_path)
    assert certificate['status'] == 'passed' and certificate['source_manifest_sha256'] == config['manifest'] and certificate['start'] == 0
    paths = evidence.proofs(certificate['proofs'])
    ready_path = parent + '/payload/study/assessment.json'
    assert ready_path in paths
    ready = evidence.json(ready_path)
    proof_paths = evidence.proofs(ready['proofs'])
    summary_path = parent + '/payload/study/summary.json'
    assert summary_path in proof_paths
    summary = evidence.json(summary_path)
    assert ready['status'] == summary['status'] == 'passed'
    assert summary['job_id'] == '29805950' and summary['mode'] == 'validate'
    module.lsf(evidence.text(parent + '/stdout.29805950.log'), '29805950')
    module.lsf(evidence.text(callback + '/stdout.29805951.log'), '29805951')
    final = [path for path in paths if '/scheduler-final-' in path]
    assert final
    assert any(module.scheduler(evidence.text(path), '29805950', 'gpul40s') == 'DONE' for path in final)
    assert ready['source_manifest_sha256'] == summary['source_manifest_sha256'] == config['manifest']
    assert ready['arm'] == 'boltz2' and ready['config_sha256'] == summary['config_sha256'] == module.digest(files['examples/configs/boltz_serial_boltz2.json'])
    model = '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16'
    context = 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'
    for value in [ready, summary, summary['native_identity'], summary['frozen_identity_after']]:
        assert value['model_digest'] == model and value['context_digest'] == context
    assert summary['frozen_identity_after']['status'] == 'passed'
    assert ready['diffusion_gradient'] is summary['diffusion_gradient'] is False
    assert ready['pocket_definition_sha256'] == summary['pocket_definition_sha256'] == module.POCKET
    assert ready['pocket_validation_status'] == summary['pocket_validation_status'] == 'passed'
    assert summary['context_file_sha256'] == module.CONTEXT
    assert summary['source']['git_revision_reference'] == module.REVISION
    assert summary['native_hard_parity'] == {'status': 'passed', 'n_atoms': 25, 'atol': 2e-5}
    assert summary['optimization_parameters'] == {'acceptance_policy': 'fixed_step', 'max_backtracks': 1, 'step_size': 0.0025}
    assert summary['initialization_execution']['dtype'] == 'float32' and summary['initialization_execution']['jax_platform'] == 'cpu' and summary['initialization_execution']['model_key'] == 17
    assert summary['exact_boundary_replay'] == 'passed'
    gradient = summary['native_gradient_checks']
    assert gradient['status'] == 'passed' and gradient['typed_bond_contrast_nonzero'] is True
    assert [row['field'] for row in gradient['rows']] == ['X', 'E_upper']
    for row in gradient['rows']:
        assert [probe['epsilon'] for probe in row['probes']] == [0.01, 0.003, 0.001]
        assert any(row['probes'][i]['agrees'] is True and row['probes'][i + 1]['agrees'] is True for i in [0, 1])
    pilot_path = parent + '/payload/study/early_pilot/report.json'
    assert pilot_path in proof_paths
    pilot = evidence.json(pilot_path)
    assert pilot['status'] == 'passed' and pilot['requested_updates'] == 10 and pilot['counts']['applied'] == 10
    assert pilot['acceptance_policy'] == 'fixed_step' and pilot['nominal_step_size'] == 0.0025
    assert pilot['counts']['forced_fallback'] == 0
    assert ready['runtime_gate']['status'] == summary['runtime_gate']['status'] == 'passed'
    assert ready['runtime_gate']['allocation_seconds'] == 82800
    assert ready['serial_execution'] == summary['serial_execution']
    assert ready['serial_execution']['start'] == 0 and ready['serial_execution']['target'] == 30000
    result = {'status': 'passed_native_readiness_byte_schema_audit', 'arm': 'boltz2', 'native_job': '29805950',
        'callback_job': '29805951', 'source_manifest_sha256': config['manifest'],
        'readiness_sha256': evidence.checksum(ready_path), 'native_certificate_sha256': evidence.checksum(certificate_path),
        'summary_sha256': evidence.checksum(summary_path), 'readiness_proof_files_verified': len(proof_paths),
        'cpu_cases_verified': cpu['cpu']['cases'], 'raw_epsilon_ladder': [0.01, 0.003, 0.001],
        'reported_adjacent_agreement_authenticated': True, 'hard_parity_frozen_runtime_passes_authenticated': True,
        'reported_ten_update_pilot_authenticated': True, 'production_optimization_started': False,
        'storage_projection': ready['storage_projection'], 'metadata_sha256': module.digest((args.bundle / 'metadata.json').read_bytes()),
        'scope': 'Local stdlib byte/schema/LSF/source proof verification only. No array decoding, derivative recomputation, inference or model execution. Original allocated numerical checks remain authoritative.'}
    with args.output.open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
