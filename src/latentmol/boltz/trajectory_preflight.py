"""HPC-only engineering gates on the historical four C13 native predictions.

No new model prediction and no optimizer update occurs here. This checks the
new audit against real pinned native artifacts, beyond synthetic unit fixtures.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from latentmol.experiments import ExperimentRun, file_sha256, write_json
from .fresh_trajectory import load_protocol
from .processed_identity import audit_processed
from .trajectory_selection import _pilot_candidates, read, verified


def run(config, expected_sha, output, historical):
    if not os.environ.get('LSB_JOBID') or not os.environ.get('LATENTMOL_SOURCE_SHA256'):
        raise ValueError('Preflight requires HPC/source identity')
    verified(config, expected_sha)
    protocol = load_protocol(config)
    parent = protocol['parents']['c13']
    record, rows, proxies, warnings = _pilot_candidates(Path(parent['pilot']), parent['source_sha256'], parent['config_sha256'])
    historical_record = read(historical / 'run.json')
    if historical_record['status'] != 'completed' or historical_record['source_archive_sha256'] != parent['source_sha256']:
        raise ValueError('Historical rescore source/status mismatch')
    attempts = read(historical / 'native_results.json')
    if len(attempts) != 4:
        raise ValueError('Require all four historical native requests, not selected successful outcomes')
    with ExperimentRun(output, 'fresh-trajectory-preflight', {'protocol_sha256': expected_sha,
                       'historical': str(historical), 'historical_run_sha256': file_sha256(historical / 'run.json')}, Path.cwd()) as run:
        audits = []
        for item in attempts:
            input_yaml = historical / 'inputs' / f"{item['stem']}.yaml"
            verified(input_yaml, item['input_sha256'])
            template = yaml.safe_load(input_yaml.read_text())
            protein = next(s['protein'] for s in template['sequences'] if 'protein' in s)
            msa = Path(protein['msa'])
            if not msa.is_absolute():
                msa = input_yaml.parent / msa
            expected = {'smiles': item['smiles'], 'target_sequence_sha256': record['target_sequence_sha256'],
                        'msa_sha256': record['msa_sha256'], 'protein_sequence': protein['sequence'],
                        'record_id': item['stem'], 'input_yaml': str(input_yaml), 'msa_path': str(msa),
                        'max_msa_seqs': 8192}
            prediction = historical / 'predictions' / item['stem'] / f"boltz_results_{item['stem']}"
            result = audit_processed(prediction, Path(protocol['cache']), expected)
            audits.append(result)
            write_json(run.path / 'audits.json', audits)
            if not result['retention_eligible']:
                raise ValueError(f"Historical native audit failed: {result['failure']}")
        run.annotate(processed_native_audits_passed=len(audits),
                     historical_completed_candidates=sum(len(v) for v in rows.values()),
                     warnings=warnings, scope='Engineering audit of existing C13 evidence; no new binding result')
    return run.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--historical', type=Path, required=True)
    a = p.parse_args()
    print(run(a.config, a.sha256, a.output_root, a.historical), flush=True)


if __name__ == '__main__':
    main()
