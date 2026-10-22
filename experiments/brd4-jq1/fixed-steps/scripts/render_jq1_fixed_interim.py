"""Allocated fixed-step progress editions from authenticated saved prefixes only.

Every arm remains visible, including an allocated/queued arm with no saved
molecule.  Only newline-committed contiguous state/update records are included;
an unfinished final line is excluded but corrupt complete rows are rejected.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from jq1_fixed_report_contract import (
    ARMS, POCKET, authenticate_inputs, authenticate_study, config_path,
    fixed_counts, read, ready_path, require, sha, source_path,
    status_or_completion, study_path,
)
from render_jq1_fixed_report import MILESTONES, VERSION, assemble, milestone, write_frames


def prefix(path):
    """Read one committed prefix, never turning corrupt complete rows into gaps."""
    raw = Path(path).read_bytes()
    boundary = raw.rfind(b'\n') + 1
    committed = raw[:boundary]
    rows = [json.loads(line) for line in committed.splitlines()]
    require(all(type(r) is dict and type(r.get('step')) is int for r in rows)
            and [r['step'] for r in rows] == list(range(len(rows))),
            'Noncontiguous live JSONL prefix')
    return rows, {'path': str(path), 'bytes': boundary,
                  'sha256': hashlib.sha256(committed).hexdigest(),
                  'uncommitted_tail_bytes': len(raw) - boundary}


def committed_case(root):
    states, sp = prefix(Path(root) / 'states.jsonl')
    updates, up = prefix(Path(root) / 'updates.jsonl')
    latest = min(len(states) - 1, len(updates))
    require(0 <= latest <= 10000, 'No committed saved state or unexpected update budget')
    # The writer may append between the two reads; use their common prefix.
    captured = {'states': states[:latest + 1], 'updates': updates[:latest]}
    for proof, rows in ((sp, captured['states']), (up, captured['updates'])):
        # Hash exact bytes for the included common prefix, while retaining the
        # full committed read boundary independently for concurrency evidence.
        proof['included_rows'] = len(rows)
        raw = Path(proof['path']).read_bytes()[:proof['bytes']]
        require(hashlib.sha256(raw).hexdigest() == proof['sha256'],
                'Previously committed JSONL bytes changed while capturing')
        included = b''.join(raw.splitlines(keepends=True)[:len(rows)])
        proof['included_bytes'] = len(included)
        proof['included_sha256'] = hashlib.sha256(included).hexdigest()
    return captured, [sp, up]


def selected_steps(latest):
    require(type(latest) is int and 0 <= latest <= 10000, 'Invalid saved update')
    return sorted({0, latest, *(s for s in MILESTONES if s <= latest)})


def read_live_report(path):
    # report.json is rewritten in place.  Retry torn JSON only; a complete
    # failed outcome or changed identity is never retried into a success.
    for attempt in range(5):
        try:
            return read(path)
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.2)


def expected_weight(arm, state):
    require(type(state) is int and 0 <= state <= 10000, 'Saved schedule index differs')
    if arm != 'boltz2_chemistry':
        return 0.0
    update = min(state + 1, 10000)
    if update <= 5000:
        return 0.001
    if update == 10000:
        return 50.0
    return math.exp(math.log(0.001) + ((update - 5000) / 5000) * (math.log(50.0) - math.log(0.001)))


def verify_prefix_identity(arm, case, captured, study):
    """Authenticate original fixed policy and entry/coupled/pocket identities."""
    pin = ARMS[arm]
    require(case['status'] in ('running', 'passed', 'failed')
            and case['requested_updates'] == 10000
            and case['parameterization'] == 'categorical_logits_v1'
            and case['acceptance_policy'] == 'fixed_step'
            and case['nominal_step_size'] == pin['rate']
            and case['key_data'] == [0, 17], 'Native case policy/initial key/scope changed')
    identity = case[pin['metric'] + '_objective_identity']
    expected = {'model_digest': pin['model'], 'context_digest': pin['context'],
                'native_name': pin['metric'], 'coefficient': 1.0, 'n_atoms': 25,
                'pocket_definition_sha256': POCKET, 'diffusion_gradient': False}
    require(all(identity.get(k) == v for k, v in expected.items()),
            'Live objective/frozen input identity differs')
    raw = study / 'native_gradient_checks/report.json'
    coupled = study / 'case-0/combined_initial_checks/report.json'
    raw_record, initial = read(raw), read(coupled)
    require(raw_record['status'] == 'passed', 'Native entry derivative gate failed')
    require(initial['status'] == 'passed' and initial['directions'] == case['initial_checks']
            and initial[pin['metric'] + '_objective_identity'] == identity,
            'Coupled entry gate differs or failed')
    require(sha(study / 'pocket_definition.json') == POCKET, 'Live pocket bytes differ')
    config = read(config_path(arm))
    require(case.get('repair_weight_schedule') == config.get('repair_schedule'),
            'Live chemistry schedule identity differs')
    for state in captured['states']:
        require(state['repair_weight'] == expected_weight(arm, state['step']),
                'Saved chemistry coefficient differs from exact original schedule')
    for update in captured['updates']:
        require(update['repair_weight'] == expected_weight(arm, update['step'])
                and update['attempts'][0]['repair_weight'] == update['repair_weight'],
                'Update trial changed its fixed chemistry coefficient')
    fixed_counts(captured['updates'], pin['rate'])
    return {'raw_entry_sha256': sha(raw), 'coupled_entry_sha256': sha(coupled),
            'pocket_definition_sha256': POCKET, 'objective_identity': identity}


def authenticate_prefix(arm, status):
    """Authenticate inputs for empty arms and entry evidence for every shown frame."""
    pin, study = ARMS[arm], study_path(arm)
    launcher = authenticate_inputs(arm)
    source = source_path(arm)
    require(sha(source / 'source_manifest.json') == pin['manifest']
            and sha(config_path(arm)) == pin['config'], 'Scientific source/configuration changed')
    proof = {'native_status': status, 'entry_readiness_sha256': pin['ready'],
             'launcher': launcher, 'terminal_checks':
             'An interim edition is not the final three-arm completion certificate'}
    state_path, report_path = study / 'case-0/states.jsonl', study / 'case-0/report.json'
    if not state_path.exists() or not report_path.exists():
        require(status != 'DONE', 'Finished native has no saved case/state evidence')
        proof['state'] = ('failed before authenticated saved trajectory' if status == 'EXIT'
                          else 'queued or preparing; no authenticated saved molecule yet')
        proof['scientific_output_present'] = state_path.exists()
        return None, proof
    case = read_live_report(report_path)
    captured, prefixes = committed_case(study / 'case-0')
    proof.update(verify_prefix_identity(arm, case, captured, study), jsonl_prefixes=prefixes,
                 native_case_status=case['status'])
    if status == 'DONE':
        # Native DONE never substitutes for terminal metadata/JSONL gates.  The
        # frame writer below separately authenticates every included NPZ.
        _, summary, full = authenticate_study(arm)
        require(captured == {'states': full['states'], 'updates': full['updates']},
                'Terminal committed prefix differs from original case')
        proof.update(summary_sha256=sha(study / 'summary.json'),
                     terminal_checks='Native terminal metadata and exact JSONL passed; included frames verified separately')
    elif case['status'] == 'failed':
        proof['failure'] = case.get('error', 'Native case reports failure')
    return captured, proof


def progress_assets(root, case, destination, metric, *, rate):
    require(len(case['updates']) == len(case['states']) - 1,
            'Committed update/state prefix differs')
    counts = fixed_counts(case['updates'], rate)
    chunks, curves = write_frames(root, case, destination, metric, rate=rate)
    return chunks, curves, counts


def status_label(status, case, proof):
    if status == 'EXIT' or proof.get('native_case_status') == 'failed':
        return 'failed partial trajectory' if case is not None else 'failed before authenticated saved trajectory'
    if status == 'DONE':
        return 'completed; native terminal metadata and included frames checked'
    if case is None:
        return 'queued; no saved molecule yet' if status == 'PEND' else 'allocated/preparing; no authenticated saved molecule yet'
    return 'running; latest captured committed prefix'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--completions', type=Path, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get('LSB_JOBID')), 'Scientific rendering requires allocated DTU CPU')
    args.output.mkdir(parents=True, exist_ok=False)
    args.proofs.mkdir(parents=True, exist_ok=False)
    studies = []
    for arm, pin in ARMS.items():
        captured = subprocess.run(['bjobs', '-l', pin['job']], capture_output=True, text=True, timeout=30)
        raw = captured.stdout + captured.stderr
        (args.proofs / (arm + '-scheduler.txt')).write_text(raw, encoding='utf-8')
        status = status_or_completion(raw, arm, args.completions)
        case, proof = authenticate_prefix(arm, status)
        entry = {
            'arm': arm, 'job_id': pin['job'], 'label': pin['label'], 'color': pin['color'],
            'metric': pin['metric'], 'learning_rate': pin['rate'], 'rate': pin['rate'],
            'selection_policy': 'fixed_step', 'acceptance_policy': 'fixed_step',
            'status': status_label(status, case, proof), 'native_status': status,
            'states': 0, 'latest_update': None, 'milestone_steps': [], 'curves': [], 'chunks': [],
            'counts': fixed_counts([], pin['rate']),
            'frame_sampling': 'Every committed saved state; no interpolation',
            'model_digest': pin['model'], 'context_digest': pin['context'],
            'source_manifest_sha256': pin['manifest'], 'config_sha256': pin['config'],
            'readiness_sha256': pin['ready'], 'progress_evidence': proof,
        }
        if case is not None:
            chunks, curves, counts = progress_assets(study_path(arm) / 'case-0', case,
                                                     args.output / arm, pin['metric'], rate=pin['rate'])
            latest = len(case['updates'])
            entry.update(chunks=chunks, curves=curves, states=len(curves),
                         milestone_steps=selected_steps(latest), counts=counts, latest_update=latest)
        studies.append(entry)
    require(any(milestone(s['latest_update']) for s in studies), 'First per-arm 1500-update milestone not reached')
    assemble(studies, args.output, interim=True, captured_utc=datetime.now(timezone.utc).isoformat())
    (args.proofs / 'capture.json').write_text(json.dumps(
        [{k: v for k, v in s.items() if k not in ('curves', 'chunks')} for s in studies],
        indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
