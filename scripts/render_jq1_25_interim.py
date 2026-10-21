"""Allocated, saved-only progress editions; never certify an unfinished study.

Only complete JSONL lines form a captured prefix. Sparse milestone probability
frames retain their original steps; every committed scalar curve point remains.
The separate final renderer still requires all native terminal gates and frames.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from jq1_report_contract import ARMS, HOME, POCKET, RELEASES, read, require, scheduler_outcome, sha, study_path
from render_jq1_25_report import assemble, frame_arrays

MILESTONES = tuple(range(1500, 10000, 1000)) + (10000,)


def prefix(path):
    """Ignore only an unfinished final line, never a corrupt committed record."""
    raw = path.read_bytes()
    boundary = raw.rfind(b'\n') + 1
    committed = raw[:boundary]
    rows = [json.loads(line) for line in committed.splitlines()]
    require([r['step'] for r in rows] == list(range(len(rows))), 'Noncontiguous live JSONL prefix')
    return rows, {'path': str(path), 'bytes': boundary,
                  'sha256': hashlib.sha256(committed).hexdigest(), 'uncommitted_tail_bytes': len(raw) - boundary}


def committed_case(root):
    states, sp = prefix(root / 'states.jsonl')
    updates, up = prefix(root / 'updates.jsonl')
    latest = min(len(states) - 1, len(updates))
    require(latest >= 0 and latest <= 10000, 'No committed saved state or unexpected update budget')
    # A producer may append between the two reads. Include only a shared prefix.
    return {'states': states[:latest + 1], 'updates': updates[:latest]}, [sp, up]


def selected_steps(latest):
    require(type(latest) is int and 0 <= latest <= 10000, 'Invalid saved update')
    return sorted({0, latest, *(s for s in MILESTONES if s <= latest)})


def milestone(latest):
    return max((m for m in MILESTONES if latest is not None and m <= latest), default=0)


def live_status(raw, job):
    normalized = re.sub(r'\s+', '', raw)
    require(f'Job<{job}>' in normalized and 'Queue<gpul40s>' in normalized, 'Live native job/queue mismatch')
    match = re.search(r'Status<([^>]+)>', normalized)
    require(match is not None and match[1] in ('RUN', 'PEND', 'DONE'), 'Native failed or scheduler outcome unavailable')
    return match[1]


def status_or_completion(raw, arm, completions):
    """An aged-out live record needs an actual authenticated completion proof."""
    pin = ARMS[arm]
    if re.search(r'Status<[^>]+>', re.sub(r'\s+', '', raw)):
        return live_status(raw, pin['job'])
    assessment = read(completions / ('complete-' + arm) / 'payload/assessment.json')
    require(assessment['status'] == 'passed' and assessment['native_job'] == pin['job'], 'Missing passed native completion proof')
    scheduler = []
    for proof in assessment['proofs']:
        require(sha(proof['path']) == proof['sha256'], 'Native completion proof changed')
        if Path(proof['path']).name.startswith(('scheduler-final', 'native-scheduler')):
            scheduler.append(Path(proof['path']).read_text())
    require(any(scheduler_outcome(text, pin['job']) == 'DONE' for text in scheduler), 'No authenticated terminal scheduler evidence')
    return 'DONE'


def read_live_report(path):
    # The producer rewrites its progress report in place. Retry torn JSON only;
    # a complete failed report or a wrong identity is never retried away.
    for attempt in range(5):
        try:
            return read(path)
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.2)


def authenticate_prefix(arm, status):
    """Bind to the passed entry evidence; terminal success remains pending."""
    from campaign25_contract import verify_readiness
    pin, study = ARMS[arm], study_path(arm)
    source = RELEASES / pin['tag'] / 'source'
    config = source / 'examples/configs' / f'brd4_jq1_25_{arm}_10000.json'
    require(sha(source / 'source_manifest.json') == pin['manifest'] and sha(config) == pin['config'], 'Native source/configuration changed')
    ready_path = HOME / pin['tag'] / ('validate-' + arm) / 'payload/study/assessment.json'
    ready = verify_readiness(ready_path, pin['ready'], config, pin['manifest'])
    require(ready['context_digest'] == pin['context'] and ready['pocket_definition_sha256'] == POCKET, 'Readiness context/pocket differs')
    require(ready['runtime_gate']['status'] == 'passed', 'Native readiness runtime failed')
    if (study / 'summary.json').exists():
        terminal = read(study / 'summary.json')
        require(terminal['status'] == 'passed' and terminal['job_id'] == pin['job'], 'Native terminal failure or wrong job')
    if not (study / 'case-0/states.jsonl').exists():
        require(status in ('PEND', 'RUN'), 'Finished native has no saved states')
        return None, {'entry_readiness_sha256': pin['ready'], 'native_status': status, 'state': 'queued or preparing; no saved molecule yet'}
    case = read_live_report(study / 'case-0/report.json')
    require(case['status'] in ('running', 'passed') and case['requested_updates'] == 10000, 'Native case failed or scope changed')
    identity = case[pin['metric'] + '_objective_identity']
    expected = {'model_digest': pin['model'], 'context_digest': pin['context'], 'native_name': pin['metric'],
                'coefficient': 1.0, 'n_atoms': 25, 'pocket_definition_sha256': POCKET, 'diffusion_gradient': False}
    require(all(identity.get(k) == v for k, v in expected.items()), 'Live objective identity differs')
    raw, coupled = study / 'native_gradient_checks/report.json', study / 'case-0/combined_initial_checks/report.json'
    require(read(raw)['status'] == 'passed', 'Native entry derivative gate failed')
    initial = read(coupled)
    require(initial['status'] == 'passed' and initial['directions'] == case['initial_checks'] and initial[pin['metric'] + '_objective_identity'] == identity, 'Coupled entry gate differs or failed')
    require(sha(study / 'pocket_definition.json') == POCKET, 'Live pocket bytes differ')
    captured, prefixes = committed_case(study / 'case-0')
    for s in captured['states']:
        require(s['repair_weight'] == 0 if arm != 'boltz2_chemistry' else 0.001 <= s['repair_weight'] <= 50, 'Unexpected chemistry coefficient')
    proof = {'native_status': status, 'entry_readiness_sha256': pin['ready'], 'raw_entry_sha256': sha(raw),
             'coupled_entry_sha256': sha(coupled), 'jsonl_prefixes': prefixes,
             'terminal_checks': 'Not certified by an interim edition; final completion collectors are authoritative'}
    return captured, proof


def sparse_assets(root, case, destination, metric):
    """Reuse exact snapshot checks, without relaxing final contiguous-frame rules."""
    destination.mkdir(parents=True, exist_ok=False)
    latest = len(case['states']) - 1
    selected, rows, binary, curves = selected_steps(latest), [], bytearray(), []
    for state in case['states']:
        step = state['step']
        update = case['updates'][step - 1] if step else None
        components, chemical = state['components'], state['components'].get('chemistry_loss')
        curves.append({'step': step, 'native': state[metric + '_loss'], 'binder': components.get('affinity_probability_binary'),
            'repair': chemical, 'weighted_repair': None if chemical is None else state['repair_weight'] * chemical,
            'weight': state['repair_weight'], 'fixed_delta': None if update is None else update['fixed_after'] - update['fixed_before'],
            'fallback': False if update is None else update['fallback_used'],
            'sanitized_connected': bool(state['readout']['sanitized'] and not state['readout']['disconnected'])})
        if step in selected:
            binary.extend(frame_arrays(root, state, metric))
            row = dict(state)
            row['readout'] = {k: v for k, v in state['readout'].items() if k != 'pairs'}
            row['previous_update'] = None if update is None else {k: update[k] for k in ('fixed_before', 'fixed_after', 'fallback_used', 'no_op', 'accepted_step_size')}
            rows.append(row)
    (destination / 'frames.bin').write_bytes(binary)
    (destination / 'frames.json').write_text(json.dumps(rows, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    (destination / 'curves.json').write_text(json.dumps(curves, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    chunk = {'start': 0, 'count': len(rows), 'binary': 'frames.bin', 'binary_sha256': sha(destination / 'frames.bin'),
             'metadata': 'frames.json', 'metadata_sha256': sha(destination / 'frames.json')}
    counts = {'applied': latest, 'forced_fallback': sum(bool(u['fallback_used']) for u in case['updates']),
              'no_op': sum(bool(u['no_op']) for u in case['updates'])}
    counts['strict_improving'] = latest - counts['forced_fallback']
    return [chunk], curves, selected, counts


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
        (args.proofs / (arm + '-scheduler.txt')).write_text(raw)
        status = status_or_completion(raw, arm, args.completions)
        case, proof = authenticate_prefix(arm, status)
        entry = {'arm': arm, 'job_id': pin['job'], 'label': pin['label'], 'color': pin['color'], 'metric': pin['metric'],
            'states': 0, 'latest_update': None, 'snapshot_steps': [], 'curves': [], 'chunks': [], 'counts': {},
            'model_digest': pin['model'], 'context_digest': pin['context'], 'source_manifest_sha256': pin['manifest'],
            'config_sha256': pin['config'], 'progress_evidence': proof}
        if case is not None:
            chunks, curves, selected, counts = sparse_assets(study_path(arm) / 'case-0', case, args.output / arm, pin['metric'])
            entry.update(chunks=chunks, curves=curves, states=len(selected), snapshot_steps=selected, counts=counts, latest_update=len(curves) - 1)
        studies.append(entry)
    require(any(milestone(s['latest_update']) for s in studies), 'First 1500-update milestone not reached')
    assemble(studies, args.output, interim=True)
    receipt = read(args.output / 'report_receipt.json')
    receipt.update(captured_utc=datetime.now(timezone.utc).isoformat(), milestone_progress={s['arm']: milestone(s['latest_update']) for s in studies})
    (args.output / 'report_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.proofs / 'capture.json').write_text(json.dumps([{k: v for k, v in s.items() if k not in ('curves', 'chunks')} for s in studies], indent=2) + '\n')


if __name__ == '__main__':
    main()
