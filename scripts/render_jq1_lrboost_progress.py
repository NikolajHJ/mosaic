"""Allocated higher-rate progress editions from authenticated saved prefixes only.

Every arm remains visible, including an allocated/queued arm with no saved
molecule.  Only newline-committed contiguous state/update records are included;
an unfinished final line is excluded but corrupt complete rows are rejected.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from jq1_lrboost_report_contract import (
    ARMS, authenticate_progress, fixed_counts, read, require,
    status_or_completion, study_path,
)
from render_jq1_lrboost_report import MILESTONES, VERSION, assemble, collect_diagnostics, milestone, write_frames


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



def authenticate_prefix(arm, status):
    """Use the rate-specific contract for saved prefixes and native outcomes."""
    return authenticate_progress(arm, status)


def progress_assets(root, case, destination, metric, *, rate):
    require(len(case['updates']) == len(case['states']) - 1,
            'Committed update/state prefix differs')
    counts = fixed_counts(case['updates'], rate)
    chunks, curves = write_frames(root, case, destination, metric, rate=rate)
    return chunks, curves, counts


def status_label(status, case, proof):
    if proof.get('overall_status') == 'failed' or status in ('EXIT', 'FAILED_PERSISTED') or proof.get('native_case_status') == 'failed':
        if proof.get('budget_status') == 'full':
            return '10,000 updates saved; terminal scientific validation failed'
        return 'failed partial trajectory' if case is not None else 'failed before authenticated saved trajectory'
    if proof.get('scientific_passed') is True:
        return 'all native validation passed; included saved frames checked'
    if case is None:
        return 'queued; no saved molecule yet' if status == 'PEND' else 'allocated/preparing; no authenticated saved molecule yet'
    return 'running; latest captured committed prefix'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proofs', type=Path, required=True)
    parser.add_argument('--completions', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path)
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
            **{k: proof[k] for k in ('readiness_status', 'terminal_status', 'terminal_error',
                 'overall_status', 'budget_status', 'scientific_passed', 'all_terminal')},
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
    assemble(studies, args.output, interim=True, captured_utc=datetime.now(timezone.utc).isoformat(),
             diagnostics=collect_diagnostics(args.diagnostics, args.output))
    (args.proofs / 'capture.json').write_text(json.dumps(
        [{k: v for k, v in s.items() if k not in ('curves', 'chunks')} for s in studies],
        indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
