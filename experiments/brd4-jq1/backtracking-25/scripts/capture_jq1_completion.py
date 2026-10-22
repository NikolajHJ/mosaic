"""Capture a native final outcome promptly, before LSF live retention expires."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from jq1_report_contract import ARMS, HOME, RELEASES, authenticate_study, require, sha, study_path, scheduler_outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get('LSB_JOBID')), 'Completion checks require allocated DTU CPU')
    args.output.mkdir(parents=True, exist_ok=False)
    pin = ARMS[args.arm]
    native = HOME / pin['tag'] / ('run-' + args.arm)
    record = {'status': 'failed', 'native_job': pin['job'], 'collector_job': os.environ['LSB_JOBID'], 'arm': args.arm}
    try:
        status = None
        stdout = native / ('stdout.' + pin['job'] + '.log')
        for attempt in range(7):
            result = subprocess.run(['bjobs', '-l', pin['job']], capture_output=True, text=True, timeout=30)
            raw = result.stdout + result.stderr
            (args.output / f'native-scheduler-{attempt}.txt').write_text(raw)
            require(result.returncode == 0, 'Native final scheduler outcome unavailable')
            try:
                status = scheduler_outcome(raw, pin['job'])
                require(status == 'DONE' and f'Subject: Job {pin["job"]}:' in stdout.read_text() and 'Successfully completed.' in stdout.read_text(), 'Native did not finish successfully')
                break
            except ValueError:
                if status == 'EXIT' or attempt == 6:
                    raise
                time.sleep(10)
        shutil.copyfile(stdout, args.output / 'native-stdout.txt')
        study, summary, case = authenticate_study(args.arm)
        sources = [study / 'summary.json', study / 'case-0/report.json', study / 'case-0/states.jsonl', study / 'case-0/updates.jsonl', study / 'initialization.npz']
        sources += [study / p / 'report.json' for p in ('native_gradient_checks', 'terminal_gradient_checks', 'case-0/combined_initial_checks')]
        for source in sources:
            target = args.output / 'native-proof' / source.relative_to(study)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        config = _layout_path(RELEASES / pin['tag'] / 'source/examples/configs' / f'brd4_jq1_25_{args.arm}_10000.json')
        shutil.copyfile(config, args.output / 'native-config.json')
        source_manifest = config.parents[2] / 'source_manifest.json'
        shutil.copyfile(source_manifest, args.output / 'native-source-manifest.json')
        ready = HOME / pin['tag'] / ('validate-' + args.arm) / 'payload/study/assessment.json'
        shutil.copyfile(ready, args.output / 'native-readiness.json')
        record.update(status='passed', states=len(case['states']), updates=len(case['updates']), n_atoms=25,
            model_digest=pin['model'], context_digest=pin['context'], source_manifest_sha256=pin['manifest'],
            proofs=[{'path': str(p), 'sha256': sha(p)} for p in sources + [config, source_manifest, ready, args.output / 'native-stdout.txt', args.output / f'native-scheduler-{attempt}.txt']])
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        if stdout.exists():
            shutil.copyfile(stdout, args.output / 'native-stdout.txt')
        summary = study_path(args.arm) / 'summary.json'
        if summary.exists():
            shutil.copyfile(summary, args.output / 'failed-native-summary.json')
        raise
    finally:
        (args.output / 'assessment.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
