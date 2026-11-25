"""Allocated fit + pilot + one-molecule 10k prototype, in a fresh output home."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    a = p.parse_args()
    if not os.environ.get('LSB_JOBID') or hashlib.sha256(a.plan.read_bytes()).hexdigest() != a.plan_sha256:
        raise ValueError('Require the allocated experiment plan')
    plan = json.loads(a.plan.read_text())
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2)+'\n')
    save('start.json', dict(job_id=os.environ['LSB_JOBID'], plan_sha256=a.plan_sha256,
                           question='Frozen45 hidden-to-native Protenix chemical embedding prototype'))
    try:
        for stage, config in [('pilot', 'embedding-pilot.json'), ('main', 'embedding-full.json')]:
            command = [sys.executable, '-m', 'latentmol.design.protenix_embedding', '--config',
                'experiments/latent-design/protenix-vae/'+config, '--output', str(output/stage),
                '--mosaic-python', plan['mosaic_python']]
            if stage == 'main':
                pilot = json.loads((output/'pilot/result.json').read_text())
                checks = pilot['design']['technical_checks']
                if pilot['status'] != 'completed' or pilot['design']['completed_step'] != 100 or not all(
                        checks.get(k) is True for k in ('finite_required_binding_path', 'finite_chemistry_path',
                            'frozen_decoder_unchanged', 'frozen_native_close_completed', 'decoder_hidden_mapping_path')):
                    raise ValueError('Pilot did not complete its finite/frozen hidden-path checks')
                command += ['--mapping', str(output/'pilot/mapping.pt')]
            with (output/(stage+'.log')).open('x') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=max(1, plan['internal_seconds'] - (time.monotonic()-started)))
            if result.returncode:
                raise RuntimeError(stage+' failed; inspect its retained log and saved state')
            value = json.loads((output/stage/'result.json').read_text())
            if value['status'] != 'completed':
                raise RuntimeError(stage+' stopped before its requested endpoint; saved state retained')
        save('result.json', dict(status='completed_10000', job_id=os.environ['LSB_JOBID'],
             elapsed_seconds=time.monotonic()-started, main=str(output/'main/result.json')))
    except BaseException as error:
        save('failure.json', dict(status='incomplete', error=str(error), elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
