"""Allocated, isolated dependency bootstrap for the official Tiny native gate."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path('/work3/s234818/latentmol/protenix_occupancy')
SOURCE=ROOT/'assets/Protenix-510faf9f5cc8fbbbb8d9055934d3fa7e6877c76d'
PACKAGES=('ml_collections==1.1.0','optree==0.17.0','biotite==1.4.0','biotraj==1.2.2','absl-py==2.3.1','fair-esm==2.0.0')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    if not os.environ.get('LSB_JOBID'):raise RuntimeError('Run only in allocated DTU CPU job')
    output=ROOT/'environment-attempt003'
    output.mkdir(exist_ok=False)
    result=dict(status='running',jobid=os.environ['LSB_JOBID'],packages=PACKAGES,
        scope='Isolated optional dependencies; base Boltz environment is not modified')
    try:
        wheels=ROOT/'wheels-attempt001'
        result['wheel_sha256']={p.name:sha(p) for p in sorted(wheels.glob('*.whl'))}
        if len(result['wheel_sha256'])!=len(PACKAGES):raise RuntimeError('Unexpected wheel cohort')
        command=['/zhome/95/e/206121/.local/bin/uv','pip','install','--python',sys.executable,'--no-deps','--no-index',
            '--find-links',str(wheels),'--target',str(output/'deps'),*PACKAGES]
        done=subprocess.run(command,capture_output=True,text=True,timeout=300)
        result['install']=dict(command=command,returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)
        if done.returncode:raise RuntimeError('Dependency installation failed')
        env=dict(os.environ,PYTHONPATH=str(output/'deps')+':'+str(SOURCE),
            LAYERNORM_TYPE='torch',PROTENIX_ROOT_DIR=str(ROOT/'data'))
        code='''import json,torch,biotite,optree,ml_collections
from protenix.model.protenix import Protenix
from protenix.data.core.featurizer import Featurizer
from protenix.data.inference.json_to_feature import SampleDictToFeatures
print(json.dumps(dict(torch=torch.__version__,biotite=biotite.__version__,cuda=torch.cuda.is_available(),native_imports=True)))'''
        done=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=env,timeout=180)
        result['native_imports']=dict(returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)
        if done.returncode:raise RuntimeError('Native import readiness failed')
        result['distributions']={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
        result.update(status='passed',runtime_note='Uses existing torch2.5.1+cu121; official repo requirements name2.7.1. Native parity/FD on this recorded runtime remain mandatory.')
    except BaseException as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        with (output/'environment-result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
