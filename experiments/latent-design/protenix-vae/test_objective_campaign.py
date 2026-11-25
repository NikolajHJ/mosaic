"""Objective dispatch metadata controls; no native model or scheduler calls."""
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def owner(monkeypatch):
    home=Path(__file__).parent
    spec=importlib.util.spec_from_file_location('base_campaign',home/'campaign.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    monkeypatch.setitem(sys.modules,'base_campaign',base)
    spec=importlib.util.spec_from_file_location('objective_campaign',home/'dbmol_campaign.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('backend,expected,wrapper',[
    ('dbmol','latentmol.design.latent_boltz','latent_boltz_run.sh'),
    ('protenix_tiny','latentmol.design.latent_protenix','protenix_latent_run.sh')])
def test_actual_worker_module_and_wrapper_match_backend(owner,monkeypatch,tmp_path,backend,expected,wrapper):
    c=owner.Campaign.__new__(owner.Campaign);c.control=tmp_path;c.root=tmp_path/'outputs'
    c.path=tmp_path/'plan.json';c.sha='a'*64;c.release=Path('/release')
    c.plan={'scorer_backend':backend,'torch_python':'/torch/python','full_config':'/full.json',
        'native_config':'/native.json','mosaic_python':'/jax/python','archive_sha256':'b'*64}
    c.capacity=lambda:{'status':'passed'}
    c.source=lambda:{'path':'/decoder.pt','sha256':'c'*64}
    commands=[]
    def run(argv,**kwargs):
        commands.append(argv)
        return type('Result',(),{'returncode':0})()
    monkeypatch.setattr(owner.subprocess,'run',run);monkeypatch.setenv('LSB_JOBID','123')
    c.run_stage('main-0001',50000,None,29000)
    command=commands[0]
    assert command[command.index('-m')+1]==expected
    assert command[command.index('--max-updates')+1]=='50000'
    assert ('--mosaic-config' in command)==(backend=='protenix_tiny')
    assert ('--native-config' in command)==(backend=='dbmol')
    assert any(str(arg).endswith(wrapper) for arg in c.launch('main-0001',50000,wall_seconds=29000))
