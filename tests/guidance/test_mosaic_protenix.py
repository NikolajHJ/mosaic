"""Model-free tests of the actual subprocess protocol and Torch VJP boundary."""
import ast
from pathlib import Path
import sys

import pytest
import torch

from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance import mosaic_protenix
from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer
from latentmol.representation import DecoderOutput, GraphSpec


WORKER = r'''
import json,sys,time
config={}
for line in sys.stdin:
 r=json.loads(line);base={'protocol':'mosaic-protenix-probability-vjp-v1','request_id':r['request_id'],'status':'ok'}
 if r['op']=='init':
  config=r['config'];base.update(identity={'frozen':'toy'},n_atoms=r['n_atoms'])
 elif r['op']=='close':base.update(frozen_unchanged=not config.get('mutated',False))
 else:
  if config.get('hang'):time.sleep(30)
  loss=0.;gradients={}
  def visit(value,scale):
   global loss
   if not isinstance(value[0],list):
    weights=[scale*(i+1) for i in range(len(value))]
    loss+=sum(p*w for p,w in zip(value,weights));return weights
   return [visit(row,scale) for row in value]
  for name,scale in [('atoms',2.),('charges',3.),('bonds',5.)]:gradients[name]=visit(r['arrays'][name],scale)
  if config.get('bad_gradient'):gradients['atoms']=[[float('nan')]]
  base.update(loss=loss,gradients=gradients,aux={'contact_score':-loss})
 print(json.dumps(base),flush=True)
 if r['op']=='close':
  if config.get('wait_for_eof'):sys.stdin.read()
  if config.get('hang_after_close'):time.sleep(30)
  sys.exit(config.get('close_exit_code',0))
'''


def graph_input():
    spec=GraphSpec(max_nodes=4)
    active=torch.tensor([True,False,True,False])
    generator=torch.Generator().manual_seed(4517)
    fields=[torch.randn(shape,generator=generator,requires_grad=True) for shape in
            [(1,4,10),(1,4,5),(1,4,4,5)]]
    decoded=DecoderOutput(*fields)
    return spec,active,fields,graph_from_decoder(decoded,spec,fixed_active=active)


def scorer(tmp_path,spec,active,**config):
    return MosaicProtenixScorer(spec,active,command=[sys.executable,'-u','-c',WORKER],
        config=config,stderr_path=tmp_path/'native.stderr',timeout=.2 if config.get('hang') else 5,startup_timeout=5)


def test_real_transport_vjp_matches_direct_torch_chain(tmp_path):
    spec,active,leaves,graph=graph_input()
    expected=graph.atoms[0,active,1:].mul(2*torch.arange(1,10)).sum()
    expected=expected+graph.charges[0,active].mul(3*torch.arange(1,6)).sum()
    expected=expected+graph.bonds[0,active][:,active].mul(5*torch.arange(1,6)).sum()
    expected_grad=torch.autograd.grad(expected,leaves,retain_graph=True)
    with scorer(tmp_path,spec,active) as native:
        loss,aux=native(graph)
        actual=torch.autograd.grad(loss,leaves)
        assert native.identity=={'frozen':'toy'}
        torch.testing.assert_close(loss,expected)
        assert aux['contact_score']==pytest.approx(-float(expected.detach()),rel=1e-6)
        for a,e in zip(actual,expected_grad):
            torch.testing.assert_close(a,e)
            assert torch.isfinite(a).all()
        assert actual[0].norm()>0 and actual[1].norm()>0 and actual[2].norm()>0
    assert native._closed


def test_rejects_changed_fixed_slots_before_native_call(tmp_path):
    spec,active,_,graph=graph_input()
    with scorer(tmp_path,spec,active) as native:
        graph.atoms=graph.atoms.detach().clone();graph.atoms[0,0]=0;graph.atoms[0,0,0]=1
        with pytest.raises(ValueError,match='slot identity'):
            native(graph)


def test_resident_worker_can_score_an_explicit_same_count_branch(tmp_path):
    spec, active, leaves, graph = graph_input()
    other = ~active
    other_graph = graph_from_decoder(DecoderOutput(*leaves), spec, fixed_active=other)
    expected = other_graph.atoms[0, other, 1:].mul(2 * torch.arange(1, 10)).sum()
    expected = expected + other_graph.charges[0, other].mul(3 * torch.arange(1, 6)).sum()
    expected = expected + other_graph.bonds[0, other][:, other].mul(5 * torch.arange(1, 6)).sum()
    with scorer(tmp_path, spec, active) as native:
        with pytest.raises(ValueError, match="prepared ligand count"):
            native.score_at(other_graph, torch.ones_like(other))
        with pytest.raises(ValueError, match="slot identity"):
            native(other_graph)
        loss, _ = native.score_at(other_graph, other)
        torch.testing.assert_close(loss, expected)
        assert all(gradient.isfinite().all() for gradient in torch.autograd.grad(loss, leaves))
        assert native(graph)[0].isfinite()


def test_timeout_terminates_owned_native_process(tmp_path):
    spec,active,_,graph=graph_input()
    native=scorer(tmp_path,spec,active,hang=True)
    with pytest.raises(TimeoutError,match='deadline'):
        native(graph)
    assert native._closed and native._process.poll() is not None


def test_frozen_native_identity_violation_is_not_silent(tmp_path):
    spec,active,_,_=graph_input()
    native=scorer(tmp_path,spec,active,mutated=True)
    with pytest.raises(RuntimeError,match='identity changed'):
        native.close()
    assert native._closed


def test_closing_command_pipe_allows_graceful_eof_exit(tmp_path):
    spec,active,_,_=graph_input()
    native=scorer(tmp_path,spec,active,wait_for_eof=True)
    native.close()
    assert native.shutdown_status=="graceful" and native._process.returncode==0


def test_acknowledged_close_timeout_reaps_worker_with_warning(tmp_path,monkeypatch):
    spec,active,_,_=graph_input()
    monkeypatch.setattr(mosaic_protenix,"_SHUTDOWN_GRACE_SECONDS",.2)
    native=scorer(tmp_path,spec,active,hang_after_close=True)
    with pytest.warns(RuntimeWarning,match="acknowledged unchanged frozen state"):
        native.close()
    assert native._closed and native._process.poll() is not None
    assert native.shutdown_status=="terminated_after_close_ack"


def test_nonzero_native_exit_after_ack_still_fails(tmp_path):
    spec,active,_,_=graph_input()
    native=scorer(tmp_path,spec,active,close_exit_code=7)
    with pytest.raises(RuntimeError,match=r"exited unsuccessfully \(7\)"):
        native.close()
    assert native._closed and native._process.returncode==7


def test_nonfinite_or_mismatched_vjp_rejected(tmp_path):
    spec,active,_,graph=graph_input()
    with scorer(tmp_path,spec,active,bad_gradient=True) as native:
        with pytest.raises(ValueError,match='VJP'):
            native(graph)


def test_worker_module_has_no_torch_import():
    path=Path(__file__).parents[2]/'src/latentmol/guidance/mosaic_protenix_worker.py'
    tree=ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):assert all(not n.name.startswith('torch') for n in node.names)
        if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('torch')
