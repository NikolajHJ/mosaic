"""Model-free test of learned-input transport and the actual Torch VJP."""
import sys

import pytest
import torch

from latentmol.guidance.mosaic_protenix_embedding import MosaicProtenixEmbeddingScorer, HEADS
from latentmol.representation import GraphSpec

WORKER = r'''
import json,sys
widths={'atom_element':3,'atom_charge':3,'pair_presence':2}
for line in sys.stdin:
 r=json.loads(line);v={'protocol':'mosaic-protenix-chemical-embedding-vjp-v1','request_id':r['request_id'],'status':'ok'}
 if r['op']=='init':
  config=r['config'];v.update(identity={'frozen':'toy'},n_atoms=r['n_atoms'],projection_bank={k:{'kind':k,'width':w,'prototypes':[]} for k,w in widths.items()})
 elif r['op']=='close':v['frozen_unchanged']=True
 else:
  loss=0
  def visit(x,scale):
   global loss
   if isinstance(x,list):return [visit(y,scale) for y in x]
   loss+=x*scale;return scale
  keys=('atoms','charges','bonds') if r['op']=='evaluate_probabilities' else widths
  gradients={k:visit(r['arrays'][k],i+1) for i,k in enumerate(keys)}
  if config.get('bad'):gradients['atom_element']=[[float('nan')]]
  v.update(loss=loss,gradients=gradients,aux={'contact_score':-loss})
 print(json.dumps(v),flush=True)
 if r['op']=='close':break
'''

def create(tmp_path, bad=False):
    return MosaicProtenixEmbeddingScorer(GraphSpec(max_nodes=4),[True,False,True,False],
        command=[sys.executable,'-u','-c',WORKER],config={'bad':bad},stderr_path=tmp_path/'native.err',
        timeout=5,startup_timeout=5)

def values():
    return {k:torch.randn(shape,requires_grad=True) for k,shape in zip(HEADS,
        ((1,4,3),(1,4,3),(1,4,4,2)))}

def test_embedding_transport_pullback_uses_only_active_slots(tmp_path):
    fields=values();mask=torch.tensor([True,False,True,False])
    expected=sum((i+1)*(v[0,mask][:,mask] if k=='pair_presence' else v[0,mask]).sum()
                 for i,(k,v) in enumerate(fields.items()))
    expected_grad=torch.autograd.grad(expected,tuple(fields.values()),retain_graph=True)
    with create(tmp_path) as scorer:
        loss,aux=scorer(fields)
        actual=torch.autograd.grad(loss,tuple(fields.values()))
        torch.testing.assert_close(loss,expected)
        assert scorer.bank is scorer.projection_bank
        assert aux['contact_score']==pytest.approx(-float(expected.detach()),abs=1e-5)
        for a,e in zip(actual,expected_grad):torch.testing.assert_close(a,e)

def test_nonfinite_native_embedding_gradient_fails(tmp_path):
    with create(tmp_path,bad=True) as scorer:
        with pytest.raises(ValueError,match='VJP'):scorer(values())

def test_selected_embedding_rows_are_accepted_without_second_mask(tmp_path):
    fields={k:torch.randn(shape,requires_grad=True) for k,shape in zip(HEADS,
        ((1,2,3),(1,2,3),(1,2,2,2)))}
    with create(tmp_path) as scorer:
        loss,_=scorer(fields)
        gradients=torch.autograd.grad(loss,tuple(fields.values()))
        for i,(g,v) in enumerate(zip(gradients,fields.values())):
            torch.testing.assert_close(g,torch.full_like(v,i+1))

def test_relaxed_control_uses_probability_vjp_in_same_process(tmp_path):
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.representation import DecoderOutput
    spec=GraphSpec(max_nodes=4);mask=torch.tensor([True,False,True,False])
    logits=DecoderOutput(torch.randn(1,4,10,requires_grad=True),
        torch.randn(1,4,5,requires_grad=True),torch.randn(1,4,4,5,requires_grad=True))
    graph=graph_from_decoder(logits,spec,temperature=.7,fixed_active=mask)
    expected=(graph.atoms[0,mask,1:].sum()+2*graph.charges[0,mask].sum()+3*graph.bonds[0,mask][:,mask].sum())
    with create(tmp_path) as scorer:
        loss,_=scorer.score_graph(graph)
        torch.testing.assert_close(loss,expected)
        actual=torch.autograd.grad(loss,(logits.atoms,logits.charges,logits.bonds),retain_graph=True)
        reference=torch.autograd.grad(expected,(logits.atoms,logits.charges,logits.bonds))
        for a,e in zip(actual,reference):torch.testing.assert_close(a,e,atol=1e-6,rtol=1e-5)
