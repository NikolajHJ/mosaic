"""Allocated-only reference promotion/autodiff/dtype controls; no model loads."""
import importlib.util
import os
from pathlib import Path
import pytest

def helpers():
    assert os.environ.get('LSB_JOBID','').isdigit() and os.environ.get('LSB_QUEUE')=='hpc'
    import torch
    path=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-release-26/source/scripts/dbmol_reference.py')
    spec=importlib.util.spec_from_file_location('_real_qualified_FP64_helpers',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return torch,m

def test_parameter_promotion_retains_original_values():
    torch,h=helpers();model=torch.nn.Linear(3,2).eval().requires_grad_(False);before={k:v.detach().clone() for k,v in model.state_dict().items()}
    result=h.promote_model(model,lambda d:str([(n,str(x.dtype),x.tolist()) for n,x in d.items()]))
    assert result['all_castbacks_exact'] and all(torch.equal(model.state_dict()[k].float(),v) for k,v in before.items())

def test_context_promotion_is_independent_and_lossless():
    torch,h=helpers();original={'f':torch.tensor([.25,.5]),'i':torch.tensor([2]),'mask':torch.tensor([True])};copy=h.promote_context(original)
    assert original['f'].dtype==torch.float32 and copy['f'].dtype==torch.float64 and torch.equal(copy['f'].float(),original['f'])
    assert copy['i'].dtype==original['i'].dtype and copy['mask'].dtype==torch.bool

def test_dtype_audit_accepts_actual_full_double_backward():
    torch,h=helpers();x=torch.tensor([.2,.8],dtype=torch.float64,requires_grad=True);mode=h.precision_mode()
    with mode:
        value=(x*x).sum();gradient=torch.autograd.grad(value,x)[0]
    assert gradient.dtype==torch.float64 and torch.allclose(gradient,2*x) and mode.assessment()['narrow_floating_operations']==0

def test_dtype_audit_rejects_a_narrow_operand():
    torch,h=helpers();x=torch.tensor([.2],dtype=torch.float32)
    with pytest.raises(ValueError):
        with h.precision_mode():x+x

@pytest.mark.parametrize('factor',[0.,-1.,1.5])
def test_zero_reversed_scaled_reference_gradients_are_not_indistinguishable(factor):
    torch,h=helpers();x=torch.tensor([.2,.8],dtype=torch.float64,requires_grad=True);scalar=(x*x).sum();g=torch.autograd.grad(scalar,x)[0]
    assert not torch.allclose(g*factor,g,rtol=.15,atol=0)

def test_native_distance_partition_survives_lossless_widening():
    torch,h=helpers()
    from mosaic.small_molecule.boltz2 import dbmol
    from boltz_objective_reference import private_dbmol
    centers=dbmol.native_bin_centers(torch.empty(1,1,1,64,dtype=torch.float32));wide=centers.double()
    assert torch.equal(centers<8,wide<8) and torch.equal(wide.float(),centers)
    private,receipt=private_dbmol(dbmol,centers)
    assert receipt['mathematical_AST_reversal_exact'] and torch.equal(private.native_bin_centers(torch.empty(1,1,1,64,dtype=torch.float64)),wide)
