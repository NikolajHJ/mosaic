"""Allocated Torch CPU controls, not a pretrained native qualification."""
import importlib.util
import os
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('dbmol_reference_fixture',
    Path(__file__).resolve().parents[1] / 'scripts/dbmol_reference.py')
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


@pytest.fixture(autouse=True)
def allocated():
    assert str(os.environ.get('LSB_JOBID', '')).isdigit()


@pytest.mark.parametrize('dtype_name', ['float16', 'bfloat16', 'float32'])
def test_every_explicit_narrowing_request_is_audited_and_promoted(dtype_name):
    import torch
    original = torch.tensor([1, -2, 3], dtype=torch.int32)
    mode = r.precision_mode()
    with mode:
        result = torch.asinh(original.to(getattr(torch, dtype_name)))
    assert result.dtype == torch.float64
    assert torch.equal(result, torch.asinh(original.double()))
    assert sum(mode.promoted_requests.values()) >= 1
    assert mode.assessment()['narrow_floating_operations'] == 0
    assert original.dtype == torch.int32


@pytest.mark.parametrize('dtype_name', ['float16', 'bfloat16', 'float32'])
def test_narrow_inputs_cannot_be_hidden_by_later_output_cast(dtype_name):
    import torch
    original = torch.ones(3, dtype=getattr(torch, dtype_name))
    with pytest.raises(ValueError, match='Narrow floating reference input'):
        with r.precision_mode(): original.double()


def test_empty_operation_audit_cannot_claim_full_double():
    with pytest.raises(ValueError, match='No actual reference operations'):
        r.precision_mode().assessment()


def test_model_promotion_preserves_parameters_integer_buffers_and_castback():
    import torch
    model = torch.nn.Linear(3, 2).float().eval().requires_grad_(False)
    model.register_buffer('indices', torch.arange(3))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    def digest(state):
        import hashlib
        h = hashlib.sha256()
        for k, v in sorted(state.items()): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    audit = r.promote_model(model, digest)
    assert audit['all_castbacks_exact']
    for k, v in before.items(): assert torch.equal(model.state_dict()[k].to(v.dtype), v)
    assert model.weight.dtype == torch.float64 and model.indices.dtype == torch.int64


def test_context_promotion_leaves_original_storage_and_integer_features_unchanged():
    import torch
    original = {'features': {'real': torch.tensor([.1, .2], dtype=torch.float32),
                             'ids': torch.arange(2)}, 'coordinates': torch.ones(2, 3)}
    promoted = r.promote_context(original)
    assert original['features']['real'].dtype == torch.float32
    assert promoted['features']['real'].dtype == torch.float64
    assert torch.equal(promoted['features']['real'].float(), original['features']['real'])
    assert torch.equal(promoted['features']['ids'], original['features']['ids'])
    assert promoted['features']['ids'].data_ptr() != original['features']['ids'].data_ptr()


@pytest.mark.parametrize('scale', [1., -1., 3.])
def test_double_native_style_linear_gradient_matches_independent_forwards(scale):
    import torch
    model = torch.nn.Linear(3, 1).double().eval().requires_grad_(False)
    x = torch.tensor([.2, .3, .5], dtype=torch.float64, requires_grad=True)
    with r.precision_mode():
        y = scale * model(x).sum(); gradient, = torch.autograd.grad(y, (x,))
        for epsilon in (.001, .0003):
            direction = torch.tensor([-1., 1., 0.], dtype=torch.float64)
            finite = (scale * model(x.detach() + epsilon * direction).sum()
                      - scale * model(x.detach() - epsilon * direction).sum()) / (2 * epsilon)
            assert torch.allclose(finite, gradient @ direction, atol=1e-10, rtol=1e-8)


def test_actual_plan_and_reference_startup_have_identical_owned_output():
    import json
    from pathlib import Path
    plan = json.loads((Path(__file__).resolve().parents[1] / 'plan.json').read_bytes())
    assert r.startup_output_path() == Path(plan['root']) / 'cpu/payload/reference-startup.json'
    assert 'dbmol-endpoint-recovery-13' in str(r.startup_output_path())


@pytest.mark.parametrize('wrong', [
    '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/dbmol-endpoint-recovery-10/cpu/payload/reference-startup.json',
    '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/dbmol-endpoint-recovery-12/cpu/payload/reference-startup.json',
    'reference-startup.json',
    '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/dbmol-endpoint-recovery-13/gpu/payload/reference-startup.json'])
def test_actual_startup_parser_rejects_wrong_owned_output_before_loading_models(monkeypatch, wrong):
    import sys
    monkeypatch.setenv('LSB_JOBID', '123')
    monkeypatch.setattr(sys, 'argv', ['reference', '--context', 'not-opened',
        '--context-sha256', '0' * 64, '--checkpoint', 'not-opened',
        '--model-digest', '0' * 64, '--startup-only', '--startup-output', wrong])
    with pytest.raises(ValueError, match='Closed startup output scope differs'):
        r.main()
