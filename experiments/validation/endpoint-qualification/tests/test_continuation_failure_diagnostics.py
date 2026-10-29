"""Two portable metadata controls and six allocated numerical controls."""
import ast
import importlib.util
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/continuation_failure_diagnostics.py'
S=importlib.util.spec_from_file_location('_failure_diagnostic_tests',P)
d=importlib.util.module_from_spec(S);S.loader.exec_module(d)


def test_metadata_cap_and_read_only_scope():assert d.CAP==128*1024**2


def test_metadata_no_optimizer_or_admission_dispatch():
    tree=ast.parse(P.read_bytes())
    calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert not calls & {'run_chemical_updates','run_updates','policy.evaluate','d.dispatch','submit'}


@pytest.mark.parametrize('change',[False,True])
def test_replay_exact_and_changed_measurements(change):
    np=d.numpy();a={'gradient':np.asarray([1.,2.],np.float32)};b={'gradient':np.asarray([1.,2.+int(change)],np.float32)}
    r=d.replay_difference(a,b)['gradient'];assert r['exact_equal'] is (not change) and r['maximum_absolute_difference']==int(change)


def test_dtype_change_cannot_be_reported_as_roundoff():
    np=d.numpy()
    with pytest.raises(ValueError):d.replay_difference({'g':np.asarray([1.],np.float32)},{'g':np.asarray([1.],np.float64)})


@pytest.mark.parametrize('sign',[-1,1])
def test_component_secants_use_original_weights_and_keep_failed_outcome(sign):
    def endpoint(scale):
        return [{'value':scale*13.5,'native_components':{n:{'value':scale*v} for n,v in [('affinity',1.),('contact',2.),('confidence',3.)]}} for _ in range(3)]
    r={'status':'failed','frozen_unchanged':True,'directions':[{'field':'E_upper','row':1,'source':0,'target':1,'pair_passed':False,
       'probes':[{'evaluated':True,'endpoint_primal_repeats':[endpoint(0),endpoint(sign)],'epsilon':.001,
                  'outcome':'failed','predicted_scalar_difference':12.*sign,'resolution_floor':1e-6}]}]}
    x=d.secant_attribution(r)[0];assert x['weighted_component_differences']=={'affinity':1.*sign,'contact':5.*sign,'confidence':7.5*sign}
    assert x['endpoint_scalar_recomposition_max_error']==0 and x['original_pair_passed'] is False and x['new_admission_authorized'] is False


def test_successful_record_cannot_replace_original_failed_input():
    with pytest.raises(ValueError):d.secant_attribution({'status':'passed','frozen_unchanged':True})
