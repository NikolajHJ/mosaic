"""Metadata guard contracts; run only in allocated CPU suite."""
import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/report_ops.py'
spec=importlib.util.spec_from_file_location('live_report_ops',PATH)
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
PIN='a'*64

@pytest.mark.parametrize('phase',['cpu','after-cpu','render'])
def test_exact_cpu_only_commands(phase):
    args=ops.argv(phase,PIN,'ended(123)' if phase=='after-cpu' else None)
    assert args[args.index('-q')+1]=='hpc'
    assert '-gpu' not in args
    assert args[-3:]==[str(ops.SOURCE),PIN,phase]

@pytest.mark.parametrize('phase,dependency',[('cpu','ended(123)'),('render','ended(123)'),('after-cpu',None),('after-cpu','done(123)'),('training',None)])
def test_wrong_dependencies_and_training_forbidden(phase,dependency):
    with pytest.raises(ValueError):ops.argv(phase,PIN,dependency)

def test_bad_pin():
    with pytest.raises(ValueError):ops.argv('cpu','invalid')

def test_binding_changed_bytes(tmp_path):
    path=tmp_path/'proof';path.write_bytes(b'first');rec=ops.binding(path)
    path.write_bytes(b'other')
    with pytest.raises(ValueError):ops.check_binding(rec)

def test_junit_failed_or_skipped_never_pass(tmp_path):
    path=tmp_path/'raw.xml';path.write_text('<testsuite><testcase classname="a" name="b"><skipped/></testcase></testsuite>')
    with pytest.raises(ValueError):ops.xml_counts(path)

def test_junit_counts_actual_rows(tmp_path):
    path=tmp_path/'raw.xml';path.write_text('<testsuite tests="99"><testcase classname="a" name="b"/><testcase classname="a" name="c"/></testsuite>')
    assert ops.xml_counts(path)=={'a':2}

def test_duplicate_junit_is_failure(tmp_path):
    path=tmp_path/'raw.xml';path.write_text('<testsuite><testcase classname="a" name="b"/><testcase classname="a" name="b"/></testsuite>')
    with pytest.raises(ValueError):ops.xml_counts(path)


def test_report_budget_adds_its_full_allowance_without_borrowing_current_futures():
    prior={'status':'passed','required_bytes':50*ops.GIB,'available_bytes':60*ops.GIB,
           'reserve_bytes':4*ops.GIB,'file_reserve':100000,'forecasts_removed':False,'original_gate_changed':False}
    gate=ops.report_budget(prior,1234)
    assert gate['required_bytes']==51*ops.GIB and gate['available_bytes']==60*ops.GIB-1234
    assert gate['reserve_bytes']==4*ops.GIB and gate['file_reserve']==100000 and gate['forecasts_removed'] is False


@pytest.mark.parametrize('change',[{'status':'failed'},{'forecasts_removed':True}])
def test_report_cannot_adopt_a_failed_or_reduced_current_campaign_gate(change):
    prior={'status':'passed','required_bytes':50*ops.GIB,'available_bytes':60*ops.GIB,
           'reserve_bytes':4*ops.GIB,'file_reserve':100000,'forecasts_removed':False,'original_gate_changed':False}
    prior.update(change)
    with pytest.raises(ValueError):ops.report_budget(prior,0)
