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
