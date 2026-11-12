import sys

import numpy as np
import pytest

from mosaic_defog import DeFoGTeacher, Graph, TeacherError
from mosaic_defog.teacher import (
    GUACAMOL_CHECKPOINT_SHA256,
    GUACAMOL_SOURCE_REVISION,
    PROTOCOL,
)


FAKE_WORKER = """
import sys,json,hashlib,time,os
mode=os.environ.get('DEFOG_TEST_MODE','normal')
def emit(r): print(json.dumps(r),flush=True)
def digest(v):
    payload=json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()
if mode=='startup_eof':
    print("ModuleNotFoundError: No module named 'torch'",file=sys.stderr,flush=True)
    sys.exit(1)
emit({'status':'ready','protocol':PROTOCOL,'elements':['C','N','O','S','F','Cl'],'feature_policy':'hard_aux_soft_core','source_revision':REVISION,'checkpoint_sha256':CHECKPOINT,'state_digest':'frozen'})
queries=0
for line in sys.stdin:
    r=json.loads(line)
    if r['action']=='frozen':
        emit({'id':r['id'],'status':'passed',
              'state_digest':'changed' if mode=='mutation' else 'frozen',
              'queries':queries,'forwards':queries})
        continue
    if mode=='timeout':time.sleep(10)
    if mode=='eof':sys.exit(1)
    if mode=='non_object':emit([]);continue
    if mode=='invalid_json':print('broken',flush=True);continue
    if mode=='failed':emit({'status':'failed','error':'model failed'});continue
    target={'X':r['X'],'E':r['E']}
    reply={'id':r['id']+(mode=='wrong_id'),'status':'passed','target':target,
           'target_sha256':digest(target),'graph_sha256':r['graph_sha256'],
           'time':r['time'],'forward_calls':2 if mode=='extra_forward' else 1}
    if mode=='wrong_graph':reply['graph_sha256']='wrong'
    if mode=='wrong_target':reply['target_sha256']='wrong'
    queries+=1
    emit(reply)
"""


@pytest.fixture
def worker(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "src/graph_discrete_flow_model.py").touch()
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    fake = tmp_path / "worker.py"
    fake.write_text(
        f"PROTOCOL={PROTOCOL!r}\nREVISION={GUACAMOL_SOURCE_REVISION!r}\nCHECKPOINT={GUACAMOL_CHECKPOINT_SHA256!r}\n"
        + FAKE_WORKER
    )
    import mosaic_defog.teacher as module

    original = module.subprocess.Popen
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda command, **kw: original([sys.executable, "-u", str(fake)], **kw),
    )
    return source, checkpoint


def graph():
    return Graph(np.full((3, 6), 1 / 6, np.float32), np.full((3, 5), 1 / 5, np.float32))


@pytest.mark.parametrize("retained_stderr", [False, True])
def test_real_pipe_transport_and_context_cleanup(worker, tmp_path, retained_stderr):
    stderr = tmp_path / "worker.stderr" if retained_stderr else None
    with DeFoGTeacher(*worker, stderr=stderr) as teacher:
        before = tuple(v.copy() for v in graph())
        prediction = teacher.predict(graph(), time=0.8)
        assert prediction.X.shape == (3, 6)
        for a, b in zip(graph(), before, strict=True):
            np.testing.assert_array_equal(a, b)
        assert teacher.verify_frozen()["queries"] == 1
        process = teacher._process
    assert process.poll() is not None
    with pytest.raises(TeacherError, match="closed"):
        teacher.predict(graph())


@pytest.mark.parametrize(
    "mode",
    [
        "wrong_id",
        "wrong_graph",
        "wrong_target",
        "extra_forward",
        "eof",
        "timeout",
        "non_object",
        "invalid_json",
        "failed",
    ],
)
def test_protocol_failures_stop_the_affected_use(worker, monkeypatch, mode):
    monkeypatch.setenv("DEFOG_TEST_MODE", mode)
    teacher = DeFoGTeacher(*worker, timeout=1)
    process = teacher._process
    with pytest.raises(TeacherError):
        teacher.predict(graph())
    assert process.poll() is not None
    assert teacher._closed
    assert not teacher._reader.is_alive()


@pytest.mark.parametrize("retained_stderr", [False, True])
def test_startup_exit_reports_worker_diagnostics(
    worker, monkeypatch, tmp_path, retained_stderr
):
    monkeypatch.setenv("DEFOG_TEST_MODE", "startup_eof")
    stderr = tmp_path / "startup.stderr" if retained_stderr else None
    with pytest.raises(TeacherError, match="No module named 'torch'"):
        DeFoGTeacher(*worker, stderr=stderr)


def test_frozen_mutation_is_not_hidden(worker, monkeypatch):
    monkeypatch.setenv("DEFOG_TEST_MODE", "mutation")
    teacher = DeFoGTeacher(*worker)
    with pytest.raises(TeacherError, match="Frozen"):
        teacher.verify_frozen()
    assert teacher._process.poll() is not None


def test_invalid_input_never_reaches_worker(worker):
    with DeFoGTeacher(*worker) as teacher:
        with pytest.raises(ValueError):
            teacher.predict(Graph(np.full((3, 6), np.nan), np.full((3, 5), 0.2)))
        assert teacher.queries == 0


def test_asset_errors_precede_process_start(tmp_path):
    with pytest.raises(FileNotFoundError):
        DeFoGTeacher(tmp_path, tmp_path / "missing.ckpt")
