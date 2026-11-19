from pathlib import Path
import shutil

from mosaic_defog import DeFoGTeacher
import mosaic_defog.teacher as transport

# Reuse the actual pipe fixture; its child replaces only the native model.
from test_transport import worker  # noqa: F401


def test_worker_uses_its_own_dependencies_even_for_wheel_install(
    worker,  # noqa: F811 -- imported pytest fixture
    monkeypatch,
):
    launch = transport.subprocess.Popen
    observed = []

    def inspect(command, **options):
        roots = options["env"]["PYTHONPATH"].split(transport.os.pathsep)
        coordinator_packages = str(Path(transport.__file__).resolve().parents[1])
        assert coordinator_packages not in roots
        assert roots == [str(worker[0]), str(worker[0] / "src")]
        assert Path(command[2]).name == "_worker_entry.py"
        assert "-m" not in command
        observed.append(True)
        return launch(command, **options)

    monkeypatch.setattr(transport.subprocess, "Popen", inspect)
    with DeFoGTeacher(*worker):
        pass
    assert observed == [True]


def test_real_entry_ignores_coordinator_site_packages(tmp_path, monkeypatch):
    coordinator_packages = tmp_path / "coordinator-site-packages"
    copied_package = coordinator_packages / "mosaic_defog"
    shutil.copytree(
        Path(transport.__file__).parent,
        copied_package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    # This incompatible dependency must never enter the teacher's import roots.
    (coordinator_packages / "numpy.py").write_text(
        "raise RuntimeError('coordinator dependency leaked into worker')\n"
    )
    (copied_package / "_worker.py").write_text(
        """
import json
import sys

def main():
    import mosaic_defog
    import numpy
    from .teacher import (
        PROTOCOL, GUACAMOL_SOURCE_REVISION, GUACAMOL_CHECKPOINT_SHA256,
    )
    print(json.dumps({
        'status': 'ready', 'protocol': PROTOCOL,
        'source_revision': GUACAMOL_SOURCE_REVISION,
        'checkpoint_sha256': GUACAMOL_CHECKPOINT_SHA256,
        'elements': ['C', 'N', 'O', 'S', 'F', 'Cl'],
        'feature_policy': 'hard_aux_soft_core', 'state_digest': 'frozen',
        'package_file': mosaic_defog.__file__, 'numpy_file': numpy.__file__,
    }), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        assert request['action'] == 'frozen'
        print(json.dumps({
            'id': request['id'], 'status': 'passed',
            'state_digest': 'frozen', 'queries': 0, 'forwards': 0,
        }), flush=True)
    return 0
"""
    )
    source = tmp_path / "upstream"
    (source / "src").mkdir(parents=True)
    (source / "src/graph_discrete_flow_model.py").touch()
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.touch()
    monkeypatch.setattr(transport, "__file__", str(copied_package / "teacher.py"))
    with DeFoGTeacher(source, checkpoint) as teacher:
        assert Path(teacher.identity["package_file"]) == copied_package / "__init__.py"
        assert not Path(teacher.identity["numpy_file"]).is_relative_to(
            coordinator_packages
        )
