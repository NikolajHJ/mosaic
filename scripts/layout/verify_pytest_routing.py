"""Exercise real pytest collection with synthetic stdlib tests only."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

root = Path(__file__).resolve().parents[2]
output = args.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
fake = Path(tempfile.mkdtemp(prefix="pytest-routing-", dir=output.parent)).resolve()
assert fake.is_relative_to(output.parent.resolve())
out = fake
for directory in ['scripts/layout', 'tests/core', 'experiments/family/study/tests']:
    (fake / directory).mkdir(parents=True)
shutil.copyfile(root / 'scripts/layout/experiment_layout.py', fake / 'scripts/layout/experiment_layout.py')
mapping = {'tests/test_alpha.py': {'path': 'tests/core/test_alpha.py'},
           'tests/test_beta.py': {'path': 'experiments/family/study/tests/test_beta.py'}}
(fake / 'experiments/migration-map.json').write_text(json.dumps({'files': mapping}), encoding='utf-8')
(fake / 'pyproject.toml').write_text('''[tool.pytest.ini_options]
testpaths = ["tests/core", "experiments"]
pythonpath = ["tests/core", "experiments/family/study/tests"]
''', encoding='utf-8')
(fake / 'tests/core/test_alpha.py').write_text('''from pathlib import Path
import sys
VALUE = "original"
def value():
    return VALUE
def test_canonical_globals(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "VALUE", "patched")
    assert value() == "patched"
    assert Path(__file__).parent.name == "core"
''', encoding='utf-8')
(fake / 'experiments/family/study/tests/test_beta.py').write_text('''from test_alpha import value
def test_shared_helper_import():
    assert value() == "original"
''', encoding='utf-8')
for old, item in mapping.items():
    (fake / old).write_text('''from pathlib import Path
import sys
root = next(p for p in Path(__file__).resolve().parents if (p / "scripts/layout/experiment_layout.py").is_file())
sys.path.insert(0, str(root / "scripts/layout"))
from experiment_layout import execute_legacy
execute_legacy(globals(), root / %r)
''' % item['path'], encoding='utf-8')
env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'addopts=']

def run(name, paths):
    result = subprocess.run(command + paths, cwd=fake, env=env, text=True, capture_output=True)
    (out / (name + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
    return result

before = run('pytest-synthetic-before-hook', ['tests/test_alpha.py'])
assert before.returncode != 0 and 'import file mismatch' in before.stdout, before.stdout + before.stderr
shutil.copyfile(root / 'tests/conftest.py', fake / 'tests/conftest.py')
explicit = run('pytest-synthetic-explicit-aliases', list(mapping))
assert explicit.returncode == 0 and '2 passed' in explicit.stdout, explicit.stdout + explicit.stderr
default = run('pytest-synthetic-default-discovery', [])
assert default.returncode == 0 and '2 passed' in default.stdout, default.stdout + default.stderr
result = {'observed_utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Real pytest on two synthetic stdlib-only modules; no Mosaic/model/numerical test imports or HPC.',
          'regression_without_hook': 'expected import-file mismatch reproduced',
          'explicit_legacy_selection': '2 passed', 'default_canonical_discovery': '2 passed',
          'function_globals_monkeypatch_and_source_paths': 'passed', 'cross_folder_helper_import': 'passed',
          'import_mismatch_checks_disabled': False, 'synthetic_source_and_logs': str(fake), 'status': 'passed'}
output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
