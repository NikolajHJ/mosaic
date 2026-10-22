"""Seal the declared exact-initialization fix from authenticated parent622."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TAG = '20261022-brd4-jq1-25-fixed-lrreset-04'


def main():
    helpers = runpy.run_path(str(HERE.parent / '20261022-brd4-jq1-25-fixed-lrreset-03/prepare_sources.py'))
    original, parent = helpers['parent_archive']('20261022-brd4-jq1-25-fixed-lrreset-03',
        '31809b975f0387db4ad15b6530c443afc1220fd501b6b8ac62e147bbcf739b01')
    assert parent == 'ff82857ecd4a985d0ba6cc16a87f1bf7b7bb2c7ffdbde18a183bbd63670ff567'
    files = dict(original)
    replace = helpers['replace_once']
    for name in ['scripts/fixed25_lrreset_ops.py', 'scripts/fixed25_lrreset_storage.py', 'tests/test_fixed25_lrreset_storage.py']:
        files[name] = files[name].replace(b'20261022-brd4-jq1-25-fixed-lrreset-03', TAG.encode()).replace(b'restart-lrreset-release-03', b'restart-lrreset-release-04').replace(b'restart-lrreset-03', b'restart-lrreset-04')
    name = 'scripts/fixed25_lrreset_storage.py'
    text = files[name].decode()
    text = replace(text, "    Path('/work3/s234818/mosaic/20261022-brd4-jq1-lrboost-analysis-02.tar.gz'),",
        "    Path('/work3/s234818/mosaic/20261022-brd4-jq1-lrboost-analysis-02.tar.gz'),\n    Path('/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-02-launcher-03'),\n    Path('/work3/s234818/mosaic/20261022-brd4-jq1-lrboost-analysis-02-launcher-03.tar.gz'),")
    files[name] = text.encode()
    name = 'tests/test_fixed25_lrreset_storage.py'
    files[name] = files[name].replace(b'len(storage.CONCURRENT_DIAGNOSTIC_PATHS) == 3', b'len(storage.CONCURRENT_DIAGNOSTIC_PATHS) == 5')
    name = 'examples/brd4_25_atoms.py'
    text = files[name].decode()
    text = replace(text, 'PROTENIX_LR03_VERSION, FIXED25_VERSIONS', 'PROTENIX_LR03_VERSION, FIXED25_VERSIONS, FIXED25_LRRESET_VERSION')
    text = replace(text, '        initial = GraphLogits.from_graph(initialize_graph(hard, config["initialization"]))',
        '''        logit_type = GraphLogits
        if arm == "protenix" and config["version"] == FIXED25_LRRESET_VERSION:
            from protenix_initialization import ProtenixReplayLogits, VERSION as INITIALIZATION_EXECUTION
            logit_type = ProtenixReplayLogits
            report["initialization_execution"] = {
                "version": INITIALIZATION_EXECUTION,
                "atoms": "unchanged FP32 jax.nn.softmax on CPU; unchanged probability copy to scorer device",
                "bonds": "unchanged FP32 jax.nn.softmax on scorer device",
                "scope": "same placement for every state and trial; no initial-point special case or tolerance",
            }
        initial = logit_type.from_graph(initialize_graph(hard, config["initialization"]))''')
    files[name] = text.encode()
    name = 'src/mosaic/small_molecule/chemistry/trajectory.py'
    files[name] = replace(files[name].decode(), '                proposal = GraphLogits(',
        '                proposal = type(evaluation.logits)(').encode()
    name = 'scripts/hpc/fixed25_lrreset_stage.lsf'
    files[name] = replace(files[name].decode(), 'export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false',
        'export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false').encode()
    name = 'scripts/hpc/fixed25_lrreset_cpu.lsf'
    files[name] = replace(files[name].decode(), 'tests/test_fixed25_lrreset_storage.py)',
        'tests/test_fixed25_lrreset_storage.py tests/test_protenix_initialization.py)').encode()
    name = 'scripts/fixed25_lrreset_ops.py'
    text = files[name].decode()
    text = replace(text, "    modules = {part for c in cases for part in c.get('classname', '').split('.')}",
        "    modules = {part for c in cases for part in c.get('classname', '').split('.')}\n    require(sum('test_protenix_initialization' in c.get('classname', '').split('.') for c in cases) == 4,\n            'Exact initialization/trajectory subclass regressions did not all execute')")
    text = replace(text, "    if arm != 'protenix':\n        require(summary['context_file_sha256']", 
        "    if arm == 'protenix':\n        require(summary.get('initialization_execution', {}).get('version') == 'protenix-fp32-cpu-atom-softmax-v1',\n                'Missing declared same-trajectory exact initialization execution policy')\n    if arm != 'protenix':\n        require(summary['context_file_sha256']")
    files[name] = text.encode()
    files['scripts/protenix_initialization.py'] = (HERE / 'protenix_initialization.py').read_bytes()
    files['tests/test_protenix_initialization.py'] = (HERE / 'fixture_protenix_initialization.py').read_bytes()
    files['docs/brd4_jq1_25_initialization_recovery04.md'] = (HERE / 'README.md').read_bytes()
    assert len(files) == 625
    scientific = [n for n in original if n.startswith('src/')]
    assert len(scientific) == 145
    assert [n for n in scientific if files[n] != original[n]] == ['src/mosaic/small_molecule/chemistry/trajectory.py']
    old = ast.parse(original[scientific[scientific.index('src/mosaic/small_molecule/chemistry/trajectory.py')]])
    changed = ast.parse(files['src/mosaic/small_molecule/chemistry/trajectory.py'])
    assert ast.dump(old) != ast.dump(changed)
    for name in original:
        if name.startswith('examples/configs/') or name.endswith('/campaign25_contract.py') or name.endswith('/chemistry/state.py'):
            assert files[name] == original[name], name
    helpers['package'](TAG, original, files, parent, HERE, output_name='source02')
    print((HERE / 'package.json').read_text())


if __name__ == '__main__':
    main()
