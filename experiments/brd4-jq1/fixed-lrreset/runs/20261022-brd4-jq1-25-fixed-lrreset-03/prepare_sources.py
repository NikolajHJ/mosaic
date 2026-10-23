"""Prepare immutable retry overlays from authenticated archives; stdlib only."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
DIAG = REPO / 'experiments/brd4-jq1/fixed-lrboost/runs/20261022-brd4-jq1-lrboost-analysis-02'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parent_archive(tag, expected):
    path = REPO / 'results/hpc' / tag / 'source01/source.tar.gz'
    assert digest(path.read_bytes()) == expected
    files = {}
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            name = member.name.removeprefix('source/')
            pp = PurePosixPath(name)
            assert member.isfile() and not pp.is_absolute() and '..' not in pp.parts and name not in files
            files[name] = tar.extractfile(member).read()
    raw = files.pop('source_manifest.json')
    manifest = json.loads(raw)
    assert set(files) == set(manifest['files'])
    for name, data in files.items():
        value = manifest['files'][name]
        assert digest(data) == (value['sha256'] if isinstance(value, dict) else value)
    return files, digest(raw)


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new)


def package(tag, original, files, parent_sha, owner, output_name='source01'):
    output = REPO / 'results/hpc' / tag / output_name
    output.mkdir(parents=True, exist_ok=False)
    changed = sorted(k for k in files if original.get(k) != files[k])
    for name in changed:
        p = owner / ('overlay-' + name.replace('/', '__'))
        p.write_bytes(files[name])
    for name, data in files.items():
        if name.endswith('.py'):
            ast.parse(data, filename=name)
        if name.endswith('.json'):
            json.loads(data)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'git_revision_reference': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'identity_policy': 'Every exact file byte authenticated; immutable parent plus declared engineering overlays',
                'parent_manifest_sha256': parent_sha, 'declared_overlays': changed,
                'files': {n: digest(b) for n, b in sorted(files.items())}}
    raw = (json.dumps(manifest, indent=2) + '\n').encode()
    (output / 'source_manifest.json').write_bytes(raw)
    with tarfile.open(output / 'source.tar.gz', 'w:gz') as tar:
        for name, data in sorted((files | {'source_manifest.json': raw}).items()):
            item = tarfile.TarInfo('source/' + name); item.size = len(data); item.mode = 0o444
            tar.addfile(item, io.BytesIO(data))
    archive = output / 'source.tar.gz'
    record = {'tag': tag, 'source_files': len(files), 'manifest_sha256': digest(raw),
              'archive_bytes': archive.stat().st_size, 'archive_sha256': digest(archive.read_bytes()),
              'parent_manifest_sha256': parent_sha, 'changed_files': changed,
              'local_scope': 'Byte/AST/JSON packaging only; no numerical tests or model execution'}
    (output / 'package.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
    (owner / 'package.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
    print(json.dumps(record, indent=2))


def lower_rate():
    original, parent_sha = parent_archive('20261022-brd4-jq1-25-fixed-lrreset-02',
        '306bdf055770e7245bb37de291800cb7da0998af7345aabb7d1b9e0b97057e25')
    assert parent_sha == '832f833dd486c2ff305f7e3eb33fc775c87810a2e12a8bc597c5df744699c5d3'
    files = dict(original)
    for name in ('scripts/fixed25_lrreset_ops.py', 'scripts/fixed25_lrreset_storage.py',
                 'tests/test_fixed25_lrreset_storage.py'):
        text = files[name].decode().replace('20261022-brd4-jq1-25-fixed-lrreset-02', '20261022-brd4-jq1-25-fixed-lrreset-03')
        text = text.replace('restart-lrreset-release-02', 'restart-lrreset-release-03').replace('restart-lrreset-02', 'restart-lrreset-03')
        files[name] = text.encode()
    name = 'scripts/fixed25_lrreset_storage.py'; text = files[name].decode()
    text = replace_once(text, "def stale_work_bytes():", "CONCURRENT_DIAGNOSTIC_PATHS = (\n    Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-02'),\n    Path('/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-02'),\n    Path('/work3/s234818/mosaic/20261022-brd4-jq1-lrboost-analysis-02.tar.gz'),\n)\n\n\ndef stale_work_bytes():")
    text = replace_once(text, 'roots = (*HISTORICAL_WORK_ROOTS, ROOT, RELEASE, UPLOAD)',
                        'roots = (*HISTORICAL_WORK_ROOTS, ROOT, RELEASE, UPLOAD, *CONCURRENT_DIAGNOSTIC_PATHS)')
    files[name] = text.encode()
    name = 'tests/test_fixed25_lrreset_storage.py'
    files[name] += b'''\n\ndef test_retry_preserves_reserves_and_charges_separate_diagnostic_outputs():
    assert storage.ROOT.name == 'restart-lrreset-03'
    assert storage.RESERVE == 4 * 1024**3 and storage.FILE_RESERVE == 100000
    assert len(storage.CONCURRENT_DIAGNOSTIC_PATHS) == 3
    assert str(storage.CONCURRENT_DIAGNOSTIC_PATHS[0]).endswith('/runs/20261022-brd4-jq1-lrboost-analysis-02')
'''
    unchanged = [n for n in original if n.startswith('src/')]
    assert len(unchanged) == 145 and all(files[n] == original[n] for n in unchanged)
    for name in original:
        if 'brd4_jq1_25_fixed_lrreset_' in name and name.endswith('.json'):
            assert files[name] == original[name]
    package('20261022-brd4-jq1-25-fixed-lrreset-03', original, files, parent_sha, HERE)


def diagnostic(output_name='source01'):
    original, parent_sha = parent_archive('20261022-brd4-jq1-lrboost-analysis-01',
        '9707bec755c9c116a03a56abceabbe1caeaf019922f4bd68cb2a99141546f3e1')
    assert parent_sha == '214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc'
    files = {n: b.replace(b'20261022-brd4-jq1-lrboost-analysis-01', b'20261022-brd4-jq1-lrboost-analysis-02') for n, b in original.items()}
    name = 'scripts/lrboost_endpoint_diagnostics.py'; text = files[name].decode()
    text = text.replace("VERSION = 'jq1-lrboost-endpoint-diagnostics-v1'", "VERSION = 'jq1-lrboost-endpoint-diagnostics-v2'")
    old = """    require(np.array_equal(np.asarray(restored.X), np.asarray(graph.X))
            and np.array_equal(np.asarray(restored.E_upper), np.asarray(graph.E_upper)),
            'Original saved logits do not reproduce exact diagnostic graph probabilities')"""
    new = """    # Raw saved-probability probes remain at graph, byte-for-byte. This is a
    # distinct coordinate control at GraphLogits.graph(original saved logits).
    # Report exact equality/differences instead of assuming FP32 replay identity.
    identity = {}
    for name in ('X', 'E_upper'):
        saved, represented = np.asarray(getattr(graph, name)), np.asarray(getattr(restored, name))
        require(saved.shape == represented.shape and saved.dtype == represented.dtype == np.float32
                and np.isfinite(saved).all() and np.isfinite(represented).all(),
                'Malformed/nonfinite saved or represented diagnostic probability graph')
        identity[name] = {'bitwise_equal': bool(np.array_equal(saved, represented)),
                          'maximum_absolute_difference': float(np.max(np.abs(saved - represented))) if saved.size else 0.0,
                          'different_entries': int(np.count_nonzero(saved != represented))}
    identity['all_bitwise_equal'] = all(identity[n]['bitwise_equal'] for n in ('X', 'E_upper'))"""
    text = replace_once(text, old, new)
    text = replace_once(text, 'native_scalar=np.asarray(value))\n    proofs.append(hashed(original))',
                        'native_scalar=np.asarray(value), saved_X=np.asarray(graph.X), saved_E_upper=np.asarray(graph.E_upper),\n                        represented_X=np.asarray(restored.X), represented_E_upper=np.asarray(restored.E_upper))\n    proofs.append(hashed(original))')
    text = replace_once(text, "'autodiff_scalar': float(value), 'analytic': analytic, 'probes': []}",
                        "'autodiff_scalar': float(value), 'analytic': analytic, 'probes': [],\n              'saved_probability_identity': identity,\n              'baseline_scope': 'GraphLogits.graph(original saved logits); raw saved probability probes are a separate unchanged point',\n              'certifies_original_raw_probability_gate': False}")
    text = replace_once(text, 'np.array_equal(np.asarray(point_graph.X), np.asarray(graph.X))\n                               and np.array_equal(np.asarray(point_graph.E_upper), np.asarray(graph.E_upper))',
                        'np.array_equal(np.asarray(point_graph.X), np.asarray(restored.X))\n                               and np.array_equal(np.asarray(point_graph.E_upper), np.asarray(restored.E_upper))')
    files[name] = text.encode()
    name = 'scripts/jq1_lrboost_report_ops.py'; text = files[name].decode()
    text = text.replace("'jq1-lrboost-endpoint-diagnostics-v1'", "'jq1-lrboost-endpoint-diagnostics-v2'")
    text = replace_once(text, 'def certify_cpu(manifest):', 'def certify_cpu(manifest, *, only_protenix=False):')
    marker = '    with (ROOT / \'dispatch.lock\').open(\'a\') as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX)\n        exclusive(ROOT / \'analysis-dispatch-started.json\''
    branch = """    if only_protenix:
        with (ROOT / 'dispatch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            exclusive(ROOT / 'protenix-only-dispatch-started.json', {'manifest': manifest, 'original_failed_diagnostic': '29780821'})
            check_storage(ROOT / 'after-cpu', 'dispatch')
            native = submit('diagnostic-protenix', 'diagnostic', manifest, arm='protenix')
            callback = submit('diagnostic-capture-protenix', 'diagnostic-capture', manifest, arm='protenix', dependency='ended(' + native['job_id'] + ')')
            jobs.update(diagnostics={'protenix': native['job_id']}, diagnostic_captures={'protenix': callback['job_id']}, no_original_study_submitted=True)
            atomic(ROOT / 'jobs.json', jobs)
        return jobs
"""
    assert text.count(marker) == 1
    text = text.replace(marker, branch + marker)
    text = text.replace("if phase == 'certify-cpu':", "if phase in ('certify-cpu', 'certify-protenix-only'):")
    text = text.replace("('cpu', 'certify-cpu', 'capture'", "('cpu', 'certify-cpu', 'certify-protenix-only', 'capture'")
    text = replace_once(text, "elif args.phase == 'certify-cpu': print(json.dumps(certify_cpu(args.manifest)))",
                        "elif args.phase == 'certify-protenix-only': print(json.dumps(certify_cpu(args.manifest, only_protenix=True)))\n    elif args.phase == 'certify-cpu': print(json.dumps(certify_cpu(args.manifest)))")
    # Charge the complete old analysis root, which contains the new lower-rate
    # root/release as well; no quota arithmetic or reserve is reduced.
    text = replace_once(text, "'20261021-brd4-jq1-25-fixed-lrboost-01')\n        charged", "'20261021-brd4-jq1-25-fixed-lrboost-01', '20261022-brd4-jq1-lrboost-analysis-01')\n        charged")
    extra = """        lower_root = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-03')
        lower_marker = lower_root / 'long-dispatch-started.json'
        if lower_marker.exists():
            projected = read(lower_marker)['projections']
            require(set(projected) == set(ARMS) and all(type(n) is int and n >= 1024**3 for n in projected.values()),
                    'Malformed actual lower-rate whole-campaign projections')
            current_lower = {a: tree_bytes(lower_root / ('run-' + a)) for a in ('protenix', 'boltz2')}
            require(all(current_lower[a] <= projected[a] for a in current_lower), 'Lower-rate output exceeded projection')
            outstanding_lower = sum(projected[a] - current_lower[a] for a in current_lower)
            record['concurrent_lower_rate_scope'] = 'Actual authenticated whole-dispatch projection minus current work3 outputs'
        else:
            outstanding_lower = 3 * 1024**3
            record['concurrent_lower_rate_scope'] = 'Full grouped three-arm native validation allowance before any long dispatch'
        budget['required_bytes'] += outstanding_lower
        record['concurrent_lower_rate_outstanding_bytes'] = outstanding_lower
"""
    text = replace_once(text, "        record.update(raw_quota=raw, quota=quota, inode=inode, filesystem_inode_observation=observed, **budget)",
                        extra + "        record.update(raw_quota=raw, quota=quota, inode=inode, filesystem_inode_observation=observed, **budget)")
    files[name] = text.encode()
    name = 'tests/test_lrboost_endpoint_diagnostics.py'
    files[name] += b'''\n\n@pytest.mark.parametrize('different', [False, True])
def test_saved_logit_control_distinguishes_exact_raw_saved_graph_without_mutation(tmp_path, different):
    import jax
    import jax.numpy as jnp
    from mosaic.small_molecule.chemistry.state import GraphLogits
    from mosaic.small_molecule.state import RelaxedGraph
    logits = GraphLogits(jnp.asarray([[1., .5, 0., 0., 0., 0.]], jnp.float32), jnp.zeros((0, 5), jnp.float32))
    represented = logits.graph()
    values = represented.X
    if different:
        values = values.at[0, 0].add(jnp.float32(1e-6)).at[0, 1].add(jnp.float32(-1e-6))
    graph = RelaxedGraph(values, represented.E_upper)
    before = np.asarray(graph.X).copy()
    loss = lambda value, key: (jnp.sum(value.X**2) + jnp.sum(value.E_upper**2), {})
    proofs = []
    result = diag.logit_control(loss, graph, logits, jax.random.key(17), 'X', {'row': 0, 'source': 0, 'target': 1}, tmp_path, proofs)
    np.testing.assert_array_equal(np.asarray(graph.X), before)
    assert result['saved_probability_identity']['all_bitwise_equal'] is (not different)
    assert result['saved_probability_identity']['X']['different_entries'] == (2 if different else 0)
    assert result['certifies_original_raw_probability_gate'] is False
    assert len(proofs) == 13 and all(p['both_points_representably_changed'] for p in result['probes'])
    with np.load(proofs[0]['path']) as saved:
        np.testing.assert_array_equal(saved['saved_X'], before)
        np.testing.assert_array_equal(saved['represented_X'], np.asarray(represented.X))
'''
    files['tests/test_jq1_lrboost_ops.py'] += b'''\n\ndef test_corrected_cpu_dispatch_only_submits_protenix_diagnostic_and_own_capture(tmp_path, monkeypatch):
    import campaign25_pocket_ops
    monkeypatch.setattr(ops, 'ROOT', tmp_path)
    payload = tmp_path / 'cpu/payload'; payload.mkdir(parents=True)
    (tmp_path / 'after-cpu').mkdir()
    (tmp_path / 'jobs.json').write_text('{"cpu":"123"}')
    xml_fixture(payload / 'report-tests.xml')
    count = len(ops.TESTS + ops.LEGACY_TESTS)
    (payload / 'report-tests.log').write_text('collected ' + str(count) + ' items\\n' + str(count) + ' passed\\n')
    (payload / 'test-invocation.json').write_text('{}')
    monkeypatch.setattr(campaign25_pocket_ops, 'final_job', lambda *a: ('DONE', 'hpc', []))
    monkeypatch.setattr(ops, 'verify_cpu', lambda *a: {})
    monkeypatch.setattr(ops, 'check_storage', lambda *a: {'status': 'passed'})
    submitted = []
    def submit(name, phase, manifest, **kwargs):
        submitted.append((name, phase, kwargs))
        return {'job_id': str(900 + len(submitted))}
    monkeypatch.setattr(ops, 'submit', submit)
    result = ops.certify_cpu('a' * 64, only_protenix=True)
    assert [x[:2] for x in submitted] == [('diagnostic-protenix', 'diagnostic'), ('diagnostic-capture-protenix', 'diagnostic-capture')]
    assert submitted[1][2] == {'arm': 'protenix', 'dependency': 'ended(901)'}
    assert result['no_original_study_submitted'] is True
    assert result['diagnostics'] == {'protenix': '901'}


def test_corrected_cpu_callback_keeps_one_cpu_and_own_dependency_resource_scope():
    resources = ops.resources('certify-protenix-only')
    assert resources[resources.index('-q') + 1] == 'hpc'
    assert resources[resources.index('-n') + 1] == '1'
    assert '-gpu' not in resources and 'rusage[mem=2GB]' in resources
'''
    package('20261022-brd4-jq1-lrboost-analysis-02', original, files, parent_sha, DIAG, output_name)


if __name__ == '__main__':
    assert REPO.name == 'mosaic'
    lower_rate()
    diagnostic()
