"""Local stdlib AST/byte packaging only; no tests, SSH, staging or submission."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/hpc/20261025-nonchem-stride20-startup-03/source02'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    old = ROOT / 'results/hpc/20261024-boltz-pro02-adoption-04/source01/source.tar.gz'
    require_sha = '5958373b3dc53a41cf6106264c611f334c6165d93f562ce167b221bf5cf6eac5'
    assert digest(old.read_bytes()) == require_sha
    with tarfile.open(old) as stream:
        original_contract = stream.extractfile('scripts/adoption_contract.py').read()
    members = {'legacy_contract.py': original_contract}
    for rel in ['README.md', 'selected_contract.py', 'selected_storage.py', 'selected_ops.py',
                'noisy_entry.py','retention.py','stage.lsf','tests/README.md',
                'tests/test_selected_launch.py','tests/test_retention.py']:
        raw = (HERE / rel).read_bytes().replace(b'\r\n', b'\n')
        if rel.endswith('.py'):
            ast.parse(raw, filename=rel)
        members[rel] = raw
    plan = {
        'version': 'independent-nonchemistry-stride20-admission-v2',
        'authorization': 'Start requested runs without delayed chemistry; preserve scientific settings/checks/saved results',
        'selected': {'boltz2': {'updates': 30000, 'lr': .0025, 'chemistry': False},
                     'boltz2_dbmol': {'updates': 10000, 'lr': .0025, 'chemistry': False},
                     'protenix_continuation': {'additional_updates': 40000, 'total_updates': 50000, 'lr': .3, 'chemistry': False},
                     'protenix_noisy': {'updates': 1000, 'lr': .3, 'chemistry': False, 'separate_cpu_and_native_validation': True}},
        'deferred': ['random_protenix_delayed_chemistry_50000', 'boltz2_delayed_chemistry_30000'],
        'deferred_policy': 'No production/unknown segment allowed; no future admission; all existing bytes remain stale-charged',
        'required_metadata_cases':78, 'reserve_bytes': 4 * 1024**3, 'file_reserve': 100000,
        'artifact_retention':{'matrix_stride':20,'loss_stride':1,'initial_terminal_boundaries':True,'rolling_latest_and_best':True},
        'retirement_assessment_sha256':'c75c65397c2b36378c8127be7da3d2dde803fa8d38ab51a594e2b1d3839896ab',
        'prior_dispatch_failure':'role status selected_owner was incorrectly sent to a passed-certificate verifier; preserved with zero production submissions',
        'original_scientific_sources_and_proofs_unchanged': True,
        'native_failure_or_terminal_failure_stops_own_chain': True,
        'noise_validation_does_not_block_other_three': True,
        'monitor_remains_paused': True,
    }
    members['plan.json'] = (json.dumps(plan, indent=2, sort_keys=True) + '\n').encode()
    manifest = {'version': 'selected-nonchemistry-metadata-release-v1',
                'git_revision_reference': 'c74c659344123937608077e3e4078f75f09cd743',
                'files': {rel: digest(raw) for rel, raw in sorted(members.items())},
                'scientific_source_changed': False, 'metadata_cases':78,
                'artifact_writers_adapted':True,'loss_every_step_matrix_stride':20}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    members['source_manifest.json'] = manifest_bytes
    (OUT / 'source_manifest.json').write_bytes(manifest_bytes)
    archive = OUT / 'source.tar.gz'
    with tarfile.open(archive, 'x:gz') as stream:
        for rel, raw in sorted(members.items()):
            item = tarfile.TarInfo(rel)
            item.size = len(raw)
            item.mode = 0o444
            item.mtime = 0
            stream.addfile(item, io.BytesIO(raw))
    with tarfile.open(archive) as stream:
        rows = stream.getmembers()
        assert len(rows) == len(members) and all(x.isfile() and not x.issym() and not x.islnk() for x in rows)
        for item in rows:
            assert stream.extractfile(item).read() == members[item.name]
    receipt = {'status': 'prepared_local_stdlib_only', 'files': len(members) - 1,
               'manifest_sha256': digest(manifest_bytes), 'archive_bytes': archive.stat().st_size,
               'archive_sha256': digest(archive.read_bytes()), 'allocated_cpu_passed': False,
               'staged': False, 'submitted': False, 'scientific_execution': False,
               'original_metadata_contract_archive_sha256': require_sha}
    (OUT / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
