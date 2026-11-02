"""Portable byte/JSON controls; no models, arrays or scientific rendering."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import report_exploratory as adapter


def original_snapshot(root):
    files={'exploratory_continuation.py':b'# frozen original entry\n','plan.json':b'{}\n'}
    for name,raw in files.items():(root/name).write_bytes(raw)
    manifest={'files':{n:hashlib.sha256(raw).hexdigest() for n,raw in files.items()}}
    (root/'snapshot.json').write_text(json.dumps(manifest),encoding='utf-8')
    return manifest


def test_original_snapshot_is_verified_and_bound(tmp_path):
    original_snapshot(tmp_path)
    proof=adapter.source_proof(tmp_path,[str(tmp_path)])
    assert proof['path']==str(tmp_path/'snapshot.json')
    assert proof['sha256']==hashlib.sha256((tmp_path/'snapshot.json').read_bytes()).hexdigest()


def test_modern_checksum_binding_stays_unchanged(tmp_path):
    (tmp_path/'checksums.sha256').write_bytes(b'original modern proof\n')
    assert adapter.source_proof(tmp_path,[str(tmp_path)])['path']==str(tmp_path/'checksums.sha256')


@pytest.mark.parametrize('problem',['changed-file','unsafe-path','missing-entry'])
def test_original_source_proof_failures_are_preserved(tmp_path,problem):
    manifest=original_snapshot(tmp_path)
    if problem=='changed-file':(tmp_path/'exploratory_continuation.py').write_bytes(b'changed')
    if problem=='unsafe-path':manifest['files']['../outside.py']='0'*64
    if problem=='missing-entry':manifest['files'].pop('exploratory_continuation.py')
    (tmp_path/'snapshot.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError):adapter.source_proof(tmp_path,[str(tmp_path)])
