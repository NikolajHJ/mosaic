"""Prepare the explicit presentation redesign capsule; local AST/bytes only.

Unlike refresh_capsule.py, this path declares a changed viewer, raw-graph display
layout and their allocated fixtures. Original science/input bytes are immutable.
Preparation does not connect, submit, render or evaluate scientific arrays.
"""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import refresh_capsule as prior

HERE = Path(__file__).resolve().parents[1]
LOCAL = prior.LOCAL
GENERATION = '04'
EDITION = 'edition-0003'
ORIGINAL_PACKAGE = LOCAL/'source03'
ORIGINAL_BYTES = 51389
ORIGINAL_ARCHIVE_SHA = '51459284779b90baf8f22d3effb7d33708d7c96d8ae5e424c73ccd13c3eeebd9'
ORIGINAL_MANIFEST_SHA = 'ee69381d752bf32f3ae55cf76f69ae1330acf5b20c52ddc7c9ee3cd7f711218c'
CHANGED = ('scripts/report_data.py', 'templates/report.html',
           'tests/test_report_data.py', 'tests/test_viewer.py')


def main():
    assert not (LOCAL/'source04').exists() and not (LOCAL/'launch-04').exists(), 'Existing/unknown preparation or launch: inspect, never replay'
    raw = (ORIGINAL_PACKAGE/'source.tar.gz').read_bytes()
    assert len(raw) == ORIGINAL_BYTES and prior.digest(raw) == ORIGINAL_ARCHIVE_SHA
    members, original = prior.archive_members(raw, ORIGINAL_MANIFEST_SHA)
    assert len(original['files']) == 17 and len(members) == 18
    manifest = deepcopy(original)
    plan = json.loads(members['plan.json'])
    plan['edition'] = EDITION
    # This edition redesigns the exact same terminal saved capture, not a new
    # observation or fresh scientific result. Preserve captured_at accordingly.
    counts = {'protenix': (10000,501), 'boltz': (10000,501),
              'dbmol': (5000,251), 'protenix_noisy': (1000,51)}
    for arm in plan['arms']:
        arm['expected_committed_updates'], arm['expected_retained_frames'] = counts[arm['id']]
    members['plan.json'] = prior.encoded(plan)
    replacements = {'nonchem-live-report-release-03':'nonchem-live-report-release-04',
                    'nonchem-live-report-03':'nonchem-live-report-04',
                    'mosaic-live-report03-':'mosaic-live-report04-',
                    'edition-0002':EDITION}
    ops = members['scripts/report_ops.py'].decode()
    for old,new in replacements.items():
        assert old in ops
        ops = ops.replace(old,new)
    ast.parse(ops)
    members['scripts/report_ops.py'] = ops.encode()
    # Keep canonical controller aligned with this new generation only. Historical
    # source03 remains sealed in its archive and on HPC.
    (HERE/'scripts/report_ops.py').write_bytes(members['scripts/report_ops.py'])
    for name in CHANGED:
        members[name] = (HERE/name).read_bytes().replace(b'\r\n',b'\n')
        if name.endswith('.py'): ast.parse(members[name],filename=name)
    fixture_counts = json.loads((LOCAL/'refined-cpu-counts.json').read_bytes())
    assert set(fixture_counts) == set(original['cpu_counts'])
    assert fixture_counts['tests.test_qualification'] == 19 and fixture_counts['tests.test_report_ops'] == 13
    manifest['cpu_counts'] = fixture_counts
    manifest['presentation_redesign'] = {
        'version':'jq1-refined-presentation-v1', 'prior_manifest_sha256':ORIGINAL_MANIFEST_SHA,
        'edition':EDITION, 'same_saved_capture':True,
        'declared_changed_members':list(CHANGED)+['scripts/report_ops.py','plan.json'],
        'graph_geometry_scope':'Deterministic display-only coordinates; every raw atom/edge retained',
        'original_scientific_results_changed':False, 'optimizer_submission':False}
    manifest['files'] = {name:prior.digest(data) for name,data in sorted(members.items()) if name!='source_manifest.json'}
    for name,pin in original['files'].items():
        if name not in (*CHANGED,'scripts/report_ops.py','plan.json'):
            assert prior.digest(members[name]) == pin
    members['source_manifest.json'] = prior.encoded(manifest)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as bundle:
            for name,data in sorted(members.items()):
                entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o444;entry.mtime=0
                bundle.addfile(entry,io.BytesIO(data))
    archive=output.getvalue();pin=prior.digest(members['source_manifest.json'])
    checked,_=prior.archive_members(archive,pin);assert checked==members
    prefix=prior.launcher_prefix()
    for old,new in replacements.items(): prefix=prefix.replace(old,new)
    anchor="assert not root.exists() and not source.parent.exists(), 'Existing/unknown source or report root: never replay'"
    assert prefix.count(anchor)==1
    claim="claim=base/'nonchem-live-report-04.redesign-claim.json'\nwith claim.open('x') as stream:json.dump({'status':'unknown','manifest_sha256':PIN,'optimizer_submission':False},stream)\n"
    prefix=prefix.replace(anchor,claim+anchor)
    import base64
    program='\n'.join(['ARCHIVE='+repr(base64.b64encode(archive).decode()),'LENGTH='+str(len(archive)),
                       'DIGEST='+repr(prior.digest(archive)),'PIN='+repr(pin),prefix])
    ast.parse(program);ast.parse(prior.STARTER)
    package=LOCAL/'source04';package.mkdir(exist_ok=False)
    artifacts={'source.tar.gz':archive,'source_manifest.json':members['source_manifest.json'],
               'plan.json':members['plan.json'],'remote-program.py':program.encode(),
               'start_prepared.py':prior.STARTER.encode()}
    for name,data in artifacts.items(): (package/name).write_bytes(data)
    receipt={'status':'prepared_only_not_submitted','generation':GENERATION,'edition':EDITION,
             'manifest_bytes':len(members['source_manifest.json']),'manifest_sha256':pin,
             'archive_bytes':len(archive),'archive_sha256':prior.digest(archive),'files':len(members)-1,
             'cpu_counts':fixture_counts,'plan_bytes':len(members['plan.json']),'plan_sha256':prior.digest(members['plan.json']),
             'program_bytes':len(program.encode()),'program_sha256':prior.digest(program.encode()),
             'same_original_saved_inputs':True,'allocated_checks_passed':False,'optimizer_submission':False,
             'monitor_remains_paused':True,'prepared_utc':datetime.now(timezone.utc).isoformat()}
    (package/'preparation.json').write_bytes(prior.encoded(receipt))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
