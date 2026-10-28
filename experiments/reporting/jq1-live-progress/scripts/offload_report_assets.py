"""Offload exact inactive generated report assets after complete local backup verification.

Never called automatically by report.py. Requires an explicit, audited archive
mapping. Scientific runs, source/config/proof files and local viewers are outside
the removable scope. Restoration comes from the original verified archive member.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

import quick_inputs as inputs
from report import Connection, write


def verify_backups(spec):
    rows, archives = [], []
    for item in spec['archives']:
        archive = Path(item['local_archive'])
        inputs.require(archive.stat().st_size == item['bytes'] and inputs.sha(archive) == item['sha256'], 'Local backup archive identity differs')
        files = {}
        with tarfile.open(archive) as bundle:
            names = set()
            for member in bundle.getmembers():
                part = PurePosixPath(member.name)
                inputs.require(not part.is_absolute() and '..' not in part.parts and '\\' not in member.name
                               and member.name.casefold() not in names and (member.isfile() or member.isdir()), 'Unsafe/duplicate backup member')
                names.add(member.name.casefold())
                if member.isfile():
                    with bundle.extractfile(member) as stream:
                        files[member.name] = {'bytes': member.size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
            receipt_name = 'evidence_manifest.json' if 'evidence_manifest.json' in files else 'report-receipt.json'
            receipt = json.loads(bundle.extractfile(receipt_name).read())
        inputs.require(set(files) == set(receipt['files']) | {receipt_name}, 'Backup member set differs from original manifest')
        for name, row in receipt['files'].items():
            inputs.require(files[name] == row, 'Backup member bytes differ: ' + name)
        prefix = item['member_prefix']
        for name, row in files.items():
            if item.get('archive_only') or not name.startswith(prefix):
                continue
            relative = name[len(prefix):]
            flat = re.fullmatch(r'[a-z0-9_-]+-(?:frames-[0-9]{4}|scalars)\.json', relative)
            old = re.fullmatch(r'(?:protenix|boltz2|boltz2_chemistry)/(?:frames-[0-9]{5}\.(?:json|bin)|curves\.json)', relative)
            if flat or old:
                rows.append({'path': item['remote_report_root'] + '/' + relative, **row,
                             'restore_archive': str(archive.resolve()), 'restore_member': name})
        archives.append({'path': str(archive.resolve()), 'bytes': archive.stat().st_size,
                         'sha256': item['sha256'], 'all_member_files_verified': len(files)})
        # A redundant generated delivery archive can also be offloaded, but only
        # when explicitly named and its complete local archive was verified above.
        # Source archives and scientific summaries are not admitted by this name.
        if item.get('remote_archive'):
            remote = PurePosixPath(item['remote_archive'])
            inputs.require(re.fullmatch(r'(?:interim|edition)-[0-9]{4}\.tar\.gz', remote.name) is not None,
                           'Only explicitly named generated interim archives are removable')
            rows.append({'path': str(remote), 'bytes': archive.stat().st_size,
                         'sha256': item['sha256'], 'restore_archive': str(archive.resolve()),
                         'restore_member': None, 'restore_method': 'copy_verified_archive_bytes'})
    inputs.require(rows and len({r['path'] for r in rows}) == len(rows), 'Missing/duplicate exact offload entries')
    inputs.require({r['path'] for r in rows}.isdisjoint(set(spec['protected'])),
                   'An offload target overlaps a protected source/input/cache/proof')
    return {'version': 'inactive-rendered-report-offload-v1', 'archives': archives, 'entries': rows,
            'remote_allowed_roots': spec['remote_allowed_roots'], 'protected': spec['protected'],
            'logical_bytes': sum(r['bytes'] for r in rows), 'no_scientific_results_removed': True}


REMOTE = r'''
import hashlib,json,os,stat
from pathlib import Path
plan=PLAN
root=Path(AUDIT_ROOT)
root.mkdir(parents=True,exist_ok=False)
(root/'plan.json').write_text(json.dumps(plan,sort_keys=True))
protected=[]
for path in plan['protected']:
 p=Path(path);assert p.is_file() and not p.is_symlink()
 with p.open('rb') as f:protected.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(f,'sha256').hexdigest()})
preflight=[]
for row in plan['entries']:
 p=Path(row['path']);assert any(p.is_relative_to(Path(r)) for r in plan['remote_allowed_roots'])
 assert all(not q.is_symlink() for q in (p,*p.parents)) and p.name not in ('summary.json','states.jsonl','updates.jsonl','report-receipt.json','manifest.json')
 if not p.exists():preflight.append({**row,'status':'already_absent'});continue
 s=p.lstat();assert stat.S_ISREG(s.st_mode)
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 preflight.append({**row,'status':'verified_exact' if s.st_size==row['bytes'] and digest==row['sha256'] else 'changed_preserved',
  'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'nlink':s.st_nlink,'blocks':s.st_blocks})
(root/'preflight.json').write_text(json.dumps(preflight,sort_keys=True))
removed=[]
with (root/'journal.jsonl').open('x') as journal:
 for row in preflight:
  result=dict(row)
  if row['status']=='verified_exact':
   p=Path(row['path']);s=p.lstat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns'])
   p.unlink();assert not p.exists();result['status']='removed_exact_backed_up_generated_asset'
  journal.write(json.dumps(result,sort_keys=True)+'\n');journal.flush();os.fsync(journal.fileno());removed.append(result)
for row in protected:
 p=Path(row['path'])
 with p.open('rb') as f:assert p.stat().st_size==row['bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
assert all(not Path(r['path']).exists() for r in removed if r['status']=='removed_exact_backed_up_generated_asset')
summary={'status':'completed_verified_exact_report_asset_offload','entries':len(removed),
 'removed_files':sum(r['status']=='removed_exact_backed_up_generated_asset' for r in removed),
 'logical_removed_bytes':sum(r['bytes'] for r in removed if r['status']=='removed_exact_backed_up_generated_asset'),
 'allocated_removed_bytes_nlink1':sum(r['blocks']*512 for r in removed if r['status']=='removed_exact_backed_up_generated_asset' and r['nlink']==1),
 'protected_before_after':protected,'journal_sha256':hashlib.sha256((root/'journal.jsonl').read_bytes()).hexdigest(),
 'native_studies_changed':False,'source_config_proofs_changed':False}
(root/'summary.json').write_text(json.dumps(summary,sort_keys=True))
print(json.dumps({'summary':summary,'journal':removed}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-audit-root', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = verify_backups(inputs.load(args.spec))
    args.output.mkdir(exist_ok=False)
    write(args.output/'restore-map.json', plan)
    if args.execute:
        program = REMOTE.replace('PLAN', repr(plan)).replace('AUDIT_ROOT', repr(args.remote_audit_root))
        connection = Connection(inputs.load(args.profile), args.output)
        result = connection.run(program, 'exact-report-offload', timeout=180)
        write(args.output/'completion.json', result)
        print(json.dumps(result['summary'], indent=2))
    else:
        print(json.dumps({'status': 'backup_verified_plan_only', 'entries': len(plan['entries']), 'logical_bytes': plan['logical_bytes']}, indent=2))


if __name__ == '__main__':
    main()
