"""Prepare a read-only exact-file/inode audit, never an approval or deletion."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BODY = r'''
import collections,hashlib,json,os,stat,sys,time
from pathlib import Path
from datetime import datetime,timezone
started=time.monotonic()
sys.dont_write_bytecode=True
source=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01/operator-source-02')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(source/'source-manifest.json')=='996c086e15e0151f3fc6484c14d4b5b34b7fce4b8b2e3835e6bb941dfe3391a3'
manifest=json.loads((source/'source-manifest.json').read_bytes())
for name,digest in manifest['files'].items():assert sha(source/name)==digest
sys.path.insert(0,str(source))
import offload_protocol as protocol
def check():
 if time.monotonic()-started>1200:raise TimeoutError('Bounded exact read-only inventory expired')
def attrs(s):return {**protocol.metadata(s),'nlink':s.st_nlink,'mode':stat.S_IMODE(s.st_mode),'ctime_ns':s.st_ctime_ns,'blocks512':s.st_blocks}
def protected(p):return protocol.is_protected(p,CONFIG['protection'])
def observe():
 q=protocol.active_jobs()
 assert {x['job_id'] for x in q['jobs']}<=set(CONFIG['audited_active_job_ids']),'Unaudited active job; re-audit before any selection'
 return q
result={'version':'mosaic-offload-exact-inventory-v1','read_only':True,'utc':datetime.now(timezone.utc).isoformat(),'input_sha256':CONFIG_SHA,'queue_before':observe(),'roots':{},'files':[],'retained':[],'inode_groups':[],'candidate_approval':False,'numerical_execution':False}
pool={};roots={Path(x['path']):x for x in CONFIG['roots']};root_before={}
def onerror(e):raise e
for root,rule in roots.items():
 check();assert protocol.permitted_candidate_root(root) and root.resolve()==root and root.is_dir() and not protected(root)
 root_before[str(root)]=attrs(root.stat());count=0
 for folder,dirs,files in os.walk(root,followlinks=False,onerror=onerror):
  check()
  for name in dirs+files:
   p=Path(folder)/name;s=p.lstat()
   if stat.S_ISDIR(s.st_mode):continue
   if not stat.S_ISREG(s.st_mode):result['retained'].append({'path':str(p),'reason':'nonregular/symlink retained'});continue
   count+=1
   if protected(p) or set(p.parts[:-1]) & protocol.FORBIDDEN_COMPONENTS:
    result['retained'].append({'path':str(p),'reason':'Protected path/component'});continue
   keep_root_metadata=p.parent==root and p.suffix.lower() in {'.json','.jsonl','.log','.txt','.xml','.yaml','.yml','.csv'}
   binary=p.suffix.lower() in {'.pt','.pth','.ckpt','.npz','.npy'}
   if keep_root_metadata or (rule['mode']=='binary_only' and not binary):continue
   assert p.resolve()==p,'Linked candidate ancestry'
   pool[str(p)]={'path':str(p),**attrs(s)}
 result['roots'][str(root)]={'mode':rule['mode'],'regular_files':count,'before':root_before[str(root)]}
assert pool
# Name every alias in the shared run/submission namespace. A group is retained
# unless its COMPLETE live link count consists only of selected safe paths.
keys={(r['device'],r['inode']) for r in pool.values()};aliases=collections.defaultdict(list);entries=0
for scope in (Path('/work3/s234818/latentmol/runs'),Path('/work3/s234818/latentmol/submissions')):
 assert scope.resolve()==scope and scope.is_dir()
 for folder,dirs,files in os.walk(scope,followlinks=False,onerror=onerror):
  check()
  for name in files:
   entries+=1;assert entries<=500000,'Alias metadata bound exceeded'
   p=Path(folder)/name;s=p.lstat()
   if stat.S_ISREG(s.st_mode) and (s.st_dev,s.st_ino) in keys:aliases[(s.st_dev,s.st_ino)].append({'path':str(p),**attrs(s)})
groups=collections.defaultdict(list)
for row in pool.values():groups[(row['device'],row['inode'])].append(row)
selected=[]
for key,rows in groups.items():
 check();actual=aliases.get(key,[]);names={x['path'] for x in rows};observed={x['path'] for x in actual}
 coherent=bool(actual) and len(observed)==len(actual)==rows[0]['nlink'] and names==observed
 coherent=coherent and all(x['nlink']==rows[0]['nlink'] and x['bytes']==rows[0]['bytes'] and not protected(Path(x['path'])) for x in actual)
 if not coherent:
  result['retained'].append({'paths':sorted(names),'reason':'Incomplete/externally retained/current inode alias group','aliases':actual});continue
 selected.extend(rows)
 result['inode_groups'].append({'device':key[0],'inode':key[1],'nlink':rows[0]['nlink'],'aliases':sorted(names),'unique_bytes':rows[0]['bytes'],'allocated_blocks512':rows[0]['blocks512']})
for index,row in enumerate(sorted(selected,key=lambda x:x['path'])):
 check();parent,name,fd,before=protocol.checked_open(row)
 try:
  assert before.st_nlink==row['nlink'] and before.st_ctime_ns==row['ctime_ns']
  with os.fdopen(fd,'rb',closefd=False) as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  protocol.unchanged(row,parent,name,fd,before)
  row['sha256']=digest;result['files'].append(row)
  if index%250==0:print(json.dumps({'hashed_files':index+1,'total_selected':len(selected),'elapsed_seconds':time.monotonic()-started}),file=sys.stderr,flush=True)
 finally:os.close(fd);os.close(parent)
for root in roots:
 after=attrs(root.stat());assert after==root_before[str(root)],'Candidate root metadata changed during inventory'
 result['roots'][str(root)]['after']=after
result.update(queue_after=observe(),alias_entries_inventoried=entries,elapsed_seconds=time.monotonic()-started,complete=True,
 logical_bytes=sum(x['bytes'] for x in result['files']),unique_logical_bytes=sum(g['unique_bytes'] for g in result['inode_groups']),
 unique_allocated_blocks512=sum(g['allocated_blocks512'] for g in result['inode_groups']),
 actual_quota_savings_established=False,delete_allowlist=[])
print(json.dumps(result))
'''


def main():
    proposal = json.loads((HERE / 'latentmol-expansion-proposal-03.json').read_bytes())
    refinement = json.loads((HERE / 'latentmol-protection-refinement-02.json').read_bytes())
    scan = json.loads((REPO / 'results/hpc/20261024-storage-offload-01/shared-closure-scan-02/stdout.bin').read_bytes())
    trees = set(refinement['protected_remote_trees']) | {
        '/work3/s234818/mosaic', '/zhome/95/e/206121/mosaic-runs',
        '/work3/s234818/latentmol/releases', '/work3/s234818/latentmol/submissions',
        '/work3/s234818/latentmol/protenix_occupancy', '/work3/s234818/latentmol/exports',
        '/work3/s234818/latentmol/logs', '/work3/s234818/latentmol/diagnostics',
        '/work3/s234818/latentmol/runs/source135-epoch36-valid-reference-attempt001',
        '/work3/s234818/latentmol/runs/source136-valid-reference-epoch40-continuation-attempt001'}
    exact = set(refinement['protected_exact_remote_files'])
    for item in refinement['five_named_full_artifact_backup_candidates']:
        exact.update(row['path'] for row in item['preserve_remote_root_metadata'])
    for tag in ('source-138-vae-cli-startup-fix-attempt001', 'source-138-epoch36-embedding-comparison-attempt003'):
        exact.add('/work3/s234818/latentmol/releases/' + tag + '.tar.gz')
    protection = {'version': 'mosaic-storage-offload-protection-v1', 'exact': sorted(exact),
        'subtrees': sorted(trees), 'negative': [],
        'provenance': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                       for name in ('protenix-protection.json', 'boltz-dbmol-protection.json',
                                    'latentmol-protection-refinement-02.json',
                                    'latentmol-expansion-proposal-03.json',
                                    'latentmol-runtime-graph-refinement-03.json',
                                    'legacy-runtime-entrypoint-addendum.json')},
        'scope': 'Earlier blanket latentmol protection narrowed ONLY for explicitly reviewed inactive legacy run candidates. All Mosaic, home, sources, proof/current inputs, caches, environments, data and unrelated scopes protected.'}
    roots = {name: 'binary_only' for name in proposal['explicit_september_roots_binary_only']}
    roots.update({item['root']: 'binary_only' for item in proposal['additional_october_roots_binary_only']})
    roots.update({name: 'full_outputs_keep_root_metadata' for name in proposal['prior_full_artifact_roots']})
    roots.update({item['root']: 'full_outputs_keep_root_metadata' for item in proposal['additional_full_artifact_roots']})
    config = {'version': 'mosaic-offload-exact-inventory-input-v1',
        'audited_active_job_ids': ['29796558', '29802716', '29802717', '29802743'],
        'roots': [{'path': p, 'mode': mode} for p, mode in sorted(roots.items())],
        'protection': protection, 'captured_scan_sha256': hashlib.sha256(json.dumps(scan, sort_keys=True).encode()).hexdigest(),
        'approval': 'Read-only candidate audit; not a deletion plan.'}
    raw = (json.dumps(config, indent=2, sort_keys=True) + '\n').encode()
    header = 'CONFIG=' + repr(config) + '\nCONFIG_SHA=' + repr(hashlib.sha256(raw).hexdigest()) + '\n'
    script = (header + BODY).encode()
    ast.parse(script)
    for name, data in (('exact-inventory-input-01.json', raw), ('scan_exact_inventory_01.py', script),
                       ('flat-protection-candidate-01.json', (json.dumps(protection, indent=2, sort_keys=True) + '\n').encode())):
        with (HERE / name).open('xb') as stream: stream.write(data)
        print(name, len(data), hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
