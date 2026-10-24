"""Prepare a read-only HPC metadata scanner; never connects or runs that scanner."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RAW = REPO / 'results/hpc/20261024-storage-offload-01/active-closure-01/stdout.bin'

REMOTE_BODY = r'''
import collections, hashlib, json, os, re, stat, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT=Path('/work3/s234818/latentmol')
RUNS=ROOT/'runs'
LIMIT_SECONDS=240
LIMIT_FILES=60000
LIMIT_TEXT_BYTES=1024**3
LIMIT_ENTRIES=600000
CHUNK=1024**2
TEXT_EXT={'.py','.sh','.lsf','.json','.jsonl','.yaml','.yml','.toml','.ini','.cfg','.conf','.md','.txt','.csv','.tsv','.xml','.out','.err','.log'}
start=time.monotonic()
r={'version':'latentmol-readonly-positive-closure-scan-v1','utc':datetime.now(timezone.utc).isoformat(),
   'read_only':True,'no_model_import_or_checkpoint_parsing':True,'input_provenance':CONFIG['provenance'],
   'source_seals':{},'scanned_text':{},'protected_scopes':{},'references':{},'symlinks':{},
   'missing_literal_references':{},'unresolved':[],'dynamic_directory_references':[],
   'candidate_basename_hits':{},'candidates':{},'live_queue':{},'limits':{'seconds':LIMIT_SECONDS,'files':LIMIT_FILES,'text_bytes':LIMIT_TEXT_BYTES,'entries':LIMIT_ENTRIES}}
q=collections.deque();queued=set();scanned=set();bytes_seen=0;entries_seen=0
def bad(reason,**details):r['unresolved'].append({'reason':reason,**details})
def check():
 if time.monotonic()-start>LIMIT_SECONDS:raise TimeoutError('Global read-only scan time bound exceeded')
 if len(scanned)>LIMIT_FILES or bytes_seen>LIMIT_TEXT_BYTES or entries_seen>LIMIT_ENTRIES:raise RuntimeError('Declared metadata scan bound exceeded')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(CHUNK),b''):check();h.update(data)
 return h.hexdigest()
def attrs(s):return {'device':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink,'mode':stat.S_IMODE(s.st_mode),'size':s.st_size,'allocated_blocks_512':getattr(s,'st_blocks',None),'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def within(path):return path==ROOT or ROOT in path.parents
def scope(path,why):
 path=str(path);r['protected_scopes'].setdefault(path,[])
 if why not in r['protected_scopes'][path]:r['protected_scopes'][path].append(why)
def run_scope(path,why):
 p=PurePosixPath(str(path));parts=p.parts;rootparts=PurePosixPath(str(RUNS)).parts
 if parts[:len(rootparts)]==rootparts and len(parts)>len(rootparts):scope(str(PurePosixPath(*parts[:len(rootparts)+1])),why)
def enqueue(path,why,expected=None,required=False):
 p=Path(path)
 if p in {ROOT,RUNS,ROOT/'releases',ROOT/'submissions'}:
  item={'owner':why,'path':str(p),'scope':'Workspace/output container existence is not a recursive input dependency; independently review computed paths'}
  if item not in r['dynamic_directory_references']:r['dynamic_directory_references'].append(item)
  return
 if not p.is_absolute() or not within(p):
  if p.is_absolute():scope(p,'External absolute dependency: '+why)
  return
 key=str(p);record=r['references'].setdefault(key,{'from':[],'expected_sha256':[],'required_seed':False})
 if why not in record['from']:record['from'].append(why)
 if expected and expected not in record['expected_sha256']:record['expected_sha256'].append(expected)
 record['required_seed']|=required
 if len(record['expected_sha256'])>1:bad('Conflicting expected dependency hashes',path=key)
 run_scope(p,'Referenced by active code/metadata: '+why)
 # Every source/env/data/cache/checkpoint root is protection, not a deletion scope.
 if key.startswith(str(ROOT/'releases')+'/'):
  parts=p.parts;scope(Path(*parts[:len((ROOT/'releases').parts)+1]),'Referenced immutable release')
 if p.suffix in {'.pt','.ckpt','.eqx','.npz','.npy','.pkl','.gz','.tgz','.zip','.tar'}:scope(p,'Binary/archive dependency retained without deserialization')
 if key not in queued:queued.add(key);q.append(p)

assert sys.platform=='linux' and ROOT.is_dir() and RUNS.is_dir(),'Execute this scanner only on the actual HPC metadata filesystem'
candidates={}
for ent in os.scandir(RUNS):
 if ent.name.startswith('202609'):
  p=Path(ent.path);s=p.lstat();candidates[ent.name]=p
  r['candidates'][str(p)]={'basename':ent.name,'root_lstat':attrs(s),'inactive_established':False,'eligible_for_deletion':False}
candidate_re=re.compile('|'.join(re.escape(x) for x in sorted(candidates,key=len,reverse=True))) if candidates else None
absolute_re=re.compile(r'/work3/s234818/latentmol(?:/[A-Za-z0-9_.%+@=,:*?{}\[\]~/-]+)?')
relative_re=re.compile(r'"(?:path|file|manifest|catalog|config|checkpoint|context|prepared|input|output|run_record|source_plan)"\s*:\s*"([^"\\\n]+)"')
def consume(text,owner):
 if candidate_re:
  for match in candidate_re.finditer(text):
   name=match.group();scope(candidates[name],'Candidate basename occurs in active code/metadata')
   hits=r['candidate_basename_hits'].setdefault(name,[])
   if len(hits)<50 and owner not in hits:hits.append(owner)
 for match in absolute_re.finditer(text.replace('\\/','/')):
  value=str(PurePosixPath(match.group().rstrip('.,:')))
  if value in {str(ROOT),str(RUNS),str(ROOT/'releases'),str(ROOT/'submissions')}:
   item={'owner':owner,'path':value,'scope':'Generic/computed root requires independent code/plan review; no inactivity inferred'}
   if item not in r['dynamic_directory_references']:r['dynamic_directory_references'].append(item)
   continue
  enqueue(value,owner)
 for match in relative_re.finditer(text):
  value=match.group(1)
  if value.startswith(('/', '$','~')) or '://' in value or len(value)>4096:continue
  path=Path(owner).parent/value
  if '..' in Path(value).parts:
   try:path=path.resolve(strict=False)
   except OSError:continue
  if path.exists() or path.is_symlink():enqueue(path,owner+' [relative-field]')

def scan_text(path,expected=None):
 global bytes_seen
 key=str(path)
 if key in scanned:
  if expected and r['scanned_text'][key]['sha256']!=expected:bad('Already-scanned text expected hash differs',path=key)
  return
 check();before=path.lstat();h=hashlib.sha256();carry=b'';count=0
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(CHUNK),b''):
   check();count+=len(chunk);bytes_seen+=len(chunk);h.update(chunk)
   data=carry+chunk
   try:text=data.decode('utf8')
   except UnicodeDecodeError:text=data.decode('utf8',errors='replace')
   consume(text,key)
   carry=data[-16384:]
 after=path.lstat()
 if attrs(before)!=attrs(after) or count!=before.st_size:bad('Text changed during metadata scan',path=key)
 got=h.hexdigest()
 if expected and got!=expected:bad('Pinned text hash differs',path=key,expected=expected,actual=got)
 r['scanned_text'][key]={'bytes':count,'sha256':got,'lstat':attrs(after),'incremental':True}
 scanned.add(key)

def handle(path):
 key=str(path);record=r['references'][key]
 try:s=path.lstat()
 except FileNotFoundError:
  r['missing_literal_references'][key]={'required_seed':record['required_seed'],'from':record['from']}
  if record['required_seed']:bad('Required seed missing',path=key)
  return
 if stat.S_ISLNK(s.st_mode):
  target=os.readlink(path)
  try:resolved=path.resolve(strict=True)
  except (OSError,RuntimeError) as e:bad('Unresolved symlink dependency',path=key,error=repr(e));return
  r['symlinks'][key]={'lstat':attrs(s),'target':target,'resolved':str(resolved)}
  scope(path,'Active symlink dependency');scope(resolved,'Target of active symlink dependency')
  enqueue(resolved,key+' [symlink target]',record['expected_sha256'][0] if record['expected_sha256'] else None,record['required_seed']);return
 if stat.S_ISDIR(s.st_mode):
  scope(path,'Referenced directory retained')
  # Do not recursively walk model/data/env trees. Read bounded direct control
  # metadata; explicit manifests enqueue their further dependencies.
  if key.startswith(tuple(str(ROOT/x)+'/' for x in ('env','cache','data','raw','boltz-preparation'))):return
  for ent in os.scandir(path):
   if ent.name.endswith(('.json','.jsonl','.yaml','.yml','.toml','.cfg','.ini')):
    enqueue(ent.path,key+' [direct metadata]')
  return
 if not stat.S_ISREG(s.st_mode):bad('Active dependency is a special file',path=key);scope(path,'Special dependency retained');return
 record['lstat']=attrs(s)
 if path.suffix.lower() in TEXT_EXT or path.name.startswith(('MANIFEST','manifest','catalog')):
  scan_text(path,record['expected_sha256'][0] if record['expected_sha256'] else None)
 else:
  scope(path,'Active non-text dependency; no checkpoint/data parsing')

try:
 for p in CONFIG['strong_protection_roots']:scope(p,'Explicit current experiment/source/env/data/cache protection')
 for source in CONFIG['sources']:
  directory=Path(source['root']);manifest_path=Path(source['manifest'])
  assert digest(manifest_path)==source['manifest_file_sha256'],'Actual source manifest bytes differ'
  manifest=json.loads(manifest_path.read_bytes());files=manifest.get('files',manifest.get('members'))
  assert isinstance(files,dict) and files and all(isinstance(k,str) and isinstance(v,str) for k,v in files.items())
  scope(directory,'Full current source138 immutable source closure')
  for rel,expected in files.items():
   check();pure=PurePosixPath(rel)
   assert not pure.is_absolute() and '..' not in pure.parts,'Unsafe source member'
   p=directory/rel;s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink(),'Source member not regular'
   assert digest(p)==expected,'Current active source member bytes differ: '+str(p)
   if p.suffix.lower() in TEXT_EXT or pure.parts[0]=='configs':scan_text(p,expected)
  actual_text=0;unmanifested=[]
  for current,dirs,fnames in os.walk(directory,followlinks=False):
   check()
   for dirname in dirs:
    link=Path(current)/dirname
    if link.is_symlink():bad('Symlink directory inside active release',path=str(link))
   for fname in fnames:
    check();p=Path(current)/fname;rel=p.relative_to(directory).as_posix()
    if p.is_symlink():bad('Symlink file inside active release',path=str(p));continue
    if p.suffix.lower() in TEXT_EXT or rel.startswith('configs/'):
     actual_text+=1
     if rel not in files and p!=manifest_path:
      unmanifested.append(rel)
      if '.pytest_cache' not in p.relative_to(directory).parts:bad('Unmanifested active release text requires review',path=str(p))
     scan_text(p,files.get(rel))
  r['source_seals'][str(directory)]={'manifest':str(manifest_path),'manifest_sha256':source['manifest_file_sha256'],'members_rehashed':len(files),'all_declared_members_rehashed':True,'actual_text_and_config_members_scanned':actual_text,'unmanifested_text':unmanifested}
  enqueue(manifest_path,'Actual current source manifest',source['manifest_file_sha256'],True)
 for row in CONFIG['seeds']:enqueue(row['path'],'Authenticated saved active-closure seed',row.get('sha256'),True)
 queue=subprocess.run(['bjobs','-u','s234818','-l'],capture_output=True,text=True,timeout=30)
 r['live_queue']={'args':['bjobs','-u','s234818','-l'],'returncode':queue.returncode,'stdout':queue.stdout,'stderr':queue.stderr,'scope':'Actual live metadata only; no aged-out status fabricated'}
 if queue.returncode==0:
  live=queue.stdout+queue.stderr
  consume(live,'ACTUAL_CURRENT_BJOBS')
  consume(re.sub(r'\r?\n[ ]{8,}','',live),'ACTUAL_CURRENT_BJOBS_UNWRAPPED')
 else:bad('Current all-job scheduler query failed; no inactivity conclusion')
 while q:
  check();handle(q.popleft())
 # Inventory candidate bytes/links only, without reading model arrays. Do not
 # feed historical candidate self-references into active dependency closure.
 for name,path in candidates.items():
  check();row=r['candidates'][str(path)];files=dirs=logical=blocks=0;special=[];links=[];multi=[];root_metadata=[]
  root_before=path.lstat()
  if not stat.S_ISDIR(root_before.st_mode) or stat.S_ISLNK(root_before.st_mode):
   row['unresolved']='Candidate root is not an ordinary directory';scope(path,'Unresolved candidate type');continue
  pending=[path]
  while pending:
   check();current=pending.pop()
   for ent in os.scandir(current):
    entries_seen+=1;check();p=Path(ent.path);s=p.lstat()
    if stat.S_ISLNK(s.st_mode):
     try:target=str(p.resolve(strict=True))
     except (OSError,RuntimeError):target=None
     links.append({'path':str(p),'target':os.readlink(p),'resolved':target,'lstat':attrs(s)})
     continue
    if stat.S_ISDIR(s.st_mode):dirs+=1;pending.append(p)
    elif stat.S_ISREG(s.st_mode):
     files+=1;logical+=s.st_size;blocks+=getattr(s,'st_blocks',0)
     if s.st_nlink>1 and len(multi)<100:multi.append({'path':str(p),'lstat':attrs(s)})
     if current==path and p.suffix in {'.json','.jsonl','.yaml','.yml','.toml','.cfg','.ini'}:
      root_metadata.append({'path':str(p),'lstat':attrs(s),'sha256':digest(p)})
    else:special.append({'path':str(p),'lstat':attrs(s)})
  row.update(files=files,directories=dirs,logical_bytes=logical,allocated_blocks_512_sum=blocks,
             symlinks=links,multiple_link_examples=multi,special_files=special,root_metadata=root_metadata,
             root_unchanged_during_inventory=attrs(root_before)==attrs(path.lstat()))
  intersections=[p for p in r['protected_scopes'] if p==str(path) or p.startswith(str(path)+'/') or str(path).startswith(p+'/')]
  row['protected_intersections']=sorted(intersections)
  if links or multi or special or not row['root_unchanged_during_inventory']:row['unresolved']='Links/specials/race require independent positive review'
  row['status']='protected_dependency' if intersections else 'unresolved' if row.get('unresolved') or r['unresolved'] else 'not_referenced_in_completed_scan_pending_independent_review'
  row['inactive_established']=False;row['eligible_for_deletion']=False
except Exception as e:
 bad('Read-only scan did not close',error=repr(e))
finally:
 r['elapsed_seconds']=time.monotonic()-start;r['text_bytes_scanned']=bytes_seen;r['text_files_scanned']=len(scanned);r['candidate_entries_inventoried']=entries_seen
 r['scan_completed_without_unresolved_errors']=not r['unresolved']
 r['positive_inactivity_certified']=False;r['delete_allowlist']=[]
 r['scope_limitation']='Read-only closure/inode metadata only. Generic/computed roots, missing historical literals, current command input use, concurrent file writes and candidate link/race cases require independent review. No numerical execution, backup verification, deletion or quota pass.'
 print(json.dumps(r,sort_keys=True))
'''


def main():
    target = HERE / 'scan_active_closure.py'
    if target.exists():
        raise ValueError('Preserve prior scanner bytes')
    data = json.loads(RAW.read_bytes())
    require = lambda ok, msg: None if ok else (_ for _ in ()).throw(ValueError(msg))
    require(data['read_only'] is True and not data['missing'], 'Actual closure seed delivery incomplete')
    pins = {}
    decoded = {}
    for path, item in data['files'].items():
        raw = item['text'].encode()
        require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], 'Delivered seed changed')
        pins[path] = item['sha256']
        if path.endswith('.json'):
            decoded[path] = json.loads(raw)
    prior = json.loads((HERE / 'latentmol-closure-seeds.json').read_bytes())
    for row in prior['active_plan_absolute_reference_seeds']:
        if row['classification'] != 'directory_scope':
            if 'sha256' in row:
                require(pins.get(row['path'], row['sha256']) == row['sha256'], 'Conflicting earlier pin')
                pins[row['path']] = row['sha256']
    vae_root = '/work3/s234818/latentmol/releases/source-138-vae-cli-startup-fix-attempt001'
    emb_root = '/work3/s234818/latentmol/releases/source-138-epoch36-embedding-comparison-attempt003'
    config = {'provenance': {'input': RAW.relative_to(REPO).as_posix(),
                           'input_sha256': hashlib.sha256(RAW.read_bytes()).hexdigest()},
              'sources': [{'root': vae_root, 'manifest': vae_root + '/release_manifest.json',
                           'manifest_file_sha256': pins[vae_root + '/release_manifest.json']},
                          {'root': emb_root, 'manifest': emb_root + '/embedding-release-manifest.json',
                           'manifest_file_sha256': pins[emb_root + '/embedding-release-manifest.json']}],
              'seeds': [{'path': path, 'sha256': digest} for path, digest in sorted(pins.items())],
              'strong_protection_roots': ['/work3/s234818/latentmol/' + name for name in
                  ['env', 'data', 'cache', 'raw', 'boltz-preparation',
                   'runs/source138-vae-cli-startup-fix-attempt001',
                   'runs/source138-epoch36-embedding-comparison-attempt003',
                   'runs/source137-epoch36-boltz-embedding-bridge-attempt002',
                   'runs/source121a002-chemistry-16core-24h-attempt002',
                   'runs/source056-random-start-study',
                   'submissions/source104-chemistry-continuation-attempt005']]}
    body = '"""READ ONLY metadata scanner. Run only on HPC; no imports of scientific packages."""\nCONFIG = ' + repr(config) + '\n' + REMOTE_BODY
    ast.parse(body)
    with target.open('x', encoding='utf8', newline='\n') as handle:
        handle.write(body)
    print(json.dumps({'path': str(target), 'bytes': target.stat().st_size,
                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                      'required_seed_files': len(config['seeds']), 'executed_remotely': False}))


if __name__ == '__main__':
    main()
