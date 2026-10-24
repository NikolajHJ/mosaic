"""READ ONLY: targeted active-runtime inputs and old-checkpoint metadata inventory."""
import ast
import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time

ROOT=Path('/work3/s234818/latentmol')
RUNS=ROOT/'runs'
T0=time.monotonic()
MAX_SECONDS=240
MAX_META_BYTES=512*1024**2
MAX_METADATA=2000
MAX_ENTRIES=500000
CODE_EXT={'.py','.sh','.lsf','.json','.yaml','.yml','.toml','.cfg','.ini'}
BINARY_EXT={'.pt','.ckpt','.npz','.npy','.eqx','.pkl','.safetensors'}
NAMED={
 'source120a004-boltz-defog-staged-24h-attempt004',
 'source-073-decoder-budget-attempt002-recovery',
 'source-071-search-decoder-attempt001-search-study',
 'source-078-dense-100epoch-attempt002-density100-study',
 'source-079-defog-contract-recovery-attempt001-random-study'}
VAEP=ROOT/'submissions/source138-vae-cli-startup-fix-attempt001/source138-vae-cli-startup-fix-attempt001.json'
EMBP=Path(CONFIG['sources'][1]['root'])/'configs/epoch36_embedding_comparison.json'
PRIMARY={str(VAEP),str(EMBP),
 str(ROOT/'data/source095-moses-triangle-v2/manifest.json'),
 str(ROOT/'submissions/source104-chemistry-continuation-attempt005/parent-config.json'),
 str(ROOT/'submissions/source104-chemistry-continuation-attempt005/parent-run.json'),
 str(ROOT/'submissions/source104-chemistry-continuation-attempt005/parent-latest_epoch.json'),
 str(ROOT/'submissions/source121a002-chemistry-16core-24h-attempt002/plans/r2_async_w12_m128_h24.json'),
 str(ROOT/'runs/source121a002-chemistry-16core-24h-attempt002/variants/r2_async_w12_m128_h24/segment-001-completion.json')}
PROOF_KEYS={'readiness_artifacts','valid_reference_adoption','artifacts','proofs','saved_artifacts',
            'valid_reference_mechanism_evidence','superseded_failed_candidate','superseded_failed_fixture_attempt',
            'stopped_predecessor','algorithm','source','raw','stdout','stderr','text','history','source_metadata'}
FOLLOW_KEYS={'config','parent_run_record','source_plan','completion','dataset_manifest','context_manifest',
             'start_data_manifest','manifest','catalog','input_manifest','prepared_manifest','bindings',
             'parent_latest_epoch','latest_epoch','plan','protocol_config','native_template'}
R={'version':'latentmol-targeted-runtime-closure-v2','utc':datetime.now(timezone.utc).isoformat(),
 'read_only':True,'input_provenance':CONFIG['provenance'],'source_seals':{},'code':{},
 'runtime_code':{},'protected_trees':{},'protected_files':{},'runtime_inputs':{},
 'hashed_proof_containers':{},'review_items':[],'unresolved':[],'queue':{},
 'top_run_inventory':{},'candidate_checkpoints':[],'candidate_metadata':{},'delete_allowlist':[]}
Q=collections.deque();SEEN=set();TOTAL=0;ENTRIES=0;CODE={};MOD={};QUEUE_CODE=collections.deque()
ABS=re.compile(r'/work3/s234818/latentmol(?:/[A-Za-z0-9_.%+@=,:*?{}\[\]~/-]+)?')

def check():
 if time.monotonic()-T0>MAX_SECONDS:raise TimeoutError('Targeted metadata deadline exceeded')
 if TOTAL>MAX_META_BYTES or len(SEEN)>MAX_METADATA or ENTRIES>MAX_ENTRIES:raise RuntimeError('Declared metadata/inventory bound exceeded')
def issue(reason,**values):R['unresolved'].append({'reason':reason,**values})
def attr(s):return {'dev':s.st_dev,'ino':s.st_ino,'nlink':s.st_nlink,'mode':stat.S_IMODE(s.st_mode),'size':s.st_size,'blocks512':getattr(s,'st_blocks',None),'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
def tree(path,why):
 path=str(Path(path));R['protected_trees'].setdefault(path,[])
 if why not in R['protected_trees'][path]:R['protected_trees'][path].append(why)
def protect(path,why,expected=None):
 path=str(Path(path));v=R['protected_files'].setdefault(path,{'reasons':[],'expected_sha256':[]})
 if why not in v['reasons']:v['reasons'].append(why)
 if expected and expected not in v['expected_sha256']:v['expected_sha256'].append(expected)
 if len(v['expected_sha256'])>1:issue('Conflicting required hash',path=path)
def runroot(path):
 p=Path(path)
 try:rel=p.relative_to(RUNS)
 except ValueError:return None
 return RUNS/rel.parts[0] if rel.parts else None
def under(path,parent):return path==parent or parent in path.parents
def hash_file(path):
 h=hashlib.sha256();before=path.lstat()
 with path.open('rb') as handle:
  for data in iter(lambda:handle.read(1024**2),b''):check();h.update(data)
 after=path.lstat()
 if attr(before)!=attr(after):issue('File changed while hashing',path=str(path))
 return h.hexdigest(),after
def input_path(value,owner,key,expected=None,*,proof=False,required=True):
 if not isinstance(value,str) or '://' in value or len(value)>4096:return
 if value.startswith(('$','~')):
  R['review_items'].append({'owner':owner,'field':key,'value':value,'reason':'Unresolved runtime expansion'});return
 p=Path(value)
 if not p.is_absolute():
  if '/' not in value and not p.suffix:return
  candidates=[Path(owner).parent/p]
  candidates += [Path(s['root'])/p for s in CONFIG['sources']]
  exists=[c for c in candidates if c.exists() or c.is_symlink()]
  if len({str(c) for c in exists})==1:p=exists[0]
  elif not exists:
   if key in FOLLOW_KEYS:R['review_items'].append({'owner':owner,'field':key,'value':value,'reason':'Relative metadata root requires code review'})
   return
  else:
   R['review_items'].append({'owner':owner,'field':key,'value':value,'reason':'Relative path has multiple existing owners'});return
 if not under(p,ROOT):return
 if p in {ROOT,RUNS,ROOT/'submissions',ROOT/'releases'}:
  R['review_items'].append({'owner':owner,'field':key,'value':str(p),'reason':'Workspace/output container, not recursive runtime input'});return
 protect(p,('Exact proof bytes only: ' if proof else 'Runtime input: ')+owner+':'+key,expected)
 try:s=p.lstat()
 except FileNotFoundError:
  if required:issue('Required runtime/proof input missing',path=str(p),owner=owner,field=key)
  return
 if stat.S_ISLNK(s.st_mode):
  try:resolved=p.resolve(strict=True)
  except (OSError,RuntimeError) as e:issue('Runtime symlink unresolved',path=str(p),error=repr(e));return
  R['protected_files'][str(p)]['symlink']={'target':os.readlink(p),'resolved':str(resolved),'lstat':attr(s)}
  input_path(str(resolved),str(p),'symlink_target',expected,proof=proof,required=required);return
 if stat.S_ISDIR(s.st_mode):
  tree(p,'Concrete current runtime directory: '+owner+':'+key);return
 if not stat.S_ISREG(s.st_mode):issue('Required input not regular',path=str(p));return
 R['protected_files'][str(p)]['lstat']=attr(s)
 if not proof:
  rr=runroot(p)
  if rr:tree(rr,'Runtime checkpoint/context/catalog/lineage input')
 if p.suffix.lower() in BINARY_EXT or p.suffix.lower() in {'.csv','.tsv','.gz','.tgz','.zip','.tar'}:
  R['protected_files'][str(p)]['scope']='Protected without deserializing or parsing dataset/model bytes';return
 follow=not proof and (key in FOLLOW_KEYS or str(p) in PRIMARY)
 if p.suffix.lower() in {'.json','.jsonl','.yaml','.yml','.toml','.cfg','.ini'} or proof:
  Q.append((p,expected,follow,proof,required))

def structured(value,owner,proof=False):
 if isinstance(value,dict):
  for key,child in value.items():
   mode=proof or key in PROOF_KEYS
   if isinstance(key,str) and key.startswith(str(ROOT)+'/'):
    input_path(key,owner,'absolute_map_key',child if isinstance(child,str) and re.fullmatch('[a-f0-9]{64}',child) else None,proof=True)
   if isinstance(child,str):
    digest=value.get(key+'_sha256')
    if key in {'path','remote','module'}:digest=value.get('sha256',digest)
    if digest and not re.fullmatch('[a-f0-9]{64}',str(digest)):digest=None
    # Only complete path values or documented path fields; never recursively
    # interpret serialized raw stdout/code/old-status strings as current plans.
    if child.startswith(str(ROOT)) or key in FOLLOW_KEYS or key in {'path','remote','module','data','dataset','prepared','boltz_cache','release','parent_release','campaign','speed_suite_root','work_root'}:
     input_path(child,owner,key,digest,proof=mode)
   elif isinstance(child,(dict,list)):structured(child,owner,mode)
 elif isinstance(value,list):
  for child in value:structured(child,owner,proof)

def metadata(path,expected,follow,proof,required):
 global TOTAL
 key=str(path)
 if key in SEEN:return
 check();SEEN.add(key);got,s=hash_file(path);TOTAL+=s.st_size
 if expected and got!=expected:issue('Pinned metadata changed',path=key,expected=expected,actual=got)
 record={'bytes':s.st_size,'sha256':got,'lstat':attr(s),'followed_as_runtime_metadata':follow}
 if proof:
  # Large historical proof containers remain exact required remote files.
  # Their embedded logs/old commands are evidence, not executed input reads.
  R['hashed_proof_containers'][key]=record;return
 R['runtime_inputs'][key]=record
 if not follow:return
 if s.st_size>16*1024**2:
  issue('True runtime metadata too large for bounded structured parse',path=key,bytes=s.st_size);return
 if path.suffix.lower()=='.json':
  try:value=json.loads(path.read_bytes())
  except (ValueError,UnicodeError) as e:issue('True runtime JSON malformed',path=key,error=repr(e));return
  structured(value,key)
 else:R['review_items'].append({'path':key,'reason':'Required non-JSON runtime metadata retained; code-specific interpretation needs review'})

def module_for(rel):
 if rel.startswith('src/') and rel.endswith('.py'):
  name=rel[4:-3].replace('/','.')
  return name.removesuffix('.__init__')
 return None
def constants_without_docstrings(t):
 docs=set()
 for n in ast.walk(t):
  body=getattr(n,'body',None)
  if isinstance(body,list) and body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):docs.add(id(body[0].value))
 return [(n.value,n.lineno) for n in ast.walk(t) if isinstance(n,ast.Constant) and isinstance(n.value,str) and id(n) not in docs]
def code_scan(source,rel,data):
 key=str(source/rel);record={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'module':module_for(rel),'imports':[],'literal_paths':[]}
 if rel.endswith('.py'):
  parsed=ast.parse(data,filename=key);constants=constants_without_docstrings(parsed)
  for node in ast.walk(parsed):
   if isinstance(node,ast.Import):record['imports'] += [x.name for x in node.names]
   elif isinstance(node,ast.ImportFrom):
    base=node.module or ''
    if node.level:
     own=record['module'] or '';parts=own.split('.')
     if not rel.endswith('/__init__.py'):parts=parts[:-1]
     parts=parts[:max(0,len(parts)-node.level+1)];base='.'.join(parts+([base] if base else []))
    record['imports'].append(base)
    record['imports'] += [base+'.'+x.name for x in node.names if x.name!='*']
  record['filesystem_calls']=[{'line':n.lineno,'call':ast.unparse(n)[:350]} for n in ast.walk(parsed)
       if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'open','read_text','read_bytes','load','load_state_dict','glob','rglob','iterdir','scandir'}][:120]
 else:constants=[(data.decode('utf8',errors='replace'),1)]
 for value,line in constants:
  for match in ABS.finditer(value):record['literal_paths'].append({'path':str(PurePosixPath(match.group().rstrip('.,:'))),'line':line})
 CODE[key]=(source,rel,record,constants);R['code'][key]=record
 if record['module']:MOD[(str(source),record['module'])]=key
def reachable(key):
 if key in R['runtime_code']:return
 source,rel,record,constants=CODE[key];R['runtime_code'][key]={'sha256':record['sha256'],'imports':record['imports'],'literal_paths':record['literal_paths']}
 # Literal paths are evidence for an actual runtime module, but an absent
 # optional/default output is a review item rather than an invented input.
 for item in record['literal_paths']:
  p=Path(item['path'])
  if p in {ROOT,RUNS,ROOT/'submissions',ROOT/'releases'}:
   R['review_items'].append({'owner':key,'line':item['line'],'value':str(p),'reason':'Generic runtime container; actual plans bind concrete inputs'});continue
  rr=runroot(p)
  if rr:tree(rr,'Literal run dependency in reachable runtime module '+key)
  if p.exists():protect(p,'Reachable runtime code literal '+key+':'+str(item['line']))
  else:R['review_items'].append({'owner':key,'line':item['line'],'value':str(p),'reason':'Runtime code default/output literal absent; inspect applicability'})
 for name in record['imports']:
  found=MOD.get((str(source),name))
  if found:QUEUE_CODE.append(found)
 for value,line in constants:
  for rel2 in re.findall(r'(?:scripts/[A-Za-z0-9_./-]+\.(?:py|sh|lsf)|latentmol\.[A-Za-z0-9_.]+)',value):
   found=str(source/rel2) if rel2.startswith('scripts/') else MOD.get((str(source),rel2))
   if found in CODE:QUEUE_CODE.append(found)

def intersects(path):
 p=Path(path)
 return [k for k in R['protected_trees'] if under(p,Path(k))] + [k for k in R['protected_files'] if p==Path(k)]

assert sys.platform=='linux' and ROOT.is_dir() and RUNS.is_dir(),'Actual HPC read-only metadata filesystem required'
try:
 for path in CONFIG['strong_protection_roots']:tree(path,'Explicit current source/env/data/context/checkpoint protection')
 for name in ['source057-protein-panel','source094-large-triangle-first-suite','source095-large-triangle-prepare-benchmark','source098-large-triangle-evaluator-recovery']:
  tree(RUNS/name,'Current input/readiness/catalog ancestor; not an offload candidate')
 for source in CONFIG['sources']:
  sr=Path(source['root']);mp=Path(source['manifest']);got,_=hash_file(mp)
  assert got==source['manifest_file_sha256'],'Actual execution manifest changed'
  manifest=json.loads(mp.read_bytes());files=manifest.get('files',manifest.get('members'));assert isinstance(files,dict) and files
  tree(sr,'Complete current immutable execution source')
  for rel,digest in files.items():
   check();pure=PurePosixPath(rel);assert not pure.is_absolute() and '..' not in pure.parts
   p=sr/rel;s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink()
   actual,_=hash_file(p);assert actual==digest,'Declared active source bytes changed: '+str(p)
   if p.suffix.lower() in CODE_EXT or rel.startswith('configs/'):code_scan(sr,rel,p.read_bytes())
  # Non-execution prose deliberately excluded: VAE's manifest excludes docs,
  # README/AGENTS/lock prose by design. Every declared execution member rehashes.
  R['source_seals'][str(sr)]={'manifest':str(mp),'manifest_file_sha256':got,'execution_members_rehashed':len(files),'all_execution_members_rehashed':True,'prose_not_used_as_runtime_inputs':True}
  protect(mp,'Pinned actual current execution manifest',got)
 for row in CONFIG['seeds']:
  p=Path(row['path']);is_primary=str(p) in PRIMARY
  input_path(str(p),'Authenticated active plan seed','plan' if is_primary else 'proof',row.get('sha256'),proof=not is_primary)
 # Explicit actual current entrypoints from authenticated live submission argv.
 for sr in [Path(s['root']) for s in CONFIG['sources']]:
  rels=['scripts/hpc/boltz_run.sh']
  if 'vae-cli' in sr.name:rels+=['scripts/verify_stability_release.py','scripts/hpc/chemistry_valid_reference_campaign.py','src/latentmol/experiments.py','src/latentmol/chemistry_continuation.py']
  else:rels+=['scripts/hpc/embedding_comparison_operator.py','scripts/embedding_comparison_campaign.py']
  for rel in rels:
   key=str(sr/rel)
   if key in CODE:QUEUE_CODE.append(key)
   else:issue('Declared runtime entrypoint absent from authenticated source',path=key)
 while QUEUE_CODE:check();reachable(QUEUE_CODE.popleft())
 while Q:check();metadata(*Q.popleft())
 queue=subprocess.run(['bjobs','-u','s234818','-l'],capture_output=True,text=True,timeout=30)
 R['queue']={'args':['bjobs','-u','s234818','-l'],'returncode':queue.returncode,'stdout':queue.stdout,'stderr':queue.stderr}
 if queue.returncode!=0:issue('Fresh current job query failed')
 else:
  raw=queue.stdout+queue.stderr
  unwrapped=re.sub(r'\r?\n[ ]{8,}','',raw)
  known={Path(s['root']).name for s in CONFIG['sources']}
  for release_name in re.findall(re.escape(str(ROOT/'releases'))+r'/([A-Za-z0-9_.-]+)',unwrapped):
   if release_name.removesuffix('.tar.gz') not in known:issue('Another current latentmol execution source needs runtime closure',release=release_name)
  for text in [raw,unwrapped]:
   for m in ABS.finditer(text):
    p=Path(str(PurePosixPath(m.group().rstrip('.,:'))));rr=runroot(p)
    if rr:tree(rr,'Actual current live job command/input/output namespace')
 # Rank every ordinary top-level run directory without opening tensors. Fully
 # protected input/current scopes are skipped, not silently ranked as zero.
 ordered=sorted(os.scandir(RUNS),key=lambda e:(not(e.name.startswith('202609') or e.name in NAMED),e.name))
 for ent in ordered:
  check();rp=Path(ent.path);rs=rp.lstat();candidate=ent.name.startswith('202609') or ent.name in NAMED
  out={'root_lstat':attr(rs),'candidate_scope':candidate,'inactive_established':False,'eligible_for_deletion':False}
  R['top_run_inventory'][str(rp)]=out
  if intersects(rp):out['protected']=intersects(rp);out['inventory_skipped']='Whole current/runtime input tree protected';continue
  if not stat.S_ISDIR(rs.st_mode) or stat.S_ISLNK(rs.st_mode):out['unresolved']='Top run is not an ordinary directory';continue
  pending=[rp];logical=blocks=files=dirs=0;links=[];hardlinks=[];metadata_rows=[];bins=[]
  while pending:
   check();cur=pending.pop()
   for sub in os.scandir(cur):
    ENTRIES+=1;check();p=Path(sub.path);s=p.lstat()
    if stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode):dirs+=1;pending.append(p);continue
    if stat.S_ISLNK(s.st_mode):
     try:resolved=str(p.resolve(strict=True))
     except (OSError,RuntimeError):resolved=None
     links.append({'path':str(p),'target':os.readlink(p),'resolved':resolved,'lstat':attr(s)});continue
    if not stat.S_ISREG(s.st_mode):links.append({'path':str(p),'special':True,'lstat':attr(s)});continue
    files+=1;logical+=s.st_size;blocks+=getattr(s,'st_blocks',0)
    if s.st_nlink>1:hardlinks.append({'path':str(p),'lstat':attr(s)})
    if candidate and p.suffix.lower() in BINARY_EXT:
     bins.append({'path':str(p),'lstat':attr(s),'protected':intersects(p),'backup_required':True,'identical_bytes_verified':False,'eligible_for_deletion':False})
    elif candidate and p.parent==rp and p.suffix.lower() in {'.json','.jsonl','.yaml','.yml','.toml','.cfg'}:
     h,_=hash_file(p);metadata_rows.append({'path':str(p),'sha256':h,'lstat':attr(s),'preserve_remote':True})
  out.update(files=files,directories=dirs,logical_bytes=logical,blocks512=blocks,symlinks_or_specials=links,hardlinks=hardlinks,root_unchanged=attr(rs)==attr(rp.lstat()))
  if candidate:
   R['candidate_metadata'][str(rp)]={'root_metadata':metadata_rows,'scope':'Metadata/logs stay remote; old binary checkpoint candidates require separate positive current-dependency review, full backup and final hash/race checks.'}
   for row in bins:
    row['review_status']='protected' if row['protected'] else 'unresolved_links_or_race' if links or hardlinks or not out['root_unchanged'] else 'no_concrete_runtime_reference_found_pending_independent_review'
    if R['unresolved']:row['review_status']='global_runtime_closure_unresolved'
    R['candidate_checkpoints'].append(row)
except Exception as e:issue('Targeted closure/inventory incomplete',error=repr(e))
finally:
 R['elapsed_seconds']=time.monotonic()-T0;R['metadata_bytes_hashed']=TOTAL;R['metadata_files_hashed']=len(SEEN);R['entries_inventoried']=ENTRIES
 R['closed_without_true_runtime_errors']=not R['unresolved'];R['positive_inactivity_certified']=False
 R['historical_text_policy']='Exact required proof files are protected/hashed but their embedded old commands and serialized status histories are not recursively adopted as current execution. Historical prose is outside runtime closure. All source execution members still rehashed; reachable-module AST paths are retained for independent review.'
 R['limitation']='No model/data parsing, no scientific execution, no backup or delete pass. Missing/ambiguous true runtime dependencies, live-job scope, links/races and runtime default applicability require independent review. Candidate file-level metadata is for a later verified backup only.'
 print(json.dumps(R,sort_keys=True))
