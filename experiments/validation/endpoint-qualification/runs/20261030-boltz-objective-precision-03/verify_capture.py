"""Independent stdlib source/command/CPU proof audit, never evaluates arrays."""
import argparse,ast,base64,hashlib,json,re
from pathlib import Path,PurePosixPath
import xml.etree.ElementTree as ET
SOURCE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-continuation-26/boltz-precision-release-03/source'
ROOT='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-continuation-26/runtime-production-03/boltz-precision-03'
SCOPE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-continuation-26'
PIN='a9b2bb83254e4c05b914f53bdabdf88436bc3c8330498f7ca0206f522036fe1d'
SHARED='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/runtime-production-release-03/source'
def sha(b):return hashlib.sha256(b).hexdigest()
def require(v,message):
 if not v:raise ValueError(message)
def verify(packet):
 files={}
 for n,r in packet['files'].items():
  b=base64.b64decode(r['base64'],validate=True);require(len(b)==r['bytes'] and sha(b)==r['sha256'],'Delivered file bytes differ');files[n]=b
 def read(n):return json.loads(files[n])
 require(sha(files[SOURCE+'/source_manifest.json'])==PIN,'Manifest pin differs');m=read(SOURCE+'/source_manifest.json')
 for n,h in m['files'].items():require(sha(files[SOURCE+'/'+n])==h,'Sealed source member differs')
 for source in packet['parent_sources']:
  require(all(re.fullmatch('[0-9a-f]{64}',r['sha256']) for r in source['members']),'Remote source rehash missing')
 require(len(packet['parent_sources'])==len(m['parent_sources']),'Parent source set differs')
 for row in m['parent_sources']:
  matches=[s for s in packet['parent_sources'] if s['manifest']['path']==row['path']+'/source_manifest.json']
  require(len(matches)==1 and matches[0]['manifest']['sha256']==row['manifest_sha256'] and len(matches[0]['members'])==row['files'],'Parent source identity/count differs')
 plan=read(SOURCE+'/plan.json');require(packet['inputs_rehashed']==plan['inputs'],'All original input file rehashes required')
 tree=ast.parse(files[SOURCE+'/scripts/boltz_precision_ops.py']);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='argv')
 ns={'ROOT':PurePosixPath(ROOT),'SOURCE':PurePosixPath(SOURCE),'SHARED':PurePosixPath(SHARED),'require':require}
 exec(compile(ast.Module([fn],[]),'actual_authenticated_metadata_argv','exec'),ns)
 for phase,row in packet['submissions'].items():
  parent=ROOT+'/'+phase;require(row['argv']==ns['argv'](phase,PIN,row['dependency']),'Actual command/resources/dependency differ')
  require(packet['registry'][parent[len(SCOPE)+1:]]==row['job_id'],'Registry differs')
  raw=read(parent+'/submission-raw.json');require(raw['returncode']==0 and re.findall(r'Job <([0-9]+)> is submitted',raw['stdout'])==[row['job_id']],'Actual accepted scheduler receipt differs')
  require(row['dispatch_state']=='held' and row['source_manifest_sha256']==PIN,'Initial source/hold differs')
  held=[read(n) for n in files if n.startswith(parent+'/held-scheduler-')]
  require(any(r['returncode']==0 and r['stdout'].split()[0:3]==[row['job_id'],'PSUSP','s234818'] for r in held),'Exact owned held proof missing')
  release=read(parent+'/release.json');r=read(parent+'/release-raw.json')
  require(release['job_id']==row['job_id'] and release['receipt_persisted_before_release'] is True and r['returncode']==0
      and r['argv']==['bresume',row['job_id']],'Actual owned release differs')
 certificate=ROOT+'/cpu-certificate.json'
 if certificate not in files:return {'status':'source_bytes_and_existing_submission_vectors_verified_CPU_not_yet_certified','jobs':{k:v['job_id'] for k,v in packet['submissions'].items()}}
 cases=list(ET.fromstring(files[ROOT+'/cpu/payload/tests.xml']).iter('testcase'));counts={k:0 for k in m['all_cpu_counts']}
 for case in cases:
  require(not any(list(case.iter(tag)) for tag in ('failure','error','skipped')),'Raw CPU fail/error/skip')
  names=[k for k in counts if case.get('classname')==k or case.get('classname','').startswith(k+'.')];require(len(names)==1,'Raw module attribution differs');counts[names[0]]+=1
 require(counts==m['all_cpu_counts'] and len(cases)==51,'Exact actual CPU counts differ')
 cpu=packet['submissions']['cpu']['job_id'];cert=read(certificate)
 require(cert['status']=='passed' and cert['job_id']==cpu and cert['cases']==51 and cert['source_manifest_sha256']==PIN,'CPU certificate differs')
 def lsf(phase,queue):
  jid=packet['submissions'][phase]['job_id'];text=''.join(files[ROOT+'/'+phase+'/stdout.'+jid+'.log'].decode().split())
  require(text.count('Subject:Job'+jid+':')==1 and 'inqueue<'+queue+'>' in text and text.count('Successfullycompleted.')==1 and 'Exitedwithexitcode' not in text,'Same-job successful LSF required: '+phase)
 lsf('cpu','hpc')
 for arm in plan['arms']:
  r=read(ROOT+'/cpu/payload/startup-'+arm+'/server-startup.json')
  require(r['job_id']==cpu and r['arm']==arm and r['model_constructed'] is False and r['updates']==0,'Actual startup sentinel differs')
 return {'status':'independent_source_bytes_raw_XML51_actual_startups_submission_hold_release_and_LSF_passed',
   'cases':51,'modules':counts,'jobs':{k:v['job_id'] for k,v in packet['submissions'].items()},'source_manifest_sha256':PIN,
   'optimizer_admission':False,'scientific_GPU_result_verified':False,'delivered_files':len(files)}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('packet',type=Path);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 require(not a.output.exists(),'Immutable audit exists');raw=a.packet.read_bytes();p=json.loads(raw);value=verify(p);value['packet_sha256']=sha(raw);value['observed_utc']=p['observed_utc']
 with a.output.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')
 print(json.dumps(value))
