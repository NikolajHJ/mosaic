"""Local stdlib AST/byte packaging only; never connect, test or run a model."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[2]
LOCAL=REPO/'results/hpc/20261025-chemistry-replay-01'
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
IMPLEMENTATION=('scripts/replay_ops.py','scripts/replay_stage.lsf','scripts/softmax_replay_diagnostic.py',
 'tests/test_replay_ops.py','tests/test_softmax_replay_diagnostic.py')


def require(value,message):
 if not value:raise ValueError(message)


def raw(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--failed-observation',type=Path,required=True)
 parser.add_argument('--boltz-observation',type=Path,required=True);args=parser.parse_args()
 require(REPO.name=='mosaic','Wrong authorized repository')
 # Only import stdlib controller definitions; no numerical module or test runs.
 import importlib.util
 spec=importlib.util.spec_from_file_location('_local_replay_metadata',HERE/'scripts/replay_ops.py')
 op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
 members={};counts={};modules=[]
 for name in IMPLEMENTATION:
  data=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
  if name.endswith('.py'):
   tree=ast.parse(data,filename=name)
   if name.startswith('tests/'):
    mod=Path(name).stem;modules.append(mod)
    for node in tree.body:
     if isinstance(node,ast.ClassDef):
      count=sum(isinstance(item,ast.FunctionDef) and item.name.startswith('test_') for item in node.body)
      if count:counts['tests.'+mod+'.'+node.name]=count
  members[name]=data
 require(counts=={'tests.test_replay_ops.ReplayControllerTests':30,
                  'tests.test_softmax_replay_diagnostic.SoftmaxReplayMetadataTests':30},
         'Exact declared allocated controller/diagnostic fixture scope differs')
 failed=args.failed_observation/'files';boltz=args.boltz_observation/'files'
 bindings={}
 def local_binding(directory,relative,remote_root):
  path=directory/relative;data=path.read_bytes()
  row={'path':remote_root+'/'+relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  previous=bindings.setdefault(row['path'],row);require(previous==row,'Conflicting original proof binding');return row,data
 # Controller Path constants are POSIX on HPC, but this packager also runs on
 # Windows. Normalize remote identities rather than serializing Windows slashes.
 pilots={arm:{**row,'root':Path(row['root']).as_posix(),
              'source':Path(row['source']).as_posix()} for arm,row in op.PILOTS.items()}
 roots={arm:row['root'] for arm,row in pilots.items()}
 for name in ('cpu-certificate.json','protenix/submission.json','protenix/stdout.29816284.log',
  'protenix/payload/assessment.json','protenix/payload/comparison/assessment.json',
  'protenix/payload/comparison/combined/assessment.json',
  'protenix/payload/comparison/combined/paired-checks/assessment.json'):
  local_binding(failed,name,roots['protenix'])
 for name in ('cpu-certificate.json','boltz/submission.json','boltz/stdout.29816184.log',
  'boltz/payload/assessment.json','boltz/payload/comparison/assessment.json'):
  local_binding(boltz,name,roots['boltz'])
 for arm,directory in (('protenix',failed),('boltz',boltz)):
  cert=bindings[roots[arm]+'/cpu-certificate.json']
  require(cert['sha256']==pilots[arm]['cpu_certificate_sha256'],'Original actual producer CPU certificate differs')
 plan={'version':'chemistry-softmax-replay-diagnostic-v1',
  'scientific_source':{'path':BASE+'/protenix-extended-release-03/source','manifest_sha256':'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562','files':643},
  'failed_pilot_source':{'path':pilots['protenix']['source'],'manifest_sha256':pilots['protenix']['manifest_sha256'],'files':30},
  'points':[]}
 for variant in ('baseline','combined'):
  prefix='protenix/payload/comparison/'+variant
  _,data=local_binding(failed,prefix+'/measurements.json',roots['protenix'])
  points=json.loads(data)['saved_points']
  require({p['label'] for p in points}=={'early-0','early-10','high-0','high-10'} and len(points)==4,'Missing original four points')
  for point in points:
   label,step=point['label'].split('-');step=int(step)
   replay_path=prefix+'/study/'+label+'_pilot/exact-replay-'+format(step,'04d')+'.json'
   replay,replay_data=local_binding(failed,replay_path,roots['protenix']);record=json.loads(replay_data)
   snapshot=record['snapshot']
   require(record==point['exact_replay_canary'] and snapshot==point['snapshot']
    and record['status']=='passed_exact_five_field_replay','Original same-child canary/snapshot binding differs')
   require(snapshot['path']==roots['protenix']+'/'+prefix+'/study/'+label+'_pilot/snapshots/step-'+format(step,'04d')+'.npz',
    'Original snapshot path differs')
   previous=bindings.setdefault(snapshot['path'],snapshot);require(previous==snapshot,'Conflicting saved NPZ proof')
   plan['points'].append({'variant':variant,'label':point['label'],'snapshot':snapshot,'replay':replay})
 parent_sources=[plan['scientific_source'],plan['failed_pilot_source'],
  {'path':BASE+'/boltz-lr0025-release-03/source','manifest_sha256':'c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6','files':635},
  {'path':pilots['boltz']['source'],'manifest_sha256':pilots['boltz']['manifest_sha256'],'files':30},
  {'path':op.GUARD.as_posix(),'manifest_sha256':op.GUARD_PIN,'files':14}]
 admission={'version':op.VERSION,'pilots':pilots,'parent_sources':parent_sources,
  'preserved_proofs':sorted(bindings.values(),key=lambda r:r['path']),
  'original_results_changed':False,'production_authorized':False}
 require(all(row['path'].startswith('/work3/') and '\\' not in row['path']
             for row in admission['preserved_proofs']+parent_sources),
         'Remote source/proof identities must retain POSIX spelling')
 members['plan.json']=raw(plan);members['admission.json']=raw(admission)
 record=HERE/'runs/20261025-chemistry-replay-01/README.md'
 require(record.is_file(),'Parent must write the dated diagnostic record before packaging')
 members['protocol.md']=record.read_bytes().replace(b'\r\n',b'\n')
 members['README.md']=b'# Separate exact saved-logit replay diagnostic\n\nNo model evaluation, optimizer update, repair, production adoption or saved-result rewrite. CPU then bounded allocated GPU only.\n'
 members['scripts/README.md']=b'# Replay diagnostic execution\n\nImmutable stdlib admission and isolated GPU replay controls; no production branch.\n'
 members['tests/README.md']=b'# Allocated metadata contracts\n\nExact declared classes require raw same-job CPU JUnit/LSF before GPU. Local packaging is AST/bytes only.\n'
 manifest={'version':op.VERSION,'persistent_cap_bytes':op.LIMIT,'group_cap_bytes':op.GROUP_LIMIT,
  'additional_allowance_bytes':0,'production_authorized':False,'monitor_remains_paused':True,
  'cpu_counts':counts,'cpu_modules':modules,'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(members.items())}}
 manifest_data=raw(manifest);members['source_manifest.json']=manifest_data
 LOCAL.mkdir(parents=True,exist_ok=True);output=LOCAL/'source01';output.mkdir(exist_ok=False)
 archive=output/'source.tar.gz'
 with tarfile.open(archive,'x:gz') as target:
  for name,data in sorted(members.items()):
   item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o444;item.mtime=0;target.addfile(item,io.BytesIO(data))
 with tarfile.open(archive) as target:
  entries=target.getmembers();require(len(entries)==len(members) and all(p.isfile() for p in entries),'Archive membership differs')
  for p in entries:require(target.extractfile(p).read()==members[p.name],'Archive bytes differ')
 (output/'source_manifest.json').write_bytes(manifest_data)
 result={'status':'local_stdlib_AST_bytes_only','manifest_sha256':hashlib.sha256(manifest_data).hexdigest(),
  'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
  'files':len(members)-1,'cpu_counts':counts,'cpu_cases':sum(counts.values()),'points':8,
  'plan_sha256':hashlib.sha256(members['plan.json']).hexdigest(),'allocated_CPU_pass':False,'jobs_submitted':False}
 (output/'preparation.json').write_bytes(raw(result));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
