"""Independent stdlib byte/schema/CPU/job audit of compact production delivery.

No numerical array is decoded and no model/checker is imported. This verifies
the delivered source and allocated metadata scope; original large scientific
bytes not copied by the observer remain explicitly external HPC references.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-chemistry-production-01'
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT=BASE+'/chemistry-production-01'
SOURCE=BASE+'/chemistry-production-release-01/source'
VERSION='chemistry-accelerated-production-v2'
COUNTS={'tests.test_production_contract.ProductionContractTests':50,
        'tests.test_production_ops.ProductionRoutingTests':30,
        'tests.test_production_integration.OriginalCallSiteTests':14}
PROFILES={
    'protenix':{'arm':'random','target':50000,'segments':5,'rate':.3,
        'root':BASE+'/chemistry-full-model-03','source':BASE+'/chemistry-full-model-release-03/source',
        'science':BASE+'/protenix-extended-release-03/source',
        'manifest':'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562',
        'config':'extended_random.json','model':'8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2',
        'context':'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4'},
    'boltz':{'arm':'boltz2_chemistry','target':30000,'segments':3,'rate':.0025,
        'root':BASE+'/chemistry-full-model-02','source':BASE+'/chemistry-full-model-release-02/source',
        'science':BASE+'/boltz-lr0025-release-03/source',
        'manifest':'c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6',
        'config':'boltz_serial_boltz2_chemistry.json','model':'395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16',
        'context':'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'},
}


def require(value,message):
    if not value:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(),parse_constant=lambda value:
                      (_ for _ in ()).throw(ValueError('Nonfinite metadata constant '+value)))


def lsf(raw,sub):
    compact=''.join(raw.split());args=sub['argv']
    subjects=re.findall(r'Subject:\s*Job\s*(\d+)\s*:\s*<([^>]+)>\s*in\s*cluster\s*<([^>]+)>\s*([A-Za-z]+)[ \t]*(?=\r?\n|$)',raw)
    subjects=[tuple(''.join(field.split()) for field in row) for row in subjects]
    require(compact.count('Subject:Job')==1 and subjects==[(sub['job_id'],args[args.index('-J')+1],'dcc','Done')]
            and re.findall(r'inqueue<([^>]+)>',compact)==[args[args.index('-q')+1]]
            and compact.count('Successfullycompleted.')==1 and 'Exitedwithexit' not in compact,
            'Unique same-job/name/queue successful raw LSF proof missing')


def lsf_exit_one(raw,sub):
    compact=''.join(raw.split());args=sub['argv']
    subjects=re.findall(r'Subject:\s*Job\s*(\d+)\s*:\s*<([^>]+)>\s*in\s*cluster\s*<([^>]+)>\s*([A-Za-z]+)[ \t]*(?=\r?\n|$)',raw)
    subjects=[tuple(''.join(field.split()) for field in row) for row in subjects]
    require(compact.count('Subject:Job')==1 and subjects==[(sub['job_id'],args[args.index('-J')+1],'dcc','Exited')]
            and re.findall(r'inqueue<([^>]+)>',compact)==[args[args.index('-q')+1]]
            and compact.count('Exitedwithexitcode1.')==1 and 'Successfullycompleted.' not in compact,
            'Unique same-job/name/queue raw callback EXIT1 proof missing')


def resources(phase,arm):
    if phase in ('native','segment'):
        mode='num=1:mode=exclusive_process' if arm=='protenix' else 'num=1:mode=shared:j_exclusive=yes:mps=no'
        return ['-q','gpul40s','-n','8','-R','span[hosts=1]','-R','rusage[mem=4GB]',
                '-gpu',mode,'-W','02:00' if phase=='native' else '23:00']
    require(phase in ('cpu','after-cpu','after-native','after-segment'),'Unknown actual phase')
    return ['-q','hpc','-n','4' if phase=='cpu' else '1','-R','span[hosts=1]',
            '-R','rusage[mem=4GB]','-W','01:00' if phase=='cpu' else '00:30']


def option(args,flag):
    require(args.count(flag)==1 and args.index(flag)+1<len(args),'Missing/duplicate actual driver option '+flag)
    return args[args.index(flag)+1]


def verify(observation,package):
    receipt=read(observation/'receipt.json');raw=(observation/'raw.json').read_bytes()
    require(receipt.get('returncode')==0 and receipt['stdout_sha256']==sha(raw)
            and receipt['program_sha256']==sha((observation/'program.py').read_bytes()),'Observation byte/return-code binding differs')
    packet=json.loads(raw);require(packet['version']=='chemistry-production-observation-v1'
        and packet['read_only'] is True and packet['root']==ROOT and packet['source']==SOURCE,'Wrong observation scope')
    files={};remote={}
    for row in packet['files']:
        rel=PurePosixPath(row['scope'])/row['relative']
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in str(rel)
                and row['scope'] in ('root','source','external') and str(rel) not in files,'Unsafe/duplicate captured path')
        data=base64.b64decode(row['base64'],validate=True)
        require(len(data)==row['bytes'] and sha(data)==row['sha256']
                and (observation/'files'/Path(*rel.parts)).read_bytes()==data,'Delivered compact bytes differ')
        require(row['path'] not in remote,'Duplicated canonical remote path')
        files[str(rel)]=data;remote[row['path']]=data
    def data(path):
        require(path in remote,'Required compact proof absent: '+path)
        return remote[path]
    def obj(path):return json.loads(data(path))
    def bound(row):
        require(type(row) is dict and {'path','sha256'}<=set(row),'Missing exact proof binding')
        require(type(row['path']) is str and PurePosixPath(row['path']).is_absolute()
                and '..' not in PurePosixPath(row['path']).parts,'Noncanonical external proof path')
        require(re.fullmatch('[a-f0-9]{64}',row['sha256']) is not None,'Malformed proof SHA')
        if 'bytes' in row:require(type(row['bytes']) is int and row['bytes']>=0,'Malformed bound proof length')
        if row['path'] not in remote:return False
        value=data(row['path']);require(sha(value)==row['sha256'],'Bound proof hash changed')
        if 'bytes' in row:require(type(row['bytes']) is int and len(value)==row['bytes'],'Bound proof length changed')
        return True
    preparation=read(package/'preparation.json');archive=package/'source.tar.gz'
    require(archive.stat().st_size==preparation['archive_bytes'] and sha(archive.read_bytes())==preparation['archive_sha256'],
            'Original local production archive changed')
    with tarfile.open(archive) as source:
        members=source.getmembers()
        require(len(members)==13 and len({row.name for row in members})==13 and all(row.isfile() for row in members),
                'Expected exact13-member immutable capsule')
        archived={}
        for row in members:
            path=PurePosixPath(row.name)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in row.name,'Unsafe archive member')
            archived[row.name]=source.extractfile(row).read()
    source_raw=data(SOURCE+'/source_manifest.json');manifest=json.loads(source_raw)
    pin=sha(source_raw);require(pin==preparation['manifest_sha256'] and source_raw==archived['source_manifest.json'],
                                'Actual remote seal differs from prepared archive')
    require(manifest['version']==VERSION and manifest['cpu_counts']==COUNTS and len(manifest['files'])==12,
            'Wrong source/version/94-case scope')
    require(set(archived)==set(manifest['files'])|{'source_manifest.json'},'Archive/source scope differs')
    require({key.removeprefix('source/') for key in files if key.startswith('source/')}==set(archived),
            'Complete source13 members were not delivered')
    for name,digest in manifest['files'].items():
        require(sha(archived[name])==digest and data(SOURCE+'/'+name)==archived[name],'Actual source member changed '+name)
    plan=obj(SOURCE+'/plan.json')
    require(set(plan)=={'version','pilots'} and plan['version']==VERSION and set(plan['pilots'])==set(PROFILES),'Wrong per-arm plan')
    for arm,spec in PROFILES.items():
        value=plan['pilots'][arm]
        require(set(value)=={'root','source','manifest_sha256','cpu_certificate_sha256','job_id'}
                and value['root']==spec['root'] and value['source']==spec['source'],'Wrong per-arm actual producer root/source')
        for key in ('manifest_sha256','cpu_certificate_sha256'):
            require(re.fullmatch('[a-f0-9]{64}',value[key]) is not None,'Missing per-arm source/CPU pin')
        require(re.fullmatch('[1-9][0-9]*',value['job_id']) is not None,'Missing actual producer job')
    require(len({value['job_id'] for value in plan['pilots'].values()})==2,'Actual producer IDs must be distinct')
    submissions={}
    for path,value in remote.items():
        if not path.startswith(ROOT+'/') or PurePosixPath(path).name!='submission.json':continue
        sub=json.loads(value)
        if sub.get('status')!='submitted':continue
        phase,arm,index=sub['phase'],sub['arm'],sub['index']
        if phase in ('cpu','after-cpu'):
            require(arm=='control' and index==0,'Wrong control owner');parent=ROOT+'/'+phase
        else:
            require(arm in PROFILES and type(index) is int and 0<=index<PROFILES[arm]['segments'],'Wrong serial production range')
            require(index>0 or phase not in ('native','after-native'),'Initial native replay forbidden')
            parent=ROOT+'/'+arm+'/'+PROFILES[arm]['arm']+'-'+phase+'-'+format(index,'04d')
        require(path==parent+'/submission.json' and re.fullmatch('[1-9][0-9]*',sub['job_id']) is not None,'Actual submission path/job differs')
        command=['bsub',*resources(phase,arm),'-J','mosaic-chem-prod01-'+arm+'-'+PurePosixPath(parent).name,
                 '-oo',parent+'/stdout.%J.log','-eo',parent+'/stderr.%J.log']
        if sub['dependency'] is not None:command+=['-w',sub['dependency']]
        command+=['bash','-l',SOURCE+'/scripts/production_stage.lsf',SOURCE,pin,phase,arm,str(index)]
        require(sub['argv']==command and sub['source_manifest_sha256']==pin and sub['returncode']==0
                and re.findall(r'Job <([1-9][0-9]*)> is submitted',sub['stdout'])==[sub['job_id']],
                'Actual bsub argv/resources/source/job output differs')
        started=obj(parent+'/submission-started.json')
        for key in ('phase','arm','index','argv','source_manifest_sha256','dependency'):
            require(started[key]==sub[key],'Exclusive unknown marker differs from actual submission')
        require(started['status']=='unknown','Submission-started marker was rewritten')
        submissions[parent]=sub
    require(len({row['job_id'] for row in submissions.values()})==len(submissions),'Duplicate actual job IDs')
    for parent,sub in submissions.items():
        phase,arm,index=sub['phase'],sub['arm'],sub['index']
        if phase=='cpu':dependency=' && '.join('ended('+plan['pilots'][name]['job_id']+')' for name in PROFILES)
        elif phase=='after-cpu':
            require(ROOT+'/cpu' in submissions,'Callback lacks actual own CPU')
            dependency=' && '.join('ended('+job+')' for job in
                (submissions[ROOT+'/cpu']['job_id'],plan['pilots']['protenix']['job_id'],plan['pilots']['boltz']['job_id']))
        elif phase.startswith('after-'):
            producer=ROOT+'/'+arm+'/'+PROFILES[arm]['arm']+'-'+phase.removeprefix('after-')+'-'+format(index,'04d')
            require(producer in submissions,'Serial callback lacks actual own producer')
            dependency='ended('+submissions[producer]['job_id']+')'
        else:dependency=None
        require(sub['dependency']==dependency,'Exact own-ended/independent dependency changed')
    if ROOT+'/jobs.json' in remote:
        for key,job in obj(ROOT+'/jobs.json').items():
            arm,label=key.split('/',1);parent=ROOT+'/'+label if arm=='control' else ROOT+'/'+key
            require(parent in submissions and submissions[parent]['job_id']==job,'Actual jobs map differs')
    query=packet['scheduler_query'];actual_ids=sorted((row['job_id'] for row in submissions.values()),key=int)
    require(query['job_ids']==actual_ids and bool(query['invoked'])==bool(actual_ids),'Combined scheduler queried invented/missing IDs')
    cpu_state='not_certified';certificate_path=ROOT+'/cpu-certificate.json'
    if certificate_path in remote:
        cert=obj(certificate_path)
        require(cert['status']=='passed' and cert['source_manifest_sha256']==pin
                and cert['plan_sha256']==sha(data(SOURCE+'/plan.json')) and cert['cases']==94,'Wrong allocated production CPU certificate')
        for row in cert['proofs']:require(bound(row),'CPU certificate has undelivered proof')
        assessment=obj(ROOT+'/cpu/payload/assessment.json')
        require(assessment['status']=='passed' and assessment['cases']==94 and assessment['source_manifest_sha256']==pin,
                'CPU assessment is not exact94-case source-bound pass')
        cases=list(ET.fromstring(data(ROOT+'/cpu/payload/tests.xml')).iter('testcase'))
        require(len(cases)==94 and len({(row.get('classname'),row.get('name')) for row in cases})==94
                and all(not list(row) for row in cases) and dict(Counter(row.get('classname') for row in cases))==COUNTS,
                'Raw allocated JUnit is incomplete/failed/error/skipped/duplicated or wrongly attributed')
        tests=obj(ROOT+'/cpu/payload/tests-command.json')
        require(tests['returncode']==0 and re.search(r'\b94\s+passed\b',tests['stdout'])
                and re.search(r'\b[1-9][0-9]*\s+deselected\b',tests['stdout']) is None,'Raw CPU execution/count scope differs')
        cpu=submissions[ROOT+'/cpu'];lsf(data(ROOT+'/cpu/stdout.'+cpu['job_id']+'.log').decode(),cpu)
        queries=sorted(path for path in remote if path.startswith(ROOT+'/after-cpu/cpu-scheduler-final-'))
        require(queries,'CPU certificate lacks actual persisted final scheduler')
        states=[]
        for path in queries:
            row=obj(path);compact=''.join(row['stdout'].split())
            require(row['returncode']==0 and row['job_id']==cpu['job_id']
                    and re.findall(r'Job<(\d+)>',compact)==[cpu['job_id']]
                    and re.findall(r'Queue<([^>]+)>',compact)==['hpc']
                    and re.findall(r'JobName<([^>]+)>',compact)==[cpu['argv'][cpu['argv'].index('-J')+1]],
                    'Actual persisted CPU scheduler identity differs')
            state=re.findall(r'Status<([^>]+)>',compact);require(state in (['RUN'],['PEND'],['DONE']),'CPU scheduler did not succeed')
            states.append(state[0])
        require(states[-1]=='DONE','Actual final CPU scheduler success absent')
        startup=obj(ROOT+'/cpu/payload/startup.json')
        require(startup['status']=='passed_metadata_only' and startup['gpu_or_gradient_passed'] is False
                and startup['optimizer_updates']==0 and len(startup['commands'])==2,'CPU startup scope differs')
        for arm,row in zip(PROFILES,startup['commands']):
            expected=[
                '/work3/s234818/mosaic/envs/protenix-tiny/bin/python',SOURCE+'/scripts/production_run.py',
                '--manifest',pin,'--arm',arm,'--phase','segment','--index','0','--metadata-startup']
            require(row['argv']==expected and row['returncode']==0,'Actual startup wrapper command differs')
            value=json.loads(row['stdout']);args=value['args'];spec=PROFILES[arm]
            require(value['status']=='actual_original_parser_metadata_accepted' and value['optimizer_updates']==0
                    and value['native_or_gradient_passed'] is False,'CPU parser execution is not a native pass')
            require(args[:2]==[expected[0],spec['science']+'/examples/brd4_25_atoms.py']
                    and option(args,'--mode')=='run' and option(args,'--offset')=='0'
                    and option(args,'--segment-steps')=='10000' and option(args,'--manifest')==spec['manifest']
                    and option(args,'--config')==spec['science']+'/examples/configs/'+spec['config']
                    and option(args,'--output')==ROOT+'/'+arm+'/'+spec['arm']+'-segment-0000/payload/study'
                    and '--resume-snapshot' not in args and '--resume-sha256' not in args
                    and option(args,'--ready')==spec['root']+'/'+arm+'/payload/comparison/combined/study/assessment.json',
                    'Actual original startup parser/source/config/global boundary/ready path differs')
            require(re.fullmatch('[a-f0-9]{64}',option(args,'--ready-sha256')) is not None,
                    'Actual original startup lacks an authenticated readiness SHA')
        cpu_state='actual_CPU_94_pass_terminal_LSF_and_startup_verified'
    unresolved=[];adoptions={}
    for arm,spec in PROFILES.items():
        path=ROOT+'/'+arm+'/adoption.json'
        if path not in remote:continue
        row=obj(path);require(row['status']=='passed' and row['arm']==arm
            and row['production_manifest_sha256']==pin and row['plan_sha256']==sha(data(SOURCE+'/plan.json')),
            'Adoption is detached from exact new source/plan')
        expected=spec['root']+'/'+arm+'/payload/comparison/combined/study/assessment.json'
        require(row['readiness']['path']==expected and row['summary']['path']==str(PurePosixPath(expected).with_name('summary.json')),
                'Adoption redirects original readiness/summary')
        for proof in [*row['proofs'],row['readiness'],row['summary']]:
            if not bound(proof):unresolved.append(proof)
        if expected in remote:
            ready=obj(expected);require(ready['status']=='passed' and ready['source_manifest_sha256']==spec['manifest']
                and ready['model_digest']==spec['model'] and ready['context_digest']==spec['context']
                and ready['runtime_gate']['status']=='passed','Delivered original readiness identity/runtime differs')
            require(ready['storage_projection']['maximum_pilot_update_bytes']==row['unit_bytes'],'Actual empirical unit changed')
        adoptions[arm]={'status':'source_plan_and_delivered_compact_adoption_bindings_verified','unit_bytes':row['unit_bytes'],
                        'all_original_scientific_bytes_downloaded':False}
    longs=[row for row in submissions.values() if row['phase'] in ('native','segment')]
    waiting_path=ROOT+'/waiting-storage.json';waiting=None
    if waiting_path in remote:
        waiting=obj(waiting_path)
        require(waiting['status']=='blocked_storage_production_not_submitted' and waiting['production_submissions'] is False
                and waiting['historical_gates_changed'] is False and waiting['source_manifest_sha256']==pin
                and not longs and not packet['production_phase_entries'],'Storage block falsely claims no production/unknown marker')
        callback=submissions[ROOT+'/after-cpu']
        require(waiting['job_id']==callback['job_id'],'Storage failure is detached from the actual callback')
        lsf_exit_one(data(ROOT+'/after-cpu/stdout.'+callback['job_id']+'.log').decode(),callback)
        require(waiting['failed_storage_receipts'],'Storage block has no actual failed gate')
        for proof in waiting['failed_storage_receipts']:
            require(bound(proof) and obj(proof['path'])['status']=='failed','Actual failed capacity evidence missing/changed')
        for proof in waiting['adoptions']:require(bound(proof),'Validated adoption receipt absent')
    return {'version':'chemistry-production-independent-delivery-v1','status':'verified_compact_source_jobs_CPU_and_storage_scope',
        'verified_utc':datetime.now(timezone.utc).isoformat(),'observation':str(observation.resolve()),
        'raw_sha256':sha(raw),'source_manifest_sha256':pin,'source_files':12,'archive_members':13,
        'compact_files':len(files),'CPU_status':cpu_state,'adoptions':adoptions,
        'actual_submissions':list(submissions.values()),'production_jobs':longs,
        'blocked_storage':waiting,'unknown_submission_records':packet['unknown_submission_records'],
        'undownloaded_external_reference_bindings':unresolved,'native_arrays_downloaded':False,
        'original_scientific_bytes_recomputed_or_recertified':False,
        'scope':'Exact compact byte/schema/actual job/CPU/startup/storage evidence; absent/aged-out scheduler history is never fabricated'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observation',type=Path,required=True)
    parser.add_argument('--package',type=Path,default=LOCAL/'source01');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Independent audit output is immutable; choose a fresh file')
    result=verify(args.observation,args.package)
    with args.output.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'CPU_status':result['CPU_status'],
        'actual_jobs':[row['job_id'] for row in result['actual_submissions']],
        'production_jobs':[row['job_id'] for row in result['production_jobs']],
        'blocked_storage':result['blocked_storage'] is not None,'audit_sha256':sha(args.output.read_bytes())},indent=2))


if __name__=='__main__':main()
