"""Independent stdlib source/actual-job/raw-XML/LSF byte audit. No models."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261026-endpoint-repair-03'
SPECS={
 'continuations':{'source':'nonchem-endpoint-release-06/source','pin':'1519d5761069f73fb6659bc1740cb04c155a04bc3218f143a3f28a73e65a1f69',
    'package':LOCAL/'continuation06/source01','cases':223,
    'counts':{'tests.test_qualification':19,'tests.test_admission':50,'tests.test_continuation_driver':49,'tests.test_continuation_ops':100,'tests.test_current_capacity':5}},
 'pro_precision':{'source':'protenix-endpoint-fix-release-03/source','pin':'a701b12bd1e69f5aa7169ad5339f2456c3d728b855257b4270437d37f9da68f3',
    'package':REPO/'results/hpc/20261026-protenix-endpoint-fix-03/source01','cases':278,
    'counts':{'tests.test_protenix_precision_reference':89,'tests.test_protenix_precision_admission':30,
              'tests.test_protenix_precision_native':82,'tests.test_protenix_precision_ops':72,'tests.test_current_capacity':5}}}

def require(v,message):
    if not v:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(raw):return json.loads(raw,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+v)))


def resources(phase):
    if phase in ('gpu','validate','segment'):
        return ['-q','gpul40s','-n','4','-R','span[hosts=1]','-R','rusage[mem=8GB]',
                '-gpu','num=1:mode=exclusive_process','-W','23:00' if phase=='segment' else '02:00']
    return ['-q','hpc','-n','4' if phase=='cpu' else '1','-R','span[hosts=1]',
            '-R','rusage[mem=4GB]' if phase=='cpu' else 'rusage[mem=2GB]',
            '-W','01:00' if phase=='cpu' else '00:20']


def lsf(raw,job,name,queue):
    compact=''.join(raw.decode().split())
    require(compact.count('Subject:Job')==1 and
        re.findall(r'Subject:Job(\d+):<([^>]+)>incluster<([^>]+)>(Done|Exited)',compact)==[(job,name,'dcc','Done')]
        and compact.count('Successfullycompleted.')==1
        and re.findall(r'inqueue<([^>]+)>',compact)==[queue], 'Same-job terminal LSF differs')


def budget(value):
    require(value.get('status')=='passed' and value.get('reserve_bytes')==4*1024**3
        and value.get('file_reserve')==100000 and type(value.get('available_bytes')) is int
        and type(value.get('required_bytes')) is int
        and value['available_bytes']>=value['required_bytes'], 'Full byte/reserve gate differs')
    return {k:value[k] for k in ('status','required_bytes','available_bytes','reserve_bytes','file_reserve')}

def verify(directory):
    packet=read((directory/'raw.json').read_bytes());files={};bindings=[]
    for row in packet['files']:
        raw=base64.b64decode(row['base64'],validate=True)
        require(len(raw)==row['bytes'] and digest(raw)==row['sha256'] and row['path'] not in files,'Duplicate/changed delivered metadata')
        files[row['path']]=raw;bindings.append({k:v for k,v in row.items() if k!='base64'})
    result={'status':'verified_repaired_CPU_source_actual_submission_scope','captured_utc':packet['captured_utc'],
        'metadata_files':len(files),'proofs':bindings,'CPU':{},'actual_jobs':{},'native_jobs':[],
        'optimizer_jobs':[],'CPU_certificates':{},'storage':{},'monitor_resumed':False,'local_scientific_evaluations':False}
    raw_scheduler=''.join(packet['scheduler']['stdout'].split())
    for label,spec in SPECS.items():
        owner=packet['roots'][label];root=owner['path']
        require(owner['source_manifest_sha256']==spec['pin'] and owner['source_files_rehashed']==16,'Different remote source')
        manifest_raw=(spec['package']/'source_manifest.json').read_bytes();manifest=read(manifest_raw)
        require(digest(manifest_raw)==spec['pin'] and manifest['cpu_counts']==spec['counts'],'Different prepared source/counts')
        with tarfile.open(spec['package']/'source.tar.gz','r:gz') as tar:
            names=set()
            for member in tar.getmembers():
                path=PurePosixPath(member.name)
                require(member.isfile() and not path.is_absolute() and '..' not in path.parts and '\\' not in member.name and member.name not in names,'Unsafe/duplicate source member')
                names.add(member.name);raw=tar.extractfile(member).read()
                require(raw==manifest_raw if member.name=='source_manifest.json' else digest(raw)==manifest['files'][member.name], 'Source member changed')
            require(names==set(manifest['files'])|{'source_manifest.json'} and len(names)==17,'Source archive scope changed')
        source='/'.join(root.split('/')[:-1])+'/'+spec['source']
        require(files[source+'/source_manifest.json']==manifest_raw,'Delivered manifest differs')
        jobs=owner['jobs'];cpu=jobs['cpu']
        for name,job in jobs.items():
            parent=root+'/'+name;sub=read(files[parent+'/submission.json']);args=sub['argv']
            require(sub['status']=='submitted' and sub['job_id']==job and sub['source_manifest_sha256']==spec['pin'],'Actual submission identity differs')
            phase=sub['phase'];prefix=['bsub',*resources(phase)]
            expected_name=('mosaic-endpoint06-'+name if label=='continuations' else 'mosaic-pro-precision03-'+name)
            expected=prefix+['-J',expected_name,'-oo',parent+'/stdout.%J.log','-eo',parent+'/stderr.%J.log']
            if phase.startswith('after-'):
                producer='cpu' if phase=='after-cpu' else name.replace('-after-','-',1)
                require(producer in jobs and sub['dependency']=='ended('+jobs[producer]+')','Callback producer differs')
                expected+=['-w',sub['dependency']]
            else:require(sub.get('dependency') is None,'Independent phase acquired a dependency')
            expected+=['bash','-l',source+'/scripts/'+('continuation_stage.lsf' if label=='continuations' else 'protenix_precision_stage.lsf'),source,spec['pin'],phase]
            if label=='continuations':expected +=[sub['arm'],str(sub['offset'])]
            require(args==expected,'Exact actual submission/resources/launcher differ')
            require(args[args.index('-q')+1]==('gpul40s' if sub['phase'] in ('validate','segment','gpu') else 'hpc'), 'Actual queue differs')
            # Restrict the scheduler block to this same job, rather than accepting
            # another job's later state when history is absent.
            block=re.search(r'Job<'+job+r'>(.*?)(?=Job<\d+>|$)',raw_scheduler)
            state=re.findall(r'Status<([^>]+)>',block.group(1)) if block else []
            require(len(state)<=1,'Ambiguous actual scheduler state')
            result['actual_jobs'][job]={'parent':parent,'state':state[0] if state else 'not_in_live_history','phase':sub['phase'],
                'submission_sha256':digest(files[parent+'/submission.json'])}
            if name=='gpu' or sub['phase']=='validate':result['native_jobs'].append(job)
            if sub['phase']=='segment':result['optimizer_jobs'].append(job)
            if state==['DONE']:
                lsf(files[parent+'/stdout.'+job+'.log'],job,expected_name,args[args.index('-q')+1])
        sub=read(files[root+'/cpu/submission.json']);args=sub['argv']
        log=files[root+'/cpu/stdout.'+cpu+'.log'].decode();compact=''.join(log.split())
        expected_name=args[args.index('-J')+1]
        require(compact.count('Subject:Job')==1 and re.findall(r'Subject:Job(\d+):<([^>]+)>incluster<([^>]+)>(Done|Exited)',compact)==[(cpu,expected_name,'dcc','Done')], 'Actual same-job CPU LSF success missing')
        require(compact.count('Successfullycompleted.')==1 and re.findall(r'inqueue<([^>]+)>',compact)==['hpc'],'CPU LSF failed/wrong queue')
        xml=ET.fromstring(files[root+'/cpu/payload/tests.xml']);cases=list(xml.iter('testcase'))
        require(len(cases)==spec['cases'] and dict(Counter(v.get('classname') for v in cases))==spec['counts'],'Actual raw CPU attribution/count differs')
        require(not any(v.tag in ('failure','error','skipped') for v in xml.iter()),'Raw CPU failure/error/skip present')
        assessment=read(files[root+'/cpu/payload/assessment.json'])
        require(assessment['status']=='passed' and assessment['cases']==spec['cases'],'CPU assessment differs')
        result['CPU'][label]={'status':'passed','cases':len(cases),'job_id':cpu,
                             'raw_xml_sha256':digest(files[root+'/cpu/payload/tests.xml']),
                             'same_job_LSF_sha256':digest(files[root+'/cpu/stdout.'+cpu+'.log'])}
        cert_raw=files[root+'/cpu-certificate.json'];cert=read(cert_raw)
        require(cert['status']=='passed' and cert['cases']==spec['cases']
            and cert['source_manifest_sha256']==spec['pin'] and cert['plan_sha256']==digest(files[source+'/plan.json'])
            and cert['original_scientific_cpu_replaced'] is False,'CPU certificate scope differs')
        for proof in cert['proofs']:
            raw=files[proof['path']]
            require(len(raw)==proof['bytes'] and digest(raw)==proof['sha256'],'CPU certificate proof changed/missing')
        result['CPU_certificates'][label]={'sha256':digest(cert_raw),'proofs_rehashed':len(cert['proofs'])}
        result['storage'][label]={}
        for parent in ('cpu','after-cpu'):
            filename='continuation-storage-entry.json' if label=='continuations' else 'precision-storage-entry.json'
            path=root+'/'+parent+'/'+filename;gate=read(files[path]);outer=budget(gate)
            current=gate['current_chemistry_and_new_scopes'];budget(current)
            require(current['new_group_allowance_bytes']==822083584 and current['original_gate_changed'] is False,
                    'New group or original full capacity changed')
            prior=current['parent_gate'];budget(prior)
            require(prior['adopted_full_production'] is True and prior['original_gate_changed'] is False
                    and set(prior['chemistry'])=={'boltz','protenix'}
                    and all(type(v['full']) is int and v['full']>0 for v in prior['chemistry'].values()),
                    'Both full chemistry futures missing')
            require(current['required_bytes']==prior['required_bytes']+current['new_group_allowance_bytes']+current['additional_projection_excess_bytes']
                    and current['available_bytes']==prior['available_bytes'],'Full group arithmetic differs')
            result['storage'][label][parent]={**outer,'proof_sha256':digest(files[path]),
                                            'both_chemistry_futures_retained':True}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    require(re.fullmatch('[a-z0-9-]{1,70}',args.name) is not None and not args.output.exists(),'Fresh safe observation/output required')
    result=verify(LOCAL/args.name)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('proofs','actual_jobs')},indent=2))

if __name__=='__main__':main()
