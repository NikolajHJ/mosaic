"""Independent local stdlib source/byte/XML/LSF/schema audit, no numerics."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[4]
LOCAL = REPO/'results/hpc/20261025-chemistry-full-model-01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def lsf(path, job, name, queue):
    raw = Path(path).read_text(); compact = ''.join(raw.split())
    assert compact.count('Subject:Job') == 1
    subjects=re.findall(r'Subject:\s*Job\s*(\d+)\s*:\s*<([^>]+)>\s*in\s*cluster\s*<([^>]+)>\s*([A-Za-z]+)[ \t]*(?=\r?\n|$)',raw)
    assert [tuple(''.join(field.split()) for field in row) for row in subjects]==[(job,name,'dcc','Done')]
    assert re.findall(r'inqueue<([^>]+)>',compact)==[queue]
    assert compact.count('Successfullycompleted.')==1


def failed_lsf(path, job, name, queue):
    """Authenticate an actual failed job without manufacturing live history."""
    raw=Path(path).read_text();compact=''.join(raw.split())
    subjects=re.findall(r'Subject:\s*Job\s*(\d+)\s*:\s*<([^>]+)>\s*in\s*cluster\s*<([^>]+)>\s*([A-Za-z]+)[ \t]*(?=\r?\n|$)',raw)
    assert [tuple(''.join(field.split()) for field in row) for row in subjects]==[(job,name,'dcc','Exited')]
    assert compact.count('Subject:Job')==1
    assert re.findall(r'inqueue<([^>]+)>',compact)==[queue]
    assert re.findall(r'Exitedwithexitcode(\d+)\.',compact)==['1']
    assert 'Successfullycompleted.' not in compact


def expected_argv(phase,pin,root,source,cpu):
    if phase=='cpu':resources=['-q','hpc','-n','16','-R','span[hosts=1]','-R','rusage[mem=2GB]','-W','02:00']
    elif phase=='after-cpu':resources=['-q','hpc','-n','1','-R','span[hosts=1]','-R','rusage[mem=2GB]','-W','00:20']
    else:resources=['-q','gpul40s','-n','8','-R','span[hosts=1]','-R','rusage[mem=4GB]','-gpu','num=1:mode=exclusive_process','-W','02:00']
    command=['bsub',*resources,'-J','mosaic-chem-full01-'+phase,'-oo',root+'/'+phase+'/stdout.%J.log','-eo',root+'/'+phase+'/stderr.%J.log']
    if phase=='after-cpu':command+=['-w','ended('+cpu+')']
    return command+['bash','-l',source+'/scripts/full_model_stage.lsf',source,pin,phase]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('observation',type=Path)
    args=parser.parse_args();out=args.observation;files=out/'files'
    transfer=load(out/'byte-verification.json')
    assert transfer['status']=='passed_compact_byte_transfer_only'
    assert sha(out/'raw.json')==transfer['raw_sha256']==load(out/'receipt.json')['raw_sha256']
    for row in transfer['files']:
        p=files/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    package=LOCAL/'source02';prep=load(package/'preparation.json')
    assert sha(package/'source.tar.gz')==prep['archive_sha256']
    assert (package/'source.tar.gz').stat().st_size==prep['archive_bytes']
    manifest=load(package/'source_manifest.json');pin=sha(package/'source_manifest.json')
    assert pin==prep['manifest_sha256']==sha(files/'_source/source_manifest.json')
    with tarfile.open(package/'source.tar.gz') as archive:
        rows=archive.getmembers()
        assert len(rows)==len(manifest['files'])+1 and all(row.isfile() for row in rows)
        assert len({row.name for row in rows})==len(rows)
        for row in rows:
            raw=archive.extractfile(row).read()
            if row.name=='source_manifest.json':assert hashlib.sha256(raw).hexdigest()==pin
            else:assert hashlib.sha256(raw).hexdigest()==manifest['files'][row.name]
            if row.name.endswith('.py'):ast.parse(raw,filename=row.name)
    raw_observation=load(out/'raw.json');jobs=raw_observation['jobs'];cpu=jobs['cpu'];sub=load(files/'cpu/submission.json')
    remote_root=raw_observation['root'];remote_source=raw_observation['source']
    assert remote_root=='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/chemistry-full-model-01'
    assert remote_source=='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/chemistry-full-model-release-01/source'
    assert sub['job_id']==cpu and sub['status']=='submitted' and sub['manifest_sha256']==pin
    assert sub['dependency'] is None and sub['argv'][-2:]==[pin,'cpu']
    assert sub['argv']==expected_argv('cpu',pin,remote_root,remote_source,cpu)
    assert sub['argv'][sub['argv'].index('-q')+1]=='hpc' and sub['argv'][sub['argv'].index('-n')+1]=='16'
    assessment=load(files/'cpu/payload/assessment.json')
    assert assessment['status']=='passed' and assessment['job_id']==cpu
    assert assessment['source_manifest_sha256']==pin and assessment['original_results_changed'] is False
    assert assessment['external_before']==assessment['external_after']
    cases=ET.parse(files/'cpu/payload/tests.xml').getroot().findall('.//testcase')
    assert cases and all(not list(case) for case in cases)
    assert len({(case.get('classname'),case.get('name')) for case in cases})==len(cases)
    counts=dict(Counter(case.get('classname') for case in cases))
    assert counts==manifest['cpu_counts']==assessment['cpu_counts']
    assert len(cases)==assessment['cpu_cases']==sum(counts.values())
    assert sha(files/'cpu/payload/tests.xml')==assessment['raw_xml_sha256']
    assert re.search(r'\b[1-9]\d*\s+deselected\b',(files/'cpu/payload/tests.log').read_text()) is None
    lsf(files/'cpu'/('stdout.'+cpu+'.log'),cpu,'mosaic-chem-full01-cpu','hpc')
    terminal=[]
    for path in (files/'after-cpu').glob('cpu-terminal-query-*.json'):
        row=load(path);text=''.join(row['stdout'].split())
        if row['returncode']==0 and re.findall(r'Job<(\d+)>',text)==[cpu] and re.findall(r'Status<([^>]+)>',text)==['DONE']:
            assert re.findall(r'Queue<([^>]+)>',text)==['hpc']
            assert re.findall(r'JobName<([^>]+)>',text)==['mosaic-chem-full01-cpu']
            terminal.append({'path':str(path.relative_to(files)),'sha256':sha(path)})
    assert terminal,'Actual recorded same-job CPU DONE proof is required'
    cert_path=files/'cpu-certificate.json'
    certificate_sha=None
    if cert_path.exists():
        cert=load(cert_path);certificate_sha=sha(cert_path)
        assert cert['version']=='chemistry-full-model-cpu-certificate-v1' and cert['status']=='passed'
        assert cert['source_manifest_sha256']==pin and cert['job_id']==cpu and cert['cpu_counts']==counts
        assert cert['plans']=={arm:manifest['files']['plans/'+arm+'.json'] for arm in ['protenix','boltz']}
        expected={'cpu_submission':'cpu/submission.json','cpu_assessment':'cpu/payload/assessment.json',
                  'cpu_xml':'cpu/payload/tests.xml','cpu_lsf':'cpu/stdout.'+cpu+'.log'}
        queries=cert['scheduler_queries'];assert 1<=len(queries)<=13
        expected.update({str(i):'after-cpu/cpu-terminal-query-'+str(i).zfill(3)+'.json' for i in range(len(queries))})
        observed=[]
        for key,relative in expected.items():
            binding=queries[int(key)] if key.isdigit() else cert[key]
            assert binding['path']==remote_root+'/'+relative
            assert (files/relative).stat().st_size==binding['bytes'] and sha(files/relative)==binding['sha256']
            if key.isdigit():
                row=load(files/relative);compact=''.join(row['stdout'].split())
                assert row['returncode']==0 and re.findall(r'Job<(\d+)>',compact)==[cpu]
                assert re.findall(r'Queue<([^>]+)>',compact)==['hpc']
                assert re.findall(r'JobName<([^>]+)>',compact)==['mosaic-chem-full01-cpu']
                states=re.findall(r'Status<([^>]+)>',compact);assert len(states)==1;observed+=states
        assert observed[-1]=='DONE' and all(state in ['PEND','RUN'] for state in observed[:-1])
    report={'status':'passed_independent_CPU_byte_XML_LSF_schema_audit','source_files':len(manifest['files']),
            'source_manifest_sha256':pin,'CPU_job':cpu,'CPU_cases':len(cases),'CPU_counts':counts,
            'raw_XML_sha256':assessment['raw_xml_sha256'],'actual_terminal_proofs':terminal,
            'files_rehashed':len(transfer['files']),'jobs':jobs,'full_model_comparisons':{}}
    report['cpu_certificate_sha256']=certificate_sha
    after=load(files/'after-cpu/submission.json')
    assert after['job_id']==jobs['after-cpu'] and after['status']=='submitted'
    assert after['manifest_sha256']==pin and after['dependency']=='ended('+cpu+')'
    assert after['argv']==expected_argv('after-cpu',pin,remote_root,remote_source,cpu)
    if any(arm in jobs for arm in ['protenix','boltz']):
        lsf(files/'after-cpu'/('stdout.'+jobs['after-cpu']+'.log'),jobs['after-cpu'],'mosaic-chem-full01-after-cpu','hpc')
        dispatch=load(files/'after-cpu/gpu-submissions-complete.json')
        assert dispatch['status']=='submitted_not_passed' and dispatch['source_manifest_sha256']==pin
        assert dispatch['external_after']==assessment['external_after']
        gate=dispatch['storage'];assert gate['status']=='passed'
        assert gate['additional_allowance_bytes']==2*1024**3 and gate['reserve_bytes']==4*1024**3
        assert gate['file_reserve']==100000 and gate['available_bytes']>=gate['required_bytes']
        assert gate['required_bytes']==gate['whole_original_gate']['work3']['required_bytes']+2*1024**3
        assert gate==load(files/'after-cpu/full-model-storage-two-gpu-dispatch.json')
        report['dispatch_required_bytes']=gate['required_bytes'];report['dispatch_available_bytes']=gate['available_bytes']
    for arm in ['protenix','boltz']:
        if arm not in jobs:continue
        s=load(files/arm/'submission.json');job=jobs[arm]
        assert s['job_id']==job and s['status']=='submitted' and s['manifest_sha256']==pin
        assert s['dependency'] is None and s['argv'][-2:]==[pin,arm]
        assert s['argv']==expected_argv(arm,pin,remote_root,remote_source,cpu)
        assert certificate_sha is not None and s['cpu_certificate_sha256']==certificate_sha
        assert dispatch['submissions'][arm]==s
        assert s['argv'][s['argv'].index('-q')+1]=='gpul40s' and s['argv'][s['argv'].index('-n')+1]=='8'
        assert s['argv'][s['argv'].index('-gpu')+1]=='num=1:mode=exclusive_process'
        path=files/arm/'payload/assessment.json'
        if not path.exists():continue
        allocated=load(path)
        if allocated['status']!='passed':
            assert allocated['status']=='failed' and allocated['job_id']==job
            assert allocated['phase']==arm and allocated['source_manifest_sha256']==pin
            assert allocated['original_results_changed'] is False and allocated['production_authorized'] is False
            assert allocated['new_cpu_certificate']['status']=='passed'
            assert allocated['new_cpu_certificate']['sha256']==certificate_sha
            assert allocated['external_before']==assessment['external_before']
            failed_lsf(files/arm/('stdout.'+job+'.log'),job,'mosaic-chem-full01-'+arm,'gpul40s')
            plan=load(files/'_source/plans'/(arm+'.json'))
            comparison=load(files/arm/'payload/comparison/assessment.json')
            assert comparison['status']=='failed' and comparison['original_results_changed'] is False
            assert comparison['plan_sha256']==sha(files/'_source/plans'/(arm+'.json'))
            nested={}
            for relative in ['payload/comparison/assessment.json',
                             'payload/comparison/combined/assessment.json',
                             'payload/comparison/combined/study/summary.json',
                             'payload/comparison/combined/paired-checks/assessment.json']:
                q=files/arm/relative
                if q.exists():
                    value=load(q);assert value['status']=='failed'
                    if 'original_results_changed' in value:assert value['original_results_changed'] is False
                    if 'job_id' in value:assert value['job_id']==job
                    nested[relative]={'sha256':sha(q),'status':value['status'],'error':value.get('error')}
            study=load(files/arm/'payload/comparison/combined/study/summary.json')
            assert study['mode']=='validate' and study['source_manifest_sha256']==plan['source']['manifest_sha256']
            assert study['config_sha256']==plan['inputs']['config']['sha256']
            baseline=load(files/arm/'payload/comparison/baseline/assessment.json')
            assert baseline['status'] in ['passed_original_baseline','passed_original_runtime_failure_preserved']
            report['full_model_comparisons'][arm]={'status':'failed_preserved',
                 'job_id':job,'assessment_sha256':sha(path),
                 'actual_terminal_proof_sha256':sha(files/arm/('stdout.'+job+'.log')),
                 'baseline_classification':baseline['status'],'nested_failures':nested,
                 'production_readiness_passed':False,
                 'scope':'Compact byte/JSON/source/CPU/submission/actual failed LSF audit; no local numerical evaluation or passed GPU comparison'}
            continue
        assert allocated['job_id']==job and allocated['source_manifest_sha256']==pin
        assert allocated['new_cpu_certificate']['status']=='passed' and allocated['new_cpu_certificate']['sha256']==certificate_sha
        assert allocated['external_before']==allocated['external_after']
        lsf(files/arm/('stdout.'+job+'.log'),job,'mosaic-chem-full01-'+arm,'gpul40s')
        comparison=load(files/arm/'payload/comparison/assessment.json')
        assert sha(files/arm/'payload/comparison/assessment.json')==allocated['comparison_sha256']
        assert comparison['status']=='passed_full_model_comparison'
        assert comparison['exact_initial_arrays_and_key'] is True and comparison['matrix_trajectory_bitwise_claim'] is False
        assert comparison['original_results_changed'] is False
        plan=load(files/'_source/plans'/(arm+'.json'))
        assert comparison['plan_sha256']==allocated['plan_sha256']==sha(files/'_source/plans'/(arm+'.json'))
        for variant in ['baseline','combined']:
            v=comparison['variants'][variant];summary=v['driver_summary']
            assert v['original_results_changed'] is False and summary['mode']=='validate'
            assert summary['source_manifest_sha256']==plan['source']['manifest_sha256']
            assert summary['config_sha256']==plan['inputs']['config']['sha256']
            assert all(summary[name]['status']=='passed' for name in ['native_hard_parity','native_gradient_checks','chemistry_assay'])
            assert set(v['pilot_reports'])=={'early','high'}
            for pilot in v['pilot_reports'].values():
                assert pilot['status']=='passed' and len(pilot['updates'])==10 and len(pilot['states'])==11
                assert [u['step'] for u in pilot['updates']]==list(range(10))
            if variant=='combined':
                assert summary['status']==summary['runtime_gate']['status']=='passed'
                paired=v['paired_checks'];assert paired['status']=='passed' and paired['atol']==2e-5
                assert paired['epsilons']==[.01,.003,.001] and len(paired['points'])==4
                for point in paired['points']:
                    assert point['status']=='passed'
                    for kind, raw_key in [('original','original_fd'),('combined','combined_fd')]:
                        assert isinstance(point[raw_key],list) and len(point[raw_key])==2
                        raw_report=load(files/arm/'payload/comparison/combined/paired-checks'/point['label']/(kind+'-fd')/'report.json')
                        assert raw_report['status']=='passed' and raw_report['directions']==point[raw_key]
            else:
                assert v['classification']['status'] in ['passed_original_baseline','passed_original_runtime_failure_preserved']
        report['full_model_comparisons'][arm]={'status':'passed_full_model_comparison',
                 'assessment_sha256':sha(path),'comparison_sha256':allocated['comparison_sha256'],
                 'scope':'Raw native arrays remain remotely authenticated references; compact original reports/LSF/schema verified locally'}
    if 'production_inventory' in raw_observation:
        inventory=raw_observation['production_inventory']
        expected={'random_protenix':'protenix-extended-03','boltz_chemistry':'boltz-lr0025-03'}
        assert set(inventory)==set(expected)
        base='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/'
        for arm,suffix in expected.items():
            assert inventory[arm]['root']==base+suffix
            assert inventory[arm]['production_entries']==[]
        report['chemistry_production_inventory']=inventory
        report['inventory_scope']='Read-only observation of declared original chemistry production marker/segment parents; no new submission'
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
