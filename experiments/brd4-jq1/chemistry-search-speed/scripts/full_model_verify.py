"""Independent local stdlib source/byte/XML/LSF/schema audit, no numerics."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
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
    assert 'Subject:Job'+job+':<'+name+'>incluster<dcc>Done' in compact
    assert re.findall(r'inqueue<([^>]+)>',compact)==[queue]
    assert compact.count('Successfullycompleted.')==1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('observation',type=Path)
    args=parser.parse_args();out=args.observation;files=out/'files'
    transfer=load(out/'byte-verification.json')
    assert transfer['status']=='passed_compact_byte_transfer_only'
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
    jobs=load(out/'raw.json')['jobs'];cpu=jobs['cpu'];sub=load(files/'cpu/submission.json')
    assert sub['job_id']==cpu and sub['status']=='submitted' and sub['manifest_sha256']==pin
    assert sub['dependency'] is None and sub['argv'][-2:]==[pin,'cpu']
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
    report={'status':'passed_independent_CPU_byte_XML_LSF_schema_audit','source_files':len(manifest['files']),
            'source_manifest_sha256':pin,'CPU_job':cpu,'CPU_cases':len(cases),'CPU_counts':counts,
            'raw_XML_sha256':assessment['raw_xml_sha256'],'actual_terminal_proofs':terminal,
            'files_rehashed':len(transfer['files']),'jobs':jobs,'full_model_comparisons':{}}
    for arm in ['protenix','boltz']:
        if arm not in jobs:continue
        s=load(files/arm/'submission.json');job=jobs[arm]
        assert s['job_id']==job and s['status']=='submitted' and s['manifest_sha256']==pin
        assert s['dependency'] is None and s['argv'][-2:]==[pin,arm]
        assert s['argv'][s['argv'].index('-q')+1]=='gpul40s' and s['argv'][s['argv'].index('-n')+1]=='8'
        assert s['argv'][s['argv'].index('-gpu')+1]=='num=1:mode=exclusive_process'
        path=files/arm/'payload/assessment.json'
        if not path.exists():continue
        allocated=load(path)
        if allocated['status']!='passed':
            report['full_model_comparisons'][arm]={'status':'failed_preserved','assessment_sha256':sha(path)};continue
        assert allocated['job_id']==job and allocated['source_manifest_sha256']==pin
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
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
