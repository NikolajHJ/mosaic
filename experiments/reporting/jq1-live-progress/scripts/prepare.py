"""Local stdlib plan/source packaging; never scientific evaluation or tests."""
import ast
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[2]
LOCAL=REPO/'results/hpc/20261025-jq1-live-report-01'


def raw(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def main():
    inspected=json.loads((LOCAL/'inputs-01/raw.json').read_bytes())
    noise=json.loads((LOCAL/'inputs-02/raw.json').read_bytes())
    definitions={
        'protenix':('Protenix native-only continuation',10000,50000,.3,
            'Stopped at absolute20,000 after10,000 additional updates. Terminal raw probability probe was infeasible before finite differences; original gate remains failed.'),
        'boltz':('Boltz-2 affinity, planned30k',0,30000,.0025,
            'Stopped after10,000/30,000 updates. Terminal native-affinity bond finite differences disagreed with AD at the original scales; cause not established.'),
        'dbmol':('Boltz-2 DBMol binding objective, planned10k',0,10000,.0025,
            'Two2,500-update segments reached5,000/10,000. Second terminal DBMol contact-component bond check failed two-adjacent-scale criterion; this is not chemistry validity loss.')}
    plan={'version':'jq1-live-saved-report-v1','edition':'edition-0002','captured_at':inspected['captured_utc'],
          'monitor_remains_paused':True,'automatic_optimizer_submission':False,'arms':[]}
    proofs={}
    def proof(row):proofs[row['path']]=row
    parents=deepcopy(inspected['sources'])+[deepcopy(noise['source'])]
    for item in inspected['arms']:
        label,start,target,rate,detail=definitions[item['id']]
        arm={'id':item['id'],'label':label,'kind':'fixed','rate':rate,'global_start':start,
             'global_target':target,'target_updates':target,'scientific_status':'failed','status_detail':detail,
             'metric_labels':{},'segments':[]}
        for row in item['segments']:
            segment={key:deepcopy(row[key]) for key in ('id','global_offset','requested_updates','anchor_directory',
                       'case_directory','states','updates','summary','config','source_manifest_sha256','job_id','stdout','stderr')}
            segment['scientific_status']='passed' if item['id']=='dbmol' and row['global_offset']==0 else 'failed'
            segment['case_report']=deepcopy(row['report_binding'])
            segment['retained_index']=deepcopy(row['small_reports']['case-0/retained-frames.json']['binding'])
            arm['segments'].append(segment)
            for key in ('stdout','stderr'):proof(row[key])
            proof(row['submission']['binding'])
            for report in row['small_reports'].values():proof(report['binding'])
        plan['arms'].append(arm)
    n=inspected['noise'];root=n['root'];summary=n['summary']['data']
    stdout=next(row['binding'] for row in noise['proofs'] if '/stdout.' in row['binding']['path'])
    stderr=next(row['binding'] for row in noise['proofs'] if '/stderr.' in row['binding']['path'])
    segment={'id':'noisy-1000','global_offset':0,'requested_updates':1000,'anchor_directory':root+'/gpu',
             'case_directory':root+'/gpu/trajectory','states':n['states'],'updates':n['updates'],
             'summary':n['summary']['binding'],'config':noise['config'],'source_manifest_sha256':noise['source']['manifest_sha256'],
             'job_id':summary['job_id'],'scientific_status':'passed','retained_index':n['retained_index'],
             'stdout':stdout,'stderr':stderr}
    plan['arms'].append({'id':'protenix_noisy','label':'Protenix noisy-logit1k pilot','kind':'noisy',
          'rate':.3,'global_start':0,'global_target':1000,'target_updates':1000,'scientific_status':'passed',
          'status_detail':'Completed1,000 updates; original initial/terminal raw and logit checks passed. Independent pilot, not the50k random chemistry run.',
          'metric_labels':{},'segments':[segment]})
    proof(n['cpu_ready']['binding'])
    proof(n['result']['binding'])
    for row in noise['proofs']:proof(row['binding'])
    old_observation=LOCAL/'observation-02/raw.json'
    old_failure=json.loads(old_observation.read_bytes())
    old_root='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-live-report-01/'
    for row in old_failure['files']:
        if row['relative'] in ('cpu/tests.xml','cpu/tests.log','cpu/stdout.29812027.log','cpu/stderr.29812027.log','cpu/submission.json'):
            proof({'path':old_root+row['relative'],'bytes':row['bytes'],'sha256':row['sha256']})
    parents.extend([
        {'path':str(Path(row['binding']['path']).parent).replace('\\','/'),'manifest_sha256':row['binding']['sha256'],'files':12}
        for row in noise['proofs'] if row['binding']['path'].endswith('/source_manifest.json')])
    # Exact originals; no archived or historical source is repinned.
    parent_by_path={row['path']:row for row in parents}
    members={'plan.json':raw(plan),'plans-readme.md':b'# Exact saved-prefix plan\n\nNo native model execution or optimizer admission.\n'}
    for directory in ('scripts','templates','tests'):
        for path in sorted((HERE/directory).glob('*')):
            if path.is_file() and (path.suffix in ('.py','.html','.lsf','.md')):
                if path.name in ('inspect_inputs.py','inspect_noise.py','prepare.py','launch.py','publish.py','verify_http.py'):continue
                data=path.read_bytes().replace(b'\r\n',b'\n')
                if path.suffix=='.py':ast.parse(data,filename=str(path))
                members[path.relative_to(HERE).as_posix()]=data
    members['README.md']=(HERE/'README.md').read_bytes().replace(b'\r\n',b'\n')
    qualification=REPO/'experiments/validation/endpoint-qualification'
    for name in ('scripts/qualification.py','tests/test_qualification.py'):
        data=(qualification/name).read_bytes().replace(b'\r\n',b'\n')
        ast.parse(data,filename=name);members[name]=data
    members['endpoint-policy.md']=(qualification/'README.md').read_bytes().replace(b'\r\n',b'\n')
    counts=json.loads((LOCAL/'cpu-counts.json').read_bytes())
    manifest={'version':'jq1-live-report-source-v1','cpu_counts':counts,'parent_sources':list(parent_by_path.values()),
              'proofs':list(proofs.values()),'persistent_cap_bytes':1024**3,'scientific_results_changed':False,
              'monitor_remains_paused':True,'endpoint_policy_scope':'Allocated toy diagnostics only; no native endpoint qualification or production gate replacement',
              'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(members.items())}}
    manifest['historical_failure']={'cpu_job_id':'29812027','status':'failed_module_attribution_metadata',
        'raw_fixture_passes':121,'scientific_render_pass':False,'preserved':True,
        'new_release_change':'Declare exact tests.-qualified raw JUnit module names; repeat allocated suite'}
    manifest['historical_visual_failure']={'source_generation':'02','render_job_id':'29812097',
        'data_presentation_assessment':'passed','publication':'edition-0001_preserved',
        'browser_QA':'failed_progress_bar_inline_layout','saved_scientific_data_changed':False,
        'new_release_change':'Block-level5px progress bars and bounded visible primary metrics; all scalar assets unchanged'}
    members['source_manifest.json']=raw(manifest)
    out=LOCAL/'source03';out.mkdir(exist_ok=False)
    (out/'source_manifest.json').write_bytes(members['source_manifest.json'])
    (out/'plan.json').write_bytes(members['plan.json'])
    with tarfile.open(out/'source.tar.gz','x:gz') as target:
        for name,data in sorted(members.items()):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o444;info.mtime=0
            target.addfile(info,io.BytesIO(data))
    with tarfile.open(out/'source.tar.gz') as target:
        items=target.getmembers();assert len(items)==len(members)
        for item in items:assert item.isfile() and target.extractfile(item).read()==members[item.name]
    receipt={'status':'local_static_only','manifest_sha256':hashlib.sha256(members['source_manifest.json']).hexdigest(),
             'archive_bytes':(out/'source.tar.gz').stat().st_size,'archive_sha256':hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest(),
             'files':len(members)-1,'cpu_counts':counts,'allocated_checks_passed':False}
    (out/'preparation.json').write_bytes(raw(receipt));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
