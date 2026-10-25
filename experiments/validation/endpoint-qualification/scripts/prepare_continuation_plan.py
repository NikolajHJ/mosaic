"""Prepare exact read-only child provenance from authenticated local JSON bytes."""
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
RUN = REPO / 'experiments/validation/endpoint-qualification/runs/20261025-continuation-admission-01'
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    report = read(REPO/'results/hpc/20261025-jq1-live-report-01/source04/plan.json')
    source = read(REPO/'results/hpc/20261025-jq1-live-report-01/source04/source_manifest.json')
    observed = read(REPO/'results/hpc/20261025-continuation-admission-01/raw.json')
    evidence = REPO/'results/hpc/20261025-nonchem-stride20-startup-03/evidence-06'
    inventory = read(evidence/'metadata.json')['files']
    all_bindings = {path:{'path':path,'bytes':row['bytes'],'sha256':row['sha256']}
                    for path,row in inventory.items()}
    for row in source['proofs']:
        all_bindings[row['path']] = row
    for row in read(REPO/'results/hpc/20261025-continuation-admission-01/arguments/bindings.json'):
        if row['path'] in all_bindings:
            assert row==all_bindings[row['path']]
        all_bindings[row['path']]=row
    pairs = {'protenix':('0117',[20000,30000,40000],10000),
             'boltz':('0040',[10000,20000],10000),
             'dbmol':('0099',[5000,7500],2500)}
    plan = {'version':'endpoint-aware-continuation-plan-v1',
            'admission_policy':'prospective-endpoint-admission-v1',
            'user_authorized_prospective_checks_before_children':True,
            'original_failures_preserved':True,'optimization_settings_changed':False,
            'monitor_remains_paused':True,'matrix_stride':20,'loss_stride':1,
            'control_root':BASE+'/nonchem-endpoint-continuation-01',
            'root':BASE+'/nonchem-endpoint-continuation-01',
            'source_root':BASE+'/nonchem-endpoint-release-01/source',
            'arms':[]}
    for arm, (number,offsets,length) in pairs.items():
        old = next(row for row in report['arms'] if row['id']==arm)
        last = old['segments'][-1]
        current = next(row for row in observed['arms'] if row['id']==arm)
        local_argv = evidence/'files'/number/'driver-invocation.json'
        data = local_argv.read_bytes()
        remote_argv = next(path for path,row in inventory.items()
                           if row['local_path']==f'files/{number}/driver-invocation.json')
        assert hashlib.sha256(data).hexdigest()==inventory[remote_argv]['sha256']
        argv = json.loads(data)['args']
        inputs = {'original_driver_argv':argv,'original_driver_invocation':all_bindings[remote_argv],
                  'original_native_report':current['native_gradient_checks/report.json']['binding']}
        for flag in ('--cpu-certificate','--reference','--pocket','--ready'):
            path = argv[argv.index(flag)+1]
            assert path in all_bindings, path
            inputs[flag[2:].replace('-','_')] = all_bindings[path]
        terminal = current['terminal_gradient_checks/report.json']['data']
        def directions(rows):
            result={'X':[],'E_upper':[]}
            for value in rows:
                item={key:value[key] for key in ('row','source')}
                item['target']=value.get('target',value.get('destination'))
                assert type(item['target']) is int
                result[value['field']].append(item)
            return result
        if arm=='dbmol':
            original = {name:directions(terminal['components'].get(name,{}).get('rows',[]))
                        for name in ('dbmol_affinity_loss','dbmol_contact_loss','dbmol_confidence_loss','dbmol_total_loss')}
        else:
            original = directions(terminal.get('directions',terminal.get('rows',[])))
        root = str(Path(last['summary']['path']).parent.parent.parent).replace('\\','/')
        failed = {key:last[key] for key in ('summary','stdout','stderr','states','updates')}
        failed.update(job_id=last['job_id'],parent=root,
                      endpoint=current['endpoint_snapshot'],
                      submission=current['producer_submission']['binding'],
                      terminal=current['terminal_gradient_checks/report.json']['binding'])
        original_source=next(row for row in source['parent_sources']
                             if row['manifest_sha256']==last['source_manifest_sha256'])
        item={'id':arm,'label':old['label'],'rate':old['rate'],
              'source':original_source,'config':last['config'],
              'start_global':offsets[0],'global_start':offsets[0],
              'target_global':old['global_target'],'global_target':old['global_target'],
              'segment_updates':length,'segment_steps':length,'offsets':offsets,
              'original_argv':argv,'inputs':inputs,'failed_parent':failed,
              'parent_snapshot':current['endpoint_snapshot'],
              'original_directions':original,
              'original_direction_scope':'saved_failed_endpoint',
              'original_direction_limits':'Authentic terminal observations only; fields not reached remain explicit empty',
              'future_endpoint_direction_scope':'not_preexisting_at_new_endpoint',
              'expected':{'saved_parent_scientific_status':'failed',
                          'parent_committed_states':current['prefixes']['states']['newline_committed_rows'],
                          'parent_committed_updates':current['prefixes']['updates']['newline_committed_rows'],
                          'original_native_report':current['native_gradient_checks/report.json']['data'],
                          'original_terminal_report':terminal}}
        plan['arms'].append(item)
    target=RUN/'plan.json'
    target.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'path':str(target),'bytes':target.stat().st_size,
                      'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                      'arms':[(a['id'],a['offsets'],a['global_target']) for a in plan['arms']]}))


if __name__=='__main__':main()
