"""Saved history adapters for dense originals and interrupted committed prefixes.

Metadata capture uses stdlib only. Numerical frame/transition/boundary validation
uses the unchanged report_data collector on allocated report CPU. No model call.
"""
from copy import deepcopy
from pathlib import Path
import types
import json
import re
import quick_inputs as inputs
import report_exploratory as exploratory

DENSE='dense-fixed-history-v1'
PARTIAL='terminal-partial-history-v1'
SCHEMAS=(DENSE,PARTIAL)
HISTORY_PROOFS=('producer_submission','science_manifest','producer_manifest','latest_metadata','latest_snapshot','retained_endpoint_snapshot')


def check_partial_lsf(path,job):
    text=''.join(Path(path).read_text().split())
    inputs.require(text.count('Subject:Job')==1 and re.search(r'Subject:Job'+re.escape(job)+r':<[^>]+>incluster<[^>]+>Exited',text),
                   'Original same-job terminal LSF failure is missing')
    inputs.require('Successfullycompleted.' not in text and (re.search(r'Exitedwithexitcode[1-9][0-9]*\.',text)
                   or re.search(r'Exitedwithsignaltermination:[1-9][0-9]*\.',text)),
                   'Missing original nonzero exit or signal termination')


def _counts(states,updates,offset,roots,dense=False):
    counts={};kept=0
    for label,record in [('states',states),('updates',updates)]:
        n=0
        with Path(record['path']).open('rb') as f:
            for line in f:
                inputs.require(line.endswith(b'\n'),'Incomplete terminal history row')
                row=json.loads(line)
                inputs.require(row.get('step')==n and row.get('global_step',offset+n)==offset+n,'Noncontiguous history prefix')
                if label=='states':
                    if dense:
                        inputs.require(row.get('snapshot') and row.get('snapshot_sha256'),'Dense history lost original snapshot binding')
                    else:
                        inputs.require(type(row.get('snapshot_retained')) is bool,'Missing original retained marker')
                        kept+=int(row['snapshot_retained'])
                n+=1
        counts[label]=n
    inputs.require(counts['states']==counts['updates']+1 and counts['updates']>0,'Incomplete or empty history prefix')
    n=counts['updates']
    if dense:kept=n//20+1+int(n%20!=0)
    return {'committed_updates':n,'scalar_states':n+1,'retained_frames':kept}


def capture_segment(spec,profile):
    roots=profile['allowed_roots'];study=Path(spec['run']);case=study/'case-0';dense=bool(spec.get('dense_saved'))
    config=inputs.binding(spec['config'],roots);cfg=inputs.load(config['path'])
    offset=spec.get('global_offset',0);budget=spec.get('requested_updates',cfg['steps']);job=str(spec['job_id'])
    inputs.require(job.isdigit() and type(offset) is int and offset>=0 and type(budget) is int and budget>0,'Invalid original history scope')
    stdout=inputs.binding(spec['stdout'],roots);stderr=inputs.binding(spec['stderr'],roots)
    submission=inputs.binding(spec['submission'],roots);submitted=inputs.load(submission['path'])
    inputs.require(str(submitted['job_id'])==job and submitted.get('status',submitted.get('outcome'))=='submitted','History submission identity differs')
    manifest=inputs.binding(spec['science_manifest'],roots)
    inputs.require(manifest['sha256']==spec['source_manifest_sha256'],'History scientific source changed')
    config_members=inputs.load(manifest['path'])['files'];inputs.require(config['sha256'] in config_members.values(),'Config is not bound by the original scientific source')
    row={'id':spec.get('id','history-'+job),'input_schema':DENSE if dense else PARTIAL,'case_directory':str(case),'anchor_directory':str(case),
         'global_offset':offset,'requested_updates':budget,'source_manifest_sha256':manifest['sha256'],'job_id':job,
         'config':config,'stdout':stdout,'stderr':stderr,'producer_submission':submission,'science_manifest':manifest,
         'states':inputs.binding(case/'states.jsonl',roots),'updates':inputs.binding(case/'updates.jsonl',roots),
         'case_report':inputs.binding(case/'report.json',roots)}
    if spec.get('producer_manifest'):
        producer=inputs.binding(spec['producer_manifest'],roots)
        inputs.require(producer['sha256']==submitted['source_manifest_sha256'],'Actual producer wrapper source changed')
        row['producer_manifest']=producer
    count=_counts(row['states'],row['updates'],offset,roots,dense)
    inputs.require(count['committed_updates']<=budget,'History exceeds original allocation budget')
    if dense:
        row['summary']=inputs.binding(study/'summary.json',roots);summary=inputs.load(row['summary']['path'])
        inputs.require(str(summary['job_id'])==job and summary['source_manifest_sha256']==manifest['sha256'] and summary['config_sha256']==config['sha256'],'Dense summary identity differs')
        inputs.require(summary['status'] in ('passed','failed'),'Dense original terminal status missing')
        row.update(scientific_status=summary['status'],producer_status=summary['status'],original_matrix_stride=1)
        inputs.check_lsf(stdout['path'],job,summary['status'])
        inputs.require(count['committed_updates']==budget,'Dense original budget incomplete')
    else:
        check_partial_lsf(stdout['path'],job)
        # This real case export anchors the original partial records. It is
        # explicitly not presented as a nonexistent terminal summary.
        row.update(summary=row['case_report'],summary_role='original_case_export_not_terminal_summary',scientific_status='unqualified',
                   producer_status='failed_original_LSF_saved_prefix',derivative_qualified=False)
        for key,name in [('latest_metadata','latest-state.json'),('latest_snapshot','latest-state.npz')]:row[key]=inputs.binding(case/name,roots)
        latest=inputs.load(row['latest_metadata']['path'])
        inputs.require(latest['snapshot_sha256']==row['latest_snapshot']['sha256'],'Latest checkpoint bytes changed')
        endpoint=count['committed_updates']
        with Path(row['states']['path']).open('rb') as stream:
            for line in stream:last=json.loads(line)
        if latest['step']!=endpoint:
            # An interrupted writer can commit its stride snapshot before updating
            # the rolling latest pointer. Use that actual retained endpoint.
            inputs.require(type(latest['step']) is int and 0<=latest['step']<endpoint
                           and last['step']==endpoint and last['snapshot_retained'] is True,
                           'Latest checkpoint does not bind committed endpoint')
            snapshot=inputs.binding(case/last['snapshot'],roots)
            inputs.require(snapshot['sha256']==last['snapshot_sha256'],'Retained endpoint snapshot changed')
            row.update(retained_endpoint_snapshot=snapshot,latest_checkpoint_role='older_auxiliary_not_endpoint')
        else:row['latest_checkpoint_role']='committed_endpoint'
        if not last['snapshot_retained']:count['retained_frames']+=1
    row['expected_counts']=count
    return row,cfg,'fixed'


def _case_equality(case,states,updates,complete):
    for key,rows in [('states',states),('updates',updates)]:
        saved=case.get(key,[]);n=min(len(saved),len(rows))
        inputs.require(n>0 and saved[:n]==rows[:n],'Original case export/committed prefix differs')
        if complete:inputs.require(len(saved)==len(rows),'Dense case export is incomplete')
    return {k:len(case.get(k,[])) for k in ['states','updates']}


def collect_segment(segment,arm):
    import report_data as data
    schema=segment.get('input_schema')
    if schema not in SCHEMAS:return exploratory.collect_segment(segment,arm)
    for name in HISTORY_PROOFS:
        if name in segment:data.bound_file(segment[name])
    dense=schema==DENSE
    states,sp=data.committed_jsonl(segment['states']);updates,up=data.committed_jsonl(segment['updates'])
    case_counts=_case_equality(data.load(data.bound_file(segment['case_report'])),states,updates,dense)
    if dense:
        raw=data.load(data.bound_file(segment['summary']))
        inputs.require(raw['source_manifest_sha256']==segment['source_manifest_sha256'] and raw['config_sha256']==segment['config']['sha256'] and str(raw['job_id'])==segment['job_id'],'Dense summary pins differ')
        for case in raw.get('cases',[]):
            if isinstance(case,dict) and 'states' in case and 'updates' in case:_case_equality(case,states,updates,True)
        status=raw['status'];error=raw.get('error');del raw
    else:status='running';error='Original terminal job failed; only authenticated committed history is presented'
    view={'status':status,'source_manifest_sha256':segment['source_manifest_sha256'],'config_sha256':segment['config']['sha256'],'job_id':segment['job_id'],'error':error}
    compatible=deepcopy(segment);compatible.pop('case_report',None);compatible['scientific_status']=status
    rows_view=states
    if dense:
        rows_view=[]
        for index,state in enumerate(states):
            # Validate every original dense NPZ, including undisplayed frames.
            path=data._saved_path(segment['anchor_directory'],state['snapshot'])
            data.validate_frame(path,state,segment['global_offset']+index)
            copied=deepcopy(state);keep=index==0 or index==len(updates) or index%20==0
            copied['snapshot_retained']=keep
            if not keep:copied.update(snapshot=None,snapshot_sha256=None)
            rows_view.append(copied)
    def read(path):return view if Path(path)==Path(segment['summary']['path']) else data.load(path)
    def committed(binding):
        if binding==segment['states']:return rows_view,sp
        if binding==segment['updates']:return updates,up
        return data.committed_jsonl(binding)
    namespace={**data.__dict__,'load':read,'committed_jsonl':committed}
    collector=types.FunctionType(data.collect_segment.__code__,namespace,'collect_saved_history')
    result=collector(compatible,arm)
    result['states']=states  # Published scalar records retain original metadata.
    if not dense and result['last_arrays'] is None:
        latest=data.load(data.bound_file(segment['latest_metadata']));last=states[-1]
        inputs.require(type(latest.get('step')) is int and latest['step']==len(updates),'Latest endpoint index differs')
        # The original latest checkpoint stores a local index; JSONL additionally
        # records the global offset. This is indexing metadata, never a new state.
        latest.setdefault('global_step',segment['global_offset']+latest['step'])
        for key,value in last.items():
            if key not in ('snapshot','snapshot_sha256','snapshot_retained','readout_retained','readout'):
                inputs.require(latest.get(key)==value,'Saved latest metadata differs from final committed state: '+key)
        latest.update(snapshot_sha256=segment['latest_snapshot']['sha256'])
        frame,arrays,proof=data.validate_frame(data.bound_file(segment['latest_snapshot']),latest,segment['global_offset']+len(updates))
        result['retained'].append((frame,proof));result['last_arrays']=arrays
    result['proof'].update(input_schema=schema,scientific_status=segment['scientific_status'],producer_status=segment['producer_status'],
        summary_role=segment.get('summary_role','original_terminal_summary'),original_case_export_counts=case_counts,
        original_dense_frames_validated=len(states) if dense else None,display_stride=20,
        additional_saved_latest_checkpoint=segment.get('latest_snapshot'),
        latest_checkpoint_role=segment.get('latest_checkpoint_role','committed_endpoint'),
        history_proofs={k:segment[k] for k in HISTORY_PROOFS if k in segment},derivative_qualified=False)
    from report_prefix import collect as collect_prefix
    return collect_prefix(result,segment)


def collect_arm(arm,output):
    import report_data as data
    namespace={**data.__dict__,'collect_segment':collect_segment}
    assembler=types.FunctionType(data.collect_arm.__code__,namespace,'assemble_saved_history')
    compatible=deepcopy(arm)
    if compatible['scientific_status']=='unqualified':compatible['scientific_status']='partial'
    result=assembler(compatible,output)
    if arm['scientific_status']=='unqualified':result['scientific_status']='unqualified'
    result['qualification_scope']='Complete saved history; all original failed/unqualified segment outcomes retained'
    result['optimization_complete']=result['latest_global_step']==result['global_target']
    return result
