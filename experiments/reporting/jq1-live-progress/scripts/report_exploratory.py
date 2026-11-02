"""Versioned terminal exploratory input adapter; uses the existing saved renderer.

No model/scoring/repair calls. Metadata capture is stdlib-only; frame collection
and the unchanged numerical renderer run only on allocated report CPU.
"""
from copy import deepcopy
from pathlib import Path, PurePosixPath
import types

import quick_inputs as inputs

VERSIONS={'mosaic-exploratory-continuation-v1','mosaic-exploratory-continuation-v2'}
SCHEMA='terminal-exploratory-fixed-v1'
PROOF_FIELDS=('producer_plan','producer_submission','producer_driver','producer_checksums','science_manifest')


def source_proof(source, roots):
    """Bind the actual producer manifest, including original v1 capsules."""
    source=Path(source)
    if (source/'checksums.sha256').is_file():
        return inputs.binding(source/'checksums.sha256',roots)
    proof=inputs.binding(source/'snapshot.json',roots)
    files=inputs.load(proof['path']).get('files')
    inputs.require(isinstance(files,dict) and 'exploratory_continuation.py' in files
                   and 'plan.json' in files,'Original producer snapshot lacks entrypoint/plan identities')
    for name,digest in files.items():
        part=PurePosixPath(name)
        inputs.require(not part.is_absolute() and '..' not in part.parts and '\\' not in name,
                       'Unsafe original producer snapshot path')
        inputs.require(isinstance(digest,str) and len(digest)==64
                       and all(c in '0123456789abcdef' for c in digest),'Invalid producer snapshot digest')
        inputs.require(inputs.binding(source.joinpath(*part.parts),roots)['sha256']==digest,
                       'Original producer snapshot file changed: '+name)
    return proof


def header(summary):
    inputs.require(summary.get('version') in VERSIONS,'Unsupported exploratory summary version')
    inputs.require(summary.get('status')=='completed_exploratory_updates','Only completed optimization outputs are reportable by this adapter; smoke/running/failed are separate')
    inputs.require(summary.get('derivative_qualified') is False,'Exploratory qualification scope changed')
    row=summary['configuration']
    inputs.require(row['research_profile']=='exploratory' and row['name']==summary['name'],'Exploratory arm identity differs')
    inputs.require(type(row['start']) is int and type(row['steps']) is int and row['steps']>0
        and row['start']+row['steps']==row['target'],'Exploratory range differs')
    inputs.require((summary['global_start'],summary['global_end'],summary['global_target'],summary['applied_updates'])
        ==(row['start'],row['target'],row['target'],row['steps']),'Incomplete completed exploratory budget')
    inputs.require(summary.get('frozen_before') is not None and summary['frozen_before']==summary.get('frozen_after')
        and 'preservation_error' not in summary,'Exploratory frozen-state preservation failed')
    if row.get('frozen_identity'):
        inputs.require(summary['frozen_before']==row['frozen_identity'],'Original frozen identities differ')
    inputs.require(not summary.get('device_smoke_updates') and summary.get('device_smoke_states_adopted',False) is False,'Disposable smoke is not production data')
    return row


def capture_segment(spec,profile,study,summary):
    row=header(summary);roots=profile['allowed_roots'];job=str(summary['job_id']);case=study/'case-0'
    inputs.require(job.isdigit(),'Original exploratory job identity missing')
    config=inputs.binding(row['config']['path'],roots);inputs.require(config==row['config'],'Original exploratory config changed')
    cfg=inputs.load(config['path']);inputs.require(cfg['initial_step_size']==row['rate'],'Original exploratory rate differs')
    inputs.require(Path(summary['case_report'])==case/'report.json','Original case report path differs')
    parent=study.parent.parent;submission=inputs.binding(parent/'submission.json',roots);submitted=inputs.load(submission['path'])
    inputs.require(submitted['job_id']==job and submitted['status']=='submitted','Original exploratory submission differs')
    plan=inputs.binding(submitted['plan']['path'],roots);inputs.require(plan==submitted['plan'],'Original producer plan changed')
    planned=inputs.load(plan['path']);inputs.require(planned['arms'][row['name']]==row,'Summary configuration differs from actual submitted plan')
    source=Path(planned['source_root']);science=Path(row['science_source']['path'])
    manifest=inputs.binding(science/'source_manifest.json',roots)
    inputs.require(manifest['sha256']==row['science_source']['manifest_sha256'],'Original scientific source changed')
    result={'id':spec.get('id','exploratory-'+row['name']),'input_schema':SCHEMA,'case_directory':str(case),'anchor_directory':str(case),
        'global_offset':row['start'],'requested_updates':row['steps'],'source_manifest_sha256':manifest['sha256'],
        'job_id':job,'scientific_status':'unqualified','producer_status':summary['status'],'derivative_qualified':False,
        'config':config,'summary':inputs.binding(study/'summary.json',roots),'producer_plan':plan,'producer_submission':submission,
        'producer_driver':inputs.binding(source/'exploratory_continuation.py',roots),
        'producer_checksums':source_proof(source,roots),'science_manifest':manifest}
    for key,name in (('states','states.jsonl'),('updates','updates.jsonl'),('case_report','report.json'),('retained_index','retained-frames.json')):
        result[key]=inputs.binding(case/name,roots)
    for key in ('stdout','stderr'):result[key]=inputs.binding(parent/(key+'.'+job+'.log'),roots)
    inputs.check_lsf(result['stdout']['path'],job,'passed')
    result['expected_counts']=inputs.counts(result['states']['path'],result['updates']['path'],row['start'],str(case),roots,'fixed')
    inputs.require(result['expected_counts']['committed_updates']==row['steps'],'Exploratory committed budget differs')
    return result,cfg,'fixed'


def collect_segment(segment,arm):
    import report_data as data
    if segment.get('input_schema')!=SCHEMA:return data.collect_segment(segment,arm)
    for name in PROOF_FIELDS:data.bound_file(segment[name])
    raw=data.load(data.bound_file(segment['summary']));row=header(raw)
    plan=data.load(data.bound_file(segment['producer_plan']));submitted=data.load(data.bound_file(segment['producer_submission']))
    data.require(plan['arms'][row['name']]==row and submitted['job_id']==segment['job_id']
        and submitted['plan']==segment['producer_plan'],'Producer handoff changed')
    data.require((row['start'],row['steps'],row['config']['sha256'],row['science_source']['manifest_sha256'])
        ==(segment['global_offset'],segment['requested_updates'],segment['config']['sha256'],segment['source_manifest_sha256']),
        'Exploratory source/config/range changed')
    # A conservative in-memory compatibility view supplies provenance to the
    # unchanged collector. "running" here never means qualified/passed. No
    # synthetic summary is written; the published proof retains the real status.
    view={'status':'running','source_manifest_sha256':segment['source_manifest_sha256'],
        'config_sha256':segment['config']['sha256'],'job_id':segment['job_id']}
    def read(path):return view if Path(path)==Path(segment['summary']['path']) else data.load(path)
    namespace={**data.__dict__,'load':read}
    collector=types.FunctionType(data.collect_segment.__code__,namespace,'collect_exploratory_segment')
    compatible=deepcopy(segment);compatible['scientific_status']='running'
    result=collector(compatible,arm)
    data.require(len(result['updates'])==row['steps'] and len(result['states'])==row['steps']+1,'Completed exploratory prefix is incomplete')
    result['proof'].update(scientific_status='unqualified',producer_status=raw['status'],input_schema=SCHEMA,
        derivative_qualified=False,primal_scope=raw.get('primal_scope'),
        producer_proofs={key:segment[key] for key in PROOF_FIELDS})
    return result


def collect_arm(arm,output):
    import report_data as data
    # Reuse the exact original assembler/renderer code object with a scoped
    # collector binding; never monkeypatch the module or copy its renderer.
    namespace={**data.__dict__,'collect_segment':collect_segment}
    assembler=types.FunctionType(data.collect_arm.__code__,namespace,'assemble_exploratory_arm')
    compatible=deepcopy(arm)
    if compatible['scientific_status']=='unqualified':compatible['scientific_status']='partial'
    result=assembler(compatible,output)
    if arm['scientific_status']=='unqualified':result['scientific_status']='unqualified'
    result['qualification_scope']='Exploratory optimization; derivative/native/binding certification not requested'
    result['optimization_complete']=result['latest_global_step']==result['global_target']
    return result
