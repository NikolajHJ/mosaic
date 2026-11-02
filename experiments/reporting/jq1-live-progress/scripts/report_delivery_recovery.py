"""Deliver an already-rendered report after LSF's live job record expires.

Standard-library metadata/byte recovery only. Reuse the original engine pin,
CPU certificate, archive, publisher and HTTP verifier; never render or submit.
Persisted success is labelled explicitly, never manufactured as live DONE.
"""
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import uuid
import xml.etree.ElementTree as ET


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_terminal(terminal, stdout, job):
    require(terminal['state'] == 'SUCCESS_PERSISTED' and terminal['job_id'] == job,
            'Wrong persisted report identity')
    live = terminal['live']
    states = re.findall(r'Status<([^>]+)>', ''.join(live['stdout'].split()))
    require(states == ['DONE'] or (not states and 'not found' in live['stderr'] and job in live['stderr']),
            'Live report is active, failed or ambiguous')
    if states:
        require(re.findall(r'Job<(\d+)>', ''.join(live['stdout'].split())) == [job], 'Live job identity differs')
    history = terminal['history']
    compact = ''.join(history['stdout'].split())
    require(history['returncode'] == 0 and re.findall(r'Job<(\d+)>', compact) == [job]
            and ('Completed<done>' in compact or 'Donesuccessfully.' in compact) and 'Exitedwithexitcode' not in compact
            and 'Exitedbysignal' not in compact, 'Same-job completed history missing')
    lsf = ''.join(stdout.split())
    require('Subject:Job' + job + ':' in lsf and lsf.count('Successfullycompleted.') == 1
            and 'inqueue<hpc>' in lsf and 'asuser<s234818>' in lsf
            and 'Exitedwithexitcode' not in lsf, 'Same-job hpc LSF success missing')


def verify_cap_terminal(terminal, stdout, stderr, job):
    require(terminal['state']=='EXIT_REPORT_CAP_ONLY' and terminal['job_id']==job,'Wrong cap-recovery identity')
    live=terminal['live'];compact=''.join(live['stdout'].split())
    states=re.findall(r'Status<([^>]+)>',compact)
    require((states==['EXIT'] and re.findall(r'Job<(\d+)>',compact)==[job]) or
            (not states and job in live['stderr'] and 'not found' in live['stderr']),
            'Cap recovery cannot take an active, successful or ambiguous job')
    history=terminal['history'];h=''.join(history['stdout'].split())
    require(history['returncode']==0 and re.findall(r'Job<(\d+)>',h)==[job]
            and 'Exitedwithexitcode1.' in h,'Original exit1 history missing')
    lsf=''.join(stdout.split())
    require('Subject:Job'+job+':' in lsf and 'inqueue<hpc>' in lsf and 'asuser<s234818>' in lsf
            and lsf.count('Exitedwithexitcode1.')==1 and 'Successfullycompleted.' not in lsf,
            'Original same-job cap-failure LSF missing')
    require(stderr.strip().splitlines()[-1]=='ValueError: Report output exceeded its declared cap'
            and stderr.count('Traceback (most recent call last):')==1
            and len(re.findall(r'^\w*(?:Error|Exception):',stderr,re.M))==1,
            'Failure was not solely the final report byte-cap guard')


def verify(value, source, pin, allow_cap=False):
    delivery, cert = value['delivery'], value['certificate']
    require(delivery['status'] == 'passed_saved_only_presentation' and delivery['engine_pin'] == pin
            and cert['status'] == 'passed' and cert['engine_pin'] == pin, 'Unqualified delivered bytes')
    raw = {}
    for row in value['proof_files']:
        data = base64.b64decode(row['base64'], validate=True)
        require(row['path'] not in raw and len(data) == row['bytes']
                and hashlib.sha256(data).hexdigest() == row['sha256'], 'Evidence bytes differ')
        raw[row['path']] = data
    for row in cert['proofs']:
        require(row['path'] in raw and hashlib.sha256(raw[row['path']]).hexdigest() == row['sha256'],
                'Original CPU certificate proof missing')
    xml = [v for p,v in raw.items() if p.endswith('/validation/tests.xml')]
    require(len(xml) == 1, 'Missing raw CPU XML')
    cases = ET.fromstring(xml[0]).findall('.//testcase')
    require(len(cases) == cert['cases'] and all(not list(c) for c in cases), 'CPU cases failed/skipped')
    cpu_parent = str(PurePosixPath(source).parent / 'validation')
    scheduler = json.loads(raw[cpu_parent + '/terminal-scheduler.json'])
    compact = ''.join(scheduler['stdout'].split())
    require(scheduler['returncode'] == 0 and re.findall(r'Job<(\d+)>', compact) == [cert['cpu_job_id']]
            and re.findall(r'Status<([^>]+)>', compact) == ['DONE'] and 'Queue<hpc>' in compact,
            'Original CPU scheduler proof missing')
    cpu_lsf = ''.join(raw[cpu_parent + '/stdout.' + cert['cpu_job_id'] + '.log'].decode().split())
    require('Subject:Job' + cert['cpu_job_id'] + ':' in cpu_lsf and cpu_lsf.count('Successfullycompleted.') == 1,
            'CPU same-job LSF success missing')
    parent = str(PurePosixPath(delivery['archive']['path']).parent)
    stdout=raw[parent+'/stdout.'+delivery['job_id']+'.log'].decode()
    if allow_cap:
        verify_cap_terminal(value['terminal'],stdout,raw[parent+'/stderr.'+delivery['job_id']+'.log'].decode(),delivery['job_id'])
        cap=value['cap_recovery'];gate=cap['capacity']
        require(gate['status']=='passed' and cap['original_failure_preserved'] is True
                and cap['original_cap_bytes']<gate['own_output']['actual_bytes']<=cap['delivery_cap_bytes']==8*1024**3
                and cap['absolute_input_bindings_verified']>0,'Corrected delivery capacity/input audit missing')
        assessment=json.loads(raw[parent+'/viewer/assessment.json'])
        receipt=json.loads(raw[parent+'/viewer/report-receipt.json'])
        require(assessment['status']=='passed_saved_only_render' and receipt['status']=='passed_saved_only_presentation'
                and receipt['job_id']==delivery['job_id'] and receipt['engine_pin']==pin
                and hashlib.sha256(raw[parent+'/viewer/report-receipt.json']).hexdigest()==delivery['receipt_sha256'],
                'Completed allocated-render assessment/receipt missing')
    else:verify_terminal(value['terminal'],stdout,delivery['job_id'])
    submitted = json.loads(raw[parent + '/submission.json'])
    require(submitted['job_id'] == delivery['job_id'] and submitted['engine_pin'] == pin,
            'Delivered job differs from accepted submission')
    require(value['source_verified'] is True, 'Original engine source was not verified')
    return raw


REMOTE = '''
import base64,subprocess
runtime.verify_source(source,pin)
parent=Path(PARENT)
profile=inputs.load(source/'profile.json')
inputs.require(parent.parent==Path(profile['remote_root'])/'requests','Recovery namespace escaped')
sub=inputs.load(parent/'submission.json');job=sub['job_id']
inputs.require(sub['engine_pin']==pin and sub['argv']==runtime.bsub_argv(profile,source,pin,'render',parent,parent/'request.json'),'Accepted command/source changed')
def command(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
 return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
allow_cap=ALLOW_CAP
terminal={'state':'EXIT_REPORT_CAP_ONLY' if allow_cap else 'SUCCESS_PERSISTED','job_id':job,'live':command(['bjobs','-l',job]),'history':command(['bhist','-l',job])}
cert=runtime.certificate(source,pin)
extra=[];cap_recovery=None
if allow_cap:
 from storage_accounting import assess_output
 error=(parent/('stderr.'+job+'.log')).read_text()
 inputs.require(error.strip().splitlines()[-1]=='ValueError: Report output exceeded its declared cap','Not a completed-artifact cap failure')
 output=parent/'viewer';receipt=inputs.load(output/'report-receipt.json');manifest=inputs.load(output/'manifest.json');plan=inputs.load(output/'input-plan.json');request=inputs.load(parent/'request.json')
 inputs.require(inputs.load(output/'assessment.json')['status']=='passed_saved_only_render' and receipt['job_id']==job
                and receipt['engine_pin']==pin and receipt['selection_key']==request['selection_key']
                and inputs.selection_key(plan['arms'],profile['validated_renderer_sha256'])==request['selection_key'],
                'Allocated complete-render/selection proof differs')
 inputs.require({p.name for p in output.iterdir()}==set(receipt['files'])|{'report-receipt.json'},'Rendered artifact inventory differs')
 for name,record in receipt['files'].items():
  p=output/name;inputs.require(p.parent==output and inputs.binding(p)=={'path':str(p),**record},'Rendered bytes changed: '+name)
 seen_inputs={}
 def audit_bindings(value):
  if isinstance(value,dict):
   if all(k in value for k in ('path','bytes','sha256')) and str(value['path']).startswith('/'):
    binding={k:value[k] for k in ('path','bytes','sha256')};path=binding['path']
    inputs.require(path not in seen_inputs or seen_inputs[path]==binding,'Conflicting source binding')
    if path not in seen_inputs:inputs.verify_binding(binding,profile['allowed_roots']);seen_inputs[path]=binding
   for item in value.values():
    if isinstance(item,(dict,list)):audit_bindings(item)
  elif isinstance(value,list):
   for item in value:
    if isinstance(item,(dict,list)):audit_bindings(item)
 audit_bindings(plan);audit_bindings(manifest)
 for arm in manifest['arms']:
  for chunk in arm['frames']['chunks']:
   for frame in inputs.load(output/chunk['path'])['frames']:audit_bindings(frame['source_snapshot'])
 gate=assess_output(parent,8*1024**3,profile.get('report_file_cap',4096))
 inputs.require(gate['status']=='passed' and profile['request_cap_bytes']<gate['own_output']['actual_bytes'], 'Corrected explicit delivery budget does not fit')
 cap_recovery={'original_failure_preserved':True,'original_cap_bytes':profile['request_cap_bytes'],'delivery_cap_bytes':8*1024**3,
               'capacity':gate,'absolute_input_bindings_verified':len(seen_inputs),'scope':'All render artifacts and source bindings rehashed; no new rendering/model/numerical evaluations or remote writes'}
 omitted={name:row for name,row in receipt['files'].items() if name not in ('manifest.json','report.html','input-plan.json','assessment.json','report-receipt.json','cpu-certificate.json') and row['sha256'] in set(request.get('known_asset_sha256',[]))}
 delivery={'status':'passed_saved_only_presentation','engine_pin':pin,'job_id':job,'edition':plan['edition'],
           'archive':inputs.binding(parent/'report-delta.tar.gz'),'receipt_sha256':inputs.sha(output/'report-receipt.json'),
           'omitted_verified_local_assets':omitted,'scientific_success_claim':False,'original_results_changed':False,
           'completion_method':'verified_complete_artifacts_original_report_cap_failure_preserved',
           'arms':{a['id']:{'updates':a['committed_updates'],'frames':a['frames']['count']} for a in manifest['arms']}}
 extra=[inputs.binding(parent/n) for n in ('stderr.'+job+'.log','viewer/assessment.json','viewer/report-receipt.json')]
else:delivery=inputs.load(parent/'delivery.json')
inputs.require(delivery['job_id']==job and delivery['engine_pin']==pin,'Delivery identity differs')
inputs.verify_binding(delivery['archive'])
inputs.require(inputs.sha(parent/'viewer/report-receipt.json')==delivery['receipt_sha256'],'Render receipt changed')
proofs=cert['proofs']+[inputs.binding(source.parent/'validation-certificate.json')]+[inputs.binding(parent/n) for n in ('submission.json','capacity-render.json','stdout.'+job+'.log')]+extra
if not allow_cap:proofs.append(inputs.binding(parent/'delivery.json'))
result={'delivery':delivery,'certificate':cert,'terminal':terminal,'source_verified':True,'proof_files':[]}
if cap_recovery is not None:result['cap_recovery']=cap_recovery
seen=set()
for row in proofs:
 if row['path'] in seen:continue
 seen.add(row['path']);p=Path(row['path'])
 inputs.require(p.is_relative_to(Path(profile['remote_root'])),'Evidence escaped namespace')
 raw=p.read_bytes();inputs.require(len(raw)<4*1024**2,'Metadata exceeds cap')
 result['proof_files'].append({**row,'base64':base64.b64encode(raw).decode()})
print(json.dumps(result,allow_nan=False))
'''


def deliver(cli, args, profile, work, state):
    require(args.resume and state.get('submission'), 'Recovery requires an accepted existing request')
    pin = state['engine_pin']
    require(state['submission']['engine_pin'] == pin, 'Original request source pin changed')
    source = str(PurePosixPath(profile['remote_root']) / 'engines' / pin / 'source')
    parent = str(PurePosixPath(profile['remote_root']) / 'requests' / state['request_id'])
    viewer = Path(state['viewer']);lock = viewer / '.report-update.lock'
    require(cli.inputs.load(lock)['request_id'] == state['request_id'], 'Another request owns this viewer')
    connection = cli.Connection(profile, work)
    code = "import sys,json\nfrom pathlib import Path\nsource=Path("+repr(source)+");pin="+repr(pin)+"\nsys.path.insert(0,str(source/'scripts'))\nimport quick_runtime as runtime,quick_inputs as inputs\n"
    allow_cap=getattr(args,'recover_cap_exceeded',False)
    complete=work/'report-delta.tar.complete.json';cached=[]
    if complete.exists() and cli.inputs.load(complete).get('status')=='passed_exact_bytes':
        for candidate in work.glob('persisted-delivery-*.json'):
            saved=cli.inputs.load(candidate)
            if (saved['delivery']['archive']==cli.inputs.load(complete)['binding']
                    and (saved['terminal']['state']=='EXIT_REPORT_CAP_ONLY')==allow_cap):cached.append((candidate,saved))
    require(len(cached)<=1,'Ambiguous cached delivery proofs; reconcile before publication')
    if cached:proof_path,value=cached[0]
    else:
        value = connection.run(code + REMOTE.replace('PARENT', repr(parent)).replace('ALLOW_CAP',repr(allow_cap)),
                               'recover_cap_delivery' if allow_cap else 'recover_completed_delivery',timeout=600 if allow_cap else 300)
        proof_path=None
    verify(value, source, pin,allow_cap)
    require(value['certificate'] == cli.inputs.load(work / 'engine-validation.json'),
            'Recovered CPU certificate differs from the original request')
    require(value['delivery']['job_id'] == state['submission']['job_id'], 'Recovered a different render')
    if proof_path is None:
        proof_path = work / ('persisted-delivery-' + uuid.uuid4().hex + '.json')
        cli.write(proof_path, value)
    delta = work / 'report-delta.tar.gz'
    cli.download(connection, value['delivery']['archive'], delta)
    pointer = cli.publish_delta(delta, value['delivery'], viewer, cli.local_assets(viewer), pin,
                                cli.inputs.sha(work / 'engine-validation.json'),
                                value['cap_recovery']['delivery_cap_bytes'] if allow_cap else profile['request_cap_bytes'])
    http = None if args.no_serve else cli.ensure_http(viewer, args.port or profile['port'], work)
    result = {'status':'published_verified_saved_report','publication':pointer,'HTTP':http,
              'runs':value['delivery']['arms'],'work':str(work),'terminal_state':value['terminal']['state'],
              'terminal_proof':str(proof_path),'engine_pin':pin,'render_job_id':value['delivery']['job_id'],
              'original_results_changed':False,'scientific_success_claim':False,'new_jobs_submitted':False}
    cli.write(work/'result.json',result)
    cli.event(profile,work,result['status'],edition=state['edition'],job_id=value['delivery']['job_id'],
              terminal_state=value['terminal']['state'],no_new_jobs=True)
    require(cli.inputs.load(lock)['request_id']==state['request_id'],'Viewer ownership changed')
    lock.unlink()
    print(json.dumps(result,indent=2))
    return result
