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


def verify(value, source, pin):
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
    verify_terminal(value['terminal'], raw[parent + '/stdout.' + delivery['job_id'] + '.log'].decode(),
                    delivery['job_id'])
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
terminal={'state':'SUCCESS_PERSISTED','job_id':job,'live':command(['bjobs','-l',job]),'history':command(['bhist','-l',job])}
cert=runtime.certificate(source,pin)
delivery=inputs.load(parent/'delivery.json')
inputs.require(delivery['job_id']==job and delivery['engine_pin']==pin,'Delivery identity differs')
inputs.verify_binding(delivery['archive'])
inputs.require(inputs.sha(parent/'viewer/report-receipt.json')==delivery['receipt_sha256'],'Render receipt changed')
proofs=cert['proofs']+[inputs.binding(source.parent/'validation-certificate.json')]+[inputs.binding(parent/n) for n in ('submission.json','delivery.json','capacity-render.json','stdout.'+job+'.log')]
result={'delivery':delivery,'certificate':cert,'terminal':terminal,'source_verified':True,'proof_files':[]}
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
    value = connection.run(code + REMOTE.replace('PARENT', repr(parent)), 'recover_completed_delivery')
    verify(value, source, pin)
    require(value['certificate'] == cli.inputs.load(work / 'engine-validation.json'),
            'Recovered CPU certificate differs from the original request')
    require(value['delivery']['job_id'] == state['submission']['job_id'], 'Recovered a different render')
    proof_path = work / ('persisted-delivery-' + uuid.uuid4().hex + '.json')
    cli.write(proof_path, value)
    delta = work / 'report-delta.tar.gz'
    cli.download(connection, value['delivery']['archive'], delta)
    pointer = cli.publish_delta(delta, value['delivery'], viewer, cli.local_assets(viewer), pin,
                                cli.inputs.sha(work / 'engine-validation.json'))
    http = None if args.no_serve else cli.ensure_http(viewer, args.port or profile['port'], work)
    result = {'status':'published_verified_saved_report','publication':pointer,'HTTP':http,
              'runs':value['delivery']['arms'],'work':str(work),'terminal_state':'SUCCESS_PERSISTED',
              'terminal_proof':str(proof_path),'engine_pin':pin,'render_job_id':value['delivery']['job_id'],
              'original_results_changed':False,'scientific_success_claim':False,'new_jobs_submitted':False}
    cli.write(work/'result.json',result)
    cli.event(profile,work,result['status'],edition=state['edition'],job_id=value['delivery']['job_id'],
              terminal_state='SUCCESS_PERSISTED',no_new_jobs=True)
    require(cli.inputs.load(lock)['request_id']==state['request_id'],'Viewer ownership changed')
    lock.unlink()
    print(json.dumps(result,indent=2))
    return result
