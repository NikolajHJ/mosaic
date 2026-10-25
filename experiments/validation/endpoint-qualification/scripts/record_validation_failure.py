"""Authenticate already-collected failed metadata, without numerical evaluation."""
import base64
import hashlib
import json
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-continuation-admission-01'
RUN=REPO/'experiments/validation/endpoint-qualification/runs/20261025-continuation-admission-01'


def main():
    observation=json.loads((LOCAL/'status-04/raw.json').read_bytes())
    metadata=json.loads((LOCAL/'failure-evidence-01/raw.json').read_bytes())
    proofs={};decoded={}
    for parent,data in observation['parents'].items():
        for name,row in data['files'].items():
            if 'base64' not in row:continue
            raw=base64.b64decode(row['base64'],validate=True);binding=row['binding']
            assert len(raw)==binding['bytes'] and hashlib.sha256(raw).hexdigest()==binding['sha256']
            proofs[binding['path']]=binding;decoded[(parent,name)]=raw
    for arm,data in metadata['arms'].items():
        for name,row in data['files'].items():
            raw=base64.b64decode(row['base64'],validate=True)
            assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
            proofs[row['path']]={k:row[k] for k in ('path','bytes','sha256')}
    assert len(observation['jobs'])==8 and not any('-segment-' in name for name in observation['jobs'])
    outcomes=[]
    for arm,offset in [('protenix',20000),('boltz',10000),('dbmol',5000)]:
        parent=arm+'-validate-'+str(offset).zfill(5)
        callback=arm+'-after-validate-'+str(offset).zfill(5)
        for name,queue in [(parent,'gpul40s'),(callback,'hpc')]:
            job=observation['jobs'][name];query=observation['scheduler'][job]
            compact=''.join(query['stdout'].split());assert query['returncode']==0
            assert re.findall(r'Job<([^>]+)>',compact)==[job]
            assert re.findall(r'Status<([^>]+)>',compact)==['EXIT']
            assert re.findall(r'Queue<([^>]+)>',compact)==[queue]
            log=''.join(decoded[(name,'stdout.'+job+'.log')].decode().split())
            assert log.count('Subject:Job')==1 and 'Subject:Job'+job+':<mosaic-endpoint01-'+name+'>incluster<dcc>Exited' in log
            assert 'Exitedwithexitcode1.' in log and 'inqueue<'+queue+'>' in log
        preserve=json.loads(decoded[(parent,'post-execution-preservation.json')])
        assert preserve['preservation_status']=='passed' and preserve['original_results_changed'] is False
        rows=metadata['arms'][arm]['files']
        if arm=='protenix':
            item=next(row for name,row in rows.items() if name.endswith('/admission.json'))
            report=json.loads(base64.b64decode(item['base64']))
            assert report['status']=='failed' and report['can_admit_new_child'] is False and report['frozen_unchanged'] is True
            assert report['field_outcomes']['E_upper']['passed'] is False
            original=[r for r in report['directions'] if 'authentic_original' in r['roles']]
            assert original and all(r['admission_evidence'] is True for r in original)
            outcome={'arm':arm,'status':'blocked_failed_prospective_endpoint_admission',
                     'primary_failure':'Additional raw-field directional agreement/precision criteria failed; no passed geometry pair for E_upper',
                     'original_atom_contrast_passed':True,'frozen_unchanged':True,
                     'admission':{k:item[k] for k in ('path','bytes','sha256')}}
        else:
            worker=rows['payload/study/worker.log'];text=base64.b64decode(worker['base64']).decode()
            assert "No module named 'mosaic'" in text
            assert not any(name.endswith('/admission.json') for name in rows)
            outcome={'arm':arm,'status':'blocked_worker_import_before_native_checks',
                     'primary_failure':"ModuleNotFoundError: No module named 'mosaic' in worker subprocess",
                     'native_gradient_result_established':False,'worker_log':{k:worker[k] for k in ('path','bytes','sha256')}}
        outcome.update(validation_job=observation['jobs'][parent],callback_job=observation['jobs'][callback],
                       optimizer_submitted=False,optimizer_started=False,original_results_changed=False)
        outcomes.append(outcome)
    result={'version':'endpoint-continuation-failure-audit-v1','status':'verified_actual_failures',
            'captured_utc':observation['captured_utc'],'outcomes':outcomes,'proofs':list(proofs.values()),
            'scope':'Local stdlib byte/LSF/JSON audit only; no scientific array computation',
            'optimization_started':False,'original_results_changed':False,'monitor_remains_paused':True}
    (RUN/'failure-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    campaign=json.loads((RUN/'campaign.json').read_bytes())
    campaign.update(status='all_native_validation_attempts_failed_no_children',captured_utc=result['captured_utc'],outcomes=outcomes)
    for row in campaign['native_jobs']:row['scheduler_state']='EXIT'
    (RUN/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'status':result['status'],'outcomes':outcomes,'verified_proof_files':len(proofs)},indent=2))


if __name__=='__main__':main()
