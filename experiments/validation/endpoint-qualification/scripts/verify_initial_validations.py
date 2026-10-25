"""Independent byte/schema/resource audit; a submission is not native passage."""
import argparse
import hashlib
import json
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[4]
RUN=REPO/'experiments/validation/endpoint-qualification/runs/20261025-continuation-admission-01'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observation',type=Path,required=True)
    parser.add_argument('--attempt',choices=('01','02'),default='01');args=parser.parse_args()
    global RUN
    if args.attempt=='02':RUN=RUN.with_name('20261025-continuation-worker-fix-02')
    raw=args.observation.read_bytes();observed=json.loads(raw)
    prep=json.loads((RUN/'preparation.json').read_bytes());plan=json.loads((RUN/'plan.json').read_bytes())
    pin=prep['manifest_sha256'];root=plan['control_root'];source=plan['source_root']
    actual=[]
    arms=[('protenix',20000),('boltz',10000),('dbmol',5000)] if args.attempt=='01' else [('boltz',10000),('dbmol',5000)]
    assert len(set(observed['jobs'].values()))==len(observed['jobs'])==2+2*len(arms)
    for arm,offset in arms:
        name=arm+'-validate-'+str(offset).zfill(5)
        callback=arm+'-after-validate-'+str(offset).zfill(5)
        native=observed['jobs'][name];after=observed['jobs'][callback]
        for parent,job,phase in [(name,native,'validate'),(callback,after,'after-validate')]:
            record=observed['parents'][parent]['submission']
            assert record['status']=='submitted' and record['job_id']==job and re.fullmatch(r'[1-9]\d*',job)
            assert record['source_manifest_sha256']==pin and record['arm']==arm and record['offset']==offset and record['phase']==phase
            resources=(['-q','gpul40s','-n','4','-R','span[hosts=1]','-R','rusage[mem=8GB]',
                        '-gpu','num=1:mode=exclusive_process','-W','02:00'] if phase=='validate'
                       else ['-q','hpc','-n','1','-R','span[hosts=1]','-R','rusage[mem=2GB]','-W','00:20'])
            expected=['bsub',*resources,'-J','mosaic-endpoint'+args.attempt+'-'+parent,
                      '-oo',root+'/'+parent+'/stdout.%J.log','-eo',root+'/'+parent+'/stderr.%J.log']
            if phase=='after-validate':expected+=['-w','ended('+native+')']
            expected+=['bash','-l',source+'/scripts/continuation_stage.lsf',source,pin,phase,arm,str(offset)]
            assert record['argv']==expected
            assert record['dependency']==('ended('+native+')' if phase=='after-validate' else None)
            query=observed['scheduler'][job];assert query['returncode']==0
            text=''.join(query['stdout'].split())
            assert re.findall(r'Job<([^>]+)>',text)==[job]
            assert re.findall(r'Queue<([^>]+)>',text)==['gpul40s' if phase=='validate' else 'hpc']
            status=re.findall(r'Status<([^>]+)>',text);assert len(status)==1 and status[0] in ('PEND','RUN','DONE')
            if phase=='validate':
                assert 'RUNLIMIT120.0min' in text and 'rusage[mem=8192.00]' in text and '4Task(s)' in text
                assert 'num=1:mode=exclusive_process' in text
            else:
                assert 'RUNLIMIT20.0min' in text and 'rusage[mem=2048.00]' in text
                assert 'DependencyCondition<ended('+native+')>' in text
            actual.append({'parent':parent,'job_id':job,'scheduler_state':status[0],'phase':phase,
                           'resources_and_source_authenticated':True})
    result={'status':'passed_submission_proof_audit','captured_utc':observed['captured_utc'],
            'source_manifest_sha256':pin,'observation_sha256':hashlib.sha256(raw).hexdigest(),
            'actual_jobs':actual,'native_validation_passed':False,'optimization_started':False,
            'scope':'Actual initial submitted resources/receipts/scheduler only',
            'original_results_changed':False,'monitor_remains_paused':True}
    target=args.observation.parent/'initial-validation-verification.json'
    with target.open('x',encoding='utf8',newline='\n') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    (RUN/'initial-validation-verification.json').write_bytes(target.read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
