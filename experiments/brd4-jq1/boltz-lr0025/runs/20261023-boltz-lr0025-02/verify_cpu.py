"""Independent stdlib byte/XML/JSON/scheduler audit; no scientific imports.

Accepts either an authentic failed CPU attempt or a full passed CPU/dispatch
bundle. A failed-attempt audit never relabels its CPU or scientific gates passed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT=BASE+'/boltz-lr0025-02'
SOURCE=BASE+'/boltz-lr0025-release-02/source'
EXPECTED_CPU='29798538'
EXPECTED_CALLBACK='29798539'
EXPECTED_MANIFEST='3689071a4e7dd135071ff5b420c026543c4d10887c951077dff63d498d9f812c'
EXPECTED_ARCHIVE='40b4a087c2735de93ed1896f7534618f32d8713dae4d83fa80d36b49b1924dfa'
POCKET='9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
REFERENCE='37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(data):return hashlib.sha256(data).hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def safe(name):
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name,'Unsafe archive member')


def scheduler(raw,job,queue,status=None):
    compact=re.sub(r'\s+','',raw)
    require(re.findall(r'Job<([^>]+)>',compact)==[job],'Wrong/ambiguous scheduler job')
    require(re.findall(r'Queue<([^>]+)>',compact)==[queue],'Wrong scheduler queue')
    states=re.findall(r'Status<([^>]+)>',compact)
    require(len(states)==1,'Missing scheduler state')
    if status is not None:require(states==[status],'Unexpected actual scheduler state')
    return states[0]


def resources(phase):
    if phase=='cpu':return ['-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','01:00']
    if phase in ('native','segment'):return ['-q','gpul40s','-n','4','-R','span[hosts=1]','-R','rusage[mem=8GB]','-gpu','num=1:mode=exclusive_process','-W','02:00' if phase=='native' else '23:00']
    require(phase in ('after-cpu','after-native','after-segment','dispatch'),'Unknown actual phase')
    return ['-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30']


def vector(label,phase,manifest,dependency):
    parent=ROOT+'/'+label
    args=['bsub',*resources(phase),'-J','mosaic-bol30k-'+label,'-oo',parent+'/stdout.%J.log','-eo',parent+'/stderr.%J.log']
    if dependency:args+=['-w',dependency]
    args+=['bash','-l',SOURCE+'/scripts/hpc/'+('boltz_serial_cpu.lsf' if phase=='cpu' else 'boltz_serial_stage.lsf'),SOURCE,ROOT,manifest]
    if phase!='cpu':args+=[phase,parent]
    return args


def lsf(raw,job,passed):
    subject=re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',raw,re.M)
    require(subject==[(job,'Done' if passed else 'Exited')],'Wrong same-job final LSF subject')
    if passed:require('Successfully completed.' in raw and 'Exited with exit' not in raw,'LSF success missing')
    else:require(re.search(r'Exited with exit code\s+\d+',raw) is not None,'LSF exit proof missing')


def source_archive(package,manifest):
    path=REPO/'results/hpc'/HERE.name/'source01/source.tar.gz'
    require(path.stat().st_size==package['archive_bytes']==39618948 and digest(path.read_bytes())==package['archive_sha256']==EXPECTED_ARCHIVE,'Local complete source archive differs')
    files={}
    with tarfile.open(path) as archive:
        for member in archive:
            safe(member.name)
            require(member.isfile() and member.name.startswith('source/'),'Nonregular/foreign source member')
            name=member.name.removeprefix('source/')
            require(name not in files,'Duplicate source member')
            files[name]=archive.extractfile(member).read()
    require(digest(files.pop('source_manifest.json'))==EXPECTED_MANIFEST,'Source manifest member differs')
    require(len(files)==633 and set(files)==set(manifest['files']),'Incomplete633 source members')
    for name,data in files.items():require(digest(data)==manifest['files'][name],'Source member changed: '+name)
    # Authenticate scientific lineage as bytes, never import any library/model.
    old=REPO/'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/source02/source.tar.gz'
    require(digest(old.read_bytes())=='4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e','Original625 source archive changed')
    with tarfile.open(old) as archive:
        scientific={m.name.removeprefix('source/'):archive.extractfile(m).read() for m in archive if m.isfile() and m.name.startswith('source/src/')}
    changed=[name for name,data in scientific.items() if files[name]!=data]
    require(len(scientific)==145 and changed==['src/mosaic/small_molecule/chemistry/trajectory.py'],'Undeclared scientific member change')
    require(digest(files[changed[0]])=='69c89b84fe0dd1c5b548a84286d6635cb63cff223dd0890a0b50ce8f3e1a8468','Optional641 global-offset trajectory bytes changed')
    # This retry must preserve every scientific/driver/config member from the
    # actual failed633 release; only unit isolation and launcher metadata change.
    parent_archive=REPO/'results/hpc/20261023-boltz-lr0025-01/source01/source.tar.gz'
    require(digest(parent_archive.read_bytes())=='d19a5d8671b12990702ee0502f78e247e36bde97d33373461a6de1de25dcc78f','Failed parent633 archive changed')
    with tarfile.open(parent_archive) as tar:
        parent={m.name.removeprefix('source/'):tar.extractfile(m).read() for m in tar if m.isfile()}
    parent_raw=parent.pop('source_manifest.json')
    require(digest(parent_raw)=='43767a8564c220821fa06fa68c3a3690870cda2e22baeb7bf956b7f849094e76','Failed parent633 exact manifest differs')
    parent_manifest=json.loads(parent_raw)
    require(set(parent)==set(parent_manifest['files']) and len(parent)==633,'Parent complete633 scope differs')
    for name,data in parent.items():require(digest(data)==parent_manifest['files'][name],'A failed-parent member changed')
    protected=[name for name in files if name.startswith('src/')]
    protected+=['examples/brd4_25_atoms.py','scripts/campaign25_contract.py','scripts/boltz_serial_protocol.py','examples/configs/boltz_serial_boltz2.json','examples/configs/boltz_serial_boltz2_chemistry.json']
    require(all(files[name]==parent[name] for name in protected),'Retry altered a protected scientific/driver/config/protocol member')
    marker=b'# Bind the real lazy coupled checker before any test monkeypatch.\nimport mosaic.small_molecule.chemistry.checks\n'
    unit='tests/test_protenix_initialization.py'
    require(files[unit].count(marker)==1 and files[unit].replace(marker,b'')==parent[unit],'Inherited unit assertions changed beyond explicit import isolation')
    require(files['tests/test_boltz_serial.py'].startswith(parent['tests/test_boltz_serial.py']),'Original baseline fixtures were modified instead of retained')
    return files


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--edition',default='01')
    parser.add_argument('--expect-failure',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    require(args.edition.isdigit() and len(args.edition)==2,'Use an exact evidence edition')
    base=REPO/'results/hpc'/HERE.name
    folder=base/('cpu-evidence-'+args.edition)
    metadata=read(folder/'metadata.json')
    receipt=read(folder/'delivery-receipt.json')
    require(digest((folder/'metadata.json').read_bytes())==receipt['metadata_sha256'],'Delivery metadata changed')
    archive=Path(receipt['archive'])
    require(archive.stat().st_size==receipt['bytes'] and digest(archive.read_bytes())==receipt['sha256'],'Compact evidence archive identity differs')
    files=metadata['files'];remote={}
    for name,record in files.items():
        safe(name);data=(folder/name).read_bytes()
        require(len(data)==record['bytes'] and digest(data)==record['sha256'],'Delivered file identity differs')
        expected=(ROOT+'/'+name[4:]) if name.startswith('run/') else (SOURCE+'/'+name[7:]) if name.startswith('source/') else None
        require(record['path']==expected and record['path'] not in remote,'Remote/local proof path mismatch')
        remote[record['path']]=record
    seen=set()
    with tarfile.open(archive) as tar:
        for member in tar:
            safe(member.name)
            prefix='cpu-evidence-'+args.edition+'/'
            require(member.isfile() and member.name.startswith(prefix),'Unsafe compact evidence member')
            name=member.name.removeprefix(prefix)
            require(name not in seen and name in set(files)|{'metadata.json'},'Unexpected/duplicate compact member')
            seen.add(name)
            require(tar.extractfile(member).read()==(folder/name).read_bytes(),'Compact archive/extracted byte mismatch')
    require(seen==set(files)|{'metadata.json'} and len(seen)==receipt['safe_members'],'Incomplete compact evidence archive')
    package=read(HERE/'package.json')
    require(package['manifest_sha256']==EXPECTED_MANIFEST and package['source_files']==633,'Wrong current source package')
    manifest_bytes=(folder/'source/source_manifest.json').read_bytes()
    require(digest(manifest_bytes)==EXPECTED_MANIFEST,'Delivered exact source manifest differs')
    manifest=json.loads(manifest_bytes)
    source=source_archive(package,manifest)
    for arm in ('boltz2','boltz2_chemistry'):
        name='examples/configs/boltz_serial_'+arm+'.json'
        require((folder/'source'/name).read_bytes()==source[name]==(HERE/(arm+'.json')).read_bytes(),'Exact baseline configuration bytes differ')
    staging=read(folder/'run/source-staging.json')
    require(staging['status']=='passed' and staging['files']==633 and staging['manifest']==EXPECTED_MANIFEST and staging['archive_sha256']==EXPECTED_ARCHIVE and staging['archive_bytes']==39618948 and staging['readonly'] is True,'Actual staging seal differs')
    jobs=read(folder/'run/jobs.json')
    require(jobs['cpu']==EXPECTED_CPU and jobs['after-cpu']==EXPECTED_CALLBACK,'Wrong actual initial jobs')
    observed={}
    for label,job in jobs.items():
        sub=read(folder/'run'/label/'submission.json')
        phase=sub['phase'];dependency=sub['dependency']
        require(sub['outcome']=='submitted' and sub['job_id']==job and sub['returncode']==0 and re.findall(r'Job <(\d+)>',sub['stdout'])==[job] and sub['args']==vector(label,phase,EXPECTED_MANIFEST,dependency),'Actual submission/resource/source vector mismatch')
        if phase=='after-cpu':require(dependency=='ended('+jobs['cpu']+')','CPU callback dependency differs')
        elif phase in ('after-native','after-segment'):
            producer=label.replace('-after-native-','-native-').replace('-after-segment-','-segment-')
            require(dependency=='ended('+jobs[producer]+')','Own-ended callback dependency differs')
        elif phase in ('cpu','native','segment'):require(dependency is None,'Unexpected interarm/native dependency')
        query=metadata['scheduler'][label]
        require(query['job_id']==job,'Current scheduler map mismatch')
        if query['returncode']==0:observed[label]=scheduler(query['raw'],job,'gpul40s' if phase in ('native','segment') else 'hpc')
        else:observed[label]='unavailable_live_history'
    gates={};gate_errors={}
    for name in files:
        if name.endswith('/gate.json'):
            gate=read(folder/name)
            require(gate['reserve_bytes']==4294967296 and gate['file_reserve']==100000,'Quota reserves weakened')
            gates[name]=gate['status']
            if gate['status']=='failed':gate_errors[name]=gate.get('error','No error string; inspect exact failed gate bytes')
            if gate['status']=='passed':
                work=gate['work3']
                require(work['status']=='passed' and work['available_bytes_conservative']>=work['required_bytes'] and work['available_files_lower']>=100000,'Passed quota proof is incoherent')
                inode=work['filesystem_inode_observation']
                if tuple(inode[k] for k in ('f_files','f_ffree','f_favail'))==(0,0,0):
                    require(work['filesystem_inode_reporting']=='unreported_all_zero','All-zero physical inode scope mislabeled')
    output=args.output or HERE/('cpu-failure-verification.json' if args.expect_failure else 'cpu-verification.json')
    require(not output.exists(),'Do not overwrite an earlier verification')
    if args.expect_failure:
        require(observed['cpu']=='EXIT' and observed['after-cpu']=='EXIT','Actual initial failure not authenticated')
        for label in ('cpu','after-cpu'):
            log=(folder/'run'/label/('stdout.'+jobs[label]+'.log')).read_text(encoding='utf-8')
            lsf(log,jobs[label],False)
        require('run/cpu-certificate.json' not in files and not any('native-' in label or '-segment-' in label for label in jobs),'Failure attempt unexpectedly advanced; inspect separate outcome')
        found_xml=[name for name in files if name.startswith('run/cpu/payload/') and name.endswith('.xml')]
        partial_counts={}
        for name in found_xml:
            try:
                tree=ET.parse(folder/name)
                partial_counts[name]={kind:len(list(tree.iter(kind))) for kind in ('testcase','failure','error','skipped')}
            except ET.ParseError as error:
                partial_counts[name]={'incomplete_xml':repr(error)}
        value={'status':'failed_attempt_evidence_verified','cpu_scientific_status':'failed',
            'cpu_passed':False,'cases_claimed_passed':None,'jobs':jobs,'actual_scheduler':observed,
            'storage_gates':gates,'storage_errors_verbatim':gate_errors,'partial_xml_counts':partial_counts,
            'files_verified':len(files),'source_members_verified':633,
            'source_manifest_sha256':EXPECTED_MANIFEST,'evidence_archive_sha256':receipt['sha256'],
            'scope':'Independent stdlib byte/schema/source/actualscheduler/LSF failure audit only. No CPU/native/optimization pass or local numerical execution.'}
        output.write_bytes((json.dumps(value,indent=2)+'\n').encode());print(json.dumps(value,indent=2));return
    certificate=read(folder/'run/cpu-certificate.json')
    require(certificate['status']=='passed' and certificate['suite']=='boltz-two-arm-serial30k-v1' and certificate['source_manifest_sha256']==EXPECTED_MANIFEST,'Wrong full CPU certificate')
    for proof in certificate['proofs']:
        require(proof['path'] in remote and remote[proof['path']]['sha256']==proof['sha256'],'CPU certificate proof omitted/mismatched')
    groups={};new_count=0;header_counts={}
    wrapper=source['scripts/hpc/boltz_serial_cpu.lsf'].decode()
    declared=set(re.findall(r'tests/(test_[A-Za-z0-9_]+)\.py',wrapper.split('contracts=(',1)[1].split(')',1)[0]))
    for group in ('hpc-contract','legacy','native-mapping','chemistry'):
        path=folder/'run/cpu/payload'/(group+'.xml')
        parsed=ET.parse(path);cases=list(parsed.iter('testcase'))
        require(not any(list(parsed.iter(kind)) for kind in ('failure','error','skipped')),'Raw allocated CPU failure/error/skip exists')
        groups[group]=len(cases)
        log=path.with_suffix('.log').read_text(encoding='utf-8')
        require(re.search(r'\b'+str(len(cases))+r'\s+passed\b',log),'Actual testcase rows/log pass count differ')
        header_counts[group]=sum(int(suite.get('tests','0')) for suite in parsed.iter('testsuite'))
        if group=='hpc-contract':
            found={part for case in cases for part in case.get('classname','').split('.')}
            require(declared<=found and all(case.get('classname') for case in cases),'A required inherited/new module is unattributed')
            new_count=sum('test_boltz_serial' in case.get('classname','').split('.') for case in cases)
            require(new_count==71 and len(cases)==467,'Exact new/inherited CPU contract rows differ')
        elif group=='legacy':require(re.search(r'\b3\s+deselected\b',log),'Historical three intentional deselections differ')
    require(groups=={'hpc-contract':467,'legacy':239,'native-mapping':34,'chemistry':59} and certificate['groups']==groups and certificate['cases']==sum(groups.values())==799,'Full actual CPU case scope differs')
    final=[proof for proof in certificate['proofs'] if re.fullmatch(r'scheduler-final-\d+\.txt',PurePosixPath(proof['path']).name)]
    require(len(final)==1,'Missing/ambiguous certificate-bound final CPU scheduler')
    local='run/'+final[0]['path'].split(ROOT+'/',1)[1]
    scheduler((folder/local).read_text(),jobs['cpu'],'hpc','DONE')
    require(observed['after-cpu']=='DONE','Actual callback completion not established')
    for label in ('cpu','after-cpu'):
        lsf((folder/'run'/label/('stdout.'+jobs[label]+'.log')).read_text(encoding='utf-8'),jobs[label],True)
    assessment=read(folder/'run/cpu/payload/assessment.json')
    require(assessment['status']=='passed' and assessment['passed']==799 and assessment['source_manifest_sha256']==EXPECTED_MANIFEST,'Allocated CPU assessment differs')
    require(digest((folder/'run/cpu/payload/pocket_definition.json').read_bytes())==POCKET and digest((folder/'run/cpu/payload/composition_reference.json').read_bytes())==REFERENCE,'Exact pocket/reference changed')
    natives={}
    for arm in ('boltz2','boltz2_chemistry'):
        label=arm+'-native-0000';callback=arm+'-after-native-0000'
        require(label in jobs and callback in jobs,'Both original fresh native/callback submissions not established')
        natives[arm]={'job':jobs[label],'callback':jobs[callback],'scheduler_state':observed[label]}
    require(gates and all(state=='passed' for state in gates.values()),'One captured actual storage gate failed')
    value={'status':'passed','cpu_cases':799,'new_cases':new_count,'groups':groups,
        'xml_header_counts':header_counts,'xml_header_limitation':'Actual testcase rows/logs are authoritative; inherited unittest subtests add3 to the contract header, not extra passed cases.',
        'files_verified':len(files),'source_members_verified':633,'source_manifest_sha256':EXPECTED_MANIFEST,
        'cpu_certificate_sha256':digest((folder/'run/cpu-certificate.json').read_bytes()),
        'jobs':jobs,'actual_scheduler':observed,'fresh_native_submissions':natives,
        'native_readiness_passed_by_this_audit':False,'optimization_started_by_this_audit':False,
        'remote_only_native_references':metadata['remote_only_native_references'],
        'evidence_archive_sha256':receipt['sha256'],
        'scope':'Independent stdlib byte/XML/schema/source/config/actualresources/scheduler/LSF/storage audit only; no local scientific arrays/models/tests. Native pending/results are separate from full CPU acceptance.'}
    output.write_bytes((json.dumps(value,indent=2)+'\n').encode());print(json.dumps(value,indent=2))


if __name__=='__main__':
    main()
