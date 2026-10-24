"""Independent stdlib audit of actual allocated adoption CPU evidence.

Prepared only: root supplies a fresh collector directory and exact sealed
preparation/manifest. A metadata CPU pass never certifies scientific gradients,
an observer, adoption, GPU startup or optimization. No network/model execution.
"""
from __future__ import annotations
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[3]
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
VERSION = 'protenix02-peer-adoption-v1'
PROFILES = {
    'boltz30k': dict(source=BASE+'/boltz-lr0025-release-03/source', root=BASE+'/boltz-lr0025-03',
        control=BASE+'/boltz-pro02-adoption-04', launcher=BASE+'/boltz-pro02-adoption-release-04/source',
        manifest='c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6', source_count=635,
        cpu_sha='06efbb027f1008a0921494af0e226c3fc7a78c0a8f4ab01b93e8b50f2c042836', cpu_cases=827,
        natives={'boltz2-native-0000':('29805950','29805951'), 'boltz2_chemistry-native-0000':('29805958','29805959')}),
    'dbmol10k': dict(source=BASE+'/boltz-dbmol-pilot-release-02/source', root=BASE+'/boltz-dbmol-pilot-02',
        control=BASE+'/dbmol-pro02-adoption-06', launcher=BASE+'/dbmol-pro02-adoption-release-06/source',
        manifest='600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed', source_count=657,
        cpu_sha='387070e8fe56b17ddbd7c8e0b82d2d6a44d9aa99e7015d3fbe4757ce82c0819b', cpu_cases=1163,
        natives={'pilot-native-0000':('29805980','29805981')}),
}
ORIGINAL_SOURCES = {
    BASE+'/boltz-lr0025-release-03/source':('c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6',635),
    BASE+'/boltz-dbmol-pilot-release-02/source':('600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed',657),
    BASE+'/boltz-dbmol-storage-release-05/source':('865303de06fc0a0bdf9297857c5bbc78c6ccd9ee05177e372a51869adc599bc3',10),
    BASE+'/protenix-extended-release-03/source':('da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562',643),
}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def strict(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, 'Duplicate JSON field'); value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def local(path, maximum=128*1024**2):
    path = Path(path); require(path.is_file() and not path.is_symlink(), 'Missing/linked local proof')
    literal_before = path.stat()
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno()); raw = stream.read(maximum+1); after = os.fstat(stream.fileno())
    literal_after = path.stat()
    full = lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_mode,x.st_nlink)
    shared = lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_mode,x.st_nlink)
    require(len(raw)==before.st_size<=maximum and full(before)==full(after)
        and full(literal_before)==full(literal_after) and shared(before)==shared(literal_before), 'Local evidence changed')
    return raw


class Evidence:
    def __init__(self, path):
        self.directory = Path(path).resolve()
        require(self.directory.is_relative_to(REPO/'results/hpc'), 'Evidence outside dedicated result scope')
        self.raw = local(self.directory/'stdout.bin'); stderr = local(self.directory/'stderr.bin')
        result = strict(local(self.directory/'result.json')); self.value = strict(self.raw)
        require(result['returncode']==0 and result['remote_mutations'] is False
            and result['stdout_sha256']==sha(self.raw) and result['stderr_sha256']==sha(stderr), 'Collection failed or changed')
        metadata = strict(local(self.directory/'metadata.json'))
        require(self.value['version']=='mosaic-peer-adoption-read-only-evidence-v1' and self.value['read_only'] is True
            and self.value['numerical_execution'] is False and self.value['scientific_or_production_pass_claimed'] is False,
            'Wrong collection scope')
        self.files = {}; self.pins = {}
        require(set(metadata['files'])==set(self.value['files']), 'Extracted metadata file scope differs')
        for path, row in self.value['files'].items():
            data = base64.b64decode(row['base64'], validate=True); index = metadata['files'][path]
            require(len(data)==row['bytes'] and sha(data)==row['sha256'] and row['remote_bytes_rehashed'] is True,
                'Raw collected original bytes differ')
            target = self.directory/index['local_path']
            require(target.resolve().is_relative_to(self.directory) and local(target)==data
                and all(index[k]==row[k] for k in ('path','bytes','sha256','remote_bytes_rehashed','local_bytes_downloaded')),
                'Extracted original bytes/index differ')
            self.files[path] = data; self.pins[path] = row
        for key in ('profile','launcher_manifest_sha256','sources','roots','external_files','scheduler','missing_references','opaque_traversal_boundaries'):
            require(self.value[key]==metadata[key], 'Raw/decoded metadata differs: '+key)
        for path, row in self.value['external_files'].items():
            require(path not in self.files and row['local_bytes_downloaded'] is False and type(row['bytes']) is int
                and row['bytes']>=0 and (row['sha256'] is None or re.fullmatch('[0-9a-f]{64}',row['sha256'])),
                'Malformed external reference')
            self.pins[path] = row

    def read(self, path):
        require(path in self.files, 'Missing compact original bytes: '+path); return self.files[path]

    def json(self, path): return strict(self.read(path))

    def proof(self, pin, fresh=True):
        path = pin['path']; require(path in self.pins, 'Missing original proof: '+path); row = self.pins[path]
        require(row['sha256']==pin['sha256'] and ('bytes' not in pin or row['bytes']==pin['bytes'])
            and (not fresh or row['remote_bytes_rehashed'] is True), 'Original proof hash/length/scope differs: '+path)
        return row

    def certificate(self, record, fresh=True):
        require(record['status']=='passed' and isinstance(record['proofs'],list) and record['proofs']
            and len({x['path'] for x in record['proofs']})==len(record['proofs']), 'Failed/empty/duplicate certificate')
        for pin in record['proofs']: self.proof(pin, fresh)


def declaration(raw, name):
    found = [ast.literal_eval(node.value) for node in ast.parse(raw).body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id==name for target in node.targets)]
    require(len(found)==1, 'Missing unique reviewed declaration: '+name); return found[0]


def verify_package(preparation, profile, expected):
    prep_raw = local(preparation); prep = strict(prep_raw)
    require(prep['status']=='prepared_local_only' and prep['profile']==profile and prep['source_file_count']==7
        and prep['manifest_sha256']==expected, 'Wrong sealed local launcher preparation')
    path = Path(prep['archive']['path']); archive_raw = local(path)
    require(len(archive_raw)==prep['archive']['bytes'] and sha(archive_raw)==prep['archive']['sha256'], 'Launcher archive changed')
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive_raw)) as archive:
        for member in archive:
            rel = PurePosixPath(member.name)
            require(member.isfile() and not member.linkname and not rel.is_absolute() and '..' not in rel.parts
                and rel.as_posix()==member.name and '\\' not in member.name and ':' not in member.name
                and member.name not in files, 'Unsafe/duplicate launcher archive member')
            files[member.name] = archive.extractfile(member).read()
    raw = files.pop('source_manifest.json'); manifest = strict(raw)
    require(sha(raw)==expected and set(files)==set(manifest['files']) and len(files)==7
        and all(sha(data)==manifest['files'][name] for name,data in files.items())
        and manifest['profile']==profile and manifest['git_revision_reference']=='c74c659344123937608077e3e4078f75f09cd743'
        and manifest['scientific_source_unchanged'] is True and manifest['metadata_cpu_is_scientific_readiness'] is False,
        'Sealed launcher/member identity differs')
    count = manifest['metadata_cpu_cases']
    require(type(count) is int and count>0 and prep['metadata_cpu_cases_planned']==count
        and declaration(files['scripts/adoption_ops.py'],'METADATA_CASES')==count
        and declaration(files['tests/test_peer_adoption.py'],'METADATA_CASES')==count,
        'Exact reviewed fixture count differs across manifest/ops/fixture')
    return files, raw, manifest, dict(path=str(path),bytes=len(archive_raw),sha256=sha(archive_raw),metadata_cases=count)


def command(p, profile, label, phase, manifest, dependency=None):
    require(phase in ('cpu','after-cpu'), 'Only metadata CPU/callback command certified here')
    parent = p['control']+'/'+label
    args = ['bsub','-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30',
        '-J','mosaic-peer-'+profile+'-'+label,'-oo',parent+'/stdout.%J.log','-eo',parent+'/stderr.%J.log']
    if dependency: args += ['-w',dependency]
    return args+['bash','-l',p['launcher']+'/scripts/hpc/'+('peer_adoption_cpu.lsf' if phase=='cpu' else 'peer_adoption_stage.lsf'),
        p['launcher'],manifest,profile,p['source'],p['root'],p['manifest'],p['control'],phase,parent]


def lsf(raw, job):
    text = raw.decode()
    require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',text,re.M)==[(job,'Done')]
        and 'Successfully completed.' in text and 'Exited with exit' not in text, 'Same-job LSF success missing')


def storage_gate(value):
    work=value['work3']
    require(value['status']==work['status']=='passed' and work['reserve_bytes']==4*1024**3
        and work['file_reserve']==100000 and work['required_bytes']==value['projected_bytes']+4*1024**3
        and work['available_bytes_conservative']>=work['required_bytes'] and work['available_files_lower']>=100000,
        'Actual metadata entry storage/reserve/headroom failed')
    inode=work['filesystem_inode_observation']
    if tuple(inode[k] for k in ('f_files','f_ffree','f_favail'))==(0,0,0):
        require(work['filesystem_inode_reporting']=='unreported_all_zero','All-zero supplemental inode tuple mislabeled')


def verify(e, args, result):
    p = PROFILES[args.profile]; files, manifest_raw, manifest, package = verify_package(args.preparation,args.profile,args.launcher_manifest)
    require(e.value['profile']==args.profile and e.value['launcher_manifest_sha256']==args.launcher_manifest,
        'Collection profile/launcher differs')
    boundaries=e.value['opaque_traversal_boundaries']
    require(isinstance(boundaries,list) and len(boundaries)<=1,'Unexpected opaque traversal scopes')
    for boundary in boundaries:
        require(boundary['path']==BASE+'/lrreset-analysis-03/complete-protenix/payload/snapshots.json'
            and boundary['bytes']==2549313 and boundary['sha256']=='ec8fa62462e5e9db4bb83f431ed43121e30b619f3bc7b43e53b7295224d5d919'
            and boundary['inventory_bytes_freshly_rehashed_and_downloaded'] is True
            and boundary['omitted_child_frame_references']==10001
            and boundary['child_probability_frames_freshly_rehashed_by_this_collection'] is False,
            'Undeclared prior-parent traversal omission')
        raw=e.read(boundary['path']);inventory=strict(raw)
        require(len(raw)==boundary['bytes'] and sha(raw)==boundary['sha256']
            and inventory['status']=='passed' and inventory['native_job']=='29792670'
            and inventory['n_atoms']==25 and len(inventory['frames'])==10001,
            'Opaque prior-parent inventory original bytes differ')
    expected_sources = {**ORIGINAL_SOURCES,p['launcher']:(args.launcher_manifest,7)}
    require(set(e.value['sources'])==set(expected_sources), 'Full source observation scopes differ')
    for root,(expected,count) in expected_sources.items():
        observed = e.value['sources'][root]; raw = e.read(root+'/source_manifest.json'); value = strict(raw)
        require(sha(raw)==expected and len(value['files'])==count and observed['manifest_sha256']==expected
            and observed['file_count']==count and observed['readonly'] is True and observed['all_members_rehashed'] is True,
            'Full original/source launcher seal not authenticated')
        require({x['path']:x['sha256'] for x in observed['members']}=={root+'/'+name:digest for name,digest in value['files'].items()}
            and len(observed['members'])==count, 'Full source proof member scope differs')
        for row in observed['members']: e.proof(row)
    require(e.read(p['launcher']+'/source_manifest.json')==manifest_raw, 'Remote/local launcher manifest differs')
    for name,data in files.items(): require(e.read(p['launcher']+'/'+name)==data, 'Remote/local launcher member differs')
    root = p['control']; jobs = e.json(root+'/jobs.json')
    require(jobs==e.value['roots'][root]['jobs'] and {'cpu','after-cpu'}<=set(jobs)
        and len(set(jobs.values()))==len(jobs) and all(re.fullmatch('[0-9]+',x) for x in jobs.values()), 'Actual unique job registry missing')
    submissions = {}
    for label,phase,dep in [('cpu','cpu',None),('after-cpu','after-cpu','ended('+jobs['cpu']+')')]:
        sub = e.json(root+'/'+label+'/submission.json'); started = e.json(root+'/'+label+'/submission-started.json')
        require(sub['outcome']=='submitted' and sub['job_id']==jobs[label] and sub['returncode']==0
            and sub['phase']==phase and sub['dependency']==dep and sub['source_manifest_sha256']==p['manifest']
            and sub['launcher_manifest_sha256']==args.launcher_manifest and sub['args']==command(p,args.profile,label,phase,args.launcher_manifest,dep)
            and started['outcome']=='unknown' and started['args']==sub['args'] and started['dependency']==dep,
            'Actual metadata submission/source/resources differ')
        submissions[label] = sub
    cert = e.json(root+'/metadata-cpu-certificate.json'); e.certificate(cert)
    count = manifest['metadata_cpu_cases']
    require(cert['version']==VERSION and cert['suite']=='mosaic-peer-adoption-metadata-v1' and cert['profile']==args.profile
        and cert['source_manifest_sha256']==p['manifest'] and cert['launcher_manifest_sha256']==args.launcher_manifest
        and cert['cases']==count and cert['scientific_readiness_passed'] is False
        and cert['future_peer_results_required_for_this_cpu'] is False and cert['original_scientific_cpu_certificate_replaced'] is False,
        'Actual metadata CPU certificate scope differs')
    xml_path=root+'/cpu/payload/adoption.xml'; xml=ET.fromstring(e.read(xml_path)); cases=list(xml.iter('testcase'))
    require(len(cases)==count and len({(x.get('classname'),x.get('name')) for x in cases})==count
        and not any(list(xml.iter(k)) for k in ('failure','error','skipped'))
        and all('test_peer_adoption' in x.get('classname','').split('.') and not list(x) for x in cases)
        and all(int(node.get(k,'0'))==0 for node in xml.iter('testsuite') for k in ('failures','errors','skipped')),
        'Raw exact metadata testcase count/module/pass differs')
    require(re.search(r'\b'+str(count)+r'\s+passed\b',e.read(root+'/cpu/payload/adoption.log').decode()), 'Actual XML/log count differs')
    require(any(x['path']==xml_path and x['sha256']==sha(e.read(xml_path)) for x in cert['proofs']), 'Certificate lacks original raw XML')
    scheduler = [x['path'] for x in cert['proofs'] if PurePosixPath(x['path']).parent==PurePosixPath(root+'/cpu')
        and re.fullmatch('scheduler-final-[0-9]+.txt',PurePosixPath(x['path']).name)]
    require(len(scheduler)==1, 'Unique final CPU scheduler proof missing')
    text = re.sub(r'\s+','',e.read(scheduler[0]).decode())
    require(re.findall('Job<([^>]+)>',text)==[jobs['cpu']] and re.findall('Status<([^>]+)>',text)==['DONE']
        and re.findall('Queue<([^>]+)>',text)==['hpc'], 'Actual final same-job CPU DONE/hpc proof differs')
    lsf(e.read(root+'/cpu/stdout.'+jobs['cpu']+'.log'),jobs['cpu'])
    gate_root=root+'/cpu/storage-metadata-cpu-entry/'
    original_gate=e.json(gate_root+'gate.json'); storage_gate(original_gate)
    adopted_gate=e.json(gate_root+'adoption-capacity.json'); storage_gate(adopted_gate)
    require(adopted_gate['no_gate_waiver'] is True and adopted_gate['original_gate']==original_gate
        and set(adopted_gate['metadata_controls'])=={'boltz30k','dbmol10k'}
        and all(row['full']==1024**3 and 0<=row['current']<=row['full']
            and row['remaining']==row['full']-row['current'] for row in adopted_gate['metadata_controls'].values())
        and adopted_gate['projected_bytes']==original_gate['projected_bytes']+
            sum(row['remaining'] for row in adopted_gate['metadata_controls'].values()),
        'Both additional metadata control allowances or original whole budget changed')
    noise=e.json(root+'/cpu/adoption-noisy-inventory-metadata-cpu-entry.json')
    require(noise['status']=='passed' and noise['inventory']['status']=='declared_capsules_only',
        'Actual external capsule inventory did not pass')
    old_raw=e.read(p['root']+'/cpu-certificate.json'); old=strict(old_raw)
    require(sha(old_raw)==p['cpu_sha'] and old['status']=='passed' and old['cases']==p['cpu_cases']
        and old['source_manifest_sha256']==p['manifest'], 'Pinned original scientific CPU certificate changed')
    if args.profile=='dbmol10k': require(old['launcher_manifest_sha256']=='865303de06fc0a0bdf9297857c5bbc78c6ccd9ee05177e372a51869adc599bc3','Original DBMol launcher changed')
    original_native=[]
    for label,(native,callback) in p['natives'].items():
        sub=e.json(p['root']+'/'+label+'/submission.json'); cb=label.replace('-native-','-after-native-')
        require(sub['job_id']==native and sub['outcome']=='submitted' and e.json(p['root']+'/'+cb+'/submission.json')['job_id']==callback,
            'Original native IDs changed')
        lsf(e.read(p['root']+'/'+label+'/stdout.'+native+'.log'),native)
        lsf(e.read(p['root']+'/'+cb+'/stdout.'+callback+'.log'),callback)
        ready=e.json(p['root']+'/'+label+'/payload/study/assessment.json')
        require(ready['status']=='passed' and ready['source_manifest_sha256']==p['manifest'], 'Original native readiness differs')
        certificate=e.json(p['root']+'/'+label+'-certificate.json');e.certificate(certificate,fresh=False)
        require(certificate['source_manifest_sha256']==p['manifest']
            and any(row['path']==p['root']+'/'+label+'/payload/study/assessment.json'
                and row['sha256']==sha(e.read(row['path'])) for row in certificate['proofs'])
            and any(row['path']==p['root']+'/'+label+'/stdout.'+native+'.log'
                and row['sha256']==sha(e.read(row['path'])) for row in certificate['proofs']),
            'Original native certificate lacks exact source/assessment/same-job LSF binding')
        original_native.append(dict(label=label,native_job_id=native,callback_job_id=callback,
            assessment_sha256=sha(e.read(p['root']+'/'+label+'/payload/study/assessment.json')),
            scope='Prior allocated original native result metadata and actual same-job persisted LSF; no local derivative re-evaluation.'))
    observations={}
    for label in jobs:
        if label=='cpu': continue
        observations[label]={'job_id':jobs[label],'live_scheduler_query':e.value['scheduler'].get(jobs[label]),
            'completion_or_start_claimed':False}
    optional={name:dict(present=root+'/'+name in e.files,
        sha256=sha(e.files[root+'/'+name]) if root+'/'+name in e.files else None,
        independently_certified_by_this_cpu_audit=False)
        for name in ('adoption-certificate.json','ownership.json')}
    result.update(status='passed_independent_metadata_cpu_evidence', profile=args.profile,
        source_manifest_sha256=p['manifest'], launcher_manifest_sha256=args.launcher_manifest,
        actual_cpu_job_id=jobs['cpu'], actual_raw_cases=count, metadata_cpu_pass_authenticated=True,
        scientific_cpu_certificate_replaced=False, original_scientific_cpu_sha256=p['cpu_sha'],
        original_native_metadata=original_native, launcher_package=package,
        metadata_entry_original_gate_sha256=sha(e.read(gate_root+'gate.json')),
        metadata_entry_adoption_capacity_sha256=sha(e.read(gate_root+'adoption-capacity.json')),
        optional_adoption_records=optional, other_actual_jobs_observed=observations,
        committed_prefix_observations={key:value.get('saved_prefixes',{}) for key,value in e.value['roots'].items()},
        scientific_readiness_or_gradients_pass_claimed=False, observer_or_adoption_pass_claimed=False,
        gpu_startup_or_production_start_claimed=False, full_production_storage_admission_pass_claimed=False,
        large_or_expired_references_not_recertified=[path for path,row in e.pins.items() if not row['remote_bytes_rehashed']],
        opaque_prior_parent_inventory_boundaries=boundaries,
        prior_parent_probability_frames_freshly_rehashed_by_this_collection=False,
        scope='Current allocated metadata CPU only: exact source hashes, actual resources/LSF/final scheduler and raw testcase rows. Optional records/prefixes remain observations, never optimization or scientific passes.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=tuple(PROFILES),required=True)
    parser.add_argument('--launcher-manifest',required=True)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args(); require(re.fullmatch('[0-9a-f]{64}',args.launcher_manifest),'Invalid launcher pin')
    directory=args.evidence.resolve(); require(directory.is_relative_to(REPO/'results/hpc'),'Evidence outside results')
    output=directory/'adoption-cpu-verification.json';require(not output.exists(),'Existing outcome; preserve and use fresh evidence')
    result=dict(version='mosaic-peer-adoption-independent-cpu-audit-v1',status='failed_or_not_ready',
        verified_utc=datetime.now(timezone.utc).isoformat(),verifier_sha256=sha(local(__file__)),
        metadata_cpu_pass_authenticated=False,scientific_readiness_or_gradients_pass_claimed=False,
        observer_or_adoption_pass_claimed=False,gpu_startup_or_production_start_claimed=False)
    code=1
    try:
        evidence=Evidence(directory);result['collected_stdout_sha256']=sha(evidence.raw);verify(evidence,args,result);code=0
    except Exception as error:result.update(status='failed_or_not_ready',error=type(error).__name__+': '+str(error))
    with output.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','metadata_cpu_pass_authenticated','scientific_readiness_or_gradients_pass_claimed','gpu_startup_or_production_start_claimed')}))
    return code


if __name__=='__main__':raise SystemExit(main())
