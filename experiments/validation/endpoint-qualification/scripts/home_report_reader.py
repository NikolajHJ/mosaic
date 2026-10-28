"""Authenticate one qualified HOME report engine as read-only for WORK3.

Using the WORK3 interpreter is not a WORK3 output allocation. This exception
requires exact engine bytes, HOME-only profile, qualified CPU proof and actual
request/submission ownership. It grants no model/optimizer or HOME admission.
"""
import hashlib,json,shlex
from pathlib import Path,PurePosixPath
import xml.etree.ElementTree as ET

PIN='4db7ca7c58a8481cd4e17c4f8a2df4aafad3e94e190408a2a4c0db30da168165'
HOME=PurePosixPath('/zhome/95/e/206121/mosaic-runs/reusable-saved-report-v2')
SOURCE=HOME/'engines'/PIN/'source'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'


def require(value,message):
    if not value:raise ValueError(message)


def binding(path):
    p=Path(path);require(p.is_file() and not p.is_symlink() and all(not a.is_symlink() for a in p.parents),'Report metadata linked/special')
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def load(path):return json.loads(Path(path).read_bytes())


def command_request(command):
    tokens=shlex.split(command)
    if len(tokens)!=11 or tokens[:4]!=[PYTHON,'-B','-u',str(SOURCE/'scripts/quick_runtime.py')]:return None
    if tokens[4] not in ('validate','render') or tokens[5:9]!=['--source',str(SOURCE),'--pin',PIN] or tokens[9]!='--request':return None
    p=PurePosixPath(tokens[10])
    if not p.is_absolute() or str(p)!=tokens[10] or '..' in p.parts:return None
    if tokens[4]=='validate':return p if p==SOURCE.parent/'validation/request.json' else None
    return p if HOME/'requests' in p.parents and p.name=='request.json' else None


def profile_home_only(profile):
    require(profile.get('output_filesystem')=='home' and profile.get('remote_root')==str(HOME)
            and profile.get('home_quota_root')==str(HOME.parent) and profile.get('python')==PYTHON,
            'Report output profile is not the qualified HOME namespace')
    require(all(PurePosixPath(p).is_relative_to(HOME.parent) and '..' not in PurePosixPath(p).parts
                for p in profile.get('additional_owned_paths',[])), 'Extra report writes could escape HOME')
    require(type(profile.get('request_cap_bytes')) is int and 0<profile['request_cap_bytes']<=1024**3
            and type(profile.get('validation_cap_bytes')) is int and 0<profile['validation_cap_bytes']<=64*1024**2,
            'Qualified HOME report caps changed')


def observe(rows):
    matched=[(row,command_request(row['COMMAND'])) for row in rows]
    matched=[(row,path) for row,path in matched if path is not None]
    if not matched:return {'work3_read_only_job_ids':[],'proofs':[],'work3_output_bytes_requested':0}
    manifest_path=SOURCE/'engine-manifest.json'
    require(binding(manifest_path)['sha256']==PIN,'Qualified HOME engine manifest changed')
    manifest=load(manifest_path)
    for name,digest in manifest['files'].items():
        p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'Report source member path invalid')
        require(binding(SOURCE/p)['sha256']==digest,'Qualified report engine member changed')
    profile=load(SOURCE/'profile.json');profile_home_only(profile)
    certificate=load(SOURCE.parent/'validation-certificate.json')
    require(certificate['status']=='passed' and certificate['engine_pin']==PIN and certificate['cases']==208,
            'HOME report allocated qualification missing')
    for pin in certificate['proofs']:require(binding(pin['path'])==pin,'HOME report qualified proof changed')
    cases=list(ET.parse(Path(SOURCE.parent/'validation/tests.xml')).iter('testcase'))
    require(len(cases)==208 and all(not list(c.iter(t)) for c in cases for t in ('failure','error','skipped')),
            'Report qualification raw XML differs')
    proofs=[binding(manifest_path),binding(SOURCE/'profile.json'),binding(SOURCE.parent/'validation-certificate.json')]
    ids=[]
    for row,request in matched:
        require(row['QUEUE']=='hpc','HOME report queue differs')
        sub=load(request.parent/'submission.json')
        tokens=shlex.split(row['COMMAND'])
        require(sub['job_id']==row['JOBID'] and sub['argv'][-len(tokens):]==tokens,
                'Actual HOME report command/submission differs')
        proofs += [binding(request),binding(request.parent/'submission.json')];ids.append(row['JOBID'])
    return {'work3_read_only_job_ids':ids,'proofs':proofs,'work3_output_bytes_requested':0,
            'HOME_capacity_and_publication_still_required':True,'report_jobs_or_inputs_changed':False}
