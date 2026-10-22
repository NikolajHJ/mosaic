"""Allocated orchestration for the separate fixed-step saved-only report.

No native job is submitted, stopped, resumed or modified here. Immutable report
editions are produced only after allocated CPU fixtures and persisted identities.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from jq1_fixed_report_contract import ARMS, read, require, sha

TAG = '20261021-brd4-jq1-25-fixed-report-05'
ROOT = Path('/work3/s234818/mosaic/runs') / TAG
SOURCE = Path('/work3/s234818/mosaic/releases') / TAG / 'source'
SCIENCE = Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-02/source')
SCIENCE_ROOT = Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-02')
ROOT03 = Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-03')
ROOT04 = Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-04')
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
SUITE = 'jq1-fixed-saved-report-contract-v1'
TESTS = ('test_jq1_fixed_report_contract.py', 'test_jq1_fixed_render.py',
         'test_jq1_fixed_publish.py', 'test_jq1_fixed_ops.py')
LEGACY_TESTS = ('test_boltz_affinity_report.py', 'test_small_molecule_chemistry_report.py',
                'test_small_molecule_chemistry_ramp_report.py')


def exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def actual_scheduler(raw, job, queue='hpc'):
    compact = re.sub(r'\s+', '', raw)
    require(re.findall(r'Job<([^>]+)>', compact) == [job]
            and re.findall(r'Queue<([^>]+)>', compact) == [queue], 'Actual job/queue identity differs')
    statuses = re.findall(r'Status<([^>]+)>', compact)
    require(statuses == ['DONE'], 'Actual CPU/report job is not DONE')


def actual_lsf(raw, job):
    subjects = re.findall(r'^Subject:\s*Job\s+(\d+):([^\r\n]*)', raw, re.M)
    require(len(subjects) == 1 and subjects[0][0] == job and subjects[0][1].rstrip().endswith(' Done')
            and re.search(r'^Successfully completed\.\s*$', raw, re.M)
            and not re.search(r'Exited\s+(?:with exit|by signal)', raw, re.I), 'Actual same-job LSF success missing')


def report_budget(current, quota_available, disk_free, stale, phase):
    full = {'protenix':1493301824, 'boltz2':1770121824}
    require(phase in ('preflight','cpu','capture','interim','final'), 'Unknown report budget phase')
    require(set(current)==set(full) and all(type(n) is int and 0<=n<=full[a] for a,n in current.items()), 'Native output exceeded pinned projection')
    require(all(type(n) is int and n>=0 for n in (quota_available,disk_free,stale)), 'Malformed quota/disk/stale accounting')
    outstanding=sum(full[a]-current[a] for a in full)
    allowance=(1 if phase in ('preflight','cpu') else 2)*1024**3
    return {'current_native_bytes':current,'full_native_projections':full,'outstanding_native_bytes':outstanding,
            'report_allowance_bytes':allowance,'stale_bytes_subtracted':stale,
            'available_bytes_conservative':min(quota_available,disk_free)-stale,
            'required_bytes':allowance+outstanding+4*1024**3}


def check_storage(parent, phase):
    """The report writes work3 only; retain exposure of both work3 native arms."""
    from fixed25_split_storage import (tree_bytes, parse_work3, inode_observation,
                                       inode_headroom, RESERVE, FILE_RESERVE)
    require(ROOT.resolve() == ROOT and Path(parent).resolve().is_relative_to(ROOT), 'Wrong report storage scope')
    folder = Path(parent) / ('storage-' + phase)
    folder.mkdir(exist_ok=False)
    record = {'status': 'failed', 'phase': phase, 'utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Report writes work3 only; full remaining Pro/Bol work3 exposure retained; no home writes',
              'reserve_bytes': RESERVE, 'file_reserve': FILE_RESERVE}
    try:
        raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, check=True, timeout=60).stdout
        quota = parse_work3(raw)
        observed = inode_observation(os.statvfs(ROOT))
        inode = inode_headroom(quota, observed)
        current = {a: tree_bytes(p) for a, p in {'protenix': ROOT03/'run-protenix', 'boltz2': ROOT04/'run-boltz2'}.items()}
        charged = sum(tree_bytes(p) for p in (SCIENCE_ROOT, ROOT03, ROOT04, ROOT, SOURCE.parent,
                      Path('/work3/s234818/mosaic')/(TAG+'.tar.gz')))
        budget=report_budget(current,quota['available_bytes_lower'],shutil.disk_usage(ROOT).free,charged,phase)
        available,required=budget['available_bytes_conservative'],budget['required_bytes']
        record.update(raw_quota=raw,quota=quota,inode=inode,filesystem_inode_observation=observed,**budget)
        record['status'] = 'passed' if available >= required and inode['available_files_lower'] >= FILE_RESERVE else 'failed'
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        exclusive(folder/'gate.json', record)
    require(record['status'] == 'passed', 'Report/native remaining byte/inode budget plus reserve does not fit')
    return record


def cpu_tests(parent):
    run = parent/'payload'
    run.mkdir(exist_ok=False)
    check_storage(parent, 'cpu')
    tmp = Path(os.environ['TMPDIR'])/'fixed-report-tests'
    command = [PYTHON, '-m', 'pytest', '--rootdir', str(SOURCE.parents[1]), *['tests/'+n for n in TESTS],
               *[str(_layout_path(SCIENCE/'tests'/n)) for n in LEGACY_TESTS], '-o', 'addopts=',
               '-o', 'cache_dir='+str(run/'cache'), '--basetemp', str(tmp),
               '--junitxml', str(run/'report-tests.xml')]
    exclusive(run/'test-invocation.json', {'args': command, 'cwd': str(SOURCE)})
    with (run/'report-tests.log').open('x') as log:
        subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)


def xml_cases(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    require(cases and not any(list(root.iter(t)) for t in ('failure','error','skipped')), 'Report fixture failures/errors/skips')
    expected = {Path(n).stem for n in TESTS + LEGACY_TESTS}
    modules = {stem for c in cases for stem in c.get('classname','').split('.') if stem in expected}
    require(modules == expected and all(expected.intersection(c.get('classname','').split('.')) for c in cases), 'Not every required report fixture module ran')
    for suite in list(root.iter('testsuite')) + list(root.iter('testsuites')):
        require(all(int(suite.get(k,'0')) == 0 for k in ('failures','errors','skipped'))
                and ('tests' not in suite.attrib or int(suite.get('tests')) == len(list(suite.iter('testcase')))),
                'Raw XML suite totals differ')
    return cases


def certify_cpu(manifest):
    from campaign25_pocket_ops import final_job
    jobs = read(ROOT/'jobs.json')
    job = jobs['cpu']
    status, queue, proofs = final_job(job, ROOT/'cpu')
    require(status == 'DONE' and queue == 'hpc', 'Report CPU final outcome failed')
    cases = xml_cases(ROOT/'cpu/payload/report-tests.xml')
    for rel in ('payload/report-tests.xml','payload/report-tests.log','payload/test-invocation.json'):
        q = ROOT/'cpu'/rel
        proofs.append({'path': str(q), 'sha256': sha(q)})
    certificate = {'status':'passed','producer_job_id':job,'job_id':job,'tests':len(cases),
                   'suite':SUITE,'source_manifest_sha256':manifest,'proofs':proofs}
    exclusive(ROOT/'cpu/payload/assessment.json', certificate)
    exclusive(ROOT/'cpu-certificate.json', certificate)
    verify_cpu(manifest)
    with (ROOT/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exclusive(ROOT/'report-dispatch-started.json', {'report_manifest':manifest,'cpu_certificate_sha256':sha(ROOT/'cpu-certificate.json')})
        collectors = {}
        for arm,pin in ARMS.items():
            collectors[arm] = submit('complete-'+arm, 'capture', manifest, arm=arm,
                                     dependency=None if arm == 'protenix' else 'ended('+pin['job']+')')['job_id']
        interim = submit('interim-0001','interim',manifest)['job_id']
        dependency = ' && '.join('done('+j+')' for j in collectors.values())
        final = submit('combined-report','final',manifest,dependency=dependency)['job_id']
        jobs.update(collectors=collectors, active_interim=interim, interim_parent='interim-0001', final=final, cpu_certificate_job=os.environ['LSB_JOBID'])
        tmp=ROOT/'jobs.json.tmp';tmp.write_text(json.dumps(jobs,indent=2)+'\n');tmp.replace(ROOT/'jobs.json')
    return jobs


def verify_cpu(manifest):
    path = ROOT/'cpu-certificate.json';c=read(path)
    require(c['status']=='passed' and c['suite']==SUITE and c['source_manifest_sha256']==manifest, 'Wrong report CPU certificate')
    for proof in c['proofs']:
        q=Path(proof['path']);require(q.resolve().is_relative_to(ROOT/'cpu') and sha(q)==proof['sha256'], 'CPU evidence changed/escaped')
    schedulers=[p for p in c['proofs'] if re.fullmatch('scheduler-final-\\d+\\.txt',Path(p['path']).name)]
    logs=[p for p in c['proofs'] if Path(p['path']).name=='stdout.'+c['producer_job_id']+'.log']
    require(len(schedulers)==len(logs)==1, 'Missing unique actual CPU/LSF proof')
    actual_scheduler(Path(schedulers[0]['path']).read_text(),c['producer_job_id'])
    actual_lsf(Path(logs[0]['path']).read_text(),c['producer_job_id'])
    require(len(xml_cases(ROOT/'cpu/payload/report-tests.xml'))==c['tests'], 'Report CPU test count differs')
    return c


def submit(name, phase, manifest, *, arm='none', dependency=None):
    require(re.fullmatch(r'(?:complete-(?:protenix|boltz2|boltz2_chemistry)|interim-\d{4}|combined-report)',name), 'Unsafe report job parent')
    parent=ROOT/name;parent.mkdir(exist_ok=False)
    command=['bsub','-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30',
             '-J','mosaic-fixed-report-'+name,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if dependency:command+=['-w',dependency]
    command+=['bash','-l',str(_layout_path(SOURCE/'scripts/hpc/jq1_fixed_report_stage.lsf')),str(SOURCE),str(ROOT),manifest,phase,arm,str(parent)]
    record={'phase':phase,'arm':arm,'args':command,'outcome':'unknown','utc':datetime.now(timezone.utc).isoformat()}
    exclusive(parent/'submission-started.json',record)
    try:
        r=subprocess.run(command,capture_output=True,text=True,timeout=60);record.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        ids=re.findall(r'Job <(\d+)>',r.stdout)
        if r.returncode==0 and len(ids)==1:record.update(outcome='submitted',job_id=ids[0])
    except Exception as error:record['error']=repr(error)
    finally:exclusive(parent/'submission.json',record)
    require(record['outcome']=='submitted','Report submission failed/unknown; inspect markers, never duplicate')
    r=subprocess.run(['bjobs','-l',record['job_id']],capture_output=True,text=True,timeout=30)
    (parent/'scheduler-at-submission.txt').write_text(r.stdout+r.stderr)
    return record


def bundle_cpu(run,manifest):
    c=verify_cpu(manifest);dest=run/'cpu-proof';dest.mkdir(exist_ok=False)
    shutil.copyfile(ROOT/'cpu-certificate.json',dest/'assessment.json')
    links=[]
    for proof in c['proofs']:
        q=Path(proof['path']);rel=q.relative_to(ROOT/'cpu');target=dest/'evidence'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,target)
        links.append({'original_path':str(q),'local_path':target.relative_to(run).as_posix(),'sha256':proof['sha256']})
    # Convenient names retain the exact raw byte copies, while evidence subpaths bind every original proof.
    for name in ('report-tests.xml','report-tests.log'):shutil.copyfile(ROOT/'cpu/payload'/name,dest/name)
    exclusive(dest/'bundled-proofs.json',links)
    shutil.copyfile(SOURCE/'source_manifest.json',run/'report-source-manifest.json')


def produce(parent,phase,manifest,arm):
    verify_cpu(manifest)
    check_storage(parent,phase)
    run=parent/'payload'
    if phase=='capture':
        subprocess.run([PYTHON,str(_layout_path(SOURCE/'scripts/capture_jq1_fixed_completion.py')),'--arm',arm,'--output',str(run),'--completions',str(ROOT),'--report-manifest',manifest,'--cpu-certificate-sha256',sha(ROOT/'cpu-certificate.json')],check=True)
        return
    run.mkdir(exist_ok=False)
    if phase=='final':
        for a,pin in ARMS.items():
            q=read(ROOT/('complete-'+a)/'payload/assessment.json')
            require(q['status']=='passed' and q['native_job']==pin['job'] and q.get('all_probability_frames_verified') is True and q.get('report_source_manifest_sha256')==manifest and q.get('cpu_certificate_sha256')==sha(ROOT/'cpu-certificate.json'),'Final native/full-frame/source/CPU completion missing')
        command=[PYTHON,str(_layout_path(SOURCE/'scripts/render_jq1_fixed_report.py')),'--output',str(run/'report'),'--completions',str(ROOT)]
    else:
        command=[PYTHON,str(_layout_path(SOURCE/'scripts/render_jq1_fixed_interim.py')),'--output',str(run/'report'),'--proofs',str(run/'native-prefix-proofs'),'--completions',str(ROOT)]
    with (run/'render.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=read(run/'report/report_receipt.json')
    require(receipt['input_jobs']==[p['job'] for p in ARMS.values()] and receipt['n_atoms']==25 and receipt['complete']==(phase=='final'), 'Report output scope changed')
    require(all((n==0 and step is None) or n==step+1 for n,step in zip(receipt['states'],receipt['included_updates'])), 'Saved update/frame correspondence differs')
    if phase=='final':require(receipt['states']==[10001]*3 and receipt['included_updates']==[10000]*3,'Full fixed report incomplete')
    bundle_cpu(run,manifest)
    for a in ARMS:
        q=ROOT/('complete-'+a)/'payload/assessment.json'
        if q.exists():
            dest=run/('completion-'+a);dest.mkdir();shutil.copyfile(q,dest/'assessment.json')
            for name in ('snapshots.json','native-stdout.txt'):
                original=q.parent/name
                if original.exists():shutil.copyfile(original,dest/name)
    exclusive(run/'assessment.json',{'status':'passed','job_id':os.environ['LSB_JOBID'],'complete':receipt['complete'],
        'report_source_manifest_sha256':manifest,'native_jobs':receipt['input_jobs'],'input_jobs':receipt['input_jobs'],
        'states':receipt['states'],'included_updates':receipt['included_updates'],'n_atoms':25,
        'report_receipt_sha256':sha(run/'report/report_receipt.json'),'cpu_certificate_sha256':sha(ROOT/'cpu-certificate.json'),
        'scope':'Saved-only fixed-step report; no scoring/repair/resampling/diffusion; original trajectories remain authoritative'})
    subprocess.run([PYTHON,str(_layout_path(SCIENCE/'scripts/collect_protenix_evidence.py')),'--storage','work3','--run',str(parent),'--output',str(parent)+'.tar.gz'],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('cpu','certify-cpu','capture','interim','final','preflight'),required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--manifest',required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--arm',default='none',choices=('none',*ARMS));args=parser.parse_args()
    require(args.root.resolve()==ROOT and args.source.resolve()==SOURCE,'Wrong immutable report scope')
    require(args.parent.resolve().is_relative_to(ROOT),'Unsafe report parent')
    if args.phase=='preflight':print(json.dumps(check_storage(args.parent,'preflight')));return
    require(bool(os.environ.get('LSB_JOBID')),'Report validation/rendering requires allocated HPC')
    if args.phase=='cpu':cpu_tests(args.parent)
    elif args.phase=='certify-cpu':print(json.dumps(certify_cpu(args.manifest)))
    else:produce(args.parent,args.phase,args.manifest,args.arm)


if __name__=='__main__':main()
