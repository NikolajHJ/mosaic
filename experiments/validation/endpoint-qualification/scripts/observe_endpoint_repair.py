"""Single read-only observation of repaired endpoint jobs; no evaluation or mutation."""
from pathlib import Path
import argparse,ast,base64,hashlib,json,re,shlex,subprocess
REPO=Path(__file__).resolve().parents[4]
PROGRAM="\nfrom datetime import datetime,timezone\nfrom pathlib import Path\nimport base64,hashlib,json,re,stat,subprocess\nbase=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')\nspecs={\n 'continuations':('nonchem-endpoint-continuation-06','nonchem-endpoint-release-06/source','1519d5761069f73fb6659bc1740cb04c155a04bc3218f143a3f28a73e65a1f69'),\n 'pro_precision':('protenix-endpoint-fix-03','protenix-endpoint-fix-release-03/source','a701b12bd1e69f5aa7169ad5339f2456c3d728b855257b4270437d37f9da68f3')}\nrecord={'captured_utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'roots':{},'files':[]}\ntotal=0;ids=set();seen=set()\ndef collect(p):\n global total\n if not p.exists() or str(p) in seen:return\n st=p.lstat();assert stat.S_ISREG(st.st_mode) and not p.is_symlink()\n seen.add(str(p))\n if st.st_size>2*1024**2:\n  record.setdefault('omitted',[]).append({'path':str(p),'bytes':st.st_size,'reason':'large scientific/diagnostic record stays remote'});return\n raw=p.read_bytes();assert len(raw)==st.st_size\n total+=len(raw);assert total<32*1024**2\n record['files'].append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()})\nfor label,(name,source,pin) in specs.items():\n root=base/name;src=base/source\n raw=(src/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==pin\n manifest=json.loads(raw)\n for rel,digest in manifest['files'].items():\n  p=src/rel;assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==digest\n collect(src/'source_manifest.json');collect(src/'plan.json')\n jobs=json.loads((root/'jobs.json').read_bytes());ids.update(jobs.values())\n row={'path':str(root),'source_manifest_sha256':pin,'source_files_rehashed':len(manifest['files']),'jobs':jobs,'parents':{}}\n for p in root.glob('*.json'):collect(p)\n for name,job in jobs.items():\n  assert re.fullmatch('[a-zA-Z0-9_-]+',name) and re.fullmatch('[0-9]+',str(job))\n  parent=root/name\n  for p in parent.glob('*.json'):collect(p)\n  for p in parent.glob('*.log'):collect(p)\n  payload=parent/'payload'\n  if payload.exists():\n   for p in payload.glob('*.json'):collect(p)\n   for p in payload.glob('*.xml'):collect(p)\n   for p in payload.glob('*.log'):collect(p)\n   for subdir in ('production','reference'):\n    for p in (payload/subdir).glob('*.json'):collect(p)\n   collect(payload/'study/assessment.json')\n  study=payload/'study';details={'study_exists':study.is_dir()}\n  if study.is_dir():\n   details['entries']=[p.name for p in study.iterdir()]\n   details['jsonl']={}\n   for p in study.glob('*.jsonl'):\n    assert not p.is_symlink() and p.stat().st_size<128*1024**2\n    lines=0;last=None;partial=False\n    with p.open('rb') as stream:\n     for rawline in stream:\n      if not rawline.endswith(b'\\n'):partial=True;break\n      v=json.loads(rawline);lines+=1\n      last={k:v[k] for k in ('step','global_step','index','update_index','state_index','global_update') if k in v}\n    details['jsonl'][p.name]={'newline_committed_rows':lines,'last_index_metadata':last,'partial_tail':partial}\n  row['parents'][name]=details\n record['roots'][label]=row\nr=subprocess.run(['bjobs','-l',*sorted(ids,key=int)],capture_output=True,text=True,timeout=45)\nrecord['scheduler']={'job_ids':sorted(ids,key=int),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}\nrecord['compact_bytes']=total\nprint(json.dumps(record))\n"

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    parser.add_argument('--repair04',action='store_true',help='Observe only the two exact new repair capsules')
    parser.add_argument('--precision-attempt',choices=('04','05'),default='04')
    args=parser.parse_args()
    assert args.repair04 or args.precision_attempt=='04', 'Precision05 requires the explicit current-repair scope'
    assert re.fullmatch('[a-z0-9-]{1,70}',args.name)
    out=REPO/('results/hpc/20261026-endpoint-repair-04' if args.repair04 else 'results/hpc/20261026-endpoint-repair-03')/args.name
    out.parent.mkdir(parents=True,exist_ok=True)
    out.mkdir(exist_ok=False)
    # Follow the CPU certificate's exact metadata bindings as well as the
    # top-level summaries. This includes the actual generated CLI/worker
    # startup receipts; never fetch native arrays or walk arbitrary trees.
    program=PROGRAM.replace("r=subprocess.run(['bjobs'", """for label, row in record['roots'].items():
 root=Path(row['path']); certificate=root/'cpu-certificate.json'
 if certificate.is_file():
  for proof in json.loads(certificate.read_bytes()).get('proofs',[]):
   p=Path(proof['path'])
   assert p.is_relative_to(root) and p.suffix in ('.json','.xml','.log')
   collect(p)
r=subprocess.run(['bjobs'""")
    if args.repair04:
        start=program.index('specs={');end=program.index("\nrecord=",start)
        specs={'pro_precision':('protenix-endpoint-fix-04','protenix-endpoint-fix-release-04/source','af2393345d9f284fb395eacf809cdf56ccfcdfdc6938dd7f4747feb728db310f'),
               'chemistry':('chemistry-production-04','chemistry-production-release-04/source','01e2de2a3413bd903590dd331c5cedf6b756e856a83df2b73ad65e02b9f9fceb')}
        if args.precision_attempt=='05':
            specs['pro_precision']=('protenix-endpoint-fix-05','protenix-endpoint-fix-release-05/source','65b4bd350386a8550c3b588eb7de2b0f6084b32ad790a907851c632dba0f023e')
        program=program[:start]+'specs='+repr(specs)+program[end:]
        program=program.replace("assert re.fullmatch('[a-zA-Z0-9_-]+',name)",
            "assert not Path(name).is_absolute() and '..' not in Path(name).parts and re.fullmatch('[a-zA-Z0-9_/-]+',name)")
        program=program.replace("for p in root.glob('*.json'):collect(p)",
            "for p in root.glob('*.json'):collect(p)\n for p in root.glob('*/adoption.json'):collect(p)")
        program=program.replace('parent=root/name',"parent=root/(name.removeprefix('control/') if label=='chemistry' else name)")
        program=program.replace("for p in study.glob('*.jsonl'):","for p in study.rglob('*.jsonl'):")
        program=program.replace("details['jsonl'][p.name]", "details['jsonl'][p.relative_to(study).as_posix()]")
        program=program.replace("study=payload/'study';details={'study_exists':study.is_dir()}",
            "study=payload/'study';details={'study_exists':study.is_dir()}\n  for rel in ('native_gradient_checks/report.json','case-0/combined_initial_checks/report.json','case-0/initial_checks/report.json'):collect(study/rel)")
        program=program.replace("jobs=json.loads((root/'jobs.json').read_bytes());ids.update(jobs.values())",
            "jobs=json.loads((root/'jobs.json').read_bytes()) if (root/'jobs.json').is_file() else {};ids.update(jobs.values())")
        program=program.replace("for p in root.glob('*/adoption.json'):collect(p)",
            "for p in root.glob('*/adoption.json'):collect(p)\n row['jobs_file_exists']=(root/'jobs.json').is_file();row['root_entries']=sorted(p.name for p in root.iterdir())\n for p in (root/'preflight').rglob('*.json'):collect(p)\n for p in root.glob('*/current-chemistry-capacity-*/*json'):collect(p)")
    ast.parse(program)
    (out/'program.py').write_text(program,encoding='utf-8',newline='\n')
    cmd=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
         'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (out/'raw.json').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
        r=subprocess.run(cmd,input=program.encode(),stdout=stdout,stderr=stderr,timeout=100)
    receipt={'returncode':r.returncode,'program_sha256':hashlib.sha256(program.encode()).hexdigest(),
             'raw_sha256':hashlib.sha256((out/'raw.json').read_bytes()).hexdigest(),'read_only':True}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    if r.returncode:raise SystemExit('Single read-only connection failed; no retry')
    packet=json.loads((out/'raw.json').read_bytes())
    records=[]
    for row in packet['files']:
        raw=base64.b64decode(row['base64'],validate=True)
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        records.append({k:v for k,v in row.items() if k!='base64'})
    (out/'byte-verification.json').write_text(json.dumps({'status':'all_compact_metadata_bytes_verified','files':records},indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'captured_utc':packet['captured_utc'],'files':len(records),
        'jobs':{k:v['jobs'] for k,v in packet['roots'].items()},'scheduler_returncode':packet['scheduler']['returncode']},indent=2))

if __name__=='__main__':main()
