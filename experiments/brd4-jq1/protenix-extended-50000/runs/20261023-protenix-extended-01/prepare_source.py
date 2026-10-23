"""New immutable extended fork of631; local stdlib AST/byte packaging only."""
import ast, hashlib, io, json, tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
def replace(s,old,new):
    assert s.count(old)==1,(old,s.count(old));return s.replace(old,new)
def main():
    path=REPO/'results/hpc/20261023-protenix-random-chem-01/source01/source.tar.gz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='c005429dc2581c929662698e80da154fabc0f3365615facfa7582641fa7fcddc'
    with tarfile.open(path) as tar:files={m.name.removeprefix('source/'):tar.extractfile(m).read() for m in tar if m.isfile()}
    raw=files.pop('source_manifest.json');parent=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()=='bb3ac9cae836b59f9a62fb33d4775c0b4388f3b200a7422ce8a620d2d405950f'
    assert len(files)==631
    for n,d in files.items():assert hashlib.sha256(d).hexdigest()==parent['files'][n]
    original=dict(files)
    s=files['scripts/campaign25_contract.py'].decode()
    s=replace(s,'def config25(config):','def legacy_config25(config):')
    marker='\ndef update_parameters(config):'
    s=replace(s,marker,"""
from extended_protocol import VERSIONS as EXTENDED_VERSIONS, validate_config, require_boundary
FIXED25_VERSIONS = (*FIXED25_VERSIONS, *EXTENDED_VERSIONS)
def config25(config):
    if config.get('version') in EXTENDED_VERSIONS:
        return validate_config(config, legacy_config25)
    return legacy_config25(config)

def update_parameters(config):""")
    s=replace(s,"    boosted = config['version'] in (FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION)","    boosted = config['version'] in (FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION, *EXTENDED_VERSIONS)")
    s=replace(s,"            if config['version'] == RANDOM_CHEM_VERSION:","            if config['version'] in EXTENDED_VERSIONS:\n                require_boundary(summary, assessment)\n            elif config['version'] == RANDOM_CHEM_VERSION:")
    s=s.replace('(FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION):','(FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION, *EXTENDED_VERSIONS):')
    s=s.replace('(FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION):','(FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION, *EXTENDED_VERSIONS):')
    files['scripts/campaign25_contract.py']=s.encode()
    s=files['examples/brd4_25_atoms.py'].decode()
    marker='    args = parser.parse_args()'
    s=replace(s,marker,"""    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--segment-steps', type=int, default=10000)
    parser.add_argument('--resume-snapshot', type=Path)
    parser.add_argument('--resume-sha256')
    args = parser.parse_args()""")
    marker='    config = config25(json.loads(args.config.read_text()))'
    s=replace(s,marker,marker+"""
    from extended_protocol import VERSIONS as EXTENDED_VERSIONS, RANDOM as EXTENDED_RANDOM, segment_scope, restore_logits
    extended = config['version'] in EXTENDED_VERSIONS
    execution = segment_scope(config, args.offset, args.segment_steps, args.resume_snapshot, args.resume_sha256) if extended else None
    if not extended and (args.offset or args.segment_steps != 10000 or args.resume_snapshot or args.resume_sha256):
        raise ValueError('Historical version cannot acquire continuation fields')""")
    s=replace(s,"if args.mode == \"validate\" and config['version'] == RANDOM_CHEM_VERSION:","if args.mode == \"validate\" and (config['version'] == RANDOM_CHEM_VERSION or extended):")
    # The first replacement above affects only the context branch in631.
    s=s.replace('config["version"] in (FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION):','config["version"] in (FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION, *EXTENDED_VERSIONS):')
    s=replace(s,'        initial = logit_type.from_graph(initialize_graph(hard, config["initialization"]))',"""        if extended and args.resume_snapshot:
            initial = restore_logits(args.resume_snapshot, args.resume_sha256, logit_type, key, sha)
        else:
            initial = logit_type.from_graph(initialize_graph(hard, config['initialization']))
        if extended:
            report.update(extended_execution=execution, exact_boundary_replay='passed')""")
    s=replace(s,"\n        if config['version'] == RANDOM_CHEM_VERSION:","\n        if config['version'] == RANDOM_CHEM_VERSION or (extended and config['version'] == EXTENDED_RANDOM and args.offset == 0):")
    s=replace(s,"and config['version'] != RANDOM_CHEM_VERSION:","and config['version'] != RANDOM_CHEM_VERSION and not extended:")
    s=replace(s,"                if config['version'] == RANDOM_CHEM_VERSION:","                if config['version'] == RANDOM_CHEM_VERSION or extended:")
    s=replace(s,"probe_type = type(initial) if config['version'] == RANDOM_CHEM_VERSION else GraphLogits","probe_type = type(initial) if config['version'] == RANDOM_CHEM_VERSION or extended else GraphLogits")
    s=replace(s,'steps=10000, config=options, schedule=schedule, **common)',"steps=args.segment_steps if extended else 10000, config=options, schedule=schedule, update_offset=args.offset if extended else None, **common)")
    marker='            report["cases"] = [{"label": config["label"], "initialization": config["initialization"], **result}]'
    s=replace(s,marker,marker+"""
            if extended and (result['global_range'] != {'start':args.offset,'end':args.offset+args.segment_steps}
                    or [v['global_step'] for v in result['states']] != list(range(args.offset,args.offset+args.segment_steps+1))):
                raise ValueError('Global contiguous segment scope differs')""")
    marker='        assessment.update(arm=arm, n_atoms=25, proofs=proofs)'
    s=replace(s,marker,marker+"\n        if extended: assessment['extended_execution'] = execution")
    files['examples/brd4_25_atoms.py']=s.encode()
    s=files['src/mosaic/small_molecule/chemistry/trajectory.py'].decode()
    s=replace(s,'    acceptance_policy="best_primal",\n):','    acceptance_policy="best_primal",\n    update_offset=None,\n):')
    s=replace(s,'    if schedule is not None and steps > schedule.total_updates:',"    if update_offset is not None and (type(update_offset) is not int or update_offset < 0):\n        raise ValueError('Invalid explicit global update offset')\n    offset = 0 if update_offset is None else update_offset\n    if schedule is not None and offset + steps > schedule.total_updates:")
    s=replace(s,'else schedule.for_state(state),','else schedule.for_state(offset + state),')
    s=replace(s,'        prepared = prepare(initial, 0)','        prepared = prepare(initial, offset)')
    s=replace(s,'            prepared = prepare(candidate.logits, step + 1)','            prepared = prepare(candidate.logits, offset + step + 1)')
    s=replace(s,'    evaluation = None\n    try:',"    if update_offset is not None:\n        report['global_range'] = {'start':offset,'end':offset+steps}\n    evaluation = None\n    try:")
    s=replace(s,'\n        _append_record(directory / "states.jsonl", report["states"][-1])',"\n        if update_offset is not None: report['states'][-1]['global_step'] = offset\n        _append_record(directory / \"states.jsonl\", report[\"states\"][-1])")
    s=replace(s,'            _append_record(directory / "states.jsonl", report["states"][-1])',"            if update_offset is not None:\n                report['states'][-1]['global_step'] = offset + step + 1\n                record['global_step'] = offset + step\n            _append_record(directory / \"states.jsonl\", report[\"states\"][-1])")
    files['src/mosaic/small_molecule/chemistry/trajectory.py']=s.encode()
    for arm in ['random','continuation']:
        cfg=json.loads(original['examples/configs/protenix_random_delayed_chem_10000.json'] if arm=='random' else original['examples/configs/brd4_jq1_25_fixed_lrreset_protenix_10000.json'])
        cfg.update(version='mosaic-brd4-jq1-protenix-random-chem-50000-v1' if arm=='random' else 'mosaic-brd4-jq1-protenix-10000-plus40000-v1',steps=50000,extension={'version':'protenix-serial50k-v1','start_global_update':0 if arm=='random' else 10000,'target_global_update':50000,'segment_updates':10000,'parent_job':None if arm=='random' else '29792670'})
        if arm=='random':cfg['repair_schedule']['total_updates']=50000
        cfg['label']='Protenix LR0.3 / '+('random Dirichlet1 seed17 / chemistry.001through5000 ->100at50000 / fresh50000' if arm=='random' else 'exact completed29792670 state10000 / native only / +40000 to50000')
        data=(json.dumps(cfg,indent=2)+'\n').encode();(HERE/(arm+'.json')).write_bytes(data);files['examples/configs/extended_'+arm+'.json']=data
    for local,target in [('extended_protocol.py','scripts/extended_protocol.py'),('extended_ops.py','scripts/extended_ops.py'),('extended_stage.lsf','scripts/hpc/extended_stage.lsf'),('fixture_extended.py','tests/test_extended_protenix.py'),('parent.json','docs/extended_parent.json'),('predecessor-retirement.json','docs/extended_retirement.json'),('README.md','docs/protenix_extended.md')]:files[target]=(HERE/local).read_bytes().replace(b'\r\n',b'\n')
    s=original['scripts/hpc/random_chem_cpu.lsf'].decode().replace('scripts/random_chem_ops.py','scripts/extended_ops.py').replace('tests/test_random_chem.py)','tests/test_random_chem.py tests/test_extended_protenix.py)')
    files['scripts/hpc/extended_cpu.lsf']=s.encode()
    for n,d in files.items():
        if n.endswith('.py'):ast.parse(d,filename=n)
    assert [n for n in original if n.startswith('src/') and original[n]!=files[n]]==['src/mosaic/small_molecule/chemistry/trajectory.py']
    changed={n:hashlib.sha256(d).hexdigest() for n,d in files.items() if original.get(n)!=d}
    for n in changed:(HERE/('overlay-'+n.replace('/','__'))).write_bytes(files[n])
    raw=(json.dumps({'files':{n:hashlib.sha256(d).hexdigest() for n,d in sorted(files.items())},'parent_manifest_sha256':hashlib.sha256(raw).hexdigest(),'scope':'Authorized serial50k exact-logit continuation; original releases/results unchanged; optional global offset preserves historical defaults'},sort_keys=True,indent=2)+'\n').encode()
    out=REPO/'results/hpc'/HERE.name/'source02';out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for n,d in {**files,'source_manifest.json':raw}.items():
            m=tarfile.TarInfo('source/'+n);m.mode=0o444;m.size=len(d);tar.addfile(m,io.BytesIO(d))
    package={'tag':HERE.name,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'source_files':len(files),'archive_bytes':(out/'source.tar.gz').stat().st_size,'archive_sha256':hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest(),'changed_members':changed}
    (out/'source_manifest.json').write_bytes(raw);(HERE/'package.json').write_bytes((json.dumps(package,indent=2)+'\n').encode())
    print(json.dumps(package,indent=2))
if __name__=='__main__':main()
