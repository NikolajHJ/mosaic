"""New versioned random-Protenix/chemistry fork; local stdlib AST/bytes only."""
import ast,hashlib,io,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
TAG=HERE.name
VERSION='mosaic-brd4-jq1-protenix-random-delayed-chem-v1'
PARENT_SHA='c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
def replace(s,old,new):
    assert s.count(old)==1,(old,s.count(old));return s.replace(old,new)
def main():
    arc=REPO/'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/source02/source.tar.gz'
    assert hashlib.sha256(arc.read_bytes()).hexdigest()=='4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e'
    with tarfile.open(arc) as t:files={m.name.removeprefix('source/'):t.extractfile(m).read() for m in t if m.isfile()}
    parent=files.pop('source_manifest.json');assert hashlib.sha256(parent).hexdigest()==PARENT_SHA
    meta=json.loads(parent);assert len(files)==625 and set(files)==set(meta['files'])
    for n,d in files.items():assert hashlib.sha256(d).hexdigest()==meta['files'][n]
    original=dict(files)
    n='scripts/campaign25_contract.py';s=files[n].decode()
    s=replace(s,'FIXED25_VERSIONS = (FIXED25_VERSION, FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION)',"RANDOM_CHEM_VERSION = "+repr(VERSION)+"\nFIXED25_VERSIONS = (FIXED25_VERSION, FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION)")
    s=replace(s,'    reset = version == FIXED25_LRRESET_VERSION',"    random_chem = version == RANDOM_CHEM_VERSION\n    reset = version == FIXED25_LRRESET_VERSION or random_chem\n    if random_chem and config.get('arm') != 'protenix':\n        raise ValueError('Random chemistry version is one Protenix arm only')")
    s=replace(s,'"initialization": {"kind": "chain_uniform", "mixing": 0.5},','"initialization": ({"kind": "dirichlet", "concentration": 1.0, "seed": 17} if random_chem else {"kind": "chain_uniform", "mixing": 0.5}),')
    s=replace(s,'    active = config["arm"] == "boltz2_chemistry"','    active = config["arm"] == "boltz2_chemistry" or random_chem')
    old="            if initial_match.get('status') != 'passed' or initial_match.get('job') != '29770179' or initial_match.get('sha256') != '351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf':"
    s=replace(s,old,"""            if config['version'] == RANDOM_CHEM_VERSION:
                expected_random = {'status':'passed','recipe':config['initialization'],'model_key':17,'draws_replayed_exactly':True,'scope':'Separate deterministic PCG64 recipe; native model key unchanged'}
                if summary.get('random_initialization') != expected_random or summary.get('initialization_execution', {}).get('version') != 'protenix-fp32-cpu-atom-softmax-v1':
                    raise ValueError('Fresh random initialization/replay execution not authenticated')
            elif initial_match.get('status') != 'passed' or initial_match.get('job') != '29770179' or initial_match.get('sha256') != '351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf':""")
    s=replace(s,"if config['version'] in (FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION):","if config['version'] in (FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION):")
    s=replace(s,"        if config['arm'] == 'boltz2_chemistry':","        if config['chemistry_enabled']:")
    s=replace(s,"            if config['version'] == FIXED25_LRRESET_VERSION:","            if config['version'] in (FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION):")
    s=replace(s,"    if config['arm'] != 'boltz2_chemistry':","    if not config['chemistry_enabled']:")
    # Apply full finite single-candidate bookkeeping to the new strict version too.
    s=replace(s,"    boosted = config['version'] in (FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION)","    boosted = config['version'] in (FIXED25_LRBOOST_VERSION, FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION)")
    files[n]=s.encode()
    n='examples/brd4_25_atoms.py';s=files[n].decode()
    s=replace(s,'FIXED25_VERSIONS, FIXED25_LRRESET_VERSION','FIXED25_VERSIONS, FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION')
    s=replace(s,'if arm == "protenix" and config["version"] == FIXED25_LRRESET_VERSION:','if arm == "protenix" and config["version"] in (FIXED25_LRRESET_VERSION, RANDOM_CHEM_VERSION):')
    s=replace(s,"if arm == 'protenix' and config[\"version\"] in (PROTENIX_LR03_VERSION, *FIXED25_VERSIONS):","if arm == 'protenix' and config[\"version\"] in (PROTENIX_LR03_VERSION, *FIXED25_VERSIONS) and config['version'] != RANDOM_CHEM_VERSION:")
    marker='        report["initialization_sha256"] = sha(args.output / "initialization.npz")'
    s=replace(s,marker,marker+'''
        if config['version'] == RANDOM_CHEM_VERSION:
            replay = logit_type.from_graph(initialize_graph(hard, config['initialization']))
            for left, right in ((initial.atoms,replay.atoms),(initial.bonds,replay.bonds),(initial.graph().X,replay.graph().X),(initial.graph().E_upper,replay.graph().E_upper)):
                np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
            report['random_initialization'] = {'status':'passed','recipe':config['initialization'],'model_key':17,'draws_replayed_exactly':True,'scope':'Separate deterministic PCG64 recipe; native model key unchanged'}
''')
    s=replace(s,'            if args.mode == "validate":','''            if args.mode == "validate" and config['version'] == RANDOM_CHEM_VERSION:
                old_ready = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04/validate-protenix/payload/study/assessment.json')
                if sha(old_ready) != '9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd':
                    raise ValueError('Original frozen context assessment bytes changed')
                old_context = json.loads(old_ready.read_text())
                context = load_context(old_ready.parent / 'context', old_context['context_files'])
                report['context_adoption'] = {'assessment_sha256':sha(old_ready),'scope':'Only frozen context input reused; all random/chemistry native checks fresh'}
            elif args.mode == "validate":''')
    s=replace(s,'                x = graph.X.at[1, 0].set(0.5 / 6).at[1, 4].set(0.5 + 0.5 / 6)','''                if config['version'] == RANDOM_CHEM_VERSION:
                    # The existing invalid assay row has a declared unit sum;
                    # random source rows cannot inherit the chain-only two-cell edit.
                    x = graph.X.at[1].set(jnp.full((6,), 0.5 / 6)).at[1, 4].set(0.5 + 0.5 / 6)
                else:
                    x = graph.X.at[1, 0].set(0.5 / 6).at[1, 4].set(0.5 + 0.5 / 6)''')
    s=replace(s,'                probe = GraphLogits.from_graph(RelaxedGraph(x, graph.E_upper))',"                probe_type = type(initial) if config['version'] == RANDOM_CHEM_VERSION else GraphLogits\n                probe = probe_type.from_graph(RelaxedGraph(x, graph.E_upper))")
    files[n]=s.encode()
    # Base 728-case CPU wrapper: unchanged original suites plus this version's contracts.
    s=original['scripts/hpc/fixed25_lrreset_cpu.lsf'].decode()
    a=s.index('"$python" - "$root" <<\'PY\'');b=s.index('\nmkdir "$root/cpu/payload"',a)
    s=s[:a]+'''"$python" "$source/scripts/random_chem_ops.py" --phase cpu-entry --root "$root" --source "$source" --manifest "$manifest"
'''+s[b:]
    s=s.replace('tests/test_protenix_initialization.py)','tests/test_protenix_initialization.py tests/test_random_chem.py)')
    files['scripts/hpc/random_chem_cpu.lsf']=s.encode()
    for local,target in [('config.json','examples/configs/protenix_random_delayed_chem_10000.json'),('random_chem_ops.py','scripts/random_chem_ops.py'),('random_chem_stage.lsf','scripts/hpc/random_chem_stage.lsf'),('fixture_random_chem.py','tests/test_random_chem.py'),('README.md','docs/protenix_random_chem.md')]:files[target]=(HERE/local).read_bytes().replace(b'\r\n',b'\n')
    assert [n for n in original if n.startswith('src/') and original[n]!=files[n]]==[]
    for n,d in files.items():
        if n.endswith('.py'):ast.parse(d,filename=n)
    changed={n:hashlib.sha256(d).hexdigest() for n,d in files.items() if original.get(n)!=d}
    for n in changed:(HERE/('overlay-'+n.replace('/','__'))).write_bytes(files[n])
    metadata={'files':{n:hashlib.sha256(d).hexdigest() for n,d in sorted(files.items())},'parent_manifest_sha256':PARENT_SHA,'scope':'One fresh random-initialized Protenix with delayed chemistry; old run/source bytes unchanged; all145 scientific library files exact base625'}
    raw=(json.dumps(metadata,sort_keys=True,indent=2)+'\n').encode();out=REPO/'results/hpc'/TAG/'source01';out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for n,d in {**files,'source_manifest.json':raw}.items():
            m=tarfile.TarInfo('source/'+n);m.mode=0o444;m.size=len(d);tar.addfile(m,io.BytesIO(d))
    package={'tag':TAG,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'source_files':len(files),'archive_bytes':(out/'source.tar.gz').stat().st_size,'archive_sha256':hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest(),'parent_manifest':PARENT_SHA,'changed_members':changed}
    (out/'source_manifest.json').write_bytes(raw)
    for p in [out/'package.json',HERE/'package.json']:p.write_text(json.dumps(package,indent=2)+'\n')
    print(json.dumps(package,indent=2))
if __name__=='__main__':main()
