"""Stdlib source fork/AST/byte packaging only; never execute numerical fixtures locally."""
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
TAG='20261023-brd4-jq1-lrreset-analysis-01'
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
REPORTROOT=BASE+'/lrreset-analysis-01'
REPORTSOURCE=BASE+'/lrreset-analysis-release-01/source'


def replace_function(source,name,value):
    tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+value.strip()+'\n'+''.join(lines[node.end_lineno:])


def assign(source,name,value):
    tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets))
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+name+' = '+value+'\n'+''.join(lines[node.end_lineno:])


def main():
    out=REPO/'results/hpc'/TAG/'source02';out.mkdir(parents=True,exist_ok=False)
    original=REPO/'results/hpc/20261022-brd4-jq1-lrboost-analysis-02/source02/source.tar.gz'
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='799869cbce1ea09b6862c30b3d3c39a258872d0948de7981cbf93790f418ab13'
    with tarfile.open(original) as tar:
        old={m.name.removeprefix('source/'):tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    parent=old.pop('source_manifest.json');assert hashlib.sha256(parent).hexdigest()=='6d547929ce3227302729b022628cc26612d9829e72c0a70861870023463dac3a'
    meta=json.loads(parent);assert set(old)==set(meta['files'])
    for name,data in old.items():assert hashlib.sha256(data).hexdigest()==meta['files'][name]
    pins=json.loads((HERE/'input-pins.json').read_bytes());assert not pins[REPORTROOT] and not pins[BASE+'/lrreset-analysis-release-01']
    sources={n:d.decode() for n,d in old.items()}
    c=sources['scripts/jq1_lrboost_report_contract.py']
    old_arms=ast.literal_eval(next(n.value for n in ast.parse(c).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='ARMS' for x in n.targets)))
    arms={}
    for arm,pin in pins['arms'].items():
        a={k:old_arms[arm][k] for k in ['metric','color','label']}
        a.update(job=pin['job'],rate=pin['rate'],manifest=pins['scientific_source']['sha256'],
            config=pin['config']['sha256'],ready=pin['ready']['sha256'],native_job={'protenix':'29792413','boltz2':'29792415','boltz2_chemistry':'29792417'}[arm],
            model=pin.get('model',old_arms[arm]['model']),context=pin.get('context',old_arms[arm]['context']),
            source=BASE+'/restart-lrreset-release-04/source',science_root=BASE+'/restart-lrreset-04',
            output_parent=pin['output_parent'],submission=pin['submission']['sha256'],decision=pin['decision']['sha256'],projected_bytes=pin['projections'])
        for key in ['driver','entry','raw_entry','coupled_entry']:
            a[key]=pin[key]['sha256'] if key in pin else None
        if arm=='boltz2':a['failed_terminal']={'applicability':'jq1-lrreset-failed-terminal-presentation-v1',**{key:pin[key]['sha256'] for key in ['summary','case','terminal','stdout']},'error':pin['error']}
        arms[arm]=a
    for name,value in {'VERSION':repr('jq1-25-lrreset-saved-report-v1'),'FAILED_APPLICABILITY':repr('jq1-lrreset-failed-terminal-presentation-v1'),
        'TAG':repr('20261022-brd4-jq1-25-fixed-lrreset-04'),'SCIENCE':"Path("+repr(BASE+'/restart-lrreset-release-04/source')+")",'ROOT':"Path("+repr(BASE+'/restart-lrreset-04')+")",'HOME_ROOT':"Path("+repr(BASE+'/restart-lrreset-04')+")",'SCIENCE_MANIFEST':repr(pins['scientific_source']['sha256']),'CPU_SHA':repr(pins['cpu']['sha256']),'DISPATCH_SHA':repr(pins['dispatch']['sha256']),'JOBS_SHA':repr(pins['submissions']['sha256']),'ARMS':repr(arms)}.items():c=assign(c,name,value)
    c=c.replace("rate in (3.0, 0.05)","rate in (0.3, 0.005)").replace('fixed_lrboost_','fixed_lrreset_').replace('50.0','100.0').replace('reach 50','reach 100')
    c+='\nLAUNCH_SOURCE = Path('+repr(BASE+'/restart-lrreset-allwork3-launch-02/source')+')\nLAUNCH_MANIFEST = '+repr(pins['launcher_source']['sha256'])+'\nLAUNCH_CPU_SHA = '+repr(pins['launcher_cpu']['sha256'])+'\n'
    replacement=(HERE/'input_authentication.py').read_text();tree=ast.parse(replacement)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):
            text='\n'.join(replacement.splitlines()[node.lineno-1:node.end_lineno])
            if node.name=='_launcher':c+='\n'+text+'\n'
            else:c=replace_function(c,node.name,text)
    c=c.replace("    raw_path, coupled_path = study /", "    require(pin.get('raw_entry') and pin.get('coupled_entry'), 'A newly started arm needs a fresh immutable entry pin')\n    raw_path, coupled_path = study /")
    c=c.replace("        initial = summary.get('baseline_initialization_match', {})", "        require(summary.get('initialization_execution', {}).get('version') == 'protenix-fp32-cpu-atom-softmax-v1', 'Exact declared FP32 placement missing')\n        initial = summary.get('baseline_initialization_match', {})")
    sources['scripts/jq1_lrboost_report_contract.py']=c
    for name in list(sources):
        if name=='scripts/jq1_lrboost_report_contract.py':continue
        s=sources[name]
        s=s.replace('jq1-25-lrboost-saved-report-v1','jq1-25-lrreset-saved-report-v1').replace('jq1-lrboost-saved-report-contract-v1','jq1-lrreset-saved-report-contract-v1')
        s=s.replace('jq1-lrboost-endpoint-diagnostics-v1','jq1-lrreset-endpoint-diagnostics-v1').replace('jq1-lrboost-endpoint-diagnostics-v2','jq1-lrreset-endpoint-diagnostics-v1')
        if name.endswith('render_jq1_lrboost_progress.py'):s=s.replace('50.0','100.0')
        if name.startswith('tests/'):
            s=s.replace('rate=0.05','rate=0.005').replace('rate=3.0','rate=0.3').replace('[3.0, 0.05]','[0.3, 0.005]').replace('[3.0, 0.05, 0.05]','[0.3, 0.005, 0.005]').replace('], 3.0)','], 0.3)')
            s=s.replace('29779177','29792670').replace('29779178','29792671').replace('29779179','29792672').replace('fixed_lrboost_','fixed_lrreset_')
            s=s.replace("if damage == 'old_rate': value['accepted_step_size'] = 0.3","if damage == 'old_rate': value['accepted_step_size'] = 3.0")
            s=s.replace("old['failed_terminal']['error']","old.get('failed_terminal', {'error': 'Synthetic terminal derivative failure'})['error']")
            s=s.replace('50.0 if i == 10000','100.0 if i == 10000').replace("'learning_rate': 3.0","'learning_rate': 0.3")
            if name.endswith('test_jq1_lrboost_ops.py'):
                s=s.replace("current = {'protenix': 100, 'boltz2': 200}","current = {'protenix': 100, 'boltz2': 200, 'boltz2_chemistry': 300}").replace('sum(ops.FULL_NATIVE.values()) - 300','sum(ops.FULL_NATIVE.values()) - 600')
                s=s.replace("allowance = 1024**3 if phase in ('preflight', 'cpu') else (2 * 1024**3 + 512 * 1024**2)","allowance = 2 * 1024**3 + 512 * 1024**2")
                s=s.replace("== [('protenix', 'diagnostic'), ('boltz2', 'diagnostic')]","== [('boltz2', 'diagnostic')]")
                s=s.replace("    result = ops.certify_cpu('a' * 64, only_protenix=True)", "    result = ops.certify_cpu('a' * 64)")
                s=s.replace("    assert [x[:2] for x in submitted] == [('diagnostic-protenix', 'diagnostic'), ('diagnostic-capture-protenix', 'diagnostic-capture')]\n    assert submitted[1][2] == {'arm': 'protenix', 'dependency': 'ended(901)'}\n    assert result['no_original_study_submitted'] is True\n    assert result['diagnostics'] == {'protenix': '901'}", "    assert [x[:2] for x in submitted] == [('diagnostic-boltz2','diagnostic'),('diagnostic-capture-boltz2','diagnostic-capture'),('complete-protenix','capture'),('complete-boltz2','capture'),('complete-boltz2_chemistry','capture'),('interim-0001','interim'),('interim-0002','interim'),('combined-report','final')]\n    assert submitted[1][2] == {'arm':'boltz2','dependency':'ended(901)'}\n    assert result['diagnostics'] == {'boltz2':'901'}\n    assert submitted[5][2]['dependency'] == 'done(903) && done(904)'\n    assert submitted[6][2]['dependency'] == 'done(906) && done(902)'\n    assert not any(phase == 'run' for _,phase,_ in submitted)")
                s=s.replace("ops.resources('certify-protenix-only')", "ops.resources('certify-cpu')")
            if name.endswith('test_lrboost_endpoint_diagnostics.py'):
                start=s.index('    source = Path(',s.index('def test_diagnostic_outputs'))
                end=s.index('\n\n',start)
                s=s[:start]+"    source = Path("+repr(REPORTSOURCE)+")\n    assert diag.diagnostic_output_path(source, 'boltz2') == Path("+repr(REPORTROOT+'/diagnostic-boltz2/payload')+")\n    with pytest.raises(ValueError):\n        diag.diagnostic_output_path(diag.SOURCE, 'boltz2')"+s[end:]
        if name.endswith('.html') or name.endswith('test_jq1_lrboost_publish.py'):
            s=s.replace('higher learning-rate','lower learning-rate').replace('higher-rate','lower-rate')
            s=s.replace('Protenix 3.0; both Boltz-2 arms 0.05','Protenix 0.3; both Boltz-2 arms 0.005')
        if name.endswith('.html'):
            anchor="byId('stepInput').addEventListener('change',"
            assert s.count(anchor)==1
            s=s.replace(anchor,"byId('stepInput').addEventListener('focus',stop);\n"+anchor)
            s=s.replace("entry.diagnostic_status:'pending'","entry.diagnostic_status:entry.requested===false?'not requested':'pending'")
            s=s.replace('<section><img', '<p class="note">Boltz’s failed final derivative check remains failed. Its saved trajectory and fixed-pose scores support exploratory analysis, not a validated endpoint gradient or evidence of binding/activity. Chemistry coefficient: 0.001 through update 5000, then geometric to 100.</p>\n<section><img')
        if name.endswith('publish_jq1_lrboost_report.py'):
            s=s.replace('jq1-lrboost-step-viewer-v1','jq1-lrreset-step-viewer-v1').replace('higher learning-rate','lower learning-rate').replace('higher-rate','lower-rate')
        if name.endswith('render_jq1_lrboost_report.py'):
            s=s.replace("'status': 'pending',\n                 'scope'", "'status': 'pending', 'requested': arm == 'boltz2',\n                 'scope'")
        sources[name]=s
    d=sources['scripts/lrboost_endpoint_diagnostics.py']
    d=assign(d,'SOURCE','Path('+repr(BASE+'/restart-lrreset-release-04/source')+')');d=assign(d,'SOURCE_MANIFEST',repr(pins['scientific_source']['sha256']));d=assign(d,'JOBS',repr({a:p['job'] for a,p in pins['arms'].items() if a!='boltz2_chemistry'}))
    d=replace_function(d,'diagnostic_output_path',"def diagnostic_output_path(source, arm):\n    require(Path(source) == Path("+repr(REPORTSOURCE)+") and arm in JOBS, 'Wrong separate immutable diagnostic scope')\n    return Path("+repr(REPORTROOT)+") / ('diagnostic-' + arm) / 'payload'")
    d=d.replace("    provenance = authenticate_inputs(args.arm)","    require(args.arm == 'boltz2', 'This authorization is Boltz-only; no repeated Protenix diagnostic')\n    provenance = authenticate_inputs(args.arm)")
    sources['scripts/lrboost_endpoint_diagnostics.py']=d
    ops=sources['scripts/jq1_lrboost_report_ops.py']
    for name,value in {'TAG':repr(TAG),'ROOT':'Path('+repr(REPORTROOT)+')','SOURCE':'Path('+repr(REPORTSOURCE)+')','SCIENCE_ROOT':'Path('+repr(BASE+'/restart-lrreset-04')+')','FULL_NATIVE':repr({a:p['projections'] for a,p in pins['arms'].items()})}.items():ops=assign(ops,name,value)
    ops=ops.replace("'--rootdir', str(SOURCE.parents[1])","'--rootdir', str(SOURCE)")
    ops=ops.replace("allowance = 1024**3 if phase in ('preflight', 'cpu') else 2 * 1024**3 + 2 * DIAGNOSTIC_BYTES","allowance = 2 * 1024**3 + 2 * DIAGNOSTIC_BYTES")
    ops=replace_function(ops,'submission_plan',"def submission_plan():\n    return [('diagnostic-boltz2','diagnostic','boltz2',None)] + [('complete-'+a,'capture',a,'ended('+ARMS[a]['job']+')' if a=='boltz2_chemistry' else None) for a in ARMS]")
    ops=replace_function(ops,'check_storage',(HERE/'analysis_storage.py').read_text())
    ops=ops.replace("or arm in ('protenix', 'boltz2')", "or arm == 'boltz2'")
    start=ops.index('    if only_protenix:',ops.index('def certify_cpu('));end=ops.index('    with (ROOT / \'dispatch.lock\').open(\'a\') as lock:',start+len('    if only_protenix:'))
    # Select the normal branch after the old optional branch's explicit return.
    end=ops.index('    with (ROOT / \'dispatch.lock\').open(\'a\') as lock:',ops.index('        return jobs',start))
    ops=ops[:start]+ops[end:]
    ops=ops.replace('def certify_cpu(manifest, *, only_protenix=False):','def certify_cpu(manifest):')
    ops=ops.replace("'certify-protenix-only', ",'')
    ops=ops.replace("    elif args.phase == 'certify-protenix-only': print(json.dumps(certify_cpu(args.manifest, only_protenix=True)))\n",'')
    ops=ops.replace("jobs['active_interim'] = submit('interim-0001', 'interim', manifest)['job_id']; jobs['interim_parent'] = 'interim-0001'; atomic(ROOT / 'jobs.json', jobs)","first_dep = ' && '.join('done(' + collectors[a] + ')' for a in ('protenix', 'boltz2'))\n        jobs['active_interim'] = submit('interim-0001', 'interim', manifest, dependency=first_dep)['job_id']; jobs['interim_parent'] = 'interim-0001'; atomic(ROOT / 'jobs.json', jobs)\n        second_dep = ' && '.join('done(' + j + ')' for j in [jobs['active_interim'], captures['boltz2']])\n        jobs['diagnostic_interim'] = submit('interim-0002', 'interim', manifest, dependency=second_dep)['job_id']; atomic(ROOT / 'jobs.json', jobs)")
    ops=ops.replace("[*collectors.values(), *captures.values()]", "[*collectors.values(), *captures.values(), jobs['diagnostic_interim']]")
    sources['scripts/jq1_lrboost_report_ops.py']=ops
    sources['scripts/hpc/jq1_lrboost_analysis_stage.lsf']=(HERE/'analysis_stage.lsf').read_text()
    sources['docs/brd4_jq1_25_endpoint_diagnostics.md']=(HERE/'README.md').read_text()
    sources['analysis-input-pins.json']=(HERE/'input-pins.json').read_text()
    # Extra allocated presentation regression for the previously observed input race.
    sources['tests/test_jq1_lrboost_publish.py']+='''\n\ndef test_new_numeric_focus_stops_playback_before_input_change():
    template = (Path(publisher.__file__).parent / 'templates/jq1_lrboost_report.html').read_text()
    focus = "byId('stepInput').addEventListener('focus',stop);"
    assert template.count(focus) == 1
    assert template.index(focus) < template.index("byId('stepInput').addEventListener('change',")
'''
    files={}
    for name,s in sources.items():
        data=s.replace('\r\n','\n').encode()
        if name.endswith('.py'):ast.parse(data,filename=name)
        files[name]=data
        local=HERE/('core-'+name.replace('/','__'))
        with local.open('wb') as f:f.write(data)
    metadata={'files':{n:hashlib.sha256(d).hexdigest() for n,d in sorted(files.items())},'parent_manifest_sha256':hashlib.sha256(parent).hexdigest(),'scientific_source_manifest_sha256':pins['scientific_source']['sha256'],'scope':'Separate lower-rate diagnostics and saved-only presentation; no saved run changes'}
    raw=(json.dumps(metadata,indent=2,sort_keys=True)+'\n').encode();manifest=hashlib.sha256(raw).hexdigest()
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for name,data in {**files,'source_manifest.json':raw}.items():
            entry=tarfile.TarInfo('source/'+name);entry.size=len(data);entry.mode=0o444;tar.addfile(entry,io.BytesIO(data))
    package={'tag':TAG,'source_files':len(files),'manifest_sha256':manifest,'archive_bytes':(out/'source.tar.gz').stat().st_size,'archive_sha256':hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest(),'parent_manifest_sha256':hashlib.sha256(parent).hexdigest()}
    (out/'source_manifest.json').write_bytes(raw);(out/'package.json').write_text(json.dumps(package,indent=2)+'\n');(HERE/'package.json').write_text(json.dumps(package,indent=2)+'\n')
    print(json.dumps(package,indent=2))


if __name__=='__main__':main()
