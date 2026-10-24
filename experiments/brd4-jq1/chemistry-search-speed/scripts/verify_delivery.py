"""Independent local stdlib proof/XML/schema and timing-metadata audit only."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import xml.etree.ElementTree as ET


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    observation = Path(sys.argv[1]).resolve()
    data = json.loads((observation / 'raw.json').read_bytes())
    files = observation / 'files'
    for rel, row in data['files'].items():
        p = files / rel
        assert p.is_relative_to(files) and p.is_file() and not p.is_symlink()
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    job = data['job_id']
    archive=(files/'source.tar.gz').read_bytes()
    assert len(archive)==17604 and hashlib.sha256(archive).hexdigest()=='5ec552bcdd5be9a98cba26b00a9db9720098214af66786eee09b7e9da9ce6b31'
    pin='a30fd3937ea79935d0c449c600d2ea42f5f26da31906508a31e306f81d1c13b0'
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:
        members=t.getmembers()
        assert len(members)==10 and len({r.name for r in members})==10
        assert all(r.isfile() and not Path(r.name).is_absolute() and '..' not in Path(r.name).parts for r in members)
        raw=t.extractfile('source_manifest.json').read()
        assert hashlib.sha256(raw).hexdigest()==pin
        manifest=json.loads(raw)
        assert set(manifest['files'])=={r.name for r in members}-{'source_manifest.json'}
        for name,digest in manifest['files'].items():
            assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==digest
    compact = ''.join(data['scheduler'].split())
    assert f'Job<{job}>' in compact and 'Status<DONE>' in compact and 'Queue<hpc>' in compact
    log = (files / f'cpu/stdout.{job}.log').read_text()
    compact_log = ''.join(log.split())
    assert compact_log.count(f'Subject:Job{job}:<mosaic-chem-speed01>incluster<dcc>Done') == 1
    assert 'Successfullycompleted.' in compact_log and 'inqueue<hpc>' in compact_log
    submission = json.loads((files / 'cpu/submission.json').read_text())
    assessment = json.loads((files / 'cpu/payload/assessment.json').read_text())
    plan = json.loads((files / 'plan.json').read_text())
    result = json.loads((files / 'cpu/payload/benchmark/assessment.json').read_text())
    assert assessment['job_id'] == submission['job_id'] == job
    assert assessment['status'] == 'passed'
    assert result['status'] == 'passed_exact_on_sampled_saved_inputs'
    assert sha(files / 'plan.json') == submission['plan_sha256'] == result['plan_sha256']
    assert submission['inputs'] == plan['inputs']
    for row in plan['inputs']:
        p=files/'input-bytes'/(row['label']+'.npz')
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    assert assessment['source_manifest_sha256'] == submission['manifest_sha256']
    assert submission['manifest_sha256']==pin
    assert assessment['original_source'] == assessment['original_source_after']
    assert assessment['original_source']['manifest_sha256']=='da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562'
    assert assessment['original_source']['files'] == 643
    xml = files / 'cpu/payload/tests.xml'
    cases = ET.parse(xml).getroot().findall('.//testcase')
    assert len(cases) == assessment['cpu_cases'] == plan['cpu_cases'] == 79
    assert len({(c.get('classname'),c.get('name')) for c in cases})==79
    assert all(c.get('classname')=='tests.test_speed' for c in cases)
    assert all(not list(case) for case in cases)
    assert sha(xml) == assessment['cpu_xml_sha256']
    assert sha(files / 'cpu/payload/benchmark/assessment.json') == assessment['benchmark_sha256']
    for gate in [assessment['storage'], submission['storage']]:
        assert gate['status'] == 'passed'
        assert gate['available_bytes'] >= gate['required_bytes']
        assert gate['reserve_bytes'] == 4*1024**3 and gate['file_reserve'] == 100000
        assert gate['additional_allowance_bytes'] == 512*1024**2
        assert gate['whole_gate']['status'] == 'passed'
    assert assessment['output_bytes'] < plan['output_cap_bytes']
    assert plan['search']=={'max_edits':3,'checks_per_depth':32,'beam_width':4,'probability_weight':1.0,'probability_floor':1e-12}
    assert plan['progress_search']=={'version':'repair-progress-v1','max_edits':6,'checks_per_depth':32,'beam_width':4,'probability_weight':1.0,'allow_atom_edits':False,'atom_edit_penalty':3.0,'max_atom_edits':1}
    rows = result['rows']
    labels = [v['label'] for v in plan['inputs']]
    modes = ['baseline','cost','cache','combined','parallel2','parallel4']
    assert len(rows) == len(labels)*len(modes)
    summaries = []
    for label in labels:
        group = {r['mode']:r for r in rows if r['input']['label'] == label}
        assert set(group) == set(modes)
        base = group['baseline']
        for mode, row in group.items():
            assert row['exact_preparation_and_gradient'] is True
            for key in ['preparation_identity','original_checks','additional_checks','routes','loss','input']:
                assert row[key] == base[key], (label,mode,key)
            assert len(row['seconds']) == 3 and all(type(t) in (float,int) and t>0 for t in row['seconds'])
            assert row['warm_median_seconds']>0 and row['first_encounter_seconds']==row['seconds'][0]
        summaries.append({'input':label,'original_checks':base['original_checks'],
                          'additional_checks':base['additional_checks'],
                          'warm_seconds':{k:v['warm_median_seconds'] for k,v in group.items()},
                          'first_encounter_seconds':{k:v['first_encounter_seconds'] for k,v in group.items()},
                          'warm_speedup':{k:base['warm_median_seconds']/v['warm_median_seconds'] for k,v in group.items()}})
    receipt = {'status':'passed_byte_xml_schema_timing_metadata_only','job_id':job,'raw_cpu_cases':79,
               'source_manifest_sha256':submission['manifest_sha256'],'source_files':9,
               'original_scientific_files':643,'saved_inputs':len(labels),'compared_rows':len(rows),
               'every_loss_and_gradient_hash_exact':True,'original_runs_changed':False,
               'evidence_raw_sha256':sha(observation/'raw.json'),'verified_files':len(data['files']),
               'verifier_source_sha256':sha(Path(__file__)),
               'source_archive_sha256':sha(files/'source.tar.gz'),
               'timings':summaries,'scope':'Independent evidence audit; numerical comparisons were allocated HPC',
               'not_claimed':['production installation','native model readiness','whole-trajectory speedup','convergence']}
    out = observation / (sys.argv[2] if len(sys.argv)>2 else 'verification.json')
    with out.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
