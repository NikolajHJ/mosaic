"""Authenticate delivered byte evidence, then atomically publish a new edition.

Only stdlib byte/XML/schema checks run locally. Scientific FP32 validation and
rendering were already performed in the original allocated CPU report job.
"""
from collections import Counter
from datetime import datetime,timezone
import argparse
import base64
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import shlex
import shutil
import subprocess
import tarfile
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-jq1-live-report-01'
WRAPPER='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mosaic · updating saved results</title><style>body{margin:0;background:#f1f5f4;font:13px system-ui;color:#17313b}header{height:38px;display:flex;align-items:center;gap:15px;padding:0 18px;border-bottom:1px solid #cbd9dc}button{font:inherit;background:white;border:1px solid #b8cbcf;border-radius:5px;padding:3px 9px;cursor:pointer}iframe{width:100%;height:calc(100vh - 39px);border:0;display:block}#state{margin-left:auto;color:#5a7079}</style><header><strong>Mosaic · saved results</strong><button id="refresh">Check for updated report</button><span id="state">Loading…</span></header><iframe title="Verified saved optimization report" id="report"></iframe><script>const frame=document.getElementById('report'),state=document.getElementById('state');let edition=null,selection={};frame.addEventListener("load",()=>{const doc=frame.contentDocument;if(!doc)return;const style=doc.createElement("style");style.dataset.owner="stable-shell-layout-v1";style.textContent=".grid{grid-template-columns:minmax(0,1.1fr) minmax(0,1fr)}.grid>.stack,.stack>.card,.heatmaps>div{min-width:0}.heatmaps{grid-template-columns:repeat(2,minmax(0,1fr))}@media(max-width:1000px){.grid{grid-template-columns:minmax(0,1fr)}}@media(max-width:560px){.heatmaps{grid-template-columns:minmax(0,1fr)}}";doc.head.append(style);});async function refresh(){try{if(frame.contentWindow?.MOSAIC_REPORT_TEST_API)selection=frame.contentWindow.MOSAIC_REPORT_TEST_API.selection();const response=await fetch('latest.json?t='+Date.now(),{cache:'no-store'});if(!response.ok)throw Error('Latest pointer unavailable');const next=await response.json();if(!/^edition-(?:[0-9]{4}|final)$/.test(next.edition)||!/^editions\\/edition-(?:[0-9]{4}|final)\\/report.html$/.test(next.report_url))throw Error('Invalid local edition pointer');if(next.edition!==edition){const url=new URL(next.report_url,location.href);if(selection.arm)url.searchParams.set('arm',selection.arm);if(Number.isSafeInteger(selection.step))url.searchParams.set('step',selection.step);frame.src=url;edition=next.edition;}state.textContent=edition+' · checks for new editions every60s';}catch(error){state.textContent=String(error.message);}}document.getElementById('refresh').onclick=refresh;setInterval(refresh,60000);refresh();</script></html>'''


LEGACY_WRAPPER = WRAPPER
WRAPPER = WRAPPER.replace('if(!doc)return;',
    'if(!doc||doc.documentElement.dataset.presentation==="refined-v1")return;')
assert WRAPPER != LEGACY_WRAPPER


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def check_lsf(path,job,phase,prefix):
    text=''.join(Path(path).read_text().split())
    assert text.count('Subject:Job')==1 and 'Subject:Job'+job+':' in text
    assert '<'+prefix+'-'+phase+'>incluster<dcc>Done' in text
    assert 'inqueue<hpc>' in text and text.count('Successfullycompleted.')==1


def download(row,out,remote_root,edition):
    assert not out.exists() and row['path']==remote_root+edition+'.tar.gz'
    assert row['path'].endswith('.tar.gz') and type(row['bytes']) is int and row['bytes']>0 and re.fullmatch('[a-f0-9]{64}',row['sha256'])
    code="import hashlib,os,pathlib,sys\np=pathlib.Path(sys.argv[1]);assert p.is_file() and not p.is_symlink()\nwith p.open('rb') as f:\n assert os.fstat(f.fileno()).st_size==int(sys.argv[2]) and hashlib.file_digest(f,'sha256').hexdigest()==sys.argv[3]\n f.seek(0)\n for b in iter(lambda:f.read(1048576),b''):sys.stdout.buffer.write(b)\n sys.stdout.buffer.flush()\n"
    remote=' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',code,row['path'],str(row['bytes']),row['sha256']])
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',remote]
    marker=out.with_suffix('.started.json');result=out.with_suffix('.result.json')
    assert not marker.exists() and not result.exists()
    record={'status':'unknown','binding':row,'argv':args};write(marker,record)
    try:
        with out.open('xb') as stream:
            process=subprocess.Popen(args,stdout=stream,stderr=subprocess.PIPE)
            try:_,error=process.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill();_,error=process.communicate(timeout=20);raise
            finally:out.with_suffix('.stderr.txt').write_bytes(error)
        assert process.returncode==0 and out.stat().st_size==row['bytes'] and sha(out)==row['sha256']
        record['status']='complete_bytes_verified'
    finally:
        record.update(observed_bytes=out.stat().st_size if out.exists() else 0,finished_utc=datetime.now(timezone.utc).isoformat())
        write(result,record)


def extract(archive,output):
    output.mkdir(exist_ok=False)
    with tarfile.open(archive) as bundle:
        items=bundle.getmembers();assert len({item.name.casefold() for item in items})==len(items)
        assert sum(item.size for item in items)<1024**3
        for item in items:
            part=PurePosixPath(item.name)
            assert item.isfile() and not part.is_absolute() and '..' not in part.parts and '\\' not in item.name
            target=output.joinpath(*part.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:shutil.copyfileobj(bundle.extractfile(item),stream)
    receipt=json.loads((output/'report-receipt.json').read_bytes())
    expected=set(receipt['files'])|{'report-receipt.json'}
    actual={path.relative_to(output).as_posix() for path in output.rglob('*') if path.is_file()}
    assert actual==expected
    for name,row in receipt['files'].items():
        path=output/name;assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    return receipt


def verify_display_only_delta(extracted, report):
    """Schema/equality comparison only; no local scientific array evaluation."""
    previous = LOCAL/'viewer/editions/edition-0002'
    old = json.loads((previous/'manifest.json').read_bytes())
    assert len(report['arms']) == 4 and len({arm['id'] for arm in report['arms']}) == 4
    assert {arm['id'] for arm in old['arms']} == {arm['id'] for arm in report['arms']}
    before = {arm['id']:arm for arm in old['arms']}
    frames = 0
    for arm in report['arms']:
        original = before[arm['id']]
        assert arm['scalars'] == original['scalars']
        assert (extracted/arm['scalars']['path']).read_bytes() == (previous/original['scalars']['path']).read_bytes()
        for field in ('committed_updates','latest_global_step','scientific_status','segment_proofs',
                      'global_start','global_target','rate','update_counts'):
            assert arm[field] == original[field], 'Presentation changed original metadata: '+field
        assert arm['frames']['steps'] == original['frames']['steps']
        assert len(arm['frames']['chunks']) == len(original['frames']['chunks'])
        for new_chunk, old_chunk in zip(arm['frames']['chunks'],original['frames']['chunks']):
            assert new_chunk['path'] == old_chunk['path']
            current = json.loads((extracted/new_chunk['path']).read_bytes())
            prior = json.loads((previous/old_chunk['path']).read_bytes())
            assert len(current['frames']) == len(prior['frames'])
            for new_frame, old_frame in zip(current['frames'],prior['frames']):
                new_graph = new_frame['graph']
                assert set(new_graph) == set(old_frame['graph']) | {'layout'}
                assert new_graph['layout']['identity_preserved'] is True
                assert len(new_graph['nodes']) == len(old_frame['graph']['nodes'])
                for new_node, old_node in zip(new_graph['nodes'],old_frame['graph']['nodes']):
                    assert set(new_node) == set(old_node) | {'x','y'}
                # Strip only the declared display additions. Exact nested JSON
                # equality preserves original matrices, keys, scalars and graph.
                stripped = {**new_frame, 'graph':{
                    **{key:value for key,value in new_graph.items() if key!='layout'},
                    'nodes':[{key:value for key,value in node.items() if key not in ('x','y')}
                             for node in new_graph['nodes']]}}
                canonical = lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
                assert canonical(stripped) == canonical(old_frame), 'Non-display saved-frame field/type changed'
                frames += 1
    return {'status':'passed_exact_saved_JSON_equality_except_display_coordinates',
            'previous_edition':'edition-0002','scalar_assets_byte_identical':len(report['arms']),
            'frames_compared':frames,'scientific_array_evaluation_performed_locally':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observation',required=True);parser.add_argument('--edition',default='edition-0002');parser.add_argument('--generation',default='03');args=parser.parse_args()
    assert re.fullmatch('[0-9]{2}',args.generation) and int(args.generation)>=3
    assert re.fullmatch('edition-[0-9]{4}',args.edition) and re.fullmatch('observation-[a-zA-Z0-9_-]+',args.observation)
    prefix='mosaic-live-report'+args.generation
    observation=LOCAL/args.observation;raw=json.loads((observation/'raw.json').read_bytes())
    assert raw['publication_ready'] is True and raw['mutated'] is False
    package=LOCAL/('source'+args.generation);manifest=json.loads((package/'source_manifest.json').read_bytes());pin=sha(package/'source_manifest.json')
    assert raw['source_manifest_sha256']==pin
    files=observation/'files';jobs=raw['jobs']
    for phase in ('cpu','after-cpu','render'):
        sub=json.loads((files/phase/'submission.json').read_bytes());assert sub['status']=='submitted' and sub['job_id']==jobs[phase] and sub['manifest_sha256']==pin
        compact=''.join(raw['queries'][phase]['stdout'].split())
        assert raw['queries'][phase]['returncode']==0 and re.findall(r'Job<(\d+)>',compact)==[jobs[phase]] and re.findall(r'Status<([^>]+)>',compact)==['DONE']
        check_lsf(files/phase/('stdout.'+jobs[phase]+'.log'),jobs[phase],phase,prefix)
    cert=json.loads((files/'cpu-certificate.json').read_bytes());assert cert['status']=='passed' and cert['manifest_sha256']==pin and cert['cpu_job_id']==jobs['cpu']
    remote_root='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-live-report-'+args.generation+'/'
    for row in cert['proofs']:
        assert row['path'].startswith(remote_root);path=files/row['path'][len(remote_root):]
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    cases=ET.parse(files/'cpu/tests.xml').getroot().findall('.//testcase')
    assert cases and all(not list(row) for row in cases)
    assert dict(Counter(row.attrib['classname'] for row in cases))==manifest['cpu_counts']
    archive_receipt=raw['archive_receipt'];assert archive_receipt['status']=='passed_saved_only_presentation' and archive_receipt['job_id']==jobs['render'] and archive_receipt['manifest_sha256']==pin
    delivery=LOCAL/('delivery-'+args.edition);delivery.mkdir(exist_ok=False)
    archive=delivery/'report.tar.gz';download(archive_receipt['archive'],archive,remote_root,args.edition)
    extracted=delivery/'extracted';receipt=extract(archive,extracted)
    assert sha(extracted/'report-receipt.json')==archive_receipt['receipt_sha256']
    assert receipt['status']=='passed_saved_only_presentation' and receipt['job_id']==jobs['render'] and receipt['manifest_sha256']==pin
    assert receipt['cpu_certificate_sha256']==sha(files/'cpu-certificate.json')==archive_receipt['cpu_certificate_sha256']
    assert receipt['plan_sha256']==sha(package/'plan.json')==sha(extracted/'input-plan.json')
    report=json.loads((extracted/'manifest.json').read_bytes());assert report['edition']==args.edition
    expected={'protenix':(10000,50000,'failed',501),'boltz':(10000,30000,'failed',501),'dbmol':(5000,10000,'failed',251),'protenix_noisy':(1000,1000,'passed',51)}
    if args.generation!='03':
        planned=json.loads((package/'plan.json').read_bytes())
        planned_ids={arm['id'] for arm in planned['arms']}
        assert planned_ids in (set(expected),set(expected)|{'protenix_chemistry','boltz_chemistry'})
        assert len(planned_ids)==len(planned['arms'])==len(report['arms'])
        assert {arm['id'] for arm in report['arms']}==planned_ids
        expected={arm['id']:(arm['expected_committed_updates'],arm['global_target'],arm['scientific_status'],arm['expected_retained_frames']) for arm in planned['arms']}
    for arm in report['arms']:
        count,target,status,frames=expected[arm['id']]
        assert (arm['committed_updates'],arm['global_target'],arm['scientific_status'],arm['frames']['count'])==(count,target,status,frames)
        for row in [arm['scalars'],*arm['frames']['chunks']]:
            path=extracted/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    display_delta = verify_display_only_delta(extracted,report) if args.generation=='04' else None
    audit={'status':'passed_independent_byte_XML_schema_delivery','archive':archive_receipt['archive'],'source_manifest_sha256':pin,
           'actual_jobs':jobs,'raw_cases':len(cases),'cpu_counts':manifest['cpu_counts'],'all_member_files_verified':len(receipt['files'])+1,
           'scientific_success_claim':False,'original_results_changed':False,'native_diagnostic_qualification':False,
           'report_manifest_sha256':sha(extracted/'manifest.json'),'display_only_delta':display_delta}
    write(delivery/'verification.json',audit)
    viewer=LOCAL/'viewer';editions=viewer/'editions';editions.mkdir(parents=True,exist_ok=True)
    destination=editions/args.edition;assert not destination.exists()
    shutil.copytree(extracted,destination)
    wrapper=viewer/'report.html'
    if wrapper.exists():
        existing = wrapper.read_text(encoding='utf8')
        assert existing in (LEGACY_WRAPPER, WRAPPER), 'Unknown stable-shell bytes: inspect before replacing'
        if existing != WRAPPER:
            # The old compatibility CSS still applies to historical editions;
            # the sealed refined edition owns its responsive layout itself.
            write(delivery/'stable-shell-upgrade.json', {
                'scope':'Mutable shell layout compatibility only; immutable editions unchanged',
                'before_sha256':sha(wrapper), 'after_sha256':hashlib.sha256(WRAPPER.encode()).hexdigest(),
                'refined_edition_uses_own_responsive_CSS':True})
            temporary = viewer/('shell-'+args.edition+'.partial')
            with temporary.open('x',encoding='utf8',newline='\n') as stream:stream.write(WRAPPER)
            temporary.replace(wrapper)
    else:wrapper.write_text(WRAPPER,encoding='utf8',newline='\n')
    pointer={'version':'jq1-live-publication-v1','edition':args.edition,'report_url':'editions/'+args.edition+'/report.html',
             'manifest_sha256':sha(destination/'manifest.json'),'delivery_verification_sha256':sha(delivery/'verification.json'),
             'published_utc':datetime.now(timezone.utc).isoformat(),'automatic_optimizer_submission':False}
    write(viewer/(args.edition+'-publication.json'),pointer)
    temporary=viewer/('latest-'+args.edition+'.partial');write(temporary,pointer);temporary.replace(viewer/'latest.json')
    print(json.dumps({'status':'published_verified_saved_only','viewer':str(viewer),'edition':args.edition,'audit':audit},indent=2))


if __name__=='__main__':main()
