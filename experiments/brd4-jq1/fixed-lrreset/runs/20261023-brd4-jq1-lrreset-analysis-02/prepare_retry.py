"""Stdlib byte/AST-only fork; no local tests or model imports."""
import ast,hashlib,io,json,tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
TAG=HERE.name
PARENT=HERE.with_name('20261023-brd4-jq1-lrreset-analysis-01')

def main():
    archive=REPO/'results/hpc'/PARENT.name/'source02/source.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='4c809367734cf6e529d1ae59e0b30cac603d9eae147914b518d992fb8818dc53'
    with tarfile.open(archive) as tar:
        files={m.name.removeprefix('source/'):tar.extractfile(m).read() for m in tar if m.isfile()}
    raw=files.pop('source_manifest.json')
    assert hashlib.sha256(raw).hexdigest()=='761d66642052fde3da2da2016736fdd91bdea477ba714787a5c393b2038f8b83'
    meta=json.loads(raw);assert set(meta['files'])==set(files)
    changes={}
    for name,data in list(files.items()):
        assert hashlib.sha256(data).hexdigest()==meta['files'][name]
        text=data.decode().replace('/lrreset-analysis-release-01/source','/lrreset-analysis-release-02/source').replace('/lrreset-analysis-01','/lrreset-analysis-02').replace('20261023-brd4-jq1-lrreset-analysis-01','20261023-brd4-jq1-lrreset-analysis-02')
        if name=='tests/test_jq1_lrboost_render.py':
            assert text.count('== 50.')==2
            text=text.replace('test_schedule_keeps_declared_5000_hold_and_terminal_50','test_schedule_keeps_declared_5000_hold_and_terminal_100').replace('== 50.','== 100.')
        if name=='analysis-input-pins.json':
            # Original input bytes remain unchanged; it is not a mutable status snapshot.
            text=data.decode()
        new=text.encode();files[name]=new
        if new!=data:changes[name]={'before':hashlib.sha256(data).hexdigest(),'after':hashlib.sha256(new).hexdigest()}
        if name.endswith('.py'):ast.parse(new,filename=name)
        (HERE/('core-'+name.replace('/','__'))).write_bytes(new)
    metadata={**meta,'files':{n:hashlib.sha256(d).hexdigest() for n,d in sorted(files.items())},'parent_analysis_manifest_sha256':hashlib.sha256(raw).hexdigest(),'engineering_change':'Correct inherited synthetic chemistry terminal50 expectations to declared100; independent analysis roots only'}
    raw=(json.dumps(metadata,indent=2,sort_keys=True)+'\n').encode()
    out=REPO/'results/hpc'/TAG/'source01';out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for name,data in {**files,'source_manifest.json':raw}.items():
            m=tarfile.TarInfo('source/'+name);m.size=len(data);m.mode=0o444;tar.addfile(m,io.BytesIO(data))
    package={'tag':TAG,'source_files':len(files),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'archive_bytes':(out/'source.tar.gz').stat().st_size,'archive_sha256':hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest()}
    (out/'source_manifest.json').write_bytes(raw)
    for p in [out/'package.json',HERE/'package.json']:p.write_text(json.dumps(package,indent=2)+'\n')
    (HERE/'source-comparison.json').write_text(json.dumps({'parent_manifest':metadata['parent_analysis_manifest_sha256'],'changes':changes,'scope':'Fixture endpoint expectation and separate root identities only; no scientific code or original-result edits'},indent=2)+'\n')
    for name in ['submit_analysis.py','observe.py']:
        text=(PARENT/name).read_text().replace('20261023-brd4-jq1-lrreset-analysis-01',TAG).replace('/lrreset-analysis-01','/lrreset-analysis-02').replace("'lrreset-analysis-01'","'lrreset-analysis-02'").replace("'lrreset-analysis-release-01'","'lrreset-analysis-release-02'").replace("'lrreset-analysis-01.tar.gz'","'lrreset-analysis-02.tar.gz'")
        if name=='submit_analysis.py':text=text.replace('source02/','source01/')
        ast.parse(text);(HERE/name).write_text(text,newline='\n')
    print(json.dumps(package))

if __name__=='__main__':main()
