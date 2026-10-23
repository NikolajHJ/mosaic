"""Stdlib attribution-only retry fork; tests execute only on allocated HPC."""
import ast,hashlib,io,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
PARENT=HERE.with_name('20261023-brd4-jq1-lrreset-analysis-02')
REGRESSION='''

def test_cpu_collection_root_contains_every_declared_source(monkeypatch,tmp_path):
    parent=tmp_path/'parent';parent.mkdir()
    monkeypatch.setenv('TMPDIR',str(tmp_path))
    monkeypatch.setattr(ops,'check_storage',lambda *args:None)
    calls=[]
    monkeypatch.setattr(ops.subprocess,'run',lambda command,**kwargs:calls.append(command))
    ops.cpu_tests(parent)
    command=calls[0];root=Path(command[command.index('--rootdir')+1])
    assert root==Path(ops.os.path.commonpath([str(ops.SOURCE),str(ops.SCIENCE)]))
    assert ops.SOURCE.is_relative_to(root) and ops.SCIENCE.is_relative_to(root)
    for module in ops.TESTS:assert str(ops.SOURCE/'tests'/module) in command
    for module in ops.LEGACY_TESTS:assert str(ops.SCIENCE/'tests'/module) in command
'''
def main():
    arc=REPO/'results/hpc'/PARENT.name/'source01/source.tar.gz'
    assert hashlib.sha256(arc.read_bytes()).hexdigest()=='013b3003443b2b2b4da543caa82dafef0032b31ef0e4cfb204938172d432abeb'
    with tarfile.open(arc) as t:files={m.name.removeprefix('source/'):t.extractfile(m).read() for m in t if m.isfile()}
    raw=files.pop('source_manifest.json');assert hashlib.sha256(raw).hexdigest()=='955872e4621b1e5a3d23a69a989aadcf16c8d0e2793e6db464782f58be15fa37'
    meta=json.loads(raw);changes={}
    for name,data in list(files.items()):
        assert hashlib.sha256(data).hexdigest()==meta['files'][name]
        s=data.decode()
        if name!='analysis-input-pins.json':s=s.replace('/lrreset-analysis-release-02/source','/lrreset-analysis-release-03/source').replace('/lrreset-analysis-02','/lrreset-analysis-03').replace('20261023-brd4-jq1-lrreset-analysis-02',HERE.name)
        if name=='scripts/jq1_lrboost_report_ops.py':
            assert s.count("'--rootdir', str(SOURCE),")==1
            s=s.replace("'--rootdir', str(SOURCE),","'--rootdir', os.path.commonpath([str(SOURCE), str(SCIENCE)]),")
        if name=='tests/test_jq1_lrboost_ops.py':s+=REGRESSION
        new=s.encode();files[name]=new
        if new!=data:changes[name]={'before':hashlib.sha256(data).hexdigest(),'after':hashlib.sha256(new).hexdigest()}
        if name.endswith('.py'):ast.parse(new)
        (HERE/('core-'+name.replace('/','__'))).write_bytes(new)
    metadata={**meta,'files':{n:hashlib.sha256(d).hexdigest() for n,d in sorted(files.items())},'parent_analysis_manifest_sha256':hashlib.sha256(raw).hexdigest(),'engineering_change':'Common pytest root attributes external legacy modules; regression; separate own roots only'}
    raw=(json.dumps(metadata,sort_keys=True,indent=2)+'\n').encode();out=REPO/'results/hpc'/HERE.name/'source01';out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for name,data in {**files,'source_manifest.json':raw}.items():
            m=tarfile.TarInfo('source/'+name);m.size=len(data);m.mode=0o444;tar.addfile(m,io.BytesIO(data))
    package={'tag':HERE.name,'source_files':17,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'archive_bytes':(out/'source.tar.gz').stat().st_size,'archive_sha256':hashlib.sha256((out/'source.tar.gz').read_bytes()).hexdigest()}
    (out/'source_manifest.json').write_bytes(raw)
    for p in [out/'package.json',HERE/'package.json']:p.write_text(json.dumps(package,indent=2)+'\n')
    (HERE/'source-comparison.json').write_text(json.dumps({'parent':metadata['parent_analysis_manifest_sha256'],'changes':changes},indent=2)+'\n')
    for name in ['submit_analysis.py','observe.py','collect_cpu.py','collect_delivery.py','download_archive.py']:
        text=(PARENT/name).read_text().replace('20261023-brd4-jq1-lrreset-analysis-02',HERE.name).replace('/lrreset-analysis-02','/lrreset-analysis-03').replace("'lrreset-analysis-02'","'lrreset-analysis-03'").replace("'lrreset-analysis-release-02'","'lrreset-analysis-release-03'").replace("'lrreset-analysis-02.tar.gz'","'lrreset-analysis-03.tar.gz'").replace('955872e4621b1e5a3d23a69a989aadcf16c8d0e2793e6db464782f58be15fa37',package['manifest_sha256'])
        if name in ['collect_cpu.py','collect_delivery.py']:text=text.replace('283','284')
        ast.parse(text);(HERE/name).write_text(text,newline='\n')
    print(json.dumps(package))
if __name__=='__main__':main()
