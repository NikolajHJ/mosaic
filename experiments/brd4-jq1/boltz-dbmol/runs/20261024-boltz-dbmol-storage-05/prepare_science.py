"""Create a new manifest with every original657 member byte unchanged."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
OLD_MANIFEST='24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
ANCESTOR_MANIFEST='c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
REVISION='c74c659344123937608077e3e4078f75f09cd743'

def sha(data):return hashlib.sha256(data).hexdigest()

def load(path,archive_sha,manifest_sha,count):
    assert sha(path.read_bytes())==archive_sha
    files={}
    with tarfile.open(path) as tar:
        for member in tar:
            assert member.isfile() and member.name.startswith('source/') and '..' not in Path(member.name).parts
            name=member.name.removeprefix('source/');assert name not in files
            files[name]=tar.extractfile(member).read()
    raw=files.pop('source_manifest.json');assert sha(raw)==manifest_sha
    manifest=json.loads(raw);assert len(files)==count and set(files)==set(manifest['files'])
    for name,data in files.items():assert sha(data)==manifest['files'][name]
    return files,manifest

def main():
    files,old=load(REPO/'results/hpc/20261023-boltz-dbmol-pilot-01/source01/source.tar.gz',
        'ec333fd3e0b33d9e5b82e0cd5e4e12c14ffb4f91782c18cd820d4bfab53de1f8',OLD_MANIFEST,657)
    _,ancestor=load(REPO/'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/source02/source.tar.gz',
        '4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e',ANCESTOR_MANIFEST,625)
    assert ancestor['git_revision_reference']==REVISION and 'git_revision_reference' not in old
    manifest={**old,'git_revision_reference':REVISION,'parent_manifest_sha256':OLD_MANIFEST,
        'inherited_parent_manifest_sha256':old['parent_manifest_sha256'],
        'revision_ancestor_manifest_sha256':ANCESTOR_MANIFEST,
        'metadata_adoption_version':'dbmol-archived-revision-metadata-v1',
        'scientific_member_bytes_unchanged':True}
    assert manifest['files']==old['files']
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output=REPO/'results/hpc/20261024-boltz-dbmol-pilot-02/source01';output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(output/'source.tar.gz','w:gz') as tar:
        for name,data in {**files,'source_manifest.json':raw}.items():
            member=tarfile.TarInfo('source/'+name);member.mode=0o444;member.size=len(data);tar.addfile(member,io.BytesIO(data))
    (output/'source_manifest.json').write_bytes(raw)
    package={'tag':'20261024-boltz-dbmol-pilot-02','source_files':len(files),'manifest_sha256':sha(raw),
        'archive_bytes':(output/'source.tar.gz').stat().st_size,'archive_sha256':sha((output/'source.tar.gz').read_bytes()),
        'parent_manifest_sha256':OLD_MANIFEST,'git_revision_reference':REVISION,
        'revision_ancestor_manifest_sha256':ANCESTOR_MANIFEST,'changed_source_members':[],
        'remote_source':'/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/boltz-dbmol-pilot-release-02/source'}
    (HERE/'science-package.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf8')
    print(json.dumps(package,indent=2))

if __name__=='__main__':main()
