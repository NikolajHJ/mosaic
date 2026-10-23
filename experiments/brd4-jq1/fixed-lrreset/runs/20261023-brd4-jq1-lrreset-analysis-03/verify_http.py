"""Byte-exact localhost presentation audit, without numerical interpretation."""
from datetime import datetime,timezone
import hashlib,json,sys,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
def main():
    edition=sys.argv[1];folder=REPO/'results/hpc'/HERE.name;viewer=folder/'viewer';files={}
    for p in viewer.rglob('*'):
        if not p.is_file():continue
        n=p.relative_to(viewer).as_posix()
        with urllib.request.urlopen('http://127.0.0.1:8779/'+n,timeout=30) as q:
            assert q.status==200;data=q.read()
        assert data==p.read_bytes();files[n]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    latest=json.loads((viewer/'latest.json').read_bytes());assert latest['path']==edition+'/report.html'
    record={'status':'passed','utc':datetime.now(timezone.utc).isoformat(),'served_files_verified':len(files),'latest':latest,'files':files,'scope':'HTTP exact bytes only; browser interaction QA recorded separately'}
    out=folder/('http-verification-'+edition+'.json');assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n')
    (HERE/('http-verification-'+edition+'.json')).write_text(json.dumps({k:v for k,v in record.items() if k!='files'},indent=2)+'\n')
    print(json.dumps({'status':'passed','files':len(files),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
