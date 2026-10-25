"""Read-only local HTTP byte verification of every published viewer file."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-jq1-live-report-01'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='http-verification.json');args=parser.parse_args()
    assert args.name.startswith('http-verification') and args.name.endswith('.json') and '/' not in args.name and '\\' not in args.name
    root=LOCAL/'viewer';rows=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(root).as_posix();url='http://127.0.0.1:8780/'+quote(relative)
        digest=hashlib.sha256();length=0
        with urlopen(url,timeout=60) as response:
            assert response.status==200
            for block in iter(lambda:response.read(1024**2),b''):length+=len(block);digest.update(block)
        with path.open('rb') as stream:expected=hashlib.file_digest(stream,'sha256').hexdigest()
        assert length==path.stat().st_size and digest.hexdigest()==expected
        rows.append({'path':relative,'bytes':length,'sha256':expected})
    out=LOCAL/args.name;assert not out.exists()
    value={'status':'passed_all_served_files_exact_bytes','captured_utc':datetime.now(timezone.utc).isoformat(),
           'url':'http://127.0.0.1:8780/report.html','files':rows,'count':len(rows),
           'scope':'Byte delivery only; no scientific array evaluation'}
    out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'status':value['status'],'count':len(rows),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
