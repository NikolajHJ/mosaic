"""Local stdlib AST/byte packaging only; no numerical imports/tests."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[2]
OUT=REPO/'results/hpc/20261025-chemistry-speed-02/source01'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    paths=['README.md','scripts/README.md','tests/README.md',
           *['scripts/'+name for name in ['exact_reuse.py','parallel_search.py','hybrid_search.py','gpu_matrix.py',
                                        'profile_oracle.py','round2_data.py','round2_cpu.py','round2_gpu.py',
                                        'round2_worker.py','round2_stage.lsf']],
           *['tests/'+name for name in ['test_speed.py','test_hybrid.py','test_gpu_matrix.py','test_oracle_profile.py','test_gpu_execution.py']]]
    members={}
    for name in paths:
        raw=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
        if name.endswith('.py'):ast.parse(raw,filename=name)
        members[name]=raw
    members['protocol.md']=(HERE/'runs/20261025-chemistry-speed-02/README.md').read_bytes().replace(b'\r\n',b'\n')
    manifest={'version':'chemistry-hybrid-gpu-matrix-v1','scientific_production_changed':False,
              'cpu_cases':143,'gpu_cases':19,'original_source_manifest_sha256':'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562',
              'value_gradient_tolerance':2e-5,'fd_epsilons':[.01,.003,.001],'fd_relative_error':.15,'fd_two_adjacent':True,
              'files':{k:hashlib.sha256(v).hexdigest() for k,v in sorted(members.items())}}
    raw=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();members['source_manifest.json']=raw
    (OUT/'source_manifest.json').write_bytes(raw)
    archive=OUT/'source.tar.gz'
    with tarfile.open(archive,'x:gz') as t:
        for name,data in sorted(members.items()):
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o444;item.mtime=0;t.addfile(item,io.BytesIO(data))
    with tarfile.open(archive) as t:
        rows=t.getmembers();assert len(rows)==len(members) and all(x.isfile() for x in rows)
        for row in rows:assert t.extractfile(row).read()==members[row.name]
    result={'status':'local_ast_byte_package_only','files':len(members)-1,'archive_bytes':archive.stat().st_size,
            'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(raw).hexdigest(),
            'allocated_cpu_passed':False,'GPU_execution_passed':False,'submitted':False}
    (OUT/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
