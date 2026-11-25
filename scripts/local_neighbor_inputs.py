"""Copy a bounded local-pilot cohort and checkpoints; never mutate the HPC.

The remote payload uses only stdlib byte reads of the existing NPY/SMILES
cache. Training, graph reconstruction and numerical validation happen locally.
No scheduler command, model import or remote output file is involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


REMOTE = r'''
import ast,base64,hashlib,json,math,random,struct
from pathlib import Path
root=Path('/work3/s234818/latentmol')
data=root/'data/source095-moses-triangle-v2'
manifest_bytes=(data/'manifest.json').read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest()=='f246cc93805b73a557cd0c09a78891ac8410c465192db615ce0b6709218c8c0e'
manifest=json.loads(manifest_bytes)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def header(p):
 with p.open('rb') as f:
  assert f.read(6)==b'\x93NUMPY'
  version=tuple(f.read(2));n=struct.unpack('<H' if version==(1,0) else '<I',f.read(2 if version==(1,0) else 4))[0]
  v=ast.literal_eval(f.read(n).decode('latin1'));offset=f.tell()
 assert v['fortran_order'] is False
 return v,offset
def subset(name,indices):
 p=data/name;v,offset=header(p);assert v['descr']=='|i1'
 row=math.prod(v['shape'][1:]);before=p.stat()
 with p.open('rb') as f:
  chunks=[]
  for i in indices:f.seek(offset+i*row);chunks.append(f.read(row))
 b=b''.join(chunks);assert len(b)==row*len(indices)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 return dict(dtype=v['descr'],shape=[len(indices),*v['shape'][1:]],base64=base64.b64encode(b).decode(),sha256=hashlib.sha256(b).hexdigest(),source_bytes=before.st_size,source_manifest_sha256=manifest['files'][name]['sha256'])
p=data/'split.npy';v,offset=header(p);assert v['descr']=='|i1'
with p.open('rb') as f:f.seek(offset);labels=f.read()
assert digest(p)==manifest['files']['split.npy']['sha256']
assert all(labels.count(code)==manifest['counts'][name] for name,code in [('train',0),('val',1),('test',2)])
rng=random.Random(148101)
selected={name:sorted(rng.sample([i for i,x in enumerate(labels) if x==code],count)) for name,code,count in [('train',0,1024),('val',1,256)]}
offsets=data/'smiles_offsets.npy';oh,oo=header(offsets);assert oh['descr']=='<i8'
cohorts={}
with offsets.open('rb') as off,(data/'smiles.utf8').open('rb') as smi:
 for name,indices in selected.items():
  smiles=[]
  for i in indices:
   off.seek(oo+8*i);a,b=struct.unpack('<qq',off.read(16));smi.seek(a);smiles.append(smi.read(b-a).decode())
  cohorts[name]=dict(source_indices=indices,smiles=smiles,arrays={n:subset(n,indices) for n in ('atom_types.npy','charges.npy','bond_types.npy')})
paths={'epoch35':root/'submissions/source104-chemistry-continuation-attempt005/parent.pt','epoch36':root/'runs/source121a002-chemistry-16core-24h-attempt002/variants/r2_async_w12_m128_h24/segments/001/20261018T175254Z_chemistry-train_89b58520/epoch_0036.pt','epoch42':root/'runs/source140-epoch40-to45-continuation-attempt002/segments/002/20261029T060320Z_chemistry-train_c3f4a7c8/epoch_0042.pt'}
expected={'epoch35':'19f438222e454ef53169f6d81082335ee7e24902b1b7cf5e0f18eced4b8e92ed','epoch36':'3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f','epoch42':'5ef851101a5f675edc8f65b31b75c23a5ec460f1c40aa5571829bf8ad6714cfd'}
checkpoints={}
for name,p in paths.items():
 h=digest(p);assert h==expected[name];checkpoints[name]=dict(path=str(p),bytes=p.stat().st_size,sha256=h)
files={}
for name,p in {'plan':root/'submissions/source140-epoch40-to45-continuation-attempt002/source140-epoch40-to45-continuation-attempt002.json','config':root/'submissions/source104-chemistry-continuation-attempt005/parent-config.json','composition':root/'runs/source140-epoch40-to45-continuation-attempt002/training_composition.json'}.items():
 b=p.read_bytes();files[name]=dict(value=json.loads(b),sha256=hashlib.sha256(b).hexdigest())
print(json.dumps(dict(manifest=json.loads(manifest_bytes),manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),cohorts=cohorts,checkpoints=checkpoints,files=files,scope='Read-only subset byte transfer; split.npy fully hashed. Graph member digests are source-manifest declarations, not a repeated full 4.3GB member audit. Selected graph arrays are revalidated locally against each source SMILES.')))
'''


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--connector", default="C:/Users/nikol/DTU-HPC/helper-v1/dtu-hpc.exe")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "inputs.json"
    if target.exists():
        value = json.loads(target.read_text())
    else:
        command = [args.connector, "exec", "--timeout", "180",
                   "bash -lc '/work3/s234818/latentmol/env/boltz311/bin/python -I -'"]
        run = subprocess.run(command, input=REMOTE, text=True, capture_output=True, timeout=190)
        receipt = dict(command=command, payload_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),
                       returncode=run.returncode, stderr=run.stderr)
        (args.output / ("transfer-receipt-" + str(time.time_ns()) + ".json")).write_text(json.dumps(receipt, indent=2))
        if run.returncode:
            raise RuntimeError("Read-only input collection failed; see receipt")
        value = json.loads(run.stdout)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(value, stream)
    for name, asset in value["checkpoints"].items():
        destination = args.output / (name + ".pt")
        if not destination.exists():
            command = [args.connector, "download", "--timeout", "180", asset["path"], str(destination)]
            run = subprocess.run(command, text=True, capture_output=True, timeout=190)
            (args.output / (name + "-transfer.json")).write_text(json.dumps(dict(command=command,
                returncode=run.returncode, stdout=run.stdout, stderr=run.stderr), indent=2))
            if run.returncode:
                raise RuntimeError("Checkpoint download failed; preserve receipt and partial file")
        if destination.stat().st_size != asset["bytes"] or sha(destination) != asset["sha256"]:
            raise ValueError("Local checkpoint does not match authenticated source: " + name)
        print(json.dumps(dict(checkpoint=name, bytes=asset["bytes"], sha256=asset["sha256"])), flush=True)
    print(json.dumps(dict(status="downloaded_hash_verified", inputs=str(target),
                         input_sha256=sha(target), hpc_mutations=False)), flush=True)


if __name__ == "__main__":
    main()
