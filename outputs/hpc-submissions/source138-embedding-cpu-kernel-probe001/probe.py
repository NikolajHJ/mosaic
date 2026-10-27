import hashlib,json,os,subprocess,sys,time
from pathlib import Path
release=Path("/work3/s234818/latentmol/releases/source-138-epoch36-embedding-comparison-attempt005")
sys.path.insert(0,str(release/"scripts/hpc"))
from embedding_operator import authenticate,quota
assert os.environ.get("LSB_JOBID")
authenticate(release,"2de705ee8d88c4699d1f12c105553b1111bf7a82c49e88611750339c9d2e6be1","3b26ae0dd1303d5a39cd42ef79f59fa088380dcc867ac57db9ee91cc85e8323d","configs/epoch36_embedding_comparison.json")
output=Path("/work3/s234818/latentmol/runs/source138-embedding-cpu-kernel-probe001")
output.mkdir(exist_ok=False)
config=json.loads((release/"configs/epoch36_embedding_comparison.json").read_text())
report=dict(status="diagnostic_only_not_readiness",hostname=os.uname().nodename,job=os.environ["LSB_JOBID"],quota=quota(),cpu=subprocess.run(["lscpu"],text=True,capture_output=True).stdout,modes={})
for mode,changes in [("baseline_four_threads",{}),("compatible_four_threads",{"MKL_CBWR":"COMPATIBLE"}),("baseline_one_thread",{"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})]:
 env=dict(os.environ);env.update(changes)
 started=time.monotonic()
 command=[sys.executable,"-m","pytest","-q",*config["cpu_tests"],"--junitxml="+str(output/(mode+".xml"))]
 run=subprocess.run(command,cwd=release,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=160)
 (output/(mode+".log")).write_text(run.stdout)
 report["modes"][mode]=dict(returncode=run.returncode,elapsed_seconds=time.monotonic()-started,environment={key:env.get(key) for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","MKL_CBWR")},log=str(output/(mode+".log")))
 (output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
report["status"]="completed_kernel_diagnostic_only"
(output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
