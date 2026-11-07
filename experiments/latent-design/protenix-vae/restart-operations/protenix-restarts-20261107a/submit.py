import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

here = Path(__file__).resolve().parent
helper = r"C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe"
base = "/work3/s234818/latentmol"
name = here.name
control = f"/zhome/95/e/206121/dbmol-control/{name}"
release = f"{base}/releases/{name}"
archive = release + ".tar.gz"
experiment = Path("experiments/latent-design/protenix-vae")
wrapper = Path("scripts/hpc/protenix_latent_run.sh")
plan_name = experiment / "restarts-gpu-plan.json"
arm_configs = json.loads((experiment / "restarts.json").read_text())["arms"]
outputs = {arm: f"{base}/runs/{name}-{arm}" for arm in arm_configs}
torch_python = f"{base}/env/boltz311/bin/python"
assets_root = f"{base}/protenix_occupancy"
fresh_deps = f"{assets_root}/environment-attempt003/deps"
fresh_data = f"{assets_root}/data"
previous = Path("outputs/flow-lr-20261106a/gpu-plan.json")
previous_plan = json.loads(previous.read_text())
plan = {key: previous_plan[key] for key in
        ("cuda_ownership_helper", "cuda_ownership_sha256", "mosaic_python")}
plan["arms"] = outputs
plan["gpu_profile"] = "v100-32gb-v1"
plan["output_cap_mib"] = 768
launcher = Path("scripts/hpc/hpc_launch.py")
storage = Path(r"C:\Users\nikol\OneDrive\Dokumenter\GitHub\mosaic\scripts\hpc\storage_accounting.py")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(file, data):
    with (here / file).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def remote(script, timeout=55):
    result = subprocess.run(
        [helper, "exec", "--timeout", str(timeout), "bash -lc " + shlex.quote("/usr/bin/python3 -I -")],
        input=script, text=True, capture_output=True, timeout=timeout + 10, check=True)
    return json.loads(result.stdout)


def upload(source, target):
    subprocess.run([helper, "upload", "--timeout", "90", str(source), target], timeout=100, check=True)


def main():
    save("gpu-plan.json", plan)
    package = here / "source.tar.gz"
    files = [experiment / filename for filename in
             ("restarts.json", "restarts-fresh-dtu.json", "native-dtu.json", "one-atom-dtu.json")]
    files.append(wrapper)
    with tarfile.open(package, "x:gz") as bundle:
        for path in [*sorted(Path("src/latentmol").rglob("*.py")), *files]:
            if path == wrapper and b"\r\n" in path.read_bytes():
                raise ValueError("Unexpected CRLF in GPU launch wrapper")
            bundle.add(path, arcname=path.as_posix())
        bundle.add(here / "gpu-plan.json", arcname=plan_name.as_posix())
    native = json.loads((experiment / "native-dtu.json").read_text())
    fresh = json.loads((experiment / "one-atom-dtu.json").read_text())
    inputs = {fresh["checkpoint"]: fresh["checkpoint_sha256"],
              fresh["native"]["weights_path"]: fresh["native"]["weights_sha256"],
              native["context_ready"]: native["context_ready_sha256"],
              native["pocket_path"]: native["pocket_sha256"],
              plan["cuda_ownership_helper"]: plan["cuda_ownership_sha256"]}
    required = [native["mosaic_source"], plan["mosaic_python"], fresh["native"]["source_root"],
                torch_python, fresh_deps, fresh_data]
    probe = f"{base}/runs/{name}-capacity-probe"
    save("preflight.json", remote(f"""
import hashlib, json
from pathlib import Path
for path in map(Path, {list(outputs.values()) + [archive, release, control, probe]!r}):
    if path.exists():
        raise FileExistsError(path)
for name, expected in {inputs!r}.items():
    digest = hashlib.sha256()
    with Path(name).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('Changed input: ' + name)
for name in {required!r}:
    if not Path(name).exists():
        raise FileNotFoundError(name)
Path({control!r}).mkdir()
for arm in {list(outputs)!r}:
    (Path({control!r}) / arm).mkdir()
print(json.dumps(dict(inputs={inputs!r}, outputs={outputs!r})))
"""))
    remote_launcher = f"{control}/hpc_launch.py"
    remote_storage = f"{control}/storage_accounting.py"
    uploads = [(package, archive), (launcher, remote_launcher), (storage, remote_storage)]
    for source, destination in uploads:
        upload(source, destination)
    save("storage.json", remote(f"""
import hashlib, subprocess, tarfile
from pathlib import Path
for name, expected in {[(destination, sha(source)) for source, destination in uploads]!r}:
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise ValueError('Upload checksum differs: ' + name)
with tarfile.open({archive!r}) as bundle:
    subprocess.run(['bash', '-n'], input=bundle.extractfile({wrapper.as_posix()!r}).read(), check=True)
probe = Path({probe!r})
probe.mkdir()
try:
    result = subprocess.run(['/usr/bin/python3', {remote_storage!r}, '--output-root', str(probe),
        '--cap-mib', '768', '--file-cap', '6000'], check=True, text=True, capture_output=True)
    print(result.stdout)
finally:
    probe.rmdir()
"""))
    for arm, output in outputs.items():
        receipts = f"{control}/{arm}"
        argv = ["bsub", "-J", f"dbmol-protenix-restarts-{arm}-1107a", "-q", "gpuv100",
                "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=4096]",
                "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G",
                "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                "-W", "08:00", "-oo", f"{receipts}/run.%J.out", "-eo", f"{receipts}/run.%J.err",
                "/usr/bin/python3", remote_launcher, "--archive", archive,
                "--archive-sha256", sha(package), "--release", release,
                "--receipt", f"{receipts}/launch.json", "--startup-seconds", "900", "--run-seconds", "28500",
                "--", "bash", wrapper.as_posix(), torch_python, "-m", "latentmol.design.protenix_restarts",
                "--config", (experiment / "restarts.json").as_posix(), "--arm", arm,
                "--output", output, "--checkpoint", fresh["checkpoint"],
                "--checkpoint-sha256", fresh["checkpoint_sha256"], "--mosaic-python", plan["mosaic_python"],
                "--native-config", (experiment / "native-dtu.json").as_posix(),
                "--fresh-config", (experiment / "restarts-fresh-dtu.json").as_posix(),
                "--fresh-deps", fresh_deps, "--fresh-data", fresh_data,
                "--plan", plan_name.as_posix(), "--plan-sha256", sha(here / "gpu-plan.json")]
        receipt = remote(f"""
import json, re, subprocess
from pathlib import Path
root = Path({receipts!r})
argv = {argv!r}
with (root / 'submit-intent.json').open('x') as stream:
    json.dump(argv, stream)
result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
job = re.search(r'Job <([0-9]+)>', result.stdout)
receipt = dict(job_id=int(job[1]) if job else None, returncode=result.returncode,
               stdout=result.stdout, stderr=result.stderr, argv=argv, output={output!r})
with (root / 'submission.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
""")
        save(f"{arm}-submission.json", receipt)
        print(arm, receipt["stdout"], receipt["stderr"], flush=True)
        if receipt["returncode"] or receipt["job_id"] is None:
            raise RuntimeError(f"Scheduler did not confirm {arm}")


if __name__ == "__main__":
    main()
