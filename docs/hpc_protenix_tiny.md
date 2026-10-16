# DTU HPC: bounded native Protenix-Tiny example

Date: 2026-10-10. Status: **completed for this bounded example**. GPU job `29674532` finished successfully, all recorded model arrays were finite, and the small result bundle was downloaded and checksum-verified at `2026-10-10T11:32:47.6274266Z`.

The user authorized DTU HPC setup, one small Protenix-Tiny example, quota inspection, and download of results. This exercised Mosaic's existing native **protein** inference interface. It does not implement or validate the Stage 0 relaxed ligand graph, molecular gradients, or the complete Mosaic dependency/model collection. Results and retained limitations are recorded in [the experiment ledger](experiment_ledger.md).

## Source and paths

| Item | Recorded identity or path |
| --- | --- |
| Remote project | `/work3/s234818/mosaic` |
| Dedicated environment | `/work3/s234818/mosaic/envs/protenix-tiny` |
| Immutable read-only source release | `/work3/s234818/mosaic/releases/20261010-tiny` |
| Unique native example run | `/work3/s234818/mosaic/runs/20261010-native-tiny-01` |
| Source commit | `6e127d60e920f25bcd27ca5c7434cb3b2ac930c9` |
| Local source/receipt directory | `results/hpc/20261010-protenix-tiny/` (ignored by Git) |
| Downloaded run receipts/results | [20261010-native-tiny-01](../results/hpc/20261010-protenix-tiny/20261010-native-tiny-01/) |
| Source archive | `source.tar.gz`, 38,660,850 bytes |
| Archive SHA-256 | `d31d13da526a7ac58281ad7de6182fa451c95d313bd79be23909d67a19b83920` (verified remotely) |
| Per-file source identity | `source_manifest.json`, including current uncommitted contents |
| Archive contents list | `archive-files.txt` |

The archive contains current tracked files and the moved project PDF, HPC runbook, and example helpers. It is the execution-source identity; the commit alone is insufficient. These documentation receipts were written after that archive was created and are not represented as part of its immutable contents. Preserve the user's moved PDF and other existing edits.

## Environment receipt

Setup took place on the `linuxsh` interactive node `n-62-30-8`, under helper allocation `714306.hnode41`, rather than on a login node. The Python module is `python3/3.12.11`; `uv` is `0.10.0`.

The reduced environment contains 57 packages. Recorded key selections are:

| Component | Selection |
| --- | --- |
| JAX | `jax[cuda12-local]==0.11.1` |
| Equinox | `0.13.8` |
| NumPy | `2.5.3` |
| JAX Protenix implementation | `protenij` revision `97cf8172f61829681582f06de7052485bc94eae4` |
| Torch and unused model stacks | Not installed in this dedicated environment |
| CUDA module for the allocated GPU job | `cuda/12.9.1` |
| cuDNN module for the allocated GPU job | `cudnn/v9.13.0.50-prod-cuda-12.X` |

The `uv` installation completed with exit code `0`. Initial environment usage was 1,573,271,552 bytes. This is a selected native-Tiny environment, not a successful full `uv sync` of Mosaic. The operator reported these exact setup commands on the interactive allocation:

```bash
module load python3/3.12.11
export TMPDIR=/work3/s234818/mosaic/cache/tmp
export UV_CACHE_DIR=/work3/s234818/mosaic/cache/uv
uv venv --python /appl9/python/3.12.11/bin/python3 /work3/s234818/mosaic/envs/protenix-tiny
uv pip compile --python /appl9/python/3.12.11/bin/python3 --output-file /work3/s234818/mosaic/runs/requirements-tiny.lock - <<'REQUIREMENTS'
jax[cuda12-local]==0.11.1
equinox>=0.13.0
jaxtyping
requests
protenix @ git+https://github.com/escalante-bio/protenij@97cf8172f61829681582f06de7052485bc94eae4
REQUIREMENTS
uv pip install --python /work3/s234818/mosaic/envs/protenix-tiny/bin/python --link-mode hardlink -r /work3/s234818/mosaic/runs/requirements-tiny.lock
```

The compile command received the displayed requirements on stdin; the heredoc is a faithful shell rendering of that input. Exact setup commands, `requirements-tiny.lock`, `environment.freeze.txt`, installation/dry-run logs, and source identity files are present in the downloaded run directory. Reuse this resolved environment for the selected example; rebuilding from the floating input requirements can select different transitive versions.

Upstream Protenix declares `requires-python ==3.12`, which the initial `uv pip check` reported as incompatible with Python `3.12.11`. The retained correction changes `Requires-Python: ==3.12` to `==3.12.*` only in this dedicated environment's `protenix-0.1.0.dist-info/METADATA`, updating its corresponding `RECORD` entry. Model source is unmodified. The recorded metadata hashes are:

- Before: `674cfe92b81c4efda15b8d09fb7fdf3892269676a33789203980cd910ad21cfc`.
- After: `b63c36bb9b96e867f063b7fdfc2e0934cc25b370e618e1ab78930c4866037cbd`.

The subsequent `uv pip check` reports all 57 packages compatible. `fix_python_metadata.py`, `metadata_fix_receipt.json`, and both original/corrected package-check receipts are downloaded. The subsequent GPU example separately establishes the selected runtime path.

`bash -n` on the LSF script and `ast.parse` on the two Python helpers passed on the interactive CPU allocation. These were static source checks, not GPU/model execution. Helper allocation `714306.hnode41` was then closed by exiting the interactive shell.

## Bounded example and GPU script

[The example](../examples/hpc_protenix_tiny.py) runs one 20-residue sequence, `ACDEFGHIKLMNPQRSTVWY`, with seed `0`, no MSA, no template, one recycle, and two coordinate-sampling steps. It loads `protenix_tiny_default_v0.5.0` once. Weights remain fixed; there is no optimizer, parameter update, or gradient check.

This native forward includes coordinate sampling and confidence outputs. It is separate from the Stage 0 trunk-only graph path. Mosaic's existing sampler overrides are recorded by the example: `gamma0=0`, `step_scale_eta=1`, `noise_scale_lambda=1`, and model `N_steps=20`; the actual invocation requests `sampling_steps=2`.

[The LSF script](../scripts/hpc/protenix_tiny.lsf) requests queue `gpua10`, one exclusive GPU, four CPU slots on one host, `rusage[mem=4GB]` per slot, and 30 minutes. It loads Python/CUDA/cuDNN explicitly, uses the existing environment without installing packages in the job, requires the CUDA JAX platform, and logs hostname, modules, and GPU/driver identity.

The actual submitted command, retained in `submit-command.txt`, was:

```bash
bsub -J mosaic-tiny -q gpua10 -n 4 -R 'span[hosts=1]' -R 'rusage[mem=4GB]' -gpu 'num=1:mode=exclusive_process' -W 00:30 -oo /work3/s234818/mosaic/runs/20261010-native-tiny-01/stdout.%J.log -eo /work3/s234818/mosaic/runs/20261010-native-tiny-01/stderr.%J.log bash -l /work3/s234818/mosaic/releases/20261010-tiny/scripts/hpc/protenix_tiny.lsf /work3/s234818/mosaic/releases/20261010-tiny /work3/s234818/mosaic/runs/20261010-native-tiny-01
```

Resources were supplied on the `bsub` command line. `bash -l` initialized modules before running the script; the file's `#BSUB` comments were not the source of the submitted resource options. The script invokes the resident environment's Python with the release's `examples/hpc_protenix_tiny.py --output "$run/artifacts"`.

Submitted resources were `gpua10`, one exclusive GPU, four cores on one host, `rusage[mem=4GB]` per slot (16 GB total requested on that host), and 30 minutes. Log files are `stdout.29674532.log` and `stderr.29674532.log`. Before submission, `classstat gpua10` reported 0 used, 32 available, and 0 pending. Execution used node `n-62-18-5`, NVIDIA A10 (23,028 MiB), driver `615.71.09`, and JAX device `cuda:0`.

The script routes `MOSAIC_CACHE_DIR`, `HF_HOME`, `XDG_CACHE_HOME`, and `TMPDIR` beneath the project's scratch `cache/` directory. It sets `HF_HUB_OFFLINE=1`, disables Hugging Face telemetry, and uses four CPU threads. The example overrides native CCD/cluster asset paths before those data modules import them; the batch run must use already staged assets and must not download weights during inference.

## Asset staging and quota evidence

[The staging helper](../scripts/hpc/stage_tiny_assets.py) downloads assets sequentially from repository `nickrb/protenij`, pinned revision `81a257b61a82a17c41a7c20638460c04c29d97ad`. The local `expected_assets.json` records expected sizes, blob identities, and available LFS SHA-256 values.

| Expected asset | Bytes |
| --- | ---: |
| `clusters-by-entity-40.txt` | 9,388,032 |
| `components.v20240608.cif` | 427,767,290 |
| `components.v20240608.cif.rdkit_mol.pkl` | 125,903,802 |
| `protenix_tiny_default_v0.5.0.eqx` | 438,089,852 |
| `protenix_tiny_default_v0.5.0.skeleton.pkl` | 34,148 |
| Total expected download | 1,001,183,124 |

Staging completed with exit code `0`; all five assets were verified against expected sizes and available LFS hashes. The helper bounds each transfer by its declared size, records actual hashes, and retains an incomplete `.part` transfer rather than silently overwriting it or retrying indefinitely. The downloaded `asset_receipt.json` records every actual hash, including the non-LFS cluster file. The Tiny `.eqx` SHA-256 is `19f5f7af00cda104e8b059289e7e641cecc03309d7e960e6e53d82ad6e3f1f57`.

| Storage observation | Receipt |
| --- | --- |
| Before environment installation | Work-pool 6 usage `95.64 / 100 GiB`; home `27.52 / 30 G` |
| Before assets, `2026-10-10 11:23:07 UTC` | Work-pool 6 usage `97.44 / 100 GiB`; home `27.52 / 30 G` (cached quota) |
| Project usage before assets | 1,655,676,928 bytes |
| Before GPU submission, `2026-10-10 11:25:44 UTC` | Work-pool 6 usage `98.37 / 100 GiB`; cached home `27.52 / 30 G`; project usage 2,656,892,928 bytes |
| After GPU run, `2026-10-10 11:31:50 UTC` | Work-pool 6 usage `98.43 / 100 GiB` (1.57 GiB headroom); cached home `27.52 / 30 G`; project usage 2,707,586,560 bytes |

Raw storage receipts are downloaded. The post-job work-pool file count was 227,913 / 20,000,000. Shared `df` reported 510 T available, which does not increase this user's 100 GiB quota. Home quota is cached, refreshed about every 240 minutes, so it is not an instantaneous reading. This task removed no unrelated files and used no quota bypass. The user's separate HPC-cleanup agent operates outside this receipt.

## Operating and download practices

Follow [the supplied DTU runbook](HPC_HELP.tex): avoid compute on login nodes, prepare the environment once on an interactive node, run inference in an allocated GPU job, and keep large caches on available scratch. Scratch is not backed up. The runbook's old module examples do not override the recorded Python 3.12 requirement or current allocation evidence.

For this task, the main operator owned one remote SSH session. Batch independent inspections, use job-specific status queries with bounded backoff, and reuse verified assets/environment rather than reconnecting or reinstalling for each check. These are task operating practices, not claimed DTU traffic limits. Use `transfer.gbar.dtu.dk` for large transfers when appropriate.

The 202,193-byte result bundle was copied by SCP; safe archive paths and the sizes/hashes of all 30 run-manifest files were verified. [The download receipt](../results/hpc/20261010-protenix-tiny/download_receipt.json) records exact remote/local paths, time and command. Bundle SHA-256: `3e5b3820d68292afc9303562f977b405e3f46cb7f8be8e0adea91a0768b35be4`. Installed environment and model/data weights remain on HPC and were not downloaded. Local checks were file-integrity and receipt inspection only, with no local numerical evaluation.

## Completed result and limitations

Downloaded `artifacts/` contains FASTA, NPZ arrays, PDB/mmCIF, summary and checksum manifest. The summary reports `passed`, all recorded arrays finite, coordinates `[1,168,3]`, distogram `[20,20,64]`, pLDDT `[20]`, and PAE `[20,20]`; both structure formats passed 168-atom round trips. Model elapsed time was 218.214 seconds, including 170.814 seconds for forward/JIT compilation, 2.834 seconds for loading, and 21.429 seconds for featurization.

LSF reports `DONE`/“Successfully completed.” Its stdout accounting reports 685 CPU seconds, 3,668 MB maximum memory, 246 seconds run time and 248 seconds turnaround. `bacct -l 29674532` returned permission denied; no retry or permission change was attempted. `accounting-limitation.txt`, `lsf-job.txt` and stdout preserve the available accounting. `bjobs -l` also contains a truncated external GPU power diagnostic. The job finished successfully; no cause or hardware-health conclusion is drawn from the truncated message.

This arbitrary short sequence with two sampling steps is not a validated fold or binding result. The completed example establishes the selected native installation/inference path only; no molecular graph gradients or Stage 0 gate were tested.

## Reproduce with the resident environment

For a later authorized repetition, check quota and current queue/module availability, create a new empty run directory, and reuse the immutable release/environment/assets. Use the actual submission command above with **all three run-path occurrences** changed to that fresh directory, including stdout/stderr. The example refuses an existing `artifacts/` directory; do not overwrite the recorded run. Do not reinstall packages or download assets inside the GPU job. Preserve the new submission, source/config/model identities, logs, summary and quota receipts separately.

The completed small-bundle transfer command was:

```bash
scp -o BatchMode=yes -o ConnectTimeout=15 dtu-hpc:/work3/s234818/mosaic/runs/20261010-native-tiny-01.tar.gz results/hpc/20261010-protenix-tiny/20261010-native-tiny-01.tar.gz
```

For a new run, package only its outputs and receipts, change the remote/local bundle names accordingly, then verify its bundle and file manifest. Keep scratch-resident weights/environment rather than transferring them back.
