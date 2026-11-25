# Compute and Execution — DTU HPC

> **Current connection route, 29 October 2026:** agents on Nikolajs_laptop must
> use the installed unattended connector described in
> [HPC_CONNECTION.md](HPC_CONNECTION.md) for new local commands and transfers.
> This supersedes older local SSH/VPN recipes for that workstation. It does not
> change scientific execution, allocation, permission or provenance requirements.

> **Workload guidance superseded (2026-09-26).** The active work is graph AE
> -> matched graph VAE -> frozen-Boltz2 latent optimization; all tests, training
> and experiments run on DTU HPC. See the
> [`latent pipeline`](04_latent_pipeline.md) and
> [`experiment ledger`](05_experiment_ledger.md). Recommendations below to train
> locally, avoid large-model resources, or use small-GNN sizing are historical
> and must be reassessed for the current workload. The standing authorization
> immediately below is preserved.

> **Current submission audit, 17 October 2026:** the current A100 workloads
> and their exact requests are recorded in the [experiment ledger](05_experiment_ledger.md),
> [long comparison](60_long_boltz_defog_comparison.md) and
> [chemistry speed suite](61_chemistry_execution_selection.md), rather than the
> historical KIBA sizing below. Read-only LSF/host inspection found the submitted
> requests structurally compatible with the eligible host; both GPU jobs still
> await resources. Live `gpua100` advertises a72h maximum, while the public DTU
> GPU page still says24h: preserve that discrepancy and recheck live policy
> before future submissions. Accepted syntax is not a promise of allocation.
> The historical `HPC_info/HPC_HELP.tex` companion is absent from the current
> checkout; its missing link below is not a current source of verified policy.

Companion to [`02_implementation_plan.md`](02_implementation_plan.md).
Derived from [`HPC_info/HPC_HELP.tex`](../HPC_info/HPC_HELP.tex).
Rewritten 2026-09-24 for the current workload.

## Standing authorization to overwrite HPC data

On 2026-09-26, the user explicitly authorized the agent to overwrite all data owned
by their DTU HPC account, including all contents of the remote home directory
(`$HOME`) and their personal `/work3/$USER` directory, when carrying out requested
HPC work. No additional confirmation is required for overwrites within this scope.
This authorization covers the user's own data; it does not extend to other users'
data or shared/system directories, including the shared `/work3` root.

## Read this first: you probably do not need the cluster

The workload is a **3–4 layer dense GNN over ≤ 2,068 unique molecules and ~117k
drug-target pairs**, trained once. On any consumer GPU this is minutes to about an hour.
CPU-only is viable if slow.

**Default: develop and train locally.** Use the cluster for three things only:

1. You have no local GPU.
2. You want a GPU node for interactive work without occupying your laptop.
3. Later, when the optimization study (protocol §10) fans out over
   methods × starting molecules × targets — and even then the jobs are small.

Do not spend time on HPC engineering at this stage. Time spent on job arrays is time not
spent on the freeze gate.

> **Historical note.** A previous version of this document sized A100-80GB job arrays for
> Boltz-2 backward passes at ~30 s each. That workload no longer exists. If you find
> references to MSA caches, CCD pickles, DeFoG checkpoints or 147-element arrays
> anywhere in the repo, they are stale.

---

## 1. Hardware, and what is actually relevant

| Queue | Hardware | Relevance here |
|---|---|---|
| `gpuv100` (32 GB / 16 GB) | ~34 GPUs | **Primary queue.** Far more than enough, and it schedules sooner than the big-GPU queues |
| `gpua10` (24 GB) | 2 GPUs | Fine. Good for smoke tests and short interactive sessions |
| `hpc` (CPU, up to 1536 GB RAM, 72 h walltime) | — | KIBA download, RDKit featurization, cache building, analysis. **Most of Step 2 belongs here** |
| `gpua100` (40/80 GB), `gpul40s`, `gpua40` | 32 GPUs total | **Not needed.** Requesting them means queueing behind real large-model jobs for no benefit |

Storage: `zhome` 30 GB (backed up), `/work3/$USER` scratch (not backed up).
**`zhome` alone is sufficient for this project** — see §3.

Check `classstat` and `nodestat -F <queue>` before submitting. The fastest queue is
whichever is idle.

---

## 2. Memory sizing

Nothing here is memory-constrained, but knowing the formula means you can pick a batch
size without guessing.

**The driver is the dense edge tensor**, `[B, N, N, H]`, one per message-passing layer:

```
bytes ≈ B · N² · H · 4        (float32)
```

At `N = 64`, `H = 128`, `B = 64`: `64 · 4096 · 128 · 4` ≈ **134 MB** per stored tensor.
With 3–4 layers and autograd retaining a few tensors each, expect **~1–3 GB**. A 16 GB
V100 is comfortable at `B = 256`.

Batch size is the only dial you need. If you hit OOM, halve `B` — do not reach for a
bigger GPU, and do not reach for gradient checkpointing.

The raw inputs are negligible: at `N = 64` with `K_atom ≈ 12`, one molecule as float32
one-hot is `X` 3 KB + `E` 80 KB + `Q`/`M` < 1 KB ≈ **84 KB**.

---

## 3. Storage layout

Total project footprint is **well under 1 GB**, so `zhome` is enough and you get backups
for free. Use `/work3` only for tool caches you do not want counting against quota.

```
$HOME/relaxed-graph/           # code, configs, cache, checkpoints, reports — in git
├── data/
│   ├── tdc/                   # TDC's KIBA download (see §4)
│   └── cache/
│       ├── graphs.pt          # ~9 MB as int8, ~174 MB as float32 one-hot
│       └── pairs.parquet      # ~117k rows, a few MB
├── checkpoints/               # kiba_predictor_v1.pt — small
└── reports/
```

### Store `E` as int8 category indices, not one-hot

`graphs.pt` at `N = 64`:

| Encoding | Per molecule | 2,068 molecules |
|---|---|---|
| float32 one-hot | ~84 KB | ~174 MB |
| int8 category indices | ~4.3 KB | **~9 MB** |

A 20× saving, and it loads faster. Expand to one-hot in the `collate_fn`.

> **This does not violate the single-path design invariant.** The invariant forbids
> index lookups *inside the forward pass*, where gradients flow. Expanding stored indices
> to one-hot in the dataloader happens before the forward pass, on constant input data,
> and no gradient passes through it. The forward pass still sees only float tensors and
> still contains no `argmax`, no `.long()`, and no `nn.Embedding`.

If you do use `/work3`, redirect caches in `~/.bashrc`:

```sh
export WORK=/work3/$USER/relaxed-graph
export HF_HOME="$WORK/cache/huggingface"
export TORCH_HOME="$WORK/cache/torch"
export WANDB_DIR="$WORK/cache/wandb"
export UV_CACHE_DIR="$WORK/cache/uv"
```

Quotas: `getquota_zhome.sh`, `getquota_work3.sh`.

---

## 4. TDC downloads — the one real gotcha

**Compute nodes typically have no outbound internet.** TDC fetches KIBA from a remote
host on first use, so a job that calls `DTI(name='KIBA')` for the first time inside an
LSF job will hang or fail with an opaque network error.

Download once on the **login node**, then point every job at that path:

```python
from tdc.multi_pred import DTI
data = DTI(name='KIBA', path='/zhome/.../relaxed-graph/data/tdc')   # login node, once
```

Then in jobs, pass the same `path=` and never re-download. Better still: make Step 2 write
`graphs.pt` and `pairs.parquet` once, and have training depend only on those. Training
should never touch TDC or the network at all.

Verify before submitting anything: build the cache, then run training with networking
assumed absent.

---

## 5. Submit scripts

### Training (Step 4) — a single job, not an array

```sh
#!/bin/sh
#BSUB -J kiba-train
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 04:00

cd "${LS_SUBCWD:-$PWD}"
mkdir -p logs

. .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=4

python -m src.train --config configs/v1.yaml --out checkpoints/
```

- `mkdir -p logs` **before** submitting — LSF will not create the log directory and the
  job fails silently otherwise.
- `rusage[mem=4GB]` is **per core**, so `-n 4` requests ~16 GB host RAM, not VRAM.
- 4 h walltime is generous. If training is anywhere near that, something is wrong —
  check that you are not re-featurizing SMILES every epoch.

Checkpoint per epoch with the config, git commit hash and RNG seed embedded. No
mid-epoch resume machinery is warranted for a job this short.

### Later: the optimization study (protocol §10)

**The scheduling advice here is the opposite of the old document's, and deliberately so.**
One optimization trajectory is 10–30 gradient steps through a small GNN — *seconds*. Job
startup, module load and environment activation dominate entirely.

So: **bundle aggressively.** One array element should run a few hundred trajectories.

```sh
#BSUB -J relaxopt[1-20]
#BSUB -q gpuv100
#BSUB -W 01:00
...
python -m src.optimize --manifest configs/opt_manifest.csv \
                       --shard "$LSB_JOBINDEX" --shards 20
```

Commit `configs/opt_manifest.csv` (method, starting molecule, target, seed, config hash)
so an array index is a reproducible pointer to an exact experiment. Because trajectories
are cheap, run **many seeds and many starting molecules** — the statistical power is free
here in a way it was not in the superseded plan.

---

## 6. Windows → Linux gotchas

You develop on Windows. All of these still apply.

1. **Line endings.** `.gitattributes` is `* text=auto`, so shell and LSF scripts checked
   out on Windows get CRLF and `sh` fails with opaque errors. Add before writing any
   `.sh`:
   ```
   *.sh  text eol=lf
   *.job text eol=lf
   ```
2. **Never copy a Windows `.venv` to the cluster.** Build the virtualenv on the cluster
   from a cluster Python module. `uv` works well.
3. **RDKit is the fragile dependency**, not PyTorch. Install it from conda-forge or a
   wheel known to work on the cluster's glibc, and pin the version — featurization output
   must be reproducible across machines, or your cache and your laptop disagree silently.
4. **Absolute paths for persistent state.** Job working directories are not where you
   expect; use `$HOME`/`$WORK` explicitly.
5. **Unbuffered logs** — `export PYTHONUNBUFFERED=1`, or `bpeek` shows nothing until the
   job ends and you cannot distinguish a hung job from a slow one.
6. **Determinism across platforms.** Set seeds for python, numpy and torch, and record
   them in the checkpoint. Expect small float differences between your laptop GPU and a
   V100; the gate checks should tolerate that, and if a check flips between machines it
   was too tight.

---

## 7. Pre-launch checklist

- [ ] KIBA downloaded on the **login node**; `data/cache/` built and committed-by-hash
- [ ] Training reads only the cache — verified with no network access
- [ ] `logs/` exists
- [ ] `tests/test_invariants.py` passes on the cluster, not just locally
- [ ] Seeds, config, git commit hash written into the checkpoint
- [ ] RDKit version pinned and recorded alongside the featurization cache
- [ ] Short smoke run (`-W 00:20`, a few hundred steps) through the exact entrypoint

---

## 8. What this project does *not* need

Listed explicitly because the superseded plan required all of it, and because each is a
plausible-looking way to lose a week:

- A100 or L40S GPUs, or any `select[gpu80gb]` resource request
- Gradient checkpointing, model parallelism, or mixed precision
- Multi-GPU or multi-node anything
- MSA generation or caching, protein structure files, CCD dictionaries
- Large model weight downloads
- Per-step checkpoint/resume with atomic renames
- `/work3` scratch for data (caches only, and even that is optional)
- Optuna or any hyperparameter sweep infrastructure — the protocol forbids sweeps
- Training-acceleration frameworks (MosaicML Composer, DeepSpeed, Lightning and the like).
  Their value is in long or repeated runs; this one takes minutes, their speedup catalogues
  are aimed at vision and transformer workloads, and adopting one means replacing the
  DIDgen training loop that protocol §3 tells you to reuse. They also default to mixed
  precision, which degrades the gradient quantities the acceptance gate measures — see
  [`02_implementation_plan.md`](02_implementation_plan.md) Step 5
