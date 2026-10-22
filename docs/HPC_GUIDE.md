# DTU HPC setup for your own account

Reviewed 22 October 2026 against repository source and the linked official
documentation. Use your identity and authorized storage. This review does not
establish connectivity, remote permissions, capacity or numerical acceptance.

## Your SSH identity and remote storage

Replace both uppercase placeholders in your local SSH configuration:

```sshconfig
Host dtu-hpc
    HostName login1.hpc.dtu.dk
    User YOUR_DTU_ID
    IdentityFile ~/.ssh/YOUR_DTU_KEY
    IdentitiesOnly yes
    ServerAliveInterval 60
```

Use the access method supported for your account/network. DTU documents login
servers and SSH-key access without VPN, as well as a VPN route. Complete your
authentication and verify host keys. See [DTU access](https://www.hpc.dtu.dk/?page_id=2501)
and [DTU VPN](https://www.cc.dtu.dk/?page_id=5685).

`ssh dtu-hpc` uses the alias **you** configured. A Windows login name need not
equal the DTU account. Run these commands **inside the remote Linux shell**:

```bash
id -un
printf '%s\n' "$HOME"
getquota_work3.sh
getquota_zhome.sh
```

A conventional personal scratch location is `/work3/<your remote username>`.
Confirm it exists and is writable, or choose separately authorized group storage.
Another user's directory is not a shared installation because a document names
it. DTU describes scratch as provisioned storage without backup; inspect actual
quota instead of assuming a fixed allocation. Large transfers use its dedicated
transfer service. [DTU storage and quotas](https://www.hpc.dtu.dk/?page_id=927).

These **remote-shell conventions for new work** do not automatically override
the constants in existing launchers:

```bash
export MOSAIC_WORK_ROOT="/work3/$(id -un)"
test -d "$MOSAIC_WORK_ROOT"
test -w "$MOSAIC_WORK_ROOT"
export MOSAIC_PROJECT_ROOT="$MOSAIC_WORK_ROOT/mosaic"
export MOSAIC_RUN_ROOT="$MOSAIC_PROJECT_ROOT/runs"
export MOSAIC_HOME_RUN_ROOT="$HOME/mosaic-runs"
export MOSAIC_PYTHON="$MOSAIC_PROJECT_ROOT/envs/protenix-tiny/bin/python"
export MOSAIC_STAGE0_CACHE="$MOSAIC_PROJECT_ROOT/cache/mosaic/protenix"
```

Choose an authorized alternative for `MOSAIC_WORK_ROOT` where appropriate.
These are intended locations: setting variables does not create an environment,
install weights or grant access. Use your own tested runtime and authorized
copies of the pinned assets. Resolve home with `$HOME`, not someone else's
physical `/zhome/...` address.

## Actual repository portability

| Component | Interface / limitation |
| --- | --- |
| Source packager | `scripts/releases/package_stage0_release.py --root ... --output ...`; local stdlib packaging |
| Source verifier | `scripts/releases/verify_stage0_release.py SOURCE_ROOT MANIFEST_SHA256`; independently recorded hash |
| Stage 0 driver | `experiments/graph-validation/stage0/scripts/small_molecule_stage0.py --config ... --cache-dir ... --output ...`; explicit cache/output, native prerequisites remain |
| Work quota check | `scripts/hpc/stage0_quota_preflight.py --run-dir ... --storage-pool ...`; actual pool required |
| Campaign controllers and many LSF scripts | Account-bound roots, runtimes, worker locations and proof identities remain in code |

The review found **104 canonical source files** with concrete account references;
see the [source inventory](documentation-audit.json). Some are historical input
bindings, some are overridable defaults, and others are controller invariants.
Exporting `MOSAIC_PROJECT_ROOT` does **not** make them portable. Before running
under another account, prepare a separate reviewed candidate with its actual
paths, config/asset/proof bindings and required validation. Preserve current
sealed releases; never replace account IDs through an archived source tree or
reuse an old certificate as proof of new paths.

Cloning Git does not supply ignored `results/hpc/`, release bundles, resident
weights or private evidence. Obtain the required inputs from authorized storage
and authenticate them. A historical path is evidence, not a public download.
The old sibling-project Boltz worker location likewise requires your own
validated worker or an explicitly shared installation.

The working `scripts/` and `tests/` roots contain only navigation/collection infrastructure. Use canonical
paths from their indexes. Old flat script/test names are stored under
[scripts/compat/](../scripts/compat/README.md) and emitted only inside a new
authenticated release for legacy launcher compatibility. Default pytest discovery
uses nested canonical tests once; do not combine them with compatibility aliases.

## Runtime, allocations and evidence

`pyproject.toml` requires **Python 3.12.x** (`==3.12.*`). Discover available
modules on the actual system; the old Python 3.10 recipe is unsuitable for this
project. Record dedicated worker interpreters separately where applicable.

The root README's general install command is `uv sync --group jax-cuda`.
Focused studies use their own pinned environment and asset contracts. Git
ignores `uv.lock`, so a fresh clone does not promise a lockfile for
`uv sync --frozen`; retain the actual resolved lock/export used by your attempt.
Do not reuse a Windows virtual environment on Linux or modify another user's
resident runtime to make an example work.

Login nodes are access/submission points. `linuxsh` provides a shared application
shell; it does not establish the LSF allocation/GPU that Mosaic's numerical
guards require. Run model imports, preprocessing, numerical tests and scientific
rendering in authorized allocations. See the [DTU access guide](https://www.hpc.dtu.dk/?page_id=2501).

Inspect actual `module avail`, `bqueues -l` and relevant node/resource information.
Old queue names, selectors, module versions and capacities are not current
guarantees. The old access page mentions Scientific Linux, while DTU's
[migration notice](https://www.hpc.dtu.dk/?page_id=5209) records the move to
AlmaLinux. Check the actual host rather than inferring its OS from copied text.

Follow the [owning experiment README](../experiments/INDEX.md), authenticate
fresh source, inputs and quota, use a unique output directory, and record exact
commands, jobs and outcomes in the [logbook](experiment_ledger.md). Scheduler
completion and scientific acceptance are separate. General queue limits do not
override an experiment's declared allocation/runtime bounds.

The [Stage 0 usage guide](stage_0_usage.md) has explicit driver commands. Its
convenience LSF launcher still hard-codes an operator's runtime root; use the
documented manual route or a separately reviewed port. This guide authorizes
no new run, retry, cleanup or change to historical evidence.
