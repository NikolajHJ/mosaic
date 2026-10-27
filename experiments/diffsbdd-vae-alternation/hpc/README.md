# Runtime and allocation boundary

`bootstrap.sh` checks two **new**, independently authenticated runtimes. It does
not install into the existing VAE, Boltz or Mosaic environments. Runtime
provisioning has not run: storage admission failed before staging or submission.
The absent runtimes are not passed gates.

The combined runtime must inherit the frozen Boltz311 package layer without
altering it (Python 3.11, Torch 2.5.1+cu121, Boltz 2.2.1, RDKit 2024.9.6), adding
only compatible Torch Scatter 2.1.2 and OpenBabel wheel 3.1.1.22 in its own venv.
Keep its wheel resolution, hashes, package inventory and base-runtime inventory.

The reference runtime must independently load the original Lightning class with
Torch 2.0.1+cu118 and Lightning 1.8.4. The full inspected upstream environment is
in the public source archive's `environment.yaml` (Python 3.10.4). Provision its
dependencies in the new reference environment and record every version; no
version fallback is implicit. Its CUDA process exits before the combined worker
creates a CUDA context. Cross-runtime replay and same-runtime native replay are
separate required receipts.

Before provisioning, measure the runtime/asset footprint and include any amount
beyond the initial 2 GiB artifact allowance in a new immutable capacity plan.
Preserve the additional 10 GiB unreserved headroom and every other campaign's
forecast. Do not install first and discover an overrun afterward. Provisioning,
all import/model checks and every fixture run belong in allocated CPU/GPU jobs.

CPU stages use four slots and at most 45 minutes. GPU stages use one exclusive
80 GB A100 and four slots, with at most 24 hours per allocation. The operator
authenticates all source/config/predecessor bytes and owns the internal timeout.
No stage submits a successor or retries an ambiguous launch.
