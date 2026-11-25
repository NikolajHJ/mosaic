# Full-data decoder comparison with fixed valid-repair targets

Status, 8 October 2026: the user authorized implementing and submitting a
full-data comparison of the matched control against search-target guidance.
The protocol/controller/HPC scripts are implemented. The user confirmed the
matched-control versus search-guided comparison; search coefficient **one**
is the prespecified default inherited from the pilot's high-weight arm, not
a separately user-tuned value. **Source075 attempt001 is uploaded,
authenticated and extracted on HPC** with the unchanged archive/manifest/
configuration identities below. CPU **29633218 DONE** passed **126 checks,
two CUDA-only skips**. Native seed 17 **29633350 PEND** and seed 23 **29633351
PEND** at **09:36 DTU-local**. Native CUDA preflight, full training and
search-guided endpoint evaluations have **not run**. HPC access is restored;
no VPN/key changes were made. Every numerical test, search, training run and
evaluation remains DTU HPC work. Local verification was metadata/file hashing.

The [archive](../outputs/source-releases/source-075-search-full-attempt001.tar.gz)
SHA-256 is `4dea426a9c999edb332df7c4d834efcba17eac193af87c193ba77a1f50cd9fcd`;
scientific manifest is
`9acbac0026c5cbcee0265ee07cc6541b0d5474dc30683656183603c8cf7cd4e3`;
full configuration is
`d802424d0e2e19c10348f8beb589f5bfc8269e34bc4ddc1e4933b3abaa36d591`.
The historical [ready receipt](../outputs/source-releases/source-075-search-full-attempt001.ready.json)
retains the pre-upload snapshot and exact commands, subsequently executed
from the authenticated release. Current submission evidence is in
[engineering-submission-evidence001](../outputs/hpc-results/source075-search-full/engineering-submission-evidence001.tar.gz),
SHA-256 `8ad28770f9c307ad948af0f7321322e2e32f5ab396673e2a18dbf2c82968281f`.
All 17 data-file hashes in 18 members were verified locally. Remote release
and receipt roots are authenticated; the output root is declared pending
native execution:
`/work3/s234818/latentmol/releases/source-075-search-full-attempt001`,
`/work3/s234818/latentmol/submissions/source075-search-full-attempt001`,
`/work3/s234818/latentmol/runs/source-075-search-full-attempt001-search-full`.
The [actual CPU run/environment](../outputs/hpc-results/source075-search-full/extracted-evidence001/cpu/run.json),
[test stdout](../outputs/hpc-results/source075-search-full/extracted-evidence001/cpu/command.log),
[seed17 receipt](../outputs/hpc-results/source075-search-full/extracted-evidence001/receipts/seed17-receipt.json),
[seed23 receipt](../outputs/hpc-results/source075-search-full/extracted-evidence001/receipts/seed23-receipt.json)
and [09:36 snapshot](../outputs/hpc-results/source075-search-full/extracted-evidence001/scheduler_snapshot.json)
are downloaded. These docs-only receipt notes were added after sealing; the
archive retains its earlier documentation snapshot and the executable
manifest remains unchanged. Existing seed17 local-defect results are separate
from this search-target experiment.

The ongoing [local-defect recovery](41_decoder_budget_recovery.md) and
[focused search pilot](40_search_repair_gradient_study.md) retain their
protocols, checkpoints, jobs and history. This new study does not mutate them
or reuse their control as a matched full-search control.

## Question and comparison

Does training the existing decoder toward detached, verified bounded repair
witnesses improve raw generation on independent random latent points, while
retaining reconstruction and diversity? The fixed-target derivative has been
verified for one frozen FP32 CPU point in the source072 diagnostic. That result
does not establish training efficacy, CUDA behavior or population-wide
gradient accuracy. The full-data experiment must establish its own engineering
gates and independent endpoint results.

The original stable source047 epoch-89 VAE remains an untouched evaluation
reference. Both trained methods start from its identical weights, freeze the
encoder parameters and buffers, and train only the decoder. The protected
initializer SHA-256 is
`1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088`;
its native path, dataset and split identities are documented in
[docs39](39_decoder_chemical_distance_training.md). The training cohort is
the same **215,000 graphs**, with **7,347 held-out test graphs**. Neither a
partially fine-tuned decoder nor a selected successful subset replaces it.

The fixed grid in [search_decoder_full.json](../configs/search_decoder_full.json)
is seeds **17 and 23**, each with two methods and **ten
completed epochs** per method:

| Method | Reconstruction and shared anchors | Search-target coefficient |
| --- | --- | --- |
| Matched control | Same original reconstruction and fresh-prior/search-pool occupancy and aromatic anchors | 0 |
| Search-guided decoder | Identical reconstruction and anchors | 1 |

Each method starts a new decoder-only AdamW, learning rate 3e-5, weight decay
1e-5 and gradient cap five. Reconstruction batch size is 64; posterior and
fresh-prior defect diagnostics use 16 points each with coefficient zero in
both methods. Ten epochs give **33,600 updates per method** on all 215,000
training graphs. Occupancy and aromatic anchors each have weight one.
Validation covers 256 reconstruction graphs and 256 prior points. The
protected dataset SHA-256 is
`560840389f7d85592ef8e16a5566e26b5ab3e5dfa30ebe392366dbf54489974f`,
with split SHA-256
`e15daf547436b66cbbd54ae82f53f153d54e2bd24edee9bf2c5787099cbe6e79`.

The control performs the same bounded search and scheduled target decoding,
even when its search coefficient is zero. All methods share reconstruction
batch order, posterior/prior RNG streams, the same fixed training latent pool,
pool schedule and search quotas. Realized checks, successful targets and wall
time may differ as decoders diverge; those differences must be reported rather
than hidden as equal compute.

The source073 control lacks the additional shared search-pool anchors. It is
useful historical evidence but **cannot replace this matched control**. The
full comparison is original versus new matched control versus new search arm,
within each training seed. No DeFoG, Boltz, new learned critic or activity
surrogate is required for this narrow decoder experiment. A selected decoder
must be frozen before later molecular latent optimization.

## Discrete search and differentiable target loss

The implementation reuses the fixed-target contract in
[docs40](40_search_repair_gradient_study.md). A prescribed **256-point
standard-normal training pool** is decoded and searched at each epoch
boundary, using that arm's current decoder. Search, argmax, RDKit validation,
repair selection and receipt/cache handling stay outside autograd. Each
accepted witness is a detached, slot-aligned target held fixed throughout
the corresponding epoch and every derivative probe of that objective.

Search preserves the raw occupied slots within each raw-to-witness repair;
occupancy can change at the next epoch refresh. Its fixed bounded policy
allows up to three edits, 256 candidate checks per depth and beam width four,
with probability-weighted cost
`edit_count + sum(max(0, log(p_old / p_new)))`. The disclosed probability floor
belongs to search ranking only. Keep the best observed valid weighted witness,
not a claimed globally nearest valid molecule. Already-valid points are
recorded without repair guidance. Censored/unsupported points receive no
invented target, stale target or successful-only replacement draw.

Target NLL follows the decoder's unconditioned categorical law: all 50 atom
decisions including NO_ATOM, charges on occupied slots, and unordered bonds
between occupied slots including NONE. Normalize each target by its declared
number of categorical decisions, then average over the fixed scheduled batch
including zeros for unavailable targets. Live stable log-softmax has no
probability floor. Decoder gradients increase support for the fixed witness;
they do not differentiate the search or an exact chemical distance.

Original reconstruction and identical occupancy/aromatic reference anchors
apply to both arms. The anchors cover **fresh prior points and the fitted
training pool**. No success-only loss normalization or changing search quota
may favor the search arm. The checked-in configuration pins loss coefficients,
optimizer settings, search limits and schedules; its digest is authenticated
by the metadata reader and submission helper. Verified release/config digests
are recorded above. The pool uses seed 71000 and cycles
eight of its 256 points per update; every epoch refresh searches the full pool.

## Independent evaluation and conclusions

The fitted pool has only 256 prescribed latent points. An improvement there
can be memorization or easier target coverage. Evaluate it separately from
an independent **4,096-point unfiltered final prior**, with argmax and **four
categorical draws per point**, and full held-out deterministic-mean
reconstruction on 7,347 molecules. Use identical final points/draws across
original, control and search decoders, preserving invalids and duplicates in
the denominators. Final generation uses seed 91000, sampling seed 91001;
the fitted pool uses sampling seed 71003. The first 64 final argmax outputs
receive the separately bounded repair audit. Final evaluation must not select
the best test checkpoint or silently substitute repaired molecules for raw outputs.

Report raw argmax and sampled validity, unique-valid yield per attempt,
novelty versus the full training split, count/element/bond concentration,
reconstruction validity and exactness, paired latent-cluster confidence
intervals, and the separately bounded repair audit. Report search coverage,
weighted edits, target NLL and actual oracle costs, but do not equate those
diagnostics with improved raw generation. Any advantage claim must compare
against both untouched original and the new matched control, with diversity
and reconstruction retention. Report each training seed's paired latent-cluster
confidence intervals separately. The common 4,096-point panel does **not**
become 8,192 independent points by evaluating two trained seeds. Two training
seeds remain limited replication; binding, synthesizability and guarantees
of chemical validity are outside the measured scope.

## HPC execution and budget plan

The metadata-only [protocol reader](../src/latentmol/search_decoder_protocol.py)
authenticates the full configuration separately from the historical pilot's
unchanged parser. The [full controller](../src/latentmol/search_decoder_full.py)
reuses the search engine with explicit full-data/native-gradient requirements,
larger evaluation reserve and full-study reporting scope. Completing a
segment requires two authenticated epoch 10 checkpoints and completed
evaluations; the audit checks initializer/data/split/protocol identity, source,
frozen encoder, shared latent pools and dedicated terminal RNG states. Missing
or mismatched artifacts prevent a complete aggregate result.

The [submission helper](../scripts/hpc/submit_search_decoder_full.py) first
plans/submits the [CPU regression stage](../scripts/hpc/search_decoder_full_tests.sh):
queue `hpc`, two slots, 4 GB per slot, one host, 45-minute limit. This runs
the historical search fixtures plus full-protocol/completion/submission
fixtures. Actual CPU results were 88 passed/one CUDA-only skipped in 12.81 s,
21 passed/one CUDA-only skipped in 10.18 s, then 17 passed in 5.26 s: 126 passed,
two skipped. This establishes engineering behavior, not native CUDA or
generation improvement. Actual runtime used Python 3.11.9, Torch 2.5.1+cu121,
pytest 8.3.5 and RDKit 2024.9.6 on host n-62-31-8.

Native submission requires the saved CPU receipt for the same source/config
and a verified DONE scheduler status. It queues two serial allocations in
`gpuv100`, each four slots, 16 GB per slot (64 GB host reservation), one
exclusive-process GPU and a 23-hour limit. Actual seed 17 job 29633350 depends on
`done(29633218) && ended(29625193)`; seed 23 job 29633351 depends on
`done(29633218) && ended(29633350)`. Thus the new study follows the existing
search pilot without replacing it. A first-segment failure still releases
the independent second seed, which must pass its own native gates.

Inside each allocation, [search_decoder_full_segment.sh](../scripts/hpc/search_decoder_full_segment.sh)
first runs the actual CUDA matched-hook fixture with
[search_full_cuda_gate.py](../scripts/search_full_cuda_gate.py). It invokes
`pytest.main` in the same process, then exits before the shell launches the
full-model Python process. It does not leave a CUDA parent alive while a
second CUDA worker starts. The full-model controller then searches the
actual 256-point pool, checks an accepted witness's fixed-target derivative
and runs a matched CUDA control/search smoke. The strict full-study gate
requires at least one accepted native witness and informative neighboring
finite-difference plateaus in **both** gradient-unit and seeded-random
directions. No-witness, zero signal or inconclusive checks do not pass this
gate. Targets remain fixed throughout each probe. Smoke consists of one
epoch on 512 training graphs/64 validation graphs per arm and establishes
finite/frozen/matched training behavior, not better generation.

Each full case has **eight hours for training**, **one hour reserved for
evaluation** and **45 minutes reserved for its smoke**. Two such cases reserve
19.5 hours inside the **21-hour controller budget**. Initial full-pool search,
derivative probes and original-reference evaluation have no separate hard
timeout and use the remaining 1.5-hour controller allowance; estimates are
not a guarantee that they fit. The controller checks available training/eval
reservation before attempting an arm. LSF's 23-hour allocation leaves
additional startup/guard time, but expensive uncapped work can still censor
or interrupt results. Retain failed, time-censored and unattempted arms.

**Search-hook checkpoint resumption is unsupported.** Teacher/cache state
is not restored by this full controller. A duplicate segment is rejected,
and partially trained checkpoints never become final comparisons. Preserve
partial artifacts rather than claiming a resume guarantee or evaluating
them as the declared ten-epoch endpoint.

Once the sealed source and unique receipt root have verified identities,
the operator can use these commands from that authenticated release on HPC.
The named shell variables must contain its actual release path, archive
SHA-256 and receipt path; no unknown digest or job is represented here:

```bash
PYTHONPATH="$task_release/src" /work3/s234818/latentmol/env/boltz311/bin/python \
  scripts/hpc/submit_search_decoder_full.py --stage cpu \
  --release "$task_release" --archive-sha256 "$task_archive_sha" \
  --receipts "$task_receipts"
```

This prints the authenticated plan only. Add `--submit` to create the CPU
job and durable intent/receipt. After the identical CPU gate is DONE, use
the same command with `--stage native --submit` to submit both serial GPU
segments. Archive/manifest/config/data identities are checked again inside
LSF. Exclusive intent records prevent blind duplicate submission when a
scheduler reply is uncertain; an uncertain receipt must be resolved first.

The coordinator reports local bundled-Python AST parsing of seven changed
Python files and JSON inspection passed, without model imports or numerical
execution. Two new shell scripts also have LF line endings and no BOM.
Archive/member verification used bundled Python 3.12.14 on Windows 11,
without model imports or numerical work. These checks establish syntax,
file identity and metadata only. Packaging commands/environment are in the
[ledger](05_experiment_ledger.md); actual HPC job IDs, runtime environments
and result paths can only be appended after successful access/submission.
A queued job is not a passed native gate or a completed training comparison.
