# V100 default and continuation migration

New Mosaic exploratory submissions prefer **32 GB V100**, with explicit L40S
fallback. See the [common GPU policy](../../../../../docs/GPU_POLICY.md).

Verified 30 October 2026 13:43 UTC:

| Run | Actual job | GPU queue | State | Confirmed updates / target |
| --- | --- | --- | --- | --- |
| Protenix + chemistry | 29853757 | gpul40s | RUN, kept unchanged | 11,510 / 50,000 |
| Plain Protenix | 29855339 | gpuv100 | RUN, migrated | 28,347 / 50,000 |
| Boltz + chemistry | 29855330 | gpuv100 | PEND, migrated | 10,000 / 30,000 |
| Plain Boltz | 29855430 | gpuv100 | PEND, migrated | 20,000 / 30,000 |
| DBMol native composite | 29853835 | gpul40s | PEND, retained | 7,500 / 30,000 |

Protenix + chemistry started during the comparison and was left running. DBMol's
whole remaining V100 horizon was estimated at 35h41 from measured timing, beyond
that queue's 24-hour limit; its existing L40S job remains queued. Targets were not
shortened. The three migrated predecessors were held, authenticated as unstarted
with no payload/history dispatch, then retired and confirmed EXIT before their
replacements were submitted/released. No active producer was stopped.

The original saved inputs, losses, precision, LRs, key17, N25, one fixed candidate,
chemistry coefficient schedules, composition0/no margin and four-worker hybrid /
matrix acceleration are unchanged. All scalar losses and stride20 matrices remain.
New V100 reservations derive from measured complete-remainder timing plus 25%
and setup/cleanup allowance: plain Pro03:44, Bolchem15:15, plain Bol07:17.

## Actual device evidence

Comparison29855106 actually DONE on a Tesla V100-SXM2-32GB. All five real model
paths completed three fixed updates each, with finite backward values and original
frozen model/context identities preserved. Sampled GPU peaks ranged from about
2.4 to9.3GiB. This establishes short-device compatibility, not FD/native gradient
certification or a guarantee for the whole trajectory. All15 smoke states are
separate disposable observations and were not adopted into any producer.

The comparison used one V100, eight CPU slots /4GiB each, 1h maximum, actually
461seconds. Each child had a540second bound; group limit3000seconds. No numerical
model calculation ran on the login node or laptop. Portable metadata/AST/fake
process controls total75passes (5 existing exploratory,21 GPU/migration,49 adapter).
All five actual pre-model CLI startups passed. Exploratory layout metadata checks
passed with an advisory historical test-registry warning. Initial Windows temp /
path fixture failures are preserved; workspace temp and POSIX normalization fixed
those metadata setup issues, without a scientific change.

## Preserved metadata repairs

Initial callback29855107 EXIT before any migration on its immediate scheduler
success observation. It also assumed a numeric `Job <ID>` header, while actual LSF
places the name there and the numeric ID in `Subject`. Immutable `control02`
records/bounds scheduler settling and authenticates the real Subject/footer;
existing device/timing code was unchanged. A scheduler propagation race is a
plausible explanation for the first rejection; its original row was not captured.

Plain Boltz's first handoff then hit a false prefix collision with the active
`boltz_chemistry` plan path. Immutable `launch03` uses exact path boundaries and
preserves real same-output/name collision checks. The previous retirement and
failed guard remain recorded; no submission had occurred. After reconciling the
empty producer parent and actual predecessor EXIT, it submitted29855430. These
repairs ran as authenticated, bounded metadata-only helper commands; the GPU
comparison was not repeated and sealed sources were never edited.

## Exact records and monitoring

Read `campaign.json`, `completion-verification.json`, `migration-proofs.json` and
`source.json`. Source snapshot is readonly at the recorded source path; original
source01 and its failed callback remain historical. Never replay a submitted or
unknown intent. The GPU-facing production code remains the tested source01;
control02 and launch03 changed only metadata handoff/ownership checks.

New jobs/individual immutable plans and outputs are under `production/` below the
run root, with actual IDs in `production/jobs.json`. The original Prochem and DBMol
roots remain in the preceding exploratory campaign. The latest reusable observer
is `results/hpc/20261030-v100-default/delivery-20261030T134321Z/program.py`; use a
fresh evidence label. It collects current producer IDs and committed JSONL counts.

Canonical reporting should consume new production prefixes via the existing
report.py engine/cache. Exclude smoke states. Presentation does not certify
binding, chemistry quality or derivatives. Existing failed scientific reports and
all historical saved results remain unchanged.

Source01's graph replay field `byte_equal` uses numeric NumPy array equality and
does not distinguish signed zeros; do not treat that field as byte proof. Raw NPZ
hashes are the input byte bindings. Cross-device FP32 replay has its predeclared
32eps-relative /32-smallest-subnormal floor, with actual differences recorded.
