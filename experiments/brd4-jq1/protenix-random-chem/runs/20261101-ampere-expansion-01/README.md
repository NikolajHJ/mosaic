# Chemistry comparison across available GPU queues

Run the existing 22-arm comparison through available queue allowances. All arms retain their original 20,000-update target, LR0.3, model/context/key17, initialization seed, objective, chemistry strategy, precision and stride20 matrix retention. No saved results or running producers were changed.

At 1 November12:48 UTC:

| Queue | Arms | Actual optimizer IDs | State |
|---|---|---|---|
| A100 | combined seeds17/23 |29865938/29865970|RUN; at least7,631/6,062 committed updates at12:46UTC|
| A10 | chemistry-first seeds17/23 |29866311/29866312|PEND|
| A40 | severity seeds17/23 |29866315/29866318|PEND|
| L40S | adaptive seeds17/23; temperature2 seed17 |29866329/29866333/29866331|PEND|

The user requested direct execution without another test bottleneck. Four unstarted A10/A40 probe/callback jobs were held, authenticated and retired. Seven original unstarted V100 producers were similarly retired with zero updates before unique full-run successors were submitted. Normal production entry and first saved updates retain finite/shape/required-gradient/frozen/input/output/capacity checks. No new test job or qualification chain was added. Whole-run reservations are A10 08:05, A40 08:00, L40S07:50/08:00, based on existing measured timings and conservative overhead. They do not reserve physical hardware before dispatch.

Adaptive seed23's first retirement observation timed out after an accepted kill. Its failure and restore attempt remain preserved. A fresh exact EXIT/TERM_OWNER, no-execution, no-payload and no-successor proof allowed the single successor29866333. Never replay these dispatch programs.

Earlier A100 probe29865922/callback29865923 completed two disposable25-update paths. Warm medians0.586/0.629s versus V1000.796/0.846s; sampled peak2535MiB. This was finite-path/timing evidence, not derivative certification. Its50 smoke states were never adopted. The two full A100 producers remain untouched.

Read campaign.json, direct-production-proof.json and each remote device run/jobs.json plus production/ARM/submission.json/release.json before recovery. Scientific outputs are ROOT/ARM/payload/study; submission receipts are ROOT/production/ARM. The old source02 registry remains historical for migrated arms. Canonical launcher/math/library files were reused; only the new L40S plan differs. Current remaining-output accounting is unchanged; no cleanup occurred. Report updates use the canonical cached engine.
