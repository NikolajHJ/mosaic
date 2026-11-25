# V100 redistribution and startup interface recovery — 3 November 2026

Question: can compatible available V100 capacity reduce the wait for both fresh
50k VAE experiments? The eligible A10 and A40 queues each had one busy host;
four eligible32GB V100 GPUs were observed idle. The original queued GPU jobs and
their callbacks were authenticated, held, rechecked as never-started/zero-update,
then retired. Original source, plans and results remain unchanged.

The first V100 attempts reached allocations immediately but failed before any
applied optimizer update: Protenix29881354 on the snapshot's graph-export function
not accepting the runner's `scope` keyword; DBMol29881364 on the input-pin helper
receiving JSON strings where it expects `Path`. Their callbacks failed correctly.
Protenix saved its initial point only; DBMol did not reach native inference.

The new shared capsule updates only those two interfaces: use the existing
optional scope-aware graph writer, and coerce native JSON paths before hashing.
The old capsule reproduces3 interface errors; all4 focused portable controls pass
against the exact corrected capsule. Probability export/defaults and bad-hash
rejection remain covered. No native-model correctness is claimed from these
stdlib controls. The original loss, decoder, LR, seeds, clipping bound, cooling,
noise and chemistry schedules are unchanged; no numerical/optimizer code changed.

| Arm | Actual GPU | Callback | Queue | Budget |
| --- | --- | --- | --- | --- |
| VAE + Protenix Tiny | 29881613 | 29881615 | V100 | full50k,8h15 |
| VAE + DBMol | 29881681 | 29881683 | V100 | first20 real updates, then measured remainder toward50k |

Both replacements were RUN at19:23:55UTC, with native startup still in progress.
DBMol's original3 allocated adapter controls had passed; full finite/frozen
native startup and completed updates were not yet established in that capture.

Checkpoint epoch45 SHA5c4946c15515e70bd01281eea7b74693c18598c4b47f6724fcc9898deb1302be;
full config SHA95dcce2b9bd7e7a80ef231d3ec978d7059db72233b64a9180b3d119c75137902
is unchanged. New source archive SHA6331974180063aae3c3d2aa252c32aec76498e43a1555adb8852fd5fdc545bec,
release `/work3/s234818/latentmol/releases/vae-startup-interfaces-20261103-01`.
Four accepted held submission vectors and exact release proofs were independently
verified. New roots/control paths and plan hashes are in the two current run JSONs.
Prior A10/A40 jobs and callbacks are retired unstarted, not scientific failures.
Every earlier actual failure remains failed and preserved. No cleanup occurred.

Evidence: `outputs/hpc-submissions/vae-startup-interfaces-20261103-01` and Mosaic's
`results/hpc/20261103-queue-review/20261103T191204Z/final02`. Follow current receipts
for successors; never rerun the redistribution or one-shot stage scripts.

Latest19:25:44UTC: all four experiments have actual GPU allocations. VAE+Protenix
has250 committed evaluated updates; VAE+DBMol is loading its native models after
3 allocated adapter controls passed. Sparse replicas continue on A100/L40S with
1,410/656 committed updates. No report rebuild for routine progress.

3 November2026 20:59:46CET: VAE+DBMol startup29881681 and callback29881683 actualDONE/same-jobLSFsuccess. All20realupdates saved; finite binding/chemistry, frozen decoder/native and sharedgraph/temperature checks passed. Existing callback submitted main29883315/callback29883316, actualRUN onV100 fromsaved20 toward7047 in24h,188global committed evaluated updates atcapture. Total50k target/settings unchanged; later measured segments remain automatic. No FD/derivative or final chemistry claim. Prorepeat29881613 RUN5115/50k; sparse23/41 RUN3980/3955. No fullrun completion/reportrefresh. Evidence Mosaic results/hpc/20261103-run-news/20261103T195943Z.

4 November 2026, 09:16 CET. The user requested A100 for the next VAE + DBMol
segment. Job 29889322 is pending on gpua100 with a 72-hour limit and depends on
successful completion of running V100 job 29883315. The existing segment
controller computes the actual update count from the validated predecessor
endpoint and measured warm timing, using a 10.6 s/update floor, its 30% margin
and 20-minute reserve. At the floor the allocation budgets 18,722 updates.
The successor starts from the saved latent, occupied slots and noise RNG.
Original source, objective, model and precision are preserved. The old unstarted
callback 29883316 was held and verified before retirement. The two new held
receipts were saved before release. Callback 29889323 is pending behind the
A100 job and retains the original automatic routing for later segments.
Verification: outputs/hpc-submissions/vae-dbmol-a100-next-20261104-01/verification.json.
