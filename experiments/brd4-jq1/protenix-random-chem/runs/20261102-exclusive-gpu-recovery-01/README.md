# Exclusive GPU recovery after DTU ticket43078

DTU support stopped account jobs on1–2November2026 and requested exclusive GPU compute mode. The old `j_exclusive=yes` allocations still requested `mode=shared`; that was the launch error. All NEW profiles/overrides now require `mode=exclusive_process:j_exclusive=yes`; these single-CUDA-client jobs use `mps=no`. Four hybrid chemistry workers are explicitly CPU-only. Historical sources and results remain unchanged.

Ten unfinished Mosaic producers were reconciled with actual `bhist`/LSF, checkpoint JSONL and no active predecessor, then resubmitted to their same GPU families. Sixteen completed strategy arms, completed original arms and the two intentionally retired duplicate beam controls were not restarted. Other-project jobs were not changed.

CPU29871025 passed70 actual focused cases, including uninterrupted versus restored adaptive/neighbor/teacher trajectories, with zero failures/errors/skips; all eight real metadata CLI startups passed. Noise/DBMol reuse byte-identical scientific drivers; their real metadata startups passed. Local work was byte/schema/XML/AST and25 portable policy assertions. Pytest was unavailable in the local venv, so the full affected tests ran on allocated HPC.

At00:29:10UTC2November, five producers RUN with actual scheduler `GPU_MODE=exclusive_process`, `J_EXCLUSIVE=yes`; five PEND. See campaign.json for all exact IDs, sources, plans and checkpoint/progress bindings. Native/FD/binding certification is not claimed.

| Arm | New job | State | Verified global updates / target |
|---|---|---|---|
| chemfirst-s17 | 29871029 | PEND | 980 / 20000 |
| chemfirst-s23 | 29871030 | PEND | 0 / 20000 |
| severity-s23 | 29871031 | PEND | 0 / 20000 |
| earlyseverity-s23 | 29871032 | RUN | 1758 / 20000 |
| delayed-dense-s17 | 29871033 | RUN | 1211 / 20000 |
| combined-dense-s17 | 29871034 | RUN | 1221 / 20000 |
| structural-s17 | 29871035 | PEND | 0 / 30000 |
| structural-s23 | 29871036 | PEND | 0 / 30000 |
| noise | 29871026 | RUN | 36508 / 50000 |
| dbmol | 29871027 | RUN | 30305 / 60000 |

Recovery uses the latest retained checkpoint that has its complete fixed-teacher/policy or noise-RNG handoff: screen offsets980/1640/1140/1180, noise36320 and DBMol30260. Later interrupted tails (10/2/2/6,9 and4 updates respectively) remain untouched in their parent studies. This makes an explicit new branch; no tail is erased or falsely added to child progress. Original targets, LR, schedule, precision, keys, contexts and losses remain unchanged; PCG64 noise continues from the checkpoint’s committed RNG state.

The screen continuation records local `step` from0 and absolute `global_step`; NPZ step and schedule/phase/sampling positions remain global. Its summary declares `global_start`, `global_target` and `resume.predecessor_study`; its config still declares the ORIGINAL total target, while requested/applied updates describe this child. To report it from update0, merge the parent through that checkpoint and the child after its duplicated boundary, retaining the parent’s later interrupted branch as history. If the canonical report adapter needs that input extension, add it there with focused tests when these children complete; never rerun a producer for reporting. All future result publication uses scripts/reporting/report.py and the existing cache.

No cleanup, saved-source edits, loss/gate changes, old dispatch replay or duplicate writer. New screen capsule/source and actual accepted/held/released receipts are under `/work3/s234818/mosaic/runs/20261102-exclusive-gpu-recovery-01/screen01`; native siblings are `native-noise01` and `native-dbmol01`. Read actual registries before recovery; do not replay this launch.
