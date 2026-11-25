# Frozen VAE + Protenix latent optimization

The [pretrained ZINC graph VAE pilot](graph-lda-contact.md) compares contact
gradient descent at learning rate 60 with matched random moves on four fixed-size
molecules, using fresh Protenix scoring for valid edits.

Question: can the trained VAE preserve useful molecular structure while a frozen
Protenix contact objective and structural-repair guidance optimize its latent?

The [completed preservation comparison](../../chemistry-guidance/preservation-protenix.md)
holds repair search fixed across seven settings, three starts and two executions.
Atom 2 / structure 2 improved validity in all six paired trajectories and
gave weaker mean hard contact scores in both executions. Atom 2 / structure 0
had zero final sulfur drift in every run.

The [completed mixed-search comparison](../../chemistry-guidance/atom-exchange-protenix.md)
used three matched starts × three arms, LR 70 and 500 updates. Increasing the
atom surcharge reduced mean sulfur drift. Trajectory validity fell from 78.9%
under current guidance to 51.1% at 0.5 and 24.9% at 2.

The [latent editability study](editability.md) maps decoder boundaries and
compares matched Protenix temperature schedules from frozen epoch 45.
The [same-molecule comparison at update 597604](editability.md#same-molecule-from-three-latent-starts-at-update-597604)
tests whether the original random vector, its re-encoded mean or a posterior sample
produces more valid edits that improve on fresh Protenix scoring.

The [DBMol objective variant](dbmol-notes.md) is a fresh random-latent50k campaign
with the same cooling/noise/chemistry schedules, submitted3November as
A10029877824 plus callback29877825. Native backward/results are pending.

Latest: [fresh Protenix repeat and DBMol packaging repair](repeat-20261103.md).
Protenix50k29879114 is queued on A10; DBMol replacement startup29879133 is queued
on A40 after the first capsule's missing test fixture was corrected. Original
attempts/results remain preserved; neither new native run has started at capture.

The separate [epoch45 hidden-to-native embedding prototype](embedding-notes.md)
fits a small mapping before one10k trajectory. Corrected job29877165 completed
all10k updates at13:22CET on3November;55.76% of source graphs were valid, while
the final graph was invalid. Results and score-gap observations are in the note.

The [mapping-versus-relaxed investigation](embedding-comparison.md) compares
saved trajectories and a separate matched-latent gradient probe. Early results
show higher raw validity with mapping, but a large gap between mapped and decoded
contact scores.

The [LR0.1/no-noise branch](embedding-no-noise-lr01.md) completed 10,000 updates
from the saved update-180 latent. Its decoded contact fell while the mapped
score rose. The [contact calibration comparison](embedding-gradient-match.md)
fits the bridge to decoded contact and finite latent steps, then compares it
with the original mapping and direct relaxed path at both saved starts.

User authorized implementation and submission on 2 November 2026. This is a new
experiment. The running VAE training and every Mosaic job remain unchanged.
Native execution has now passed the two pilots and one full 50k trajectory.
These are finite/frozen-path engineering checks, not native derivative parity
or experimental binding qualification.

## Completed first trajectory — 3 November 2026 09:30 CET

Queues reopened overnight. Both the 200-update pilot and independent 20-update
pilot at chemistry weight 10 passed. The callback submitted **main 29875634**,
native **DONE at 08:38:04 CET** after starting at 02:43:12. All 50,000 updates
finished in one allocation: runner elapsed 21,234.39 s (**5.90 h**), mean
0.42455 s/update. Main callback 29875635 DONE wrote `completed_50000`.
The exact epoch-45 checkpoint SHA, `5c4946c1...1302be`, is recorded.
All finite/attached/shared-graph checks pass; decoder/native weights and context
are unchanged. No models were rerun to collect these results.

| Saved state | Relaxed contact score (higher better) | Raw argmax |
| --- | ---: | --- |
| Initial | 0.151316 | valid |
| Best observed score, update 27 | 0.345866 | valid |
| Final, update 50,000 | 0.245647 | valid |

The score is evaluated on the relaxed graph, not the displayed discrete
molecule. Temperature changed 0.7→0.4, so this is not a fixed-condition hard-graph
affinity comparison. A same-protocol native rescore and a matched control are
still needed for stronger claims. The final graph differs from the initial by
**4 atom identities, 0 charges and 12 unordered bond labels**; 25 slots stay fixed.
Final SMILES: `CC1C(Cl)N2CC23OC2(CC12)C3CCCCCCCCCCC(N)O`.

Across 50k source states, 16,289 were raw-valid (32.58%) and 33,711 had accepted
repair targets; **zero searches were censored**. Validity by schedule phase was
746/20,000 (**3.73%**) before 20k, 1,395/10,000 (**13.95%**) during the ramp, and
14,148/20,000 (**70.74%**) after 30k. Temperature, noise and chemistry changed
together; this does not isolate which caused the difference. There were 50
distinct valid canonical SMILES along this trajectory, not 50 independent runs
or established novel/synthesizable molecules. Only 4 steps hit the displacement
bound. The final argmax at the saved 30k/40k/50k states is the same valid identity.

The standalone HTML was generated on HPC and downloaded with byte/SHA checks:
[interactive report (148 MiB)](C:/Users/nikol/DBMol_Local/reports/vae-protenix-50000-20261103/index.html).
The [compact interactive report](../../../outputs/hpc-results/source157-vae-protenix-latent-attempt001/report-compact-20261106/index.html)
summarizes the full trajectory in 250-update blocks and offers playback at 22 saved states.
[Main/pilot/queue receipts](../../../outputs/hpc-results/source157-vae-protenix-latent-attempt001/status-1793694544453956600.json),
[initial/final, phase counts and native terminal history](../../../outputs/hpc-results/source157-vae-protenix-latent-attempt001/outcome-1793694608638934300.json),
[verified download](../../../outputs/hpc-results/source157-vae-protenix-latent-attempt001/report-download-1793694689308318500.json).

**Queue audit 2 November 2026 20:56 CET:** every queue returned by `bqueues`
is Open:Inact, including V100, A100, L40S, A40, A10, H100 and CPU `hpc`.
Detailed standard GPU/CPU queue status is `Open:Inact_Adm`. There is no active
alternative to migrate to. Open permits submission; administrative inactivity
prevents new starts while existing jobs can continue. Pilot/callback remain
queued unchanged; the scheduler output provides no reopening ETA.
[Complete native queue audit](../../../outputs/hpc-results/source157-vae-protenix-latent-attempt001/queue-audit-1793649397288293500.json).

**Status 2 November 2026 20:53 CET:** VAE45 prerequisite finished DONE18:48:17
CET; checkpoint saved, final validity75.10%, reconstruction256/256 and all
health/repair checks pass. Pilot29873719 remains PEND because `gpuv100` is
inactivated by the administrator; the final VAE CPU callback is likewise waiting
on inactive `hpc`. No pilot result or50k main yet. Existing queued jobs remain
unchanged. [Terminal/queue evidence](../../../outputs/hpc-results/source157-vae-protenix-latent-attempt001/terminal-check-1793649216712916700.json),
[completed45 metrics](../../../outputs/hpc-results/source157-vae-protenix-latent-attempt001/status-1793649179553656800.json).

**Status 2 November 2026 17:59 CET:** pilot29873719 and callback29873720 remain
PEND on their intended dependencies. VAE45 is RUN at6,000/6,596 updates with
all18 checks passing; no completed checkpoint or Protenix pilot result yet.
[Current native status](../../../outputs/hpc-results/source157-vae-protenix-latent-attempt001/status-1793638793077848500.json).

**Submitted 2 November 2026 13:12 CET:** GPU pilot **29873719 PEND**, callback
**29873720 PEND**. Native LSF confirms `done(29871778)` before the pilot and
`ended(29873719)` before its assessor. The pilot uses one exclusive32GB V100,
8 CPU slots ×4GB and2h. The50k main is conditional, not yet submitted or run.
Frozen source157 archive SHA
`2c3082c8d8b32ac162c6f29c13e00e6bc9d0f710de0a7ef35433607e61656216`,
plan SHA `6ee042ae34ac227b3b2251a86a4578116a548413060b6c89f55aad1459f2879f`.
[Actual plan](../../../outputs/hpc-submissions/source157-vae-protenix-latent-attempt001.json),
[submission and native dependencies](../../../outputs/hpc-submissions/source157-submit-1793621563210104100.json).
The existing VAE continues unchanged; its epoch45 checkpoint is resolved and
SHA-bound only after successful completion. There is no unfinished/epoch44
fallback in this scientific pilot.

## Pipeline

1. Wait for the existing epoch-45 job29871778 to finish successfully. Authenticate
   its completed result, original health checks and checkpoint hash. No fallback
   to an unfinished checkpoint. The full run uses that exact frozen decoder.
2. Sample one256-dimensional `z ~ N(0,I)` at seed17. Choose the25 highest-occupancy
   slots once to match Mosaic's prepared BRD4/JQ1 context. This is explicit count
   conditioning, not learned atom birth/death. Keep those slot IDs fixed.
3. Decode a single temperature-scaled graph with all nine occupied element types,
   five charge classes and five bond classes. Both branches consume this graph.
4. A resident Mosaic JAX process injects expected elements into the native128
   element channels, expected formal charge, and bond presence. It reuses the
   frozen geometry-off Protenix-Tiny trunk and **negative mean ligand best-pocket
   contact probability below8Å**, fixed model key17. No diffusion or confidence
   head is used. This is a contact surrogate, not predicted affinity.
5. The native process returns probability derivatives. A Torch custom VJP carries
   them through the same decoder probabilities to `z`. The decoder runs on CPU;
   the JAX process is the sole GPU model owner. Neither model is trained.
6. Structural repair searches the current graph outside autograd. Its target
   preserves element/charge/occupancy labels, joins components, prunes excessive
   bonds while protecting connectivity, then uses bounded beam fallback. Every
   accepted target passes native chemistry/schema checks. Its fixed-target NLL
   differentiates through the decoder. Failed repairs and already-valid graphs
   retain explicit zero repair guidance. No partial-valence fallback is enabled.

Chemistry uses fixed-occupancy normalization `2*K + K*(K-1)/2`, including every
unordered NONE bond once. This differs from unconditional VAE training's
`N+K+K*(K-1)/2`. VAE population composition, occupancy35 and reference36 penalties
are not copied into this single-molecule optimization.

## Schedules and updates

| Setting | Declared policy |
| --- | --- |
| Chemistry coefficient | 1 through20k, geometric to10 at30k, then10 |
| Temperature | Linear0.7 to0.4 over50k |
| Latent noise | Unit Gaussian direction scaled by0.4 of binding-only latent SGD displacement; linear to0 by30k |
| Step | SGD LR0.3 as a pilot starting guess; bound final displacement to0.5, logging every bound |
| Main budget | 50,000 updates, preserving global schedules/RNG between allocations |

Temperature, target and coefficient are fixed within each evaluation/update.
All branches use the same temperature. The total objective changes when targets,
weights or temperature change; a jump is not automatically an optimization gain.
Nonmonotone loss, flat gradients and low validity are recorded as findings.
Nonfinite/incompatible/detached computations, frozen writes and duplicate writers
stop the affected run. Bound use and zero chemistry guidance are explicit.

The actual pilot is200 updates on the original50k schedule, plus an independent
20-update weight10 test from the same initial latent (no compressed schedule).
Both are discarded. After native DONE and technical checks, the main trajectory
starts again from the original seed. Pilot timing sets chunks with30% margin
and20-minute reserve, at most23h internally/24h LSF. Successful chunks resume
`state.pt` in a new output directory until50k; failures do not auto-retry.

## Code and outputs

- [Runner](../../../src/latentmol/design/latent_protenix.py): schedules, latent
  updates, structural targets, frozen checks and resumable RNG/state.
- [Torch bridge](../../../src/latentmol/guidance/mosaic_protenix.py) and
  [native worker](../../../src/latentmol/guidance/mosaic_protenix_worker.py).
- [Pilot](pilot.json), [weight10 test](pilot-weight10.json), [main](full.json),
  [DTU native assets](native-dtu.json); [controller](campaign.py).
- Losses/gradient norms/cosine, noise, latent displacement and raw validity are
  saved per state. All latent vectors and raw hard graphs, including invalid
  ones, are packed in NPZ blocks. Probabilities and repaired target labels are
  saved every100 updates plus endpoints. Censoring, validity, repair coverage,
  unique molecule identities and step bounds remain observable.
- Existing interactive HTML reports are generated from saved JSON only: loss
  overlays, playback, raw molecule diagrams and probability heatmaps. JSONL
  metadata avoids tens of thousands of tiny files. Presentation failures have
  separate receipts and never trigger new model predictions.

Protenix collapses typed bonds to presence; chemistry/decoder retain the typed
distinctions. Expected-charge features do not represent charge uncertainty in
full. This pilot does not establish native derivative parity, affinity, synthesis
or experimental binding. A future direct-graph control should start from the
same decoded graph and match objective, chemistry, schedules and compute.

Local evidence: **85** focused runner/bridge/controller/report cases passed
against the exact packaged snapshot; actual runner CLI and shell syntax passed.
A
two-update CPU interface smoke with the real epoch44 decoder and an **analytic
scorer** passed and rendered HTML; this was not native Protenix execution and
does not replace the epoch45 scientific pilot.

Preparation notes: a too-long Windows operator command failed before remote
execution. A separate metadata extraction attempt hit the old shared
`releases/source.tar.gz` name before any submission; the new empty inspection
directory was then filled safely from the hash-checked archive, without touching
that older file. Compute launches still use the tested isolated temporary
extraction. Both failures/receipts are preserved; neither changed the source,
old runs or scientific settings. One actual pilot submission was made.
