# Standalone delayed-chemistry report

> **Scope and access note (22 October 2026):** this document contains an
> experiment protocol and dated operator evidence. Concrete account paths,
> environments, jobs and source hashes describe that operator/attempt; they
> are not shared-account defaults or a fresh status/acceptance claim. Use
> [your own HPC setup](HPC_GUIDE.md). Keep the scientific gates and historical
> outcomes below intact when preparing a separately validated candidate.
> Canonical working home: [brd4/chemistry-ramp-10000](../experiments/brd4/chemistry-ramp-10000/README.md).


19 October 2026. User requested a **new report** for the completed 50% mix,
ready to include random when it finishes. Historical five-thousand-update
chemistry reports and earlier six-initialization reports remain separate.

This report family includes only the two fresh ten-thousand-update BRD4 arms:

| Input | Job | Inclusion |
| --- | --- | --- |
| 50% chain/uniform, delayed chemistry | 29753667 | Complete; 10,001 saved states |
| Random Dirichlet(1), delayed chemistry | 29753668 | Complete; 10,001 saved states |

The first edition uses one real study, with explicit pending metadata for the
second study. There are no fabricated random frames, padded states, hidden
partial outputs or historical scalar overlays. The included study must pass
all existing native, budget, coefficient, snapshot and frozen-identity gates.
The report distinguishes a completed included arm from an incomplete comparison.

Render through `scripts/render_chemistry_ramp.py` with one `--study`, both
authenticated `--config` files and `--pending-job 29753668`. The second config
binds the expected future input without loading its running trajectory. Once
the random input is complete, use the existing two-study invocation, which
requires both frozen model/context identities and schedules to match. This
changes presentation only; `src/` and `examples/` must remain byte-identical
to the passed native source01 release.

The existing pair report job **29753669** already waits for both native jobs
and renders only these two new arms. Keep it unchanged. It can provide the
full comparison after authenticating its actual outcome, native gates and
archive. Preserve the first standalone edition as a separate saved version;
publish the verified updated edition at the new report family's viewer link.
Never overwrite or merge into the earlier report viewers.

New preview job `scripts/hpc/brd4_chemistry_preview.lsf` runs allocated CPU
report/collector fixtures, authenticates source01 and completed mix50 summary
SHA `0236e16f8da2aa333bf469ec5cf07b2fbe962b3a5e4d09a631d90ae60e0dfca8`,
then validates and renders every saved snapshot. It retains source/config,
summary/case, native derivative/pilot, original readiness and scheduler proofs
in a compact presentation evidence bundle. Full scientific trial/snapshot
archives remain under the immutable native directories and the queued pair
evidence job. No inference, repair, training or new scientific readout occurs.

Download only after CPU success, verify exact archive path/length/SHA, safe
extraction and every evidence file. Check input job29753667, 10,001 states,
pending input29753668, no historical input and expected frozen identities.
Serve on a new loopback port with the detached helper and inspect playback,
readable atom labels, probability sizing, heatmaps and honest pending status
in the browser. Local work is static code/byte verification and presentation QA;
all tests and scientific plot generation remain allocated HPC.

## Progress

- [x] Explicit completed-arm/pending-input report mode, preserving strict default pair mode.
- [x] Allocated fixture tests for pending provenance/no invented frames and failed-native rejection.
- [x] Unique report-only wrapper with unchanged scientific-source gate.
- [x] Static checks, immutable packaging and actual CPU submission recorded.
- [x] Allocated presentation checks/rendering/packaging passed.
- [x] Download verified and standalone report opened.
- [x] Random endpoint and complete pair report authenticated and added as a new edition.


Preview source01:365 files, manifest
`34462df7a3ba5001bced1c6266fa1a2851d77f11022694339f194e6c38a88671`;
archive39,195,593bytes SHA
`7ced77a30b6613011d538b017eb7b720539a8a64dc14b43e79eacc7e1c3e638c`.
Remote source `/work3/s234818/mosaic/releases/20261019-brd4-chemistry-preview-01/source`.
Actual CPU report job **29755226** submitted to hpc, two slots,4GiB per slot,
one hour, after done(29753667), unique run
`/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-preview-01`.
Source transfer/archive/per-file authentication passed. Current home cached
quota12.41/30GiB; per-job gate remains required. Initial scheduler state RUN,
19 report/collector tests selected. This is not yet a passed render/download.


Allocated preview tests: **19 passed**, zero failures/errors/skips,72.67s.
This includes six ramp/pending report fixtures, three existing report fixtures
and ten collector boundary fixtures. Actual render remains running at this
entry; model inference and chemistry evaluation are not part of this job.

Local publication layout will preserve the downloaded first edition under
`results/hpc/20261019-brd4-delayed-chemistry-report/edition-01/`, and serve a
verified copy from the new family's `viewer/` directory. Later, authenticate
29753668 and29753669, download the two-arm edition into `edition-02/`, then
replace only this new family's viewer files after byte/identity checks. Preserve
the first edition and all prior report families/servers. The HTML is self-contained,
so replacing it last allows the existing detached server to serve the new
edition on refresh without terminating earlier viewers.


## First edition delivered

Preview **29755226 DONE**, native source/config byte-map parity and terminal
proof authentication passed;19 allocated report/collector tests passed, no
failures/errors/skips. Every10001saved snapshot and coefficient was validated
before rendering; no molecular work was rerun. Entry8GiB and packaging
4.123253824GiB quota gates passed with17GiB conservatively available.

Archive `/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-preview-01.tar.gz`:
15,963,546bytes,154files, SHA
`8aba45395a1ce5ad6a67030bc42d89272db95742a4c749555975cfe38a03fb9b`.
Final scheduler/LSF success separately authenticated, exact receipt path/bytes/
SHA checked, `scripts/verify_hpc_evidence.py` safe extraction and154/154file
checks passed. `scripts/hpc/receipts/20261019-brd4-chemistry-ramp/verify_preview_download.py`
passed stdlib saved-HTML/JSON/XML/schema checks: included native29753667,
10001contiguous frames, six-slot dimensions, pending29753668 with no frames,
no historical input, correct coefficient endpoints/frozen identities/counts.
Local download under `results/hpc/20261019-brd4-delayed-chemistry-report/edition-01/`.
Full native trial archives remain on HPC and in the future pair evidence job.

HTML35,486,566bytes SHA
`981dec0a28d9e7835e9ffdcefae664e3fa0078ae881257d29da87919f55bc014`.
New viewer **http://127.0.0.1:8773/report.html**; detached HTTP server HEAD200
and exact byte length/hash verified. Existing report servers preserved. Browser
checks passed: explicit completed50%/random-not-included status and disabled
random selector; playback advanced; endpoint10000/weight50; readable C1–C6;
probability sizes, heatmaps and four loss/ramp panels. Tab marked deliverable.
Screenshot `results/hpc/20261019-brd4-delayed-chemistry-report/viewer-final.png`.

Saved first-edition findings: contact -0.2778551579 to -0.4092489183; all10001
raw hard readouts hexane and all repair losses zero. Reported directly from
saved evidence, not reevaluated chemistry or an affinity result.

Reused the prior paused report heartbeat with updated scope/name **Finish new
BRD4 chemistry report** (id `finish-combined-brd4-trajectory-report`), now ACTIVE
hourly. It checks random29753668 and pair report29753669, stays quiet on unchanged
queued/running/outage state, authenticates terminal/native/archive gates,
preserves edition-01 and updates ONLY the new viewer after verification.
It pauses after successful delivery or a native blocker requiring user action.
No duplicate monitor or scientific rerun was created.


## Second edition verification in progress

Random **29753668 DONE/passed**, completed15:57:35 Europe/Paris,10000applied
strict improvements/10001states, zero fallback/no-op; final model/context pins
unchanged. Summary SHA
`9a2ebae13049babdeaca670bb13e707927058b8f2cd273a706d12fbf2177bdb2`.
Saved final raw hard readout `FCCC(F)F`, connected/sanitized; saved repair loss
1.4735963345 to0 and contact -.2969741225 to -.4144608378.

Pair report **29753669 DONE**. Render receipt complete=true, ONLY native
29753667/29753668,10001frames each. HTML71,949,822bytes SHA
`8575ad053dd80d9b53f6f34594b1b6e25982303f361b51b950b73fc3399467d7`.
Full evidence archive739,279,500bytes/61,267files SHA
`fea18fc354e3185dbb3b6b38058cd992a8713ac068bbeb5dfe2e8df4616d41e0`,
path `/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-ramp-01-report.tar.gz`.
Exact receipt and final random/report scheduler + all native/report LSF success
authenticated before download. Mix50/earlier readiness bjobs entries have aged
out; preserved native job-final and LSF success records remain authoritative.
The archive's not-found scheduler snapshots are not failures or waived gates.

Downloading to new `edition-02/`; first edition/viewer retained until safe
extraction, every-file integrity and native/report saved-record checks pass.
All checks use stdlib bytes/records; no local chemistry/models/plot generation.
If the original Windows serial byte loop is slow after safe extraction, retain
its incomplete status and finish every archive/path/manifest/length/hash gate
with the bounded eight-I/O-worker continuation in `verify_pair_download.py`.


## Second edition delivered

Report29753669 completed16:11:22 Europe/Paris. The739,279,500byte archive
matched its independently authenticated exact receipt path, size and SHA.
`scripts/verify_hpc_evidence.py` completed safe member validation/extraction.
Its slow Windows serial hash pass was intentionally stopped after45seconds
following completed extraction; this incomplete serial pass is not claimed
passed. The bounded stdlib continuation reauthenticated archive SHA, exact
archived/extracted manifest and safe unique member set, then verified
**61,267/61,267 paths, lengths and file hashes**, eight I/O workers,160seconds.
The manifest itself is byte-bound to the authenticated archive. No gates lost.

`verify_pair_download.py` passed native summary/case equality, both completed
budgets,274-test readiness/XML/source/reference/initialization pins, before/
after frozen model/context identities, all native contact/low/high-coefficient
derivative/pilot statuses, frame metadata and snapshot-SHA bindings to the
verified manifest. Both native arms applied10000strict improvements with zero
fallback/no-op; final coefficients50 and fixed six occupied neutral slots.
No local numerical, chemistry or model execution/plot generation.

Local proofs in `edition-02/`: `20261019-brd4-chemistry-ramp-01-report-download.json`
and `pair-verification.json`; actual final scheduler/LSF files retained there.
The original native/data archives remain unchanged on HPC.

Saved findings: mix50 retains hexane and zero repair in all10001states; contact
-.2778551579 to -.4092489183. Random repair1.4735963345 to0, contact
-.2969741225 to -.4144608378; final connected/sanitized hard readout `FCCC(F)F`.
Its first saved native/schema-valid state is8959;1012states pass this saved
criterion in total, so validity is not monotone thereafter. Late repair spikes,
invalid intermediate graphs and contact increases remain visible. Changing
repair targets/coefficients preclude treating all combined totals as one
objective. Contact is not affinity; these two exploratory initializations do
not establish general chemical/binding performance.

`publish_pair.py` copied only verified artifacts to this new family's viewer,
HTML last, retaining `edition-01/` and its publication receipt. Existing detached
8773server reused; HTTP200,71,949,822byte length and streamed full-body SHA
`8575ad053dd80d9b53f6f34594b1b6e25982303f361b51b950b73fc3399467d7` passed.
Updated link: **http://127.0.0.1:8773/report.html**. Older report families/servers
untouched. Browser QA passed both selectors/colors, random0/playback/final10000,
raw invalid and final valid graphs, readable element/slot labels, sizes, exact
heatmaps and all four comparison panels. The browser's large accessibility
snapshot transport hit a frame limit; actual page loading/screenshot-and-keyboard
verification succeeded, so no scientific/report rerun or gate relaxation occurred.
Tab marked deliverable; `viewer-complete.png` and `viewer-overlays.png` retained.
Heartbeat `finish-combined-brd4-trajectory-report` explicitly PAUSED after delivery.
