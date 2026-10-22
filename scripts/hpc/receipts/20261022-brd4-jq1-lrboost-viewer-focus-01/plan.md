# Presentation-only exact-update focus correction

Browser QA of the already byte-verified second higher-rate report found a real
playback/input race: while playback advances, `draw()` writes the current saved
step into the numeric input. Typing a chosen step before pressing Enter can lose
the typed value. The correction adds exactly one listener immediately before the
existing input change listener:

```javascript
byId('stepInput').addEventListener('focus',stop);
```

The existing `stop()` changes the playback epoch before typing begins. The
current `draw()` writes the input before its first await; a draw already awaiting
a chunk does not subsequently overwrite that input. Numeric change/Enter retain
their original handling. Browser QA must exercise actual focus, continued typing,
Enter and a delayed stability observation while playback was running.

This is a new immutable presentation tooling release. It changes no scientific
source, loss, rate, trajectory, native result, endpoint diagnostic, collector or
original report. The original analysis source16 and CPU278 remain renderer
provenance. The new source and allocated CPU certificate are additive patch
provenance. Protenix and normal Boltz remain scientifically failed; chemistry's
authenticated terminal success remains separate.

The first adoption declares original render job 29781041, archive 157404869 bytes,
SHA256 `e92c477b8d173c1c0a9e17813e9cb0f8549cd24c9e97578d43e7591823cbb30a`.
Its source manifest is
`214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc`,
and original rendering CPU certificate is
`42b2b000ecce5d661da9b75ad85f03a41c3b303010ccfa03f87835a25993ca69`.

`viewer_focus.py` is a byte-only helper. The allocated patch job streams the
complete original archive and every member against its evidence manifest,
without extracting or copying its roughly 200 MiB of saved frame assets. It
compares stable archived payload bytes to the existing original report parent,
validates the original report using the authenticated original publisher, and
requires actual final DONE/hpc and same-job successful LSF/submission evidence.
The archive's own scheduler snapshot can still say RUN because it was captured
during packaging; that snapshot is preserved and does not establish completion.

The output is a small delta: patched HTML, updated report receipt and patch
assessment, explicit adoption, verbatim original receipt/assessment, original
producer final proofs, new source manifest and new CPU proof bundle. The patch
assessment truthfully identifies the new allocated patch job and explicitly
identifies the original render job. Its renderer source/CPU fields remain
original. The original embedded manifest, all 610 other served assets, captured
steps, outcomes, scalar metadata, probability chunks and hashes remain exact.
There is no scientific rerender, probability decoding or native evaluation.

All new source/upload/control/output bytes are nested beneath the original
analysis root. Its existing sealed diagnostic entry accounting therefore charges
the new bytes. The new controller independently retains the entire remaining
native/report/diagnostic byte projections, 4 GiB reserve and 100000-file headroom.
No old gate is overwritten, reduced or silently treated as a reservation.

Before a patch, allocated CPU must rerun all original 278 report fixtures plus
`test_viewer_focus.py` and `test_viewer_focus_ops.py`. Exact new module counts
come from actual collected nodes and actual raw XML, never a guessed total.
No failures, errors, skips or deselections are allowed. Actual CPU DONE/hpc,
same-job LSF success, complete raw log/XML and byte/source certificate bindings
must authenticate before the patch job is submitted. No native or diagnostic
job is created by this controller.

Focused fixtures cover the exact listener insertion and original manifest byte
preservation, missing/duplicate/already patched input rejection, 610 unchanged
byte assets, separate renderer and patch identities, altered data/adoption/CPU
proof rejection, safe paths, actual same-job proof identity and the exact patch
submission vector plus its exclusive start marker. Fixtures use explicitly tiny
fake byte assets solely to exercise metadata integrity. They cannot certify
scientific frames or replace the original allocated saved-frame checks.

Publication requires both independently verified local original full archive
and independently verified new delta archive. The additive publisher validates
both archives/all members, original report semantics, new CPU/source proof,
actual new patch DONE/hpc/LSF, exact patch source/root/manifest/resources/phase
and exclusive submission-started identity. It reconstructs a fresh full report
using only local byte copies of the already verified original assets and the
allocated patch metadata. It does not claim the small delta archive contained
the full assets. Publication records both original and delta archive SHA256s.
Only after all checks may it create a new immutable edition and change the
atomic latest pointer. All prior editions and their publication records remain
unchanged. Verify every served HTTP byte and repeat browser sliders, numeric
input, arrows, playback/manual pause, all run selectors, latest updates, raw
graphs and probability heatmaps.

The existing separate diagnostics and final analysis renderer remain their
original jobs. A future completed final render can use the same generic byte
patch helper with its declared authenticated original archive/job pins; this
initial controller pins only the known second interim. No final render,
diagnostic or scientific study is restarted or duplicated here.
