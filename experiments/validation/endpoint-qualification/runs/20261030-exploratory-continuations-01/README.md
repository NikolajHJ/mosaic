# Exploratory saved-endpoint continuations, 30 October 2026

Question: can the five partial studies progress under the user's new exploratory
policy without repeating the qualification gridlock? The user explicitly
authorized adopting exploratory admission for these continuations, especially
both chemistry arms. This is a new execution protocol, not a pass of any old
failed gradient contract.

At12:12:36UTC all five actual optimization jobs were PEND, released and waiting
for a GPU. None depends on the old qualification jobs or a new CPU campaign.
Actual scheduler, commands, held/release receipts and74 compact file bytes were
verified in campaign.json/submission-proofs.json. Native/model backward smoke
has not run yet; no new optimizer progress is claimed.

| Run | Job | Saved start → target | LR | Wall allocation |
| --- | --- | --- | --- | --- |
| Protenix + chemistry |29853757|10,000 →50,000|0.3|14:15|
| Boltz + chemistry |29853831|10,000 →30,000|0.0025|13:30|
| Protenix |29853759|27,175 →50,000|0.3|02:45|
| Boltz |29853833|20,000 →30,000|0.0025|06:45|
| DBMol composite |29853835|7,500 →30,000|0.0025|23:00|

Each child uses its exact saved FP32 logits/probabilities/key17, original frozen
native scorer/context, original objective and original one-candidate fixed update.
Composition remains0; no confidence margin or neighbor loss was added. Chemistry
stays0.001 through global5,000 then rises geometrically to100 at50k/30k.
Targets are detached and fixed during each step's trial evaluations. Existing
hybrid4-worker/matrix chemistry acceleration and stride20 matrices/latest/best
with every scalar loss remain. Frozen/input mutation, nonfinite values, incompatible
shapes, actual native backward errors, output collision and duplicate writers fail.

The original backward and update expressions are reused from sealed source.
Only two redundant independent-primal calls are replaced by the values already
computed in the backward call. This changes measurement scope, not the objective
or optimizer. The outer summary explicitly records primal_from_backward_call
and derivative_qualified=false; legacy scalar field names inside loop artifacts
must be interpreted under that scope. Smoke is finite-backward evidence only;
it does not certify precise derivatives, reproducibility or chemical improvement.
Old FD/replay failures, coarse contradictions and original results stay visible.
Flat components, inconclusive FD, validity drops and nonmonotone loss are findings.

Own bounded output caps are2GiB/12,000files per arm. Live exact quota admission
passed with57.04GB available versus11.01GB required for the original five-arm
group (including its reserve). The three-arm replacement passed with about50.6GB
headroom. Existing bytes are counted once; unrelated account provenance is
advisory. Entry and300-second periodic checks enforce own caps/capacity. No cleanup.
Whole measured remaining horizons were requested. Timings from completed actual
updates are recorded in each plan; DBMol uses the largest23h allocation and a
bounded internal deadline, preserving any partial prefix if its target cannot fit.

Plain Pro's earlier29853431 stopped at27,175 because the historical periodic
account guard rejected unrelated queued DeFoG29853638, despite ample capacity.
Its exact latest-state checkpoint is preserved and bound in plan-protenix.json;
the new child continues from it. Old source05/callback/failure remain unchanged.

Source01 is frozen at ROOT/source for the two Pro jobs. A post-submission import
origin check caught absent inherited science/src for native Boltz children before
any execution. The site rejected bmod with "Only administrators can run bmod".
Only exact owned unstarted29853758/29853760/29853761 were verified PEND/PSUSP,
never dispatched and without payload, then intentionally retired. No model or
update ran. Source02 and the rejected repair remain preserved. Fresh source03
adds child PYTHONPATH binding and a new retry output root for the three replacement
jobs; settings/input bytes are identical. The two Pro jobs, pre-existing queued
diagnostics, callbacks, other projects and saved studies were not modified.

ROOT is /work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/exploratory-continuations-20261030-01.
Boltz/DB replacements write ROOT/retry02/ARM/payload/study; Pro jobs write
ROOT/ARM/payload/study. There are no optimizer callbacks. Monitor actual saved
prefixes/errors and refresh the existing canonical report when data changes.
Do not replay any submission intent or restart a live writer.

Commands: bash -l SOURCE/exploratory_continuation_stage.sh SOURCE ARM run.
The exact bsub vectors are in submission-proofs.json. Five local stdlib metadata/
AST tests passed; actual original evaluation AST shapes matched. All5 real
metadata startup commands passed before first submission; replacement commands
also passed startup and child package-origin checks. These are not native tests.
Evidence is results/hpc/20261030-exploratory-continuations, especially
delivery-20261030T121234Z; packetSHAaea004354bacdae80cd05fb0523b571a846fbd42bf7019df4c691e1bf9f5ccd0.
Tracked plan mirrors use LF; the authenticated packet retains exact remote bytes
and their original SHA/length bindings. No remote source was normalized or changed.
