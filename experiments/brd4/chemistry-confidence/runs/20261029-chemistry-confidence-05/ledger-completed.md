### 2026-10-29T11:53:23Z — Bounded-slope confidence guidance CPU validated; opt-in only

Implemented optional valid-confidence-huber-v1, coefficient0.01/margin0.2/
slope cap1.0. All atom/bond labels and validity/occupancy masks come from the
detached authenticated source preparation, fixed during each step/trial; target
refresh only between updates. Invalid source rows and confident decisions have
zero margin. Composition remains off; no live atom-valence mixture, repair-ramp
multiplier or optimizer-gradient clipping. Existing failure-aware repair and
default objective remain unchanged, with bitwise prechange-control checks.

Source001/002 pre-submit queue rejections and003 actual29848043 EXIT1 at the
node-local scratch guard before numerical imports remain preserved.004 CPU
29848047 actualDONE/35pass/12matched pairs passed, then child TMPDIR locality
was strengthened in005;004's numerical pass is not relabelled. Exact142
scientific/test/control/compare files are unchanged between004/005.

Final005 CPU29848105 actualDONE/hpcint/n-62-29-6,12:50:51--12:53:18Paris,
35rawcases/zero failures/errors/skips in21.14seconds. Twelve matched pairs,
24chemistry-only128-update trajectories,3seeds/4starts and72raw sample archives
completed. Independent stdlib146source/82evidence-member/rawXML/scheduler/LSF/
schema audit passed. Source manifestdf24a831267b9303748c228fb2aa9d9e31bb471bc0b615946de72a4d2a0716b7,
source archive405849bytes SHA564d994e2cd336cf4ef20d502794e00a1b70f4cea856335edb2ef6c0f87ae58c;
returned339473bytes SHA2b971b0a7e67c36e586b35e5e1cae477ca5cc450a01a09b95814d3813e6884b8.
Both source/output/cache and actual verified child temp paths are node-local;
raw proof/arrays streamed to laptop, no saved-run writes/restarts or current
scientific setting/job changes. General monitor remains paused.

Sampled strict validity improved only0--4/256 in small uncertain cases, was
unchanged for15atoms/3sulfurs and confident controls. C->O integration fixture
passed, but paired chemistry-only runs do not establish exploration under real
binding gradients. Native models/weights/pockets were not evaluated or changed,
VAE training unchanged; no native binding/runtime benefit or production
promotion claimed. Keep experimental/opt-in pending matched full-model evidence.
Own metadata/AST/byte checks passed; full legacy layout audit still reports
preexisting unregistered modules and historical missing READMEs, not a global
layout pass. New shared test has a canonical self-entry, no legacy alias.
Owner: experiments/brd4/chemistry-confidence/runs/20261029-chemistry-confidence-05/.
