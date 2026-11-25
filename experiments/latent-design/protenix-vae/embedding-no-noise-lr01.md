# Lower-rate, no-noise mapping branch

Question: does the trained epoch-45 VAE/Protenix mapping preserve the early
decoded-graph gain when continued with SGD LR0.1 and no latent noise?

The user authorized this branch on 3 November 2026. It starts from the saved
latent at source update180, preserving the original25 occupied slots and global
temperature/chemistry schedules. That source state was valid and had observed
decoded relaxed-graph contact0.390198. This value is not a discrete hard-graph
rescore or evidence of affinity/binding improvement.

[Config](embedding-no-noise-lr01.json) requests10,000 additional updates,
absolute schedule steps180--10180. It uses the same frozen epoch45 checkpoint,
trained mapping, native Protenix context and contact objective. The mapping is
reused rather than refitted. The source raw-block SHA is
`e164312e9a66a27ed75c0300f04faa34d915be9663bef42bc88ca81c4a97269f`.
Snapshots remain every10 updates and resumable
checkpoints every100; the maximum displacement bound remains0.5.

Relative to the original branch, LR changes0.3→0.1 and latent noise changes to
zero. Both changes occur together, so this branch cannot isolate the learning
rate effect from the noise effect. Chemistry guidance is unchanged; the ordinary
decoded graph remains observable while the binding path consumes learned native
chemical projections.

No separate mapping qualification or pilot is required for this reuse. The
ordinary run retains finite/attached-path, shape and frozen-model checks. Poor
scores, invalid molecules and nonmonotone loss are recorded findings.

Submitted as **29881664**, queue `gpuv100`, one32GB V100/fourCPUslots/32GiB
hostmemory, four-hour maximum with3h50 internal budget. LSF confirms RUN from
20:23:16CET on3November, host n-62-11-14. At20:23:28 it was loading the native
model; no optimization update was saved yet. Twenty focused local cases passed
for the latent loop, explicit warm start and zero-noise behavior. Output home:
`/work3/s234818/latentmol/runs/protenix-embedding-no-noise-lr01-20261103a`.
Snapshot SHA256 `62ba53a06f6946fc8f0a90cb7fcdb96a8d8335e1b2b3022929e2d68bd6d37848`.
Receipts are under `outputs/hpc-submissions/protenix-embedding-no-noise-lr01-20261103a`.

At20:24:27CET, actual optimization has begun: the first metric at source180
reproduces decoded relaxed contact0.3901983798 and mapped contact0.6516512632;
latest logged state182 means two new updates have been applied. All recorded
noise norms are zero. The first displacement was0.08641, compared with0.28149
for the original noisy LR0.3 update from the same point. No outcome conclusion
is drawn from two updates; the latest raw graph is invalid and the run continues.

At20:43:55CET, job29881664 remains RUN at global1705, i.e.1525additional
updates. The decoded relaxed contact did not hold the starting0.3902: it fell
to0.2121 at190 (ten new updates), and0.2042 at1700. The original noisyLR0.3
trajectory scores0.2273 at that same global1700. Over the shared180--1705
window the new mean decoded score is0.2121 versus0.2563 originally. Raw-valid
states are12/1526 (0.79%) versus421/1526 (27.59%) in the original window.
The latest new state is invalid. Mapped self-contact at1700 is0.8345, again
substantially above decoded contact. This early paired result does not support
an LR/noise-only explanation of the original decline; the two settings changed
together and their individual effects are not isolated. The bounded run
continues without a quality-based stop.
[Progress receipt](../../../outputs/hpc-results/protenix-embedding-no-noise-lr01-20261103a/progress-1793735032702672900.json).

Job 29881664 completed all 10,000 additional updates on 3 November at
22:32:42 CET. The terminal result passed the finite-path and frozen-model
checks. Decoded relaxed contact fell from 0.3902 to 0.1988, while contact
on the learned embedding rose from 0.6517 to 0.8464. The endpoint failed
raw graph validity. The starting state remained the best valid state
among the observations scored every 10 updates. The only valid scored
observation after the start was at step 200, with decoded contact 0.2151.
Raw graph validity was checked at each update, and 145 of 10,000
post-start states were valid.

Over the shared schedule steps 180–10,000, mean decoded contact was
0.2022 versus 0.2208 on the original noisy LR0.3 trajectory, and raw
validity was 112/9,821 (1.14%) versus 5,512/9,821 (56.12%). At step
10,000, decoded contact was 0.1983 versus 0.2582. This branch changed
both the learning rate and noise, so their individual effects remain
unresolved.
[Terminal receipt](../../../outputs/hpc-results/protenix-embedding-no-noise-lr01-20261103a/terminal-check-1793791956403520400.json).
