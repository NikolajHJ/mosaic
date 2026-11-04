# Projection feedback over 20,000 updates

Projection feedback kept more optimization steps valid and finished at a higher
hard-graph contact score. The archive-only arm found the highest-scoring valid
candidate and yielded more distinct valid molecules. For candidate discovery at
this starting checkpoint, the archive-only arm performed better.

| Outcome | Archive only | Projection feedback |
| --- | ---: | ---: |
| Best archived valid hard contact | 0.546988 | 0.524529 |
| Final hard contact, both valid | 0.376663 | 0.524529 |
| Valid argmax at update sources | 16,306 / 20,000 | 19,070 / 20,000 |
| Distinct valid canonical SMILES | 128 | 20 |
| Valid unfiltered draws across 40 boundaries | 629 / 5,120 | 29 / 5,120 |
| Valid unfiltered draws at the endpoint | 31 / 128 | 0 / 128 |
| Feedback resets | 0 | 6 |
| Elapsed per arm | 2 h 45 min 7 s | 2 h 48 min 51 s |

Both arms started from the same saved valid repair at parent update 30,000,
with 2% uniform support, temperature 0.4, chemistry weight 30, learning rate
0.3, zero noise and model key 17. Both used 40 blocks of 500 updates and 5,161
hard scoring calls. At each boundary both arms considered the valid raw argmax
and valid unfiltered proposals for their archive. The feedback arm could project the current best
archived candidate into the next block. It applied six resets through update
16,000. The diversity count includes distinct valid trajectory and sampled
identities. Valid update sources exclude the extra endpoint evaluation in each
block.

The 2,000-update pilot favored feedback on best archived contact, 0.434865 versus
0.414022. The longer comparison reversed that ranking. It evaluates one parent
and one model key, and the scores measure geometry-off Protenix contact in a
fixed context. Affinity and experimental binding remain unmeasured.
The arms ran serially, so compilation order limits runtime comparison.

DTU HPC job `29888169` completed on V100 on 4 November 2026 at 10:53:53 CET.
Both update prefixes, same-job LSF success, source and input bindings, and
frozen model/context preservation were verified. The original result is
`/work3/s234818/mosaic/runs/20261104-feedback-confirmation-01/payload/result.json`
with SHA-256 `3359b5a86f79a62f5bb80686a253f1bdab9df48c429e0229f9a2207ef413e38c`.

[Terminal verification and artifact bindings](https://github.com/NikolajHJ/mosaic/blob/30ff53ec821b62f6afbd6fa7309527b248b14c22/experiments/brd4-jq1/protenix-random-chem/runs/20261103-adaptive-validity-research-01/feedback-confirmation-terminal-verification.json).
