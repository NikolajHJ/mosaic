# Follow-ups to the November 10 restart results

The [completed searches](restart-search.md#completed-results-on-10-november-2026)
left three questions about branch allocation, small parent kicks, and the
stability of the fresh-score ranking. Each comparison uses the frozen
BRD4 scoring setup and keeps the conditions being compared on one GPU.

| Question | Config | Budget and readout |
| --- | --- | --- |
| Does reserving two of six slots for independent starts help? | [Hybrid allocation](restart-lineage-hybrid.json) | Three matched parent panels. Each compares six adaptive starts, four adaptive plus two independent, and six independent. Five cycles of 96 updates per child give 2,880 updates and up to 90 fresh attempts per policy. Compare best loss, mean top-three loss, and diverse low-loss yield at matched budgets. |
| Does a radius of 0.003 still split sibling trajectories at 500 updates? | [Parent kick](restart-tiny-kick.json) | Two panels reuse the four parents and seed streams of the 128-update comparison. Compare 0, 0.003, and 0.03 with three repeats per parent, LR 70, cap 0.25, and no update noise. Each job spends 9,000 updates and up to 36 fresh attempts per radius. Read endpoint identity, path similarity, parent retention, validity, and fresh loss together. |
| Does the wave-7 winner still beat the wave-3 incumbent under fresh geometry? | [Conformer re-score](restart-conformer-rescore.json) | A frozen eight-molecule panel from job 29943712, including both incumbents. One historical seed and three additional conformer seeds give 32 attempts and zero optimizer updates. Compare the paired incumbent gap and the panel ranking across seeds, including scoring failures. |

The [shared launcher](restart-operations/submit_campaign.py) stages these
configs as `protenix-restart-lineage-hybrid-20261110a`,
`protenix-restart-tiny-kick-20261110a`, and
`protenix-restart-conformer-rescore-20261110a`. Each submission directory
under `outputs/` holds its sealed code, commands, and scheduler receipts.

| Task | Queue | DTU job |
| --- | --- | ---: |
| Hybrid panel A | A100 | 29973702 |
| Hybrid panel B | L40S | 29973703 |
| Hybrid panel C | V100 | 29973704 |
| Parent kick panel A | L40S | 29973695 |
| Parent kick panel B | V100 | 29973697 |
| Conformer re-score | L40S | 29973777 |

Hybrid panel B finished at 17:45 CET on 10 November. All three policies
completed 2,880 updates and 90 fresh attempts. Adaptive scored 89 and the
other two scored 90. Adaptive and hybrid found the same best molecule
at loss 2.25195. Independent reached 2.28914. The diverse low-loss counts
were 6, 5, and 2 for adaptive, hybrid, and independent, respectively.
[Panel B result](../../../outputs/restart-finished-20261110/29973703-result.json).

On 11 November, the conformer re-score (**29973777**) completed at 04:42
CET with 29 successful scores from 32 attempts. The wave-7 winner beat
the wave-3 incumbent on all three held-out geometry seeds. The mean gap
was 0.00786, versus the original 0.00949. One other candidate failed
embedding on all three held-out seeds. [Re-score result](../../../outputs/restart-finished-20261111/29973777-result.json).

The no-bond weighting pilot (**29948080**) completed at 05:40 CET with
36 successful fresh scores per treatment. At 25% mixing, 8× no-bond
weight gave the largest mean gain (0.29351) and lowest loss (2.28174),
winning on two of three parents against 1× weighting. The stronger
weight lowered initial soft bond mass and reduced trajectory validity
at both tested mixing doses. [No-bond result](../../../outputs/restart-finished-20261111/29948080-result.json).

At 23:09 CET on 11 November, hybrid panels A (**29973702**) and C
(**29973704**) were also verified completed, with worker return code zero.
Each policy completed 2,880 updates and 90 fresh attempts per panel.
Panel A favored hybrid: best losses were 2.24636 adaptive, 2.21349 hybrid
and 2.22391 independent; diverse low-loss counts were 6/13/5. Panel C
favored independent on best loss: 2.21838 adaptive, 2.24785 hybrid,
2.19553 independent. Adaptive had the slightly lower top-three mean in C
(2.22486 versus 2.22618 independent), and diverse counts were 21/7/4.
Together with the previously completed B tie, no policy wins all panels.
[Panel A](../../../outputs/restart-finished-20261111/29973702-result.json)
and [panel C](../../../outputs/restart-finished-20261111/29973704-result.json).

The second 500-update parent-kick panel (**29973697**) completed 9,000
updates and 108 fresh attempts. Across its two parents, mean best-score
gains were 0.15546 with no kick, 0.24004 at radius 0.003, and 0.31148 at
0.03. The lowest loss was 2.21614 at 0.03. Each treatment used 3,000
updates and 36 attempts, with 36/35/36 successful scores respectively.
Both perturbation strengths helped relative to no kick on both parents;
0.03 won one parent and 0.003 the other. [Panel B result](../../../outputs/restart-finished-20261111/29973697-result.json).

The earlier 128-update noise-timing panel B (**29948076**) also completed:
6,912 updates and 216 attempts across nine treatments. Parent-only radius
0.003 had the largest mean per-parent best-score gain (0.28526); adding
early update noise lowered that mean to 0.23812 at 1% and 0.20904 at 5%.
This small two-parent panel supports keeping optimization noise off, while
the 500-update panel shows that the preferred parent-kick magnitude can
depend on the optimization horizon. [Timing panel B](../../../outputs/restart-finished-20261111/29948076-result.json).
