# Completed chemistry comparisons — 1 November 2026

Copied from original saved production summaries at 2026-11-01T20:36:03.958365+00:00. All13 rows reached20,000 applied updates and report zero strict-valid connected argmax states. Original fields are retained in completed-summary.json; this summary performs no new chemistry, scoring, sampling or array analysis.

The interactive report remains edition0014 while the queued update29870705 waits: all DTU queues report Open:Inact_Adm. The requested edition0015 will contain20 panels after allocated rendering and publication complete.

| Run | Contact loss, lower is better | Valid argmax states | Valid endpoint samples | Recorded loop seconds |
| --- | ---: | ---: | ---: | ---: |
| Sparse seed17 - delayed chemistry | -0.882516622543335 | 0 | 0/128 | 15016.573369299062 |
| Sparse seed23 - delayed chemistry | -0.669877827167511 | 0 | 0/128 | 16110.804249790031 |
| Sparse seed17 - constant chemistry 1 | -0.8737822771072388 | 0 | 0/128 | 15141.484446319984 |
| Sparse seed23 - constant chemistry1 | -0.7456281781196594 | 0 | 0/128 | 15122.523607730982 |
| Sparse seed17 - constant chemistry10 | -0.7074814438819885 | 0 | 0/128 | 15623.014323750045 |
| Sparse seed23 - constant chemistry10 | -0.3667180836200714 | 0 | 0/128 | 14826.975841340027 |
| Sparse seed17 - severity guidance | -0.8799918293952942 | 0 | 0/128 | 15249.812994687818 |
| adaptive s17 (gpul40s) | -0.5219058990478516 | 0 | 0/128 | 12837.664728051983 |
| adaptive s23 (gpul40s) | -0.6119710803031921 | 0 | 0/128 | 12923.012131436 |
| neighbor s17 (gpuv100) | -0.7371864914894104 | 0 | 0/128 | 19308.174604333006 |
| temperature2 s17 (gpul40s) | -0.5963305234909058 | 0 | 0/128 | 13136.790095995006 |
| Sparse seed17 - combined strategies (A100) | -0.14969739317893982 | 0 | 0/128 | 13035.51235351013 |
| Sparse seed23 - combined strategies (A100) | -0.10922345519065857 | 0 | 0/128 | 13069.733997041127 |

Contact values are relaxed fixed-pose model measurements, not experimental binding evidence. Strategies differ in chemistry schedules, targets, temperature and sometimes GPU, so these timings are observations rather than a controlled speedup comparison. All invalid/disconnected outcomes remain visible in the saved data. No native derivative certification is implied.
