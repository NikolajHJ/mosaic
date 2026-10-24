# Exact local backup audit

Local stdlib byte/hash/tar/JSON only; no SSH/deletion/numerics. Remote existence, inactivity and dependency exclusions must be authenticated by the root operator before any removal.

| Backup | Files | Exact member bytes | Potential saved/report bytes before remote checks |
| --- | ---: | ---: | ---: |
| higher-rate-report-0001 | 636 | 275673157 | 270784155 |
| higher-rate-report-0002 | 647 | 281789246 | 274409777 |
| lower-rate-report-0001 | 441 | 183494769 | 178228357 |
| original-jq1-edition-0001 | 36 | 1381511 | 1123492 |
| original-jq1-edition-0002 | 154 | 52337075 | 52077624 |
| original-jq1-edition-0003 | 208 | 74695439 | 74436035 |
| original-jq1-edition-0004 | 236 | 88072653 | 87813118 |
| original-jq1-edition-0005 | 294 | 112308604 | 112052552 |
| original-jq1-edition-0006 | 310 | 118855751 | 118599793 |
| historical-brd4-best-primal | 186506 | 1715980264 | 733232757 |
| historical-brd4-delayed-chemistry | 61267 | 1289880745 | 385937734 |
| historical-brd4-chemistry | 15995 | 436122258 | 155691981 |

Previous cleanup already removed 6999317068 logical bytes; those exact paths are excluded.

Each compressed JSONL inventory gives exact remote path, archive/member, length and SHA. These are candidate mappings, not a claim the remote copies still exist or can be removed without checking current dependencies.

Report probability JSON, plots and exported chunk assets are backed only as those report files. They do not back original NPZ or JSONL. The separately listed original Protenix summary is a single exact original file backup.
