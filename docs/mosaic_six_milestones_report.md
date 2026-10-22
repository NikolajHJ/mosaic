# Mosaic: six research milestones, 16–20 October 2026

> **Scope and access note (22 October 2026):** this document contains an
> experiment protocol and dated operator evidence. Concrete account paths,
> environments, jobs and source hashes describe that operator/attempt; they
> are not shared-account defaults or a fresh status/acceptance claim. Use
> [your own HPC setup](HPC_GUIDE.md). Keep the scientific gates and historical
> outcomes below intact when preparing a separately validated candidate.
> Canonical working home: [reporting/milestones](../experiments/reporting/milestones/README.md).


Prepared 21 October at the user's request for one report covering the most
important early and recent Protenix/Boltz-2 results. This is a chronological
selection, not a ranking by loss. The user clarified to include DeFoG only if
it actually ran in Mosaic. Repository code, plans and experiment history contain
no completed Mosaic DeFoG study. The DBMol-inspired rule-based repair loss is
included under its actual name; no separate DBMol experiment is imported.

## Selected trajectories

| Milestone | Native job | Selected input | Updates / saved states | Final raw output |
| --- | --- | --- | --- | --- |
| Original Protenix carbon-chain proof | 29735673 | 30% mixture, case 1 | 100 / 101 | Valid decane |
| BRD4 contact-only | 29745503 | 50% mixture, combined-report case 2 | 5,000 / 5,001 | Invalid fluorine valence |
| First chemical-repair optimization | 29753152 | 50% mixture, constant repair coefficient 0.5 | 5,000 / 5,001 | Valid hexane; repair zero throughout |
| Protenix delayed chemistry, mix | 29753667 | 50% mixture | 10,000 / 10,001 | Valid hexane; repair zero throughout |
| Protenix delayed chemistry, random | 29753668 | Dirichlet(1), seed 20261019 | 10,000 / 10,001 | Valid `FCCC(F)F` |
| Boltz-2 affinity + delayed chemistry | 29759212 | 50% mixture | 10,000 / 10,001 | Invalid chlorine valence |

All **40,106** saved states of these selected inputs remain available at unit
update resolution. Initial/final metrics come directly from the saved reports.
Selected intermediate jumps include Protenix random state 8959 and Boltz states
8294/9639/9999/10000. The latter retain the invalid best-affinity hard graph,
valid `SCCCCCl`, last valid state and invalid final graph respectively.

## Presentation and provenance

`scripts/assemble_milestone_report.py` composes a self-contained HTML using
Python standard-library byte/JSON operations. It authenticates all five source
HTML files and their existing PNG figures against their preserved evidence
manifests or report receipts before extracting the selected saved records.
Legacy fields `loss`/`E_upper` are exposed as `contact`/`E`; their values are
unchanged. Redundant pre-rendered SVG strings are omitted. All raw probability,
readout and captured component records are retained; absent scientific values
are not reconstructed. Payload serialization roundtrips, exact gzip/unpacked
SHA bindings, contiguous state schemas and declared slot/category counts are
checked as standard-library data-integrity work.

`scripts/templates/mosaic_milestones.html` presents six selectors, endpoint
summaries, playback, single-state navigation, probability heatmaps, raw graph
depictions, original loss figures, a comparison table and source provenance.
Graph presentation uses the saved hard readout, existing fixed-slot viewer
conventions, winning-element circle sizes and bond-existence line widths.
Invalid and disconnected outputs remain visible. Every original scientific
plot is copied byte for byte, with its original control curves clearly labeled.
No new scientific plot, model output, chemical evaluation, repair, sampling or
numerical validation was performed locally or on HPC for this composition.

Targets/atom counts differ for the original synthetic-pocket study, and contact
and affinity use different definitions/scales. They are never overlaid on one
shared score axis. Logit versus projected-probability optimization and historical
acceptance-policy exposure are disclosed. Refreshed repair totals are not one
fixed comparable objective. Boltz retains its fixed-pose, zero-recycle,
no-molecular-weight-correction partial-gradient scope; binder probability remains
separate. No hard molecule is assigned a fabricated fresh affinity.

## Verified delivery

Local command, standard-library presentation only, Windows Python 3.13:

```powershell
py -3 -u scripts/assemble_milestone_report.py --output results/hpc/20261021-mosaic-six-milestones/edition-01
py -3 -u scripts/serve_protenix_report.py results/hpc/20261021-mosaic-six-milestones/edition-01 --port 8775
```

Fresh output, preserving all earlier reports and servers:
`results/hpc/20261021-mosaic-six-milestones/edition-01/`.
HTML **52,444,743 bytes**, SHA-256
`625c7d8a6b88dcbf98bad1b0d2bdb9182ee40a69ee8786a582b6c8c5506babe4`.
Detached loopback **http://127.0.0.1:8775/report.html** returned HTTP 200;
the full served length/SHA exactly matched the file. The self-contained file
can be copied independently; its download button is included. Browser QA used
loopback HTTP, not a separate `file:` navigation test.

Actual browser checks passed all six selectors/endpoints and plot loads,
10 versus 6 readable atom-slot labels, Boltz valid state 9639/invalid endpoint,
six atom/fifteen bond heatmap rows, playback/pause at observed updates
25/75/100 and absence of console warnings/errors. The tab is marked deliverable.
`presentation_receipt.json`, `browser-qa.json`, `report-preview.png` and the
parent's `server.json` retain exact publication/evidence details. Static Python
AST parsing, Node JavaScript syntax and Git whitespace checks passed. Ruff was
unavailable in the local presentation Python (`No module named ruff`); no Ruff
or numerical-test pass is claimed. No native experiment or gate was changed.

The separate random Boltz diagnostic remains recorded with its original failed
native gate; its 10,000-update optimization is not submitted. Continuous-chemistry
and learning-rate/schedule proposals remain untested and are not included as
completed results. Unrelated `project_desc.pdf` moves and OneNote files remain
untouched.
