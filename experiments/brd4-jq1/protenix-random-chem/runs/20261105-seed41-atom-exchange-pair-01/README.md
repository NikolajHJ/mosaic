# Seed 41 matched optimization with atom exchange

DTU job **29898292** completed on one L40S on 5 November 2026 at 12:34 CET. Both arms started from the exact FP32 logits saved at update **50,000** in seed 41 job **29880095** and completed 2,000 additional updates. The experiment stores local steps 0–2,000 and their parent offset 50,000. All 34 allocated checks passed. [Campaign and input pins](campaign.json)

## Matched settings

Both arms use the original frozen Protenix Tiny model, context, pocket and key 17. The endpoint policy holds learning rate 0.3, temperature 0.4, chemistry coefficient 30, zero update noise and always-on contact. The original optimizer applies one finite fixed candidate per update. Both arms use the same repair loss, common-organic scope and atom-preserving fallback policies. Preparation runs afresh before the first update and holds the selected target fixed during each trial.

The control uses bond-only structural proposals. The treatment can also try up to four supported one-atom substitutions, selecting at most one per target and adding one edit unit to the cost. Both arms enforce the sulfur coordination rule. Introduced sulfur also has to fit the previous atom's bond-order cap.

## Measurements

- Count raw valid connected argmax graphs under native chemistry and under native chemistry plus the shared common-organic rule. The saved seed 41 start passes native validation and falls outside the stricter scope. Keep raw validity distinct from repair-target availability.
- Save relaxed contact at every step. Rescore the hard argmax on the same 21-point grid in each arm, including start and endpoint. Report best and endpoint scores and whether those scored graphs satisfy each validity rule. This is frozen Protenix contact.
- Record exact atom labels and a hash of the slot-aligned bond labels for each complete target. Count flips between adjacent applied update sources with an active complete target at both steps. Record the denominator, atom-only, bond-only and joint flips, target activations and deactivations, and atom substitutions relative to the source state. The unused endpoint target is outside the churn denominator.

The immutable job runs allocated preparation and gradient fixtures before model startup. The result is written to `/work3/s234818/mosaic/runs/20261105-seed41-atom-exchange-pair-01/payload/result.json` and includes both arms' raw ledger bindings and per-arm summaries.

## Results

| Measure | Bond only | Atom exchange |
| --- | ---: | ---: |
| Native-valid connected states | 1,997 / 2,001 | 1,861 / 2,001 |
| Valid connected states under the shared sulfur scope | 0 / 2,001 | 1,841 / 2,001 |
| Distinct scoped-valid graphs | 0 | 1 |
| Scoped-valid endpoint | No | Yes |
| Final relaxed contact | 0.263162 | 0.316562 |
| Final hard contact | 0.215035 | 0.186648 |
| Best scoped-valid hard contact on the grid | — | 0.186648 |
| Update sources with a complete target | 0 / 2,000 | 160 / 2,000 |
| Target label flips among consecutive active pairs | Undefined (0 pairs) | 1 / 62 |

The atom-exchange arm first reached scoped validity at local update 63 and ended with eight consecutive scoped-valid states. Its one flip changed both atom and bond labels. Complete targets also activated 97 times and deactivated 98 times between update sources. Of the 22 targets that changed an atom relative to that step's source, 11 changed sulfur to carbon, nine changed sulfur to nitrogen, and two changed nitrogen to carbon. The bond-only arm's sulfur environment stayed out of scope throughout. Its inactive complete-target metric provides no flip-rate comparison.

The starting hard contact was 0.203096 for both arms. Atom exchange resolved the scoped validity problem on this checkpoint, with a lower hard endpoint score despite a higher relaxed endpoint score. The hard comparison uses 21 equally spaced saved states per arm. [Terminal verification](terminal-verification.json)
