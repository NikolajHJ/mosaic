# Seed 41 matched optimization with atom exchange

DTU job **29898292** runs both arms on one L40S, the same GPU family as the parent. Both start from the exact FP32 logits saved at update **50,000** in seed 41 job **29880095**. They each take 2,000 additional updates in fresh output directories. The experiment stores local steps 0–2,000 and their parent offset 50,000. The first post-release check was pending on 5 November 2026 at 12:03 CET. [Campaign and input pins](campaign.json)

## Matched settings

Both arms use the original frozen Protenix Tiny model, context, pocket and key 17. The endpoint policy holds learning rate 0.3, temperature 0.4, chemistry coefficient 30, zero update noise and always-on contact. The original optimizer applies one finite fixed candidate per update. Both arms use the same repair loss, common-organic scope and atom-preserving fallback policies. Preparation runs afresh before the first update and holds the selected target fixed during each trial.

The control uses bond-only structural proposals. The treatment can also try up to four supported one-atom substitutions, selecting at most one per target and adding one edit unit to the cost. Both arms enforce the sulfur coordination rule. Introduced sulfur also has to fit the previous atom's bond-order cap.

## Measurements

- Count raw valid connected argmax graphs under native chemistry and under native chemistry plus the shared common-organic rule. The saved seed 41 start passes native validation and falls outside the stricter scope. Keep raw validity distinct from repair-target availability.
- Save relaxed contact at every step. Rescore the hard argmax on the same 21-point grid in each arm, including start and endpoint. Report best and endpoint scores and whether those scored graphs satisfy each validity rule. This is frozen Protenix contact.
- Record exact atom labels and a hash of the slot-aligned bond labels for each complete target. Count flips between adjacent applied update sources with an active complete target at both steps. Record the denominator, atom-only, bond-only and joint flips, target activations and deactivations, and atom substitutions relative to the source state. The unused endpoint target is outside the churn denominator.

The immutable job runs allocated preparation and gradient fixtures before model startup. The result is written to `/work3/s234818/mosaic/runs/20261105-seed41-atom-exchange-pair-01/payload/result.json` and includes both arms' raw ledger bindings and per-arm summaries.
