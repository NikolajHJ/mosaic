"""Paired search readouts for three latent representations of the same hard molecule."""

import math

import torch

from latentmol.design.decoder_comparison import permute_for_native


ARM_NAMES = ("original_random", "reencoded_mean", "posterior_sample")


class OrderedScorer:
    """Present the same atom ordering to the resident protein/ligand pose in every arm."""

    def __init__(self, resident, active, order, native_active):
        self.resident = resident
        self.order = torch.as_tensor(order, device=active.device, dtype=torch.long)
        if (self.order.shape != active.shape
                or not torch.equal(self.order.sort().values, torch.arange(active.numel(), device=active.device))
                or not torch.equal(active[self.order], native_active.to(active))):
            raise ValueError("Native order must map each fixed active slot into the shared ligand prefix once")
        self.identity = dict(resident.identity, native_to_decoder_slots=self.order.cpu().tolist())

    def __call__(self, graph):
        # Torch indexing carries the native VJP back through the inverse permutation.
        return self.resident(permute_for_native(graph, self.order))

    def close(self):
        self.resident.assert_unchanged()


def matched_fresh_comparison(study, fresh, candidates):
    """Count confirmed distinct hits per parent and arm under the same fresh budget.

    Each selected candidate, including an uncommitted trial, is eligible. A
    missing baseline makes its comparisons unevaluable. Failed edit rescoring
    stays visible as unknown even when other edits have confirmed improvements.
    """
    planned = study["optimization_scorer_budget_per_arm"]
    edit_cap = study["fresh_edit_limit_per_arm"]
    if fresh["max_edit_candidates_per_arm"] != edit_cap or planned < 1:
        raise ValueError("Study and fresh-score budgets differ")
    scored = fresh["candidates"]
    keys = ("start", "arm", "role", "step", "smiles", "on_committed_trajectory")
    if [tuple(row[k] for k in keys) for row in candidates] != [tuple(row[k] for k in keys) for row in scored]:
        raise ValueError("Fresh scores do not match the preselected candidates")
    panels = {}
    for row in scored:
        if (row["loss"] is None) != (row["status"] != "scored") or (
                row["loss"] is not None and not math.isfinite(row["loss"])):
            raise ValueError("Missing or nonfinite scored candidate loss")
        panels.setdefault((row["start"], row["arm"]), []).append(row)
    expected = {(parent, arm) for parent in study["starts"] for arm in ARM_NAMES}
    first, stop = study.get("parent_slice", [0, study["same_molecule_panel"]["parents"]])
    if set(panels) != expected or len(study["starts"]) != stop - first:
        raise ValueError("Incomplete matched-parent comparison")
    by_parent = []
    for parent, arms in study["starts"].items():
        parent_smiles = None
        parent_loss = None
        measures = {}
        for index, arm in enumerate(ARM_NAMES):
            trajectory = arms[arm]
            actual = trajectory["scorer_budget"]["actual"]
            if actual != planned:
                raise ValueError("A matched arm did not consume the declared search budget")
            panel = panels[parent, arm]
            base, *edits = panel
            if (base["role"] != "initial" or base["step"] != 0 or len(edits) > edit_cap
                    or len({r["smiles"] for r in panel}) != len(panel)
                    or any(r["role"] == "initial" for r in edits)):
                raise ValueError("Invalid baseline or duplicate fresh candidate")
            if index == 0:
                parent_smiles, parent_loss = base["smiles"], base["loss"]
            if (base["smiles"] != parent_smiles or base["loss"] != parent_loss
                    or trajectory["candidates"]["initial"]["smiles"] != parent_smiles):
                raise ValueError("Arms did not share the same fresh parent")
            available = [r for r in edits if r["loss"] is not None]
            hits = [] if parent_loss is None else [r for r in available if r["loss"] < parent_loss]
            best_gain = max((parent_loss - r["loss"] for r in hits), default=None)
            measures[arm] = dict(optimization_calls=actual,
                distinct_valid_edits=trajectory["fresh_selection"]["distinct_valid_edits"],
                committed_distinct_valid_edits=trajectory["candidates"]["distinct_valid_edits"],
                raw_distinct_valid_edits=trajectory["all_scored"]["unconditioned"]["distinct_valid_edits"],
                selected_edits=len(edits), scored_edits=len(available),
                unscored_edits=len(edits) - len(available),
                confirmed_improving_edits=None if parent_loss is None else len(hits),
                committed_improving_edits=None if parent_loss is None else sum(r["on_committed_trajectory"] for r in hits),
                trial_only_improving_edits=None if parent_loss is None else sum(not r["on_committed_trajectory"] for r in hits),
                best_improvement=best_gain)
        by_parent.append(dict(parent=parent, smiles=parent_smiles, baseline_loss=parent_loss, arms=measures))
    totals = {}
    for arm in ARM_NAMES:
        all_rows = [p["arms"][arm] for p in by_parent]
        comparable = [r for r in all_rows if r["confirmed_improving_edits"] is not None]
        hits = sum(r["confirmed_improving_edits"] for r in comparable)
        calls = sum(r["optimization_calls"] for r in comparable)
        totals[arm] = dict(parents=len(all_rows), scorable_parents=len(comparable),
            optimization_calls_from_scorable_parents=calls,
            parents_with_confirmed_improvement=sum(r["confirmed_improving_edits"] > 0 for r in comparable),
            confirmed_improving_edits=hits if comparable else None,
            confirmed_improvements_per_100_search_calls=100 * hits / calls if calls else None,
            committed_improving_edits=sum(r["committed_improving_edits"] for r in comparable) if comparable else None,
            trial_only_improving_edits=sum(r["trial_only_improving_edits"] for r in comparable) if comparable else None,
            **{k: sum(r[k] for r in all_rows) for k in ("optimization_calls", "distinct_valid_edits",
                "committed_distinct_valid_edits", "raw_distinct_valid_edits", "selected_edits", "scored_edits", "unscored_edits")})
    return dict(status="completed", per_parent=by_parent, totals=totals, parent_slice=[first, stop],
                panel_sha256=study["same_molecule_panel"]["sha256"],
                native_config_sha256=study["native_config_sha256"],
                checkpoint_sha256=study["checkpoint_sha256"], geometry_seed=fresh["geometry_seed"],
                search_calls_per_parent_arm=planned, fresh_edit_cap_per_parent_arm=edit_cap,
                unique_fresh_scores_attempted=fresh["unique_candidates"],
                improvement="Lower fresh native FILE contact loss than the same parent at the same geometry seed.",
                yield_denominator="Optimization calls for parents whose shared fresh baseline could be scored.",
                coverage="Counts include only selected edits with a measured gain. An unselected or unscored edit has unknown benefit.")
