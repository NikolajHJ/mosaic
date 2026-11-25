"""Select unmodified CPU prior draws for the first wave of a restart search."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from math import inf
from time import perf_counter, process_time

import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import choose_active, hard_from_graph, structural_target
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.evaluation.quality import diagnose_graph
from latentmol.graph_policy import graph_from_decoder


def _draw(decoder, design, seed, name):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn((1, decoder.config.latent), generator=generator)
    decoded = decoder.decode(latent)
    if not all(torch.isfinite(x).all() for x in (decoded.atoms, decoded.charges, decoded.bonds)):
        raise ValueError("Nonfinite prior decoder logits")
    active = choose_active(decoded, decoder.spec, design.occupied_atoms)
    graph = graph_from_decoder(decoded, decoder.spec, fixed_active=active, temperature=design.temperature_initial)
    hard = hard_from_graph(graph)
    return SearchPoint(Candidate(None, latent, hard, 0., name, 0), active, None, (name,)), graph


@torch.no_grad()
def bootstrap(decoder, design, *, seed=None, name="bootstrap"):
    point, _ = _draw(decoder, design, design.seed if seed is None else seed, name)
    return point


def _smiles(hard, spec):
    return inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec).smiles


@dataclass(frozen=True)
class _Draw:
    seed: int
    point: SearchPoint
    category: str
    smiles: str | None
    components: int
    problem_burden: int | None
    overvalence: float | None
    witness_smiles: str | None = None
    edits: int | None = None
    checks: int = 0

    @property
    def molecule(self):
        return self.smiles or self.witness_smiles

    @property
    def defect_rank(self):
        burden = self.problem_burden if self.problem_burden is not None else inf
        return (self.components - 1 + burden, self.components,
                self.overvalence if self.overvalence is not None else inf, self.seed)


def _classify(decoder, design, seed, index):
    point, _ = _draw(decoder, design, seed, f"bootstrap-{index + 1:02d}")
    hard = point.candidate.hard
    diag = diagnose_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec)
    return _Draw(seed, point, "valid" if diag["valid"] else "unchecked_invalid", diag["smiles"],
                 diag["component_count"], diag.get("problem_burden"), diag.get("overvalence_excess"))


def _check_witness(draw, decoder, design, near_edit_limit):
    point = draw.point
    graph = graph_from_decoder(decoder.decode(point.candidate.latent), decoder.spec,
                               fixed_active=point.active, temperature=design.temperature_initial)
    target, report = structural_target(graph, decoder.spec, point.active, design, step=0)
    edits = report["edit_count"]
    if target is not None and type(edits) is int and edits >= 1:
        witness = _smiles(target.graph, decoder.spec)
        if witness is None:
            raise ValueError("Bootstrap repair witness failed raw graph validation")
        return replace(draw, category="near_valid" if edits <= near_edit_limit else "farther_witness",
                       witness_smiles=witness, edits=edits, checks=report["checks"])
    return replace(draw, category="no_witness_found", checks=report["checks"])


def _select(draws, count):
    # Reserve the same first prior draw as a paired unscreened panel. Keep a
    # near-valid slot when possible, then fill with distinct raw-valid molecules.
    selected = [(draws[0], "random")]
    seen = {draws[0].molecule}
    near = sorted((draw for draw in draws[1:] if draw.category == "near_valid"),
                  key=lambda draw: (draw.edits, draw.seed))
    valid = [draw for draw in draws[1:] if draw.category == "valid"]
    farther = sorted((draw for draw in draws[1:] if draw.category == "farther_witness"),
                     key=lambda draw: (draw.edits, draw.defect_rank))

    def add(draw, role):
        selected.append((draw, role))
        seen.add(draw.molecule)

    if count >= 3:
        reserve = next((draw for draw in near if draw.molecule not in seen), None)
        if reserve:
            add(reserve, "near_valid")
    for draw in [*valid, *near, *farther]:
        if len(selected) == count:
            break
        if draw.molecule not in seen:
            add(draw, draw.category)
    chosen_seeds = {draw.seed for draw, _ in selected}
    for draw in sorted(draws[1:], key=lambda d: d.defect_rank):
        if len(selected) == count:
            break
        if draw.seed not in chosen_seeds:
            add(draw, "fallback_duplicate" if draw.molecule in seen and draw.molecule else "graded_fallback")
    return selected


@torch.no_grad()
def screened_bootstrap(decoder, design, *, seed, count=6, pool_size=128, near_edit_limit=2, repair_pool_size=128):
    """Pre-screen a bounded seeded pool by chemistry on CPU, without binding calls.

    A repair witness classifies the original draw. Returned latent, hard graph
    and active mask always come from that unmodified draw.
    """
    if (type(count) is not int or type(pool_size) is not int or not 2 <= count <= pool_size
            or type(seed) is not int or not 0 <= seed < 2**63 - pool_size + 1
            or type(near_edit_limit) is not int or near_edit_limit < 1
            or type(repair_pool_size) is not int or repair_pool_size < 1):
        raise ValueError("Require a bounded pool covering every bootstrap slot, valid seeds and positive edit limit")
    wall, cpu = perf_counter(), process_time()
    draws = [_classify(decoder, design, seed + i, i) for i in range(pool_size)]
    invalid = sorted((draw for draw in draws[1:] if draw.smiles is None), key=lambda d: d.defect_rank)
    checked = invalid[:repair_pool_size]
    for draw in checked:
        draws[draw.seed - seed] = _check_witness(draw, decoder, design, near_edit_limit)
    chosen = _select(draws, count)
    receipt = dict(seed=seed, pool_size=pool_size, near_edit_limit=near_edit_limit, draws=len(draws),
                   pool_categories=dict(Counter(draw.category for draw in draws)),
                   repair_pool_size=repair_pool_size, repair_attempted=len(checked),
                   repair_checks=sum(draw.checks for draw in draws), native_scoring_calls=0,
                   defect_order="component_count - 1 + RDKit problem_burden, component_count, observed overvalence, seed",
                   selected=[dict(slot=i, seed=draw.seed, role=role, category=draw.category,
                                  smiles=draw.smiles, witness_smiles=draw.witness_smiles, witness_edits=draw.edits,
                                  component_count=draw.components, problem_burden=draw.problem_burden,
                                  observed_overvalence=draw.overvalence)
                             for i, (draw, role) in enumerate(chosen, 1)],
                   elapsed_seconds=perf_counter() - wall, cpu_seconds=process_time() - cpu)
    return [draw.point for draw, _ in chosen], receipt
