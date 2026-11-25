"""Plan the three paired fragmented-start comparisons of DeFoG and repair guidance."""

import argparse
from pathlib import Path

from stage3_campaign import utc, write_json
from stage3_design import _base, constant, expand_runs, load_design, make_bundles


def make_plan(output, campaign_id=None, mock=False):
    output = Path(output)
    campaign_id = campaign_id or output.name
    shared = dict(family="repair-pair", interior_uniform_mass=1e-4,
                  question="Which guidance visits and retains valid connected raw argmax molecules?",
                  requirements=[], comparisons=["repair-pair"])
    repair = dict(_base(), **shared, id="repair-w1", paired_with=None)
    repair["teacher"]["weight"] = constant(0)
    repair.update(chemistry_loss="failure_aware_repair", chemistry_weight=1.0)
    defog = dict(_base(), **shared, id="defog-w0p5", paired_with=repair["id"])
    defog["teacher"]["weight"] = constant(0.5)
    conditions = [repair, defog]

    # Reuse Stage 3 validation and its run/bundle ID and pairing rules.
    # Select only fragmented starts from the validated three-start design.
    base = load_design()
    base.update(conditions=conditions, campaign_id=campaign_id, seeds=[17, 23, 42])
    runs = [r for r in expand_runs(base) if r["initialization"] == "fragmented"]
    for r in runs:
        r["campaign_id"] = campaign_id
    bundles = [{k: v for k, v in b.items() if k != "runs"}
               for b in make_bundles(base, max_arms=2) if b["initialization"] == "fragmented"]
    shared_config = {k: base[k] for k in ("model", "native_weight", "learning_rate",
                                         "updates", "n_atoms", "seeds", "initialization_parameters",
                                         "evaluation")}
    shared_config.update(initializations=["fragmented"], interior_uniform_mass=1e-4)
    plan = {
        "schema": "defog-stage3-v1", "campaign_id": campaign_id, "created_utc": utc(),
        "config": shared_config, "wave": "paired", "mock": mock,
        "execution_mode": "mock" if mock else "native",
        "conditions": conditions, "runs": runs, "bundles": bundles,
        "primary_endpoint": "highest hard Protenix contact among raw sanitized connected argmax molecules",
        "evaluation": "Paired seeds and model keys. Repairs supply loss targets only. "
                      "Score the actual raw argmax before any repair.",
        "shared_graph_policy": "After initialization and after each complete projection, "
                               "mix each atom and bond row with 0.0001 uniform probability mass.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    (output / "bundles").mkdir()
    write_json(output / "plan.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-id")
    parser.add_argument("--mock", action="store_true", help="Create a separate CPU fixture plan")
    args = parser.parse_args()
    plan = make_plan(args.output, args.campaign_id, args.mock)
    print(args.output / "plan.json")
    for bundle in plan["bundles"]:
        print(bundle["bundle_id"])
