"""Authenticate the diagnostic and unchanged protocol for the precision retry."""

import json
from pathlib import Path

from .embedding_experiment import verified, write


VERSION = "epoch36-final-policy-precision-correction-v1"
POLICY = "final_float64_cast_back_v1"
EXPERIMENT_VERSION = "epoch36-matched-embedding-comparison-experiment-v2"
DIAGNOSTIC_SOURCE = "bd677171c8890c888191e1280ed62c8b84bb622d6b8e8b0f13cdbe5656ac8740"
DIAGNOSTIC_PLAN = "517d1086d9eb69c9ea101cb1125f8d87e31ee2341fc6e053b5e989467ac3df7a"
EXTRA_TESTS = {"tests/test_embedding_comparison_policy.py",
               "tests/test_embedding_comparison_correction.py",
               "tests/test_embedding_comparison_cpu_profile.py",
               "tests/test_embedding_gradient_diagnostic.py"}
CPU_KERNEL_PROFILE = "mkl_compatible_four_threads_v1"
CPU_ENVIRONMENT = dict(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", MKL_CBWR="COMPATIBLE")
COMPONENTS = {"dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss", "dbmol_total_loss"}
FIELDS = {"atoms", "charges", "bonds"}
CUT_FIELDS = {"z": {"z"}, "logits": FIELDS, "graph": FIELDS,
              "native": {"ref_element", "ref_charge", "token_bonds", "bond_types"}}


def verify_correction(config):
    """A fixed-cotangent result selects a candidate, never a native gate pass."""
    correction = config["precision_correction"]
    if (config["version"] != EXPERIMENT_VERSION or correction["version"] != VERSION
            or config["comparison_graph_policy"] != POLICY):
        raise ValueError("Unknown comparison precision correction")
    if correction.get("cpu_kernel_profile") != CPU_KERNEL_PROFILE:
        raise ValueError("Unknown comparison CPU kernel profile")
    parent = json.loads(verified(correction["parent_plan"]).read_text())
    candidate = {key: value for key, value in config.items()
                 if key not in ("comparison_graph_policy", "precision_correction")
                 and not key.startswith("_")}
    candidate["version"] = parent["version"]
    candidate["cpu_tests"] = [name for name in candidate["cpu_tests"] if name not in EXTRA_TESTS]
    if candidate != parent or not EXTRA_TESTS.issubset(config["cpu_tests"]):
        raise ValueError("Precision retry changed inherited protocol or omitted correction tests")
    diagnostic = json.loads(verified(correction["diagnostic"]).read_text())
    terminal = json.loads(verified(correction["diagnostic_result"]).read_text())
    if (diagnostic["status"] != "completed_diagnostic_no_fix_selected"
            or terminal["status"] != diagnostic["status"]
            or terminal["source_sha256"] != DIAGNOSTIC_SOURCE
            or terminal["plan_sha256"] != DIAGNOSTIC_PLAN
            or diagnostic["optimization"] != "not_run"
            or diagnostic["case"]["case_id"] != "case-003"
            or diagnostic["native_oracle"] != "passed_native_oracle_only"
            or not diagnostic["frozen_identity_unchanged"]
            or not diagnostic["frozen_versions_unchanged"]
            or not diagnostic["full_forward_primal_exact"]
            or diagnostic["parent_failure_binding"] != dict(scores_exact=True, gradient_exact=True, direction_exact=True)
            or diagnostic["parent_starts"] != correction["parent_starts"]):
        raise ValueError("Precision correction lacks completed original native diagnostic binding")
    cuts = diagnostic["cut_primal_exact"]
    if (set(cuts) != set(CUT_FIELDS) or any(set(cuts[key]) != fields
            or not all(value is True for value in cuts[key].values()) for key, fields in CUT_FIELDS.items())
            or set(diagnostic["local_replays"]) != COMPONENTS):
        raise ValueError("Diagnostic cut replay was incomplete")
    for stages in diagnostic["local_replays"].values():
        if set(stages) != {"decoder", "graph_policy", "adapter"}:
            raise ValueError("Diagnostic cut replay was incomplete")
        for name, row in stages.items():
            capture = row["captured_input_cotangent_comparison"]
            expected = {"z"} if name == "decoder" else FIELDS
            if not (row["captured_output_primal_exact"] and row["primal_exact_between_ad_modes"]
                    and capture["complete"] and capture["bitwise_exact"]
                    and set(capture["rows"]) == expected
                    and capture["maximum_absolute_difference"] == capture["difference_l2"] == 0
                    and all(item["bitwise_exact"] and item["maximum_absolute_difference"]
                            == item["difference_l2"] == 0 for item in capture["rows"].values())):
                raise ValueError("Diagnostic cut replay was incomplete")
    control = diagnostic["numerical_controls"]["final_policy_float64"]
    if set(control) != COMPONENTS or not all(
            row["within_inherited_threshold"] and row["primal_exact_between_ad_modes"] for row in control.values()):
        raise ValueError("Selected final-policy control did not resolve diagnostic discrepancy")
    return dict(status="authenticated_candidate_not_native_pass", version=VERSION,
                policy=POLICY, diagnostic=correction["diagnostic"],
                parent_plan=correction["parent_plan"], parent_starts=correction["parent_starts"])


def cpu_environment_receipt(config, environ):
    if (config["precision_correction"].get("cpu_kernel_profile") != CPU_KERNEL_PROFILE
            or any(environ.get(key) != value for key, value in CPU_ENVIRONMENT.items())):
        raise ValueError("CPU readiness requires its declared compatible four-thread kernel profile")
    return dict(profile=CPU_KERNEL_PROFILE, environment=dict(CPU_ENVIRONMENT),
                scope="CPU readiness only; native GPU arithmetic unchanged")


def verify_retry_starts(config, output):
    """All original starts must survive the candidate's tiny primal changes."""
    import torch
    from latentmol.train import state_digest
    from .embedding_experiment import load_vae
    from .embedding_comparison_policy import comparison_graph

    receipt = verify_correction(config)
    correction = config["precision_correction"]
    parent = torch.load(verified(correction["parent_starts"]), map_location="cpu", weights_only=True)
    fresh = torch.load(Path(output) / "starts.pt", map_location="cpu", weights_only=True)
    identical = (set(parent) == set(fresh) == {"cases", "graph_spec"}
        and parent["graph_spec"] == fresh["graph_spec"]
        and len(parent["cases"]) == len(fresh["cases"]) == 3)
    if identical:
        identical = all(set(left) == set(right) == {"record", "z", "active"}
            and left["record"] == right["record"]
            and left["z"].dtype == right["z"].dtype
            and left["active"].dtype == right["active"].dtype
            and torch.equal(left["z"], right["z"])
            and torch.equal(left["active"], right["active"])
            for left, right in zip(parent["cases"], fresh["cases"], strict=True))
    if not identical:
        raise ValueError("Precision retry changed original matched starts")
    vae, checkpoint = load_vae(config, "cpu")
    del checkpoint
    before = state_digest(vae.state_dict())
    rows = []
    with torch.no_grad():
        for item in fresh["cases"]:
            decoded = vae.decode(item["z"])
            original = comparison_graph(decoded, item["active"], vae.spec, policy="decoder_dtype_v1")
            corrected = comparison_graph(decoded, item["active"], vae.spec, policy=POLICY)
            changes = {}
            for key in ("atoms", "charges", "bonds"):
                a, b = getattr(original, key), getattr(corrected, key)
                if a.dtype != b.dtype or not torch.equal(a.argmax(-1), b.argmax(-1)):
                    raise ValueError("Precision correction changed a prescribed hard starting graph")
                changes[key] = float((a.double() - b.double()).abs().max())
                if changes[key] > 5e-7:
                    raise ValueError("Precision correction changed starting probabilities materially")
            rows.append(dict(case_id=item["record"]["case_id"], hard_identity_unchanged=True,
                             maximum_probability_change=changes))
    if state_digest(vae.state_dict()) != before:
        raise ValueError("Precision readiness changed original frozen VAE")
    receipt.update(all_three_parent_starts_exact=True, cases=rows,
                   frozen_vae_unchanged=True,
                   claim="CPU candidate readiness only; fresh native gates remain mandatory")
    write(Path(output) / "precision-correction.json", receipt)
    return receipt
