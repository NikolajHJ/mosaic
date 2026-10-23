"""Restored native counterfactuals for the frozen valid-target correction.

At original36 the new anchor must be exactly the same objective/gradient/AdamW
step as current-valid self guidance. After an explicit invalid-live perturbation
it must retain its original labels and a useful gradient. These are disposable
mechanism checks, not evidence that a longer training trajectory is healthy.
"""
from copy import deepcopy
import math
from pathlib import Path


GATES = frozenset(("reference_initial_step_parity", "reference_invalid_live_gradient",
                   "reference_directional_derivative", "reference_anchor_unchanged"))


def _fields(value):
    return value.atoms,value.charges,value.bonds


def _fd(decoded, prepared, spec, policy):
    import torch
    from .representation import DecoderOutput
    from .chemistry_valid_reference_guidance import valid_prior_reference_nll
    values=DecoderOutput(*(v.detach().double().requires_grad_() for v in _fields(decoded)))
    def score(v): return valid_prior_reference_nll(v,prepared,spec,policy)["weighted_loss"]
    loss=score(values)
    gradients=torch.autograd.grad(loss,_fields(values),allow_unused=True)
    gradients=[torch.zeros_like(v) if g is None else g for v,g in zip(_fields(values),gradients)]
    length=torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
    if not bool(torch.isfinite(length)) or float(length)<=1e-10:
        raise ValueError("Invalid-live reference check lacks an informative logit derivative")
    directions=[g/length for g in gradients]
    analytic=sum((g*d).sum() for g,d in zip(gradients,directions))
    eps=1e-3
    plus,minus=[score(DecoderOutput(*(v.detach()+sign*eps*d for v,d in zip(_fields(values),directions))))
                for sign in (1,-1)]
    numeric=(plus-minus)/(2*eps)
    return dict(passed=bool(torch.isfinite(numeric)&torch.isclose(analytic,numeric,rtol=1e-4,atol=1e-8)),
                analytic=float(analytic),finite_difference=float(numeric),epsilon=eps,rtol=1e-4,atol=1e-8,
                scope="Frozen original36 valid labels, deliberately invalid live logits, float64 fixed-target FD")


def native_reference_check(plan,model,reference,optimizer,config,data,cohorts,generators,teacher,epoch,updates,composition):
    import torch
    from . import chemistry_continuation as c
    from .chemistry_execution_validation import _snapshot,_restore
    from .chemistry_objective_gradient_diagnostics import ObjectiveGradientDiagnostics
    from .chemistry_valid_reference_guidance import valid_prior_reference_nll
    from .chemistry_valid_self_guidance import prepare_valid_self_guidance,valid_prior_self_nll
    from .data import collate_graphs
    from .representation import DecoderOutput
    from .train import state_digest
    from .experiments import write_json

    anchor=teacher.valid_reference_anchor
    output=Path(plan["campaign"])/"valid_reference_native.json"
    if output.exists():
        raise FileExistsError("Preserve the native reference counterfactual; require a new immutable attempt")
    before=anchor.assert_unchanged()
    if before["model_sha256"]!=state_digest(model.state_dict()) or epoch!=36:
        raise ValueError("Native reference counterfactual must start at original36")
    initial=_snapshot(model,reference,optimizer,generators)
    outcomes={}
    report=dict(version="frozen-valid-reference-native-v1",status="running",plan_sha256=plan["_plan_sha256"],
        archive_sha256=plan["archive_sha256"],manifest_sha256=plan["manifest_sha256"],
        valid_reference_test=plan["valid_reference_test"],reference_before=before,
        scope="Two restored same-start first updates and one fixed-logit dropout check; not a training arm")
    try:
        hard=collate_graphs([data["train"][i] for i in cohorts["train"][:config.batch_size]])
        for name in ("current_valid","frozen_reference"):
            teacher.drain();teacher.clear_cache()
            _restore(initial,model,reference,optimizer,generators)
            protocol=deepcopy(plan["protocol"])
            if name=="current_valid":
                protocol.pop("valid_prior_reference_guidance")
                protocol["valid_prior_self_guidance"]=dict(version="valid-prior-self-nll-v1",weight=.1)
            with ObjectiveGradientDiagnostics(model,real_points=256,prior_points=512,microbatch_size=128) as observer:
                metrics=c.chemistry_update(model,reference,optimizer,hard,config,protocol,composition,teacher,
                    generators,updates,37,plan["parent_epoch"],overlap=True,
                    step_control=plan["optimizer_step_control"],component_observer=observer)
            outcomes[name]=dict(metrics=metrics,state=_snapshot(model,reference,optimizer,generators),
                                component_gradients=observer.receipt())
        old,new=outcomes["current_valid"],outcomes["frozen_reference"]
        state_checks={key:c.nested_equal(old["state"][key],new["state"][key]) for key in initial}
        scalar_keys=("reconstruction","kl","repair_real","repair_prior","occupancy_anchor","element_js","charge_js","total")
        scalar_checks={key:old["metrics"][key]==new["metrics"][key] for key in scalar_keys}
        scalar_checks.update(reference_nll=old["metrics"]["valid_prior_self_nll"]==new["metrics"]["valid_prior_reference_nll"],
            reference_weighted=old["metrics"]["valid_prior_self_weighted"]==new["metrics"]["valid_prior_reference_weighted"],
            same_eligible=old["metrics"]["valid_prior_self_guidance"]["eligible"]
                          ==new["metrics"]["valid_prior_reference_guidance"]["eligible"],
            same_live_repair_targets=old["metrics"]["repair_targets"]==new["metrics"]["repair_targets"],
            same_live_repair_statuses=old["metrics"]["repair_statuses"]==new["metrics"]["repair_statuses"])
        report["initial_step_parity"]=dict(passed=all(state_checks.values()) and all(scalar_checks.values()),
            state_checks=state_checks,scalar_checks=scalar_checks,
            component_gradients={name:value["component_gradients"] for name,value in outcomes.items()},
            current_valid_eligible=old["metrics"]["valid_prior_self_guidance"]["eligible"],
            frozen_reference_eligible=new["metrics"]["valid_prior_reference_guidance"]["eligible"])
        teacher.drain();teacher.clear_cache();_restore(initial,model,reference,optimizer,generators)
        # Reuse the actual fresh-prior stream, then discard/restore this draw.
        z=torch.randn(128,model.config.latent,generator=generators["fresh_prior"]).to(next(model.parameters()).device)
        prepared=anchor.prepare(teacher,z,candidate_ids=list(range(128)),update=updates,overlap=False)
        if not prepared.eligible_rows:
            raise ValueError("Native frozen reference cohort has no strict-valid targets")
        model.eval()
        live=model.decode(z)
        graph=prepared.graph.to(live.atoms.device)
        occupied=graph.atoms!=0
        pair=occupied[:,:,None]&occupied[:,None,:]
        pair=pair & ~torch.eye(model.spec.max_nodes,dtype=torch.bool,device=z.device)[None]
        bias=torch.zeros_like(live.bonds)
        bias[...,model.spec.bond_names.index("TRIPLE")]=pair*100.
        broken=DecoderOutput(live.atoms,live.charges,live.bonds+bias)
        base=teacher.build_targets(broken,candidate_ids=list(range(128)),update=updates)
        dynamic=prepare_valid_self_guidance(DecoderOutput(*(v.detach().cpu().clone() for v in _fields(broken))),base,model.spec)
        if dynamic.eligible_rows:
            raise ValueError("Native invalid-live perturbation did not eliminate current strict validity")
        dynamic_score=valid_prior_self_nll(broken,dynamic,model.spec,dict(version="valid-prior-self-nll-v1",weight=.1))
        anchored=valid_prior_reference_nll(broken,prepared,model.spec,plan["protocol"]["valid_prior_reference_guidance"])
        gradients=torch.autograd.grad(anchored["weighted_loss"],tuple(model.decoder.parameters()),retain_graph=True,allow_unused=True)
        norm=math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))
        report["invalid_live"]=dict(attempts=128,reference_eligible=len(prepared.eligible_rows),
            current_eligible=len(dynamic.eligible_rows),current_loss=float(dynamic_score["weighted_loss"].detach()),
            reference_loss=float(anchored["weighted_loss"].detach()),decoder_gradient_norm=norm,
            target_receipt=prepared.receipt,reference_searches=0,
            current_loss_zero=bool(dynamic_score["weighted_loss"]==0),
            persistent_reference=anchored["eligible_rows"]==prepared.eligible_rows,
            deliberately_invalid_live=True)
        report["directional_derivative"]=_fd(broken,prepared,model.spec,plan["protocol"]["valid_prior_reference_guidance"])
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        teacher.drain();teacher.clear_cache()
        _restore(initial,model,reference,optimizer,generators)
        restored=_snapshot(model,reference,optimizer,generators)
        report["restoration"]={key:c.nested_equal(value,restored[key]) for key,value in initial.items()}
        report["reference_after"]=anchor.assert_unchanged()
        write_json(output,report)
        if not all(report["restoration"].values()) or report["reference_after"]!=before:
            raise ValueError("Native reference counterfactual did not restore original36 state")
    invalid=report["invalid_live"]
    report["gates"]=dict(reference_initial_step_parity=report["initial_step_parity"]["passed"],
        reference_invalid_live_gradient=invalid["current_loss_zero"] and invalid["persistent_reference"]
            and math.isfinite(invalid["reference_loss"]) and invalid["reference_loss"]>0
            and math.isfinite(invalid["decoder_gradient_norm"]) and invalid["decoder_gradient_norm"]>1e-10,
        reference_directional_derivative=report["directional_derivative"]["passed"],
        reference_anchor_unchanged=report["reference_before"]==report["reference_after"] and all(report["restoration"].values()))
    report["status"]="passed" if all(report["gates"].values()) else "failed"
    write_json(output,report)
    return report
