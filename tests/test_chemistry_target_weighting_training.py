"""Allocated-only production update checks for the weighted chemistry candidate.

Use actual valid C4-chain self targets with both present and absent pairs.
The fixed per-slot bias makes this fixture informative about NONE weighting;
it neither changes the production model nor fabricates chemistry receipts.
"""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from latentmol import chemistry_continuation as c
from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
from latentmol.chemistry_classification import VERSION as CLASSIFICATION_VERSION
from latentmol.chemistry_execution_validation import _snapshot, _restore
from latentmol.chemistry_objective_gradient_diagnostics import ObjectiveGradientDiagnostics
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput
from test_chemistry_continuation import _captured_gradients, _hard
from test_chemistry_valid_self_guidance import device
from test_chemistry_valid_self_training import _models, _teacher


WEIGHTING=dict(version="bond-weighted-target-ce-v1",none_bond_weight=.1)
STEP=dict(version="adamw-displacement-scale-v1",factor=.25)


def _chain_models(device):
    model,reference,config,protocol,composition=_models(device)
    native=model.decoder.forward
    n=model.spec.max_nodes
    single=model.spec.bond_names.index("SINGLE")
    bias=torch.zeros((1,n,n,len(model.spec.bond_names)),device=device)
    for left in range(n):
        for right in range(n):
            bias[0,left,right,0 if abs(left-right)!=1 else single]=8.
    def forward(z):
        value=native(z)
        return DecoderOutput(value.atoms,value.charges,value.bonds+bias)
    model.decoder.forward=forward
    config=replace(config,clip_grad=1e6)
    protocol["valid_prior_self_guidance"]=dict(version="valid-prior-self-nll-v1",weight=.1)
    return model,reference,config,protocol,composition


def _run(model,reference,config,protocol,composition,generators,optimizer,*,size=2,overlap=False,observer=None,step=True):
    with _teacher(model) as teacher:
        return c.chemistry_update(model,reference,optimizer,_hard(model.spec),config,protocol,composition,
            teacher,generators,7,37,35,microbatch_size=size,overlap=overlap,do_step=step,
            step_control=STEP,component_observer=observer)


def test_production_weighting_changes_informative_actual_repair_loss_and_gradient_with_self_disabled(device):
    """A real one-bond repair isolates production repair-policy propagation.

    The decoder emits O#C-C-C: neutral oxygen has excessive valence, while
    replacing the triple bond by a single bond yields a supported valid graph.
    Every row has present and NONE pairs. The detached teacher still chooses
    and authenticates its own target; the fixture supplies no repair witness.
    """
    model,reference,config,protocol,composition=_models(device)
    assert "valid_prior_self_guidance" not in protocol
    config=replace(config,clip_grad=1e6)
    native=model.decoder.forward
    spec=model.spec
    assert spec.max_nodes==4
    atom_bias=torch.zeros((1,spec.max_nodes,len(spec.atomic_numbers)),device=device)
    atom_bias[0,0,spec.atomic_numbers.index(8)]=8.
    bond_bias=torch.zeros((1,spec.max_nodes,spec.max_nodes,len(spec.bond_names)),device=device)
    for left in range(spec.max_nodes):
        for right in range(spec.max_nodes):
            bond_bias[0,left,right,0 if abs(left-right)!=1 else spec.bond_names.index("SINGLE")]=8.
    bond_bias[0,0,1,spec.bond_names.index("SINGLE")]=0.
    bond_bias[0,1,0,spec.bond_names.index("SINGLE")]=0.
    bond_bias[0,0,1,spec.bond_names.index("TRIPLE")]=12.
    bond_bias[0,1,0,spec.bond_names.index("TRIPLE")]=12.

    def forward(z):
        value=native(z)
        return DecoderOutput(value.atoms+atom_bias,value.charges,value.bonds+bond_bias)

    model.decoder.forward=forward

    class RecordingTeacher(AtomPreservingRepairTeacher):
        """Record authentic targets and logits without changing teacher work."""

        def __init__(self):
            super().__init__(spec,ProbabilityRepairConfig(max_edits=1,checks_per_depth=32),
                             classification_version=CLASSIFICATION_VERSION)
            self.records=[]

        def build_targets(self,decoded,*,candidate_ids,update):
            batch=super().build_targets(decoded,candidate_ids=candidate_ids,update=update)
            self.records.append(dict(candidate_ids=list(batch.candidate_ids),
                decoded=tuple(value.detach().clone() for value in (decoded.atoms,decoded.charges,decoded.bonds)),
                targets=tuple(None if target is None else target.record() for target in batch.targets)))
            return batch

    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    generators={"fresh_prior":torch.Generator().manual_seed(901)}
    initial=_snapshot(model,reference,optimizer,generators)
    outputs=[]
    for weighted in (False,True):
        _restore(initial,model,reference,optimizer,generators)
        current=deepcopy(protocol)
        if weighted:current["chemistry_target_weighting"]=dict(WEIGHTING)
        with RecordingTeacher() as teacher:
            metrics=c.chemistry_update(model,reference,optimizer,_hard(spec),config,current,composition,
                teacher,generators,7,37,35,microbatch_size=2,do_step=False,step_control=STEP)
            records=teacher.records
        assert metrics["repair_targets"]==dict(real=3,prior=6)
        assert metrics["repair_statuses"]==dict(real={"accepted":3},prior={"accepted":6})
        assert metrics["repair_only_decoder_gradient_norm"]>0.
        assert metrics["repair_real"]>0. and metrics["repair_prior"]>0.
        assert "valid_prior_self_guidance" not in metrics
        assert all(target is not None and target["edit_count"]==1
                   for batch in records for target in batch["targets"])
        outputs.append((metrics,_captured_gradients(model),_snapshot(model,reference,optimizer,generators),records))
    old,new=outputs
    assert c.nested_equal(old[3],new[3])
    assert c.nested_equal(old[2]["rng"],new[2]["rng"])
    assert old[0]["repair_real"]!=pytest.approx(new[0]["repair_real"],rel=1e-5,abs=1e-7)
    assert old[0]["repair_prior"]!=pytest.approx(new[0]["repair_prior"],rel=1e-5,abs=1e-7)
    assert old[0]["reconstruction"]==new[0]["reconstruction"]
    assert old[0]["kl"]==new[0]["kl"]
    assert any(not c.nested_equal(old[1][name],value) for name,value in new[1].items()
               if name.startswith("decoder."))
    assert new[0]["chemistry_target_weighting"]==WEIGHTING
    _restore(initial,model,reference,optimizer,generators)


def test_production_weighting_changes_decoder_gradient_without_changing_targets_draws_or_encoder(device):
    model,reference,config,protocol,composition=_chain_models(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    generators={"fresh_prior":torch.Generator().manual_seed(901)}
    initial=_snapshot(model,reference,optimizer,generators)
    outputs=[]
    for weighted in (False,True):
        _restore(initial,model,reference,optimizer,generators)
        current=deepcopy(protocol)
        if weighted:current["chemistry_target_weighting"]=dict(WEIGHTING)
        metrics=_run(model,reference,config,current,composition,generators,optimizer,step=False)
        outputs.append((metrics,_captured_gradients(model),_snapshot(model,reference,optimizer,generators)))
    old,new=outputs
    assert old[0]["valid_prior_self_guidance"]["eligible"]==new[0]["valid_prior_self_guidance"]["eligible"]==6
    assert old[0]["repair_statuses"]==new[0]["repair_statuses"]
    assert c.nested_equal(old[2]["rng"],new[2]["rng"])
    assert all(c.nested_equal(old[1][name],value) for name,value in new[1].items() if name.startswith("encoder."))
    assert any(not c.nested_equal(old[1][name],value) for name,value in new[1].items() if name.startswith("decoder."))
    assert new[0]["chemistry_target_weighting"]==WEIGHTING
    assert new[0]["chemistry_target_score_kind"]=="weighted_target_cross_entropy_not_exact_graph_NLL"


def test_full_vs_microbatch_weighted_production_scalar_gradients_and_rng(device):
    model,reference,config,protocol,composition=_chain_models(device)
    protocol["chemistry_target_weighting"]=dict(WEIGHTING)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    generators={"fresh_prior":torch.Generator().manual_seed(901)}
    initial=_snapshot(model,reference,optimizer,generators)
    outputs=[]
    for size in (2,6):
        _restore(initial,model,reference,optimizer,generators)
        metrics=_run(model,reference,config,protocol,composition,generators,optimizer,size=size,step=False)
        outputs.append((metrics,_captured_gradients(model),_snapshot(model,reference,optimizer,generators)))
    old,new=outputs
    for key in ("total","valid_prior_self_weighted","repair_real","repair_prior"):
        assert old[0][key]==pytest.approx(new[0][key],rel=2e-5,abs=2e-7)
    assert c.nested_equal(old[2]["rng"],new[2]["rng"])
    for name,gradient in old[1].items():
        other=new[1][name]
        if gradient is None:assert other is None
        else:torch.testing.assert_close(gradient,other,rtol=3e-4,atol=3e-7)


def test_weighted_update_sync_async_restore_native_adamw_and_complete_component_observation(device):
    model,reference,config,protocol,composition=_chain_models(device)
    protocol["chemistry_target_weighting"]=dict(WEIGHTING)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    generators={"fresh_prior":torch.Generator().manual_seed(901)}
    initial=_snapshot(model,reference,optimizer,generators)
    outputs=[]
    for overlap in (False,True):
        _restore(initial,model,reference,optimizer,generators)
        with ObjectiveGradientDiagnostics(model,real_points=3,prior_points=6,microbatch_size=2) as observer:
            metrics=_run(model,reference,config,protocol,composition,generators,optimizer,
                         overlap=overlap,observer=observer)
        receipt=observer.receipt()
        assert receipt["complete_logical_update"] and receipt["component_sum_matches_native_backward"]
        outputs.append((metrics,_snapshot(model,reference,optimizer,generators)))
    assert c.nested_equal(outputs[0][1],outputs[1][1])
    assert outputs[0][0]["optimizer_step_control"]==outputs[1][0]["optimizer_step_control"]
    _restore(initial,model,reference,optimizer,generators)
    assert c.nested_equal(initial,_snapshot(model,reference,optimizer,generators))
