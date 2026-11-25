from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
import gzip
import json
from pathlib import Path

import pytest
import torch

from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from latentmol.train import TrainConfig, capture_rng_state
from latentmol.training import motif_activation as gate
from latentmol.training import occupancy_monitor
from latentmol.training.combined_continuation import _train
from latentmol.training.projected_continuation import _student


def panel_scores():
    return {k: [False] * (256 if k == "exact" else 4096) for k in ("aromatic", "amide", "valid", "exact")}


def test_paired_qualification_thresholds_ties_and_se():
    left, right = panel_scores(), panel_scores()
    right["aromatic"][:4] = [True]*4
    select = lambda: gate.paired_decision(left,right)[0]
    assert select() == "candidate"
    left["valid"][:40] = [True]*40
    left["amide"][:20] = [True]*20
    left["exact"][:2] = [True]*2
    assert select() == "candidate"
    for k,pos in (("valid",40),("amide",20),("exact",2)):
        left[k][pos] = True
        assert select() == "control"
        left[k][pos] = False
    left["aromatic"][4:24] = [True]*20
    right["aromatic"][24:44] = [True]*20
    arm, counts, checks = gate.paired_decision(left,right)
    assert arm == "control" and counts["aromatic"]["net"] == 4 and not checks["aromatic_above_one_paired_se"]
    right["exact"].pop()
    with pytest.raises(ValueError,match="Incomplete"):
        select()


def test_raw_panel_and_holdout_reproduce_with_validity_failure_details(tmp_path, monkeypatch):
    monkeypatch.setattr(gate,"PRIOR_COUNT",8)
    monkeypatch.setattr(gate,"RECON_COUNT",4)
    spec = GraphSpec(max_nodes=4)
    config = ModelConfig(hidden=12,latent=8,layers=1,global_hidden=12,pair_hidden=6,triangle_hidden=3,
                         decoder_variant="triangle_refine",refinement_rounds=1,encoder_pooling="canonical")
    model = GraphAutoencoder(spec,config)
    with torch.no_grad():
        # Force the four occupied carbon slots apart. This must count as an invalid raw draw.
        model.decoder.atom_head.weight.zero_()
        model.decoder.atom_head.bias.fill_(-100)
        model.decoder.atom_head.bias[1] = 100
        model.decoder.bond_head[-1].weight.zero_()
        model.decoder.bond_head[-1].bias.fill_(-100)
        model.decoder.bond_head[-1].bias[0] = 100
    state = {"model_config":asdict(config),"model_state":model.state_dict()}
    class Validation(list):
        split = "val"
        manifest = {"provenance":{"source_counts":{"val":1}}}
        smiles = ["CC"] * 6
    val = Validation([dict(atoms=torch.tensor([1,1,0,0]), charges=torch.tensor([2]*4),
                           bonds=torch.zeros((4,4),dtype=torch.long)) for _ in range(6)])
    outputs = []
    for name in ("a","b"):
        d = tmp_path/name
        d.mkdir()
        scores = gate.evaluate_panel(state,spec,val,dict(seed=15,validation_points=4,batch_size=2,ring_activation=True),42,d,"cpu",None)
        assert scores["valid"] == [False]*8 and len(scores["exact"]) == 4
        assert scores["triangle"] == scores["any_ring"] == [False]*8 and scores["scaffold"] == [""]*8
        with gzip.open(d/"paired-panel.jsonl.gz","rt") as stream:
            raw = [json.loads(line) for line in stream]
        assert len(raw) == 12 and all(v["failure_reason"] and v["occupied_atoms"] == 4 and v["ring8plus"] is None
                                      for v in raw[:8])
        outputs.append(raw)
    assert outputs[0] == outputs[1]


def test_ring_gate_selects_paired_size_improvement_and_rejects_shortcuts():
    from copy import deepcopy
    from latentmol.training.ring_activation import ring_decision
    from latentmol.training.occupancy_monitor import _motif_panel

    policy = {"ring_size":{"weight":.1}, "score_draws":240,
              "activation_gate":{"version":2, "control":{"score_draws":240}, "seed":6}}
    assert gate.policies({"functional_groups":policy})[1]["control"] == {"score_draws":240}
    wrong = dict(policy, neighbor_weight=1.)
    with pytest.raises(ValueError, match="matched ring"):
        gate.policies({"functional_groups":wrong})
    n = 128
    a = {k:[True]*n for k in ("valid","any_ring","triangle","ring8plus","aromatic","amide")}
    a.update(exact=[True]*16, occupied_atoms=[24]*n, scaffold=[str(i) for i in range(n)])
    def choose(b):
        return ring_decision(a,b,n,16)[0]
    b = deepcopy(a)
    assert choose(b) == "control"
    b["triangle"][:10] = b["ring8plus"][:10] = [False]*10
    assert choose(b) == "candidate"
    for k, value in (("valid",False),("any_ring",False),("occupied_atoms",15),("scaffold","")):
        bad = deepcopy(b)
        bad[k][:10] = [value]*10
        assert choose(bad) == "control", k
    no_valid = deepcopy(b)
    no_valid["valid"] = [False]*n
    assert choose(no_valid) == "control"
    monitor = _motif_panel(["C1CC1", "C1CCCCCCC1", "C", None], monitor_ring_sizes=True)
    assert monitor["valid"] == 3 and monitor["distinct_nonempty_scaffolds"] == 2
    assert monitor["ring_size"]["ring_any_hits"] == 2
    assert monitor["ring_size"]["ring_reference_supported_valid"] == 1


@pytest.mark.parametrize("winner", ["candidate", "control"])
def test_roundtrip_handoff_retains_ring_winner_and_neighbor_across_resumes(tmp_path, winner):
    recipe = Path(__file__).resolve().parents[2] / "experiments/vae/epoch48-seven-round-ring-sizes.json"
    old = json.loads(recipe.read_bytes())
    new = dict(old, generated_roundtrip=dict(draws=16, max_parents=8, every_updates=4, weight=.1))
    _, arms = gate.policies(old)
    selected = arms[winner]
    stamp = dict(protocol="ring-protocol", winner=winner, policy_sha256=gate._digest(selected),
                 adopted_at_update=123, decision_file="ring-decision", decision_sha256="decision-digest")
    resume = dict(config_sha256="ring-protocol", continuation={"updates": 130}, experiment_state={"motif_activation": stamp})
    receipt = dict(updates=130)
    binding = dict(source_protocol="ring-protocol", target_protocol="new-protocol")

    def train(config, identity, *args, experiment_state):
        assert config["functional_groups"] == selected and config["generated_roundtrip"] == new["generated_roundtrip"]
        assert selected["neighbor_weight"] == 3. and ("ring_size" in selected) == (winner == "candidate")
        assert identity == "new-protocol" and experiment_state == {"motif_activation": stamp, "roundtrip_handoff": binding}
        return {"saved_metadata": experiment_state}

    def run(previous_policy):
        return gate.run_activation(new, "new-protocol", None, None, tmp_path, "cpu", 240, resume, None, None, 48,
                                   previous_policy=previous_policy, input_receipt=receipt, train=train)

    switched = run(old["functional_groups"])
    assert switched["motif_activation"] == stamp and switched["roundtrip_handoff"] == binding
    resume.update(config_sha256="new-protocol", experiment_state=switched["saved_metadata"])
    assert run(None)["motif_activation"] == stamp
    resume["experiment_state"] = dict(resume["experiment_state"], roundtrip_handoff=dict(binding, target_protocol="other"))
    with pytest.raises(ValueError, match="authenticated round-trip"):
        run(None)
    resume["experiment_state"] = dict(switched["saved_metadata"], motif_activation=dict(stamp, policy_sha256="wrong"))
    with pytest.raises(ValueError, match="authenticated selection"):
        run(None)


@pytest.mark.parametrize(("winner","at_wall","version"), [
    ("candidate",False,1),("control",False,1),("control",True,1),("candidate",False,2),("control",False,2)])
def test_gate_adopts_winners_exact_state_history_and_only_runs_once(tmp_path,monkeypatch,winner,at_wall,version):
    monkeypatch.setattr(gate,"TRIAL_UPDATES",2)
    class Monitor:
        def __init__(self,policy,val,model,output,initial):
            self.path = output / "quality-panel.jsonl"
            self.path.touch(exist_ok=False)
            self.last = None
        def record(self,model,cursor,force=False):
            if cursor["updates"] != self.last:
                with self.path.open("a") as handle:
                    handle.write(json.dumps(cursor)+"\n")
                self.last = cursor["updates"]
    monkeypatch.setattr(occupancy_monitor,"OccupancyMonitor",Monitor)
    class Rows(list):
        manifest = {"dataset_sha256": "test"}
    spec = GraphSpec(max_nodes=4)
    arch = ModelConfig(hidden=12,latent=8,layers=1,global_hidden=12,pair_hidden=6,triangle_hidden=3,
                       decoder_variant="triangle_refine",refinement_rounds=1,encoder_pooling="canonical")
    model = GraphAutoencoder(spec,arch)
    tx = TrainConfig(model=arch,mode="vae",device="cpu",batch_size=2)
    parent = dict(train_config=asdict(tx),loss_config=asdict(tx.loss),
                  optimizer_state=torch.optim.AdamW(model.parameters()).state_dict(),
                  continuation=dict(updates=700,rng=capture_rng_state({"fresh_prior":torch.Generator().manual_seed(32)},False)))
    assets = (model,parent,None,None,None)
    original = {"weight":1}
    candidate = {"weight":4} if version == 1 else dict(original, ring_size={"weight":.1})
    activation = dict(version=version,control=original,seed=72)
    cfg = dict(parent_epoch=47,epochs=1,shuffle_seed=13,quality_panel={"seed":19},checkpoint_every_updates=2,
               functional_groups=dict(candidate,activation_gate=activation))
    rows = Rows([dict(atoms=torch.tensor([1,i%4+1,0,0]),charges=torch.zeros(4,dtype=torch.long),
                      bonds=torch.zeros((4,4),dtype=torch.long)) for i in range(16)])
    def step(config,m,ref,opt,graph,settings,composition,teacher,gen,update,epoch,micro):
        assert "activation_gate" not in config["functional_groups"]
        noise = torch.rand(()) + torch.rand((),generator=gen["fresh_prior"])
        policy = config["functional_groups"]
        value = graph.atoms.double().sum() + noise + policy["weight"] + policy.get("ring_size",{}).get("weight",0)
        loss = (m.decoder.atom_head.bias.mean()-value).square()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {"observed":float(value),"objective":float(loss.detach())}
    kwargs = dict(student_factory=partial(_student,count=0,seed=7,atom_order="graph"),
                  update_fn=step,teacher_factory=lambda *_:nullcontext(),validation=rows)
    def directory(name):
        path = tmp_path/name
        path.mkdir()
        return path
    def state(rep):
        return torch.load(rep["latest"]["checkpoint"],weights_only=True,map_location="cpu")
    start_dir = directory("first")
    old = _train(dict(cfg,functional_groups=original),"old",rows,assets,start_dir,"cpu",2,None,1,None,2,**kwargs)
    start = state(old)
    visits = []
    def panel(s,spec,val,q,seed,directory,device,deadline):
        visits.append(directory.name)
        flags = panel_scores()
        if version == 2:
            assert q["ring_activation"]
            flags.update(any_ring=[True]*4096, triangle=[True]*4096, ring8plus=[True]*4096,
                         scaffold=["a"]*4096, valid=[True]*4096, occupied_atoms=[24]*4096)
            if winner == "candidate" and directory.name == "candidate":
                flags["ring8plus"][:32] = [False]*32
        if winner == "candidate" and directory.name == "candidate":
            flags["aromatic"][:12] = [True]*12
        return flags
    output = directory("segment-two")
    def short_root(*args,**kw):
        positional = list(args)
        if positional[4] == output:
            if at_wall:
                positional[9] = 0. # soft deadline already reached: promote saved winning endpoint
            else:
                positional[8] = 1 # ordinary early pause after one production update
        return _train(*positional,**kw)
    result = gate.run_activation(cfg,"protocol",rows,assets,output,"cpu",2,start,None,None,2,
            previous_policy=original if version == 1 else {"weight":.3}, input_receipt=old["latest"],train=short_root,panel=panel,**kwargs)
    assert visits == ["control","candidate"] and result["motif_activation"]["winner"] == winner
    assert result["discarded_trial_updates"] == 2 and result["segment_updates"] == (2 if at_wall else 3)
    if at_wall:
        assert result["stop_reason"] == "allocation_budget_reached"
    selected = state(result)
    assert selected["config_sha256"] == "protocol" and selected["experiment_state"]["motif_activation"] == result["motif_activation"]
    assert Path(result["latest"]["checkpoint"]).parent == output
    accepted = [json.loads(line)["updates"] for line in (output/"training.jsonl").read_text().splitlines()]
    assert accepted == list(range(702,result["updates"]+1))
    panel_updates = [json.loads(line)["updates"] for line in (output/"quality-panel.jsonl").read_text().splitlines()]
    assert panel_updates == list(range(701,result["updates"]+1))
    final = gate.run_activation(cfg,"protocol",rows,assets,directory("segment-three"),"cpu",2,selected,None,None,2,
                                previous_policy=None,input_receipt=result["latest"],panel=panel,**kwargs)
    assert visits == ["control","candidate"] and final["motif_activation"] == result["motif_activation"]
    chosen = candidate if winner=="candidate" else original
    expected = _train(dict(cfg,functional_groups=chosen),"reference",rows,assets,directory("reference"),"cpu",2,
                      start,None,None,2,**kwargs)
    actual, expected = state(final), state(expected)
    assert actual["continuation"]["updates"] == expected["continuation"]["updates"] == 708
    for key in ("model_state","optimizer_state"):
        torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
    for key in ("torch","loaders"):
        torch.testing.assert_close(actual["continuation"]["rng"][key],expected["continuation"]["rng"][key],atol=0,rtol=0)
