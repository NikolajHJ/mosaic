"""Exercise the real controller with fake scheduler IDs and small checkpoint files."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

from experiments.vae import epoch48_segment_chain as chain
from experiments.vae import epoch48_cutover as cut
from experiments.vae import epoch48_early_cutover as early_cut
import shlex


def put(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(("segment", "parent_visible", "purge", "started_race", "nested", "rollback"),
                         [(2, True, False, False, False, False), (2, False, True, False, False, False),
                          (3, True, False, False, False, False), (2, True, False, True, False, False),
                          (3, False, True, False, True, False), (3, False, True, False, True, True)])
def test_cutover_retires_only_pending_jobs_and_keeps_one_lineage(
        tmp_path, monkeypatch, segment, parent_visible, purge, started_race, nested, rollback):
    oldc = tmp_path/"original"
    oldc.mkdir()
    old = fixture_plan(oldc)
    if nested:
        before = tmp_path / "motif"
        before.mkdir()
        base = dict(old, output_root=str(before/"runs"))
        put(before/"plan.json", base)
        for label, ident in ((chain.name(2), 8001), (chain.name(2, after=True), 8002)):
            put(before/(label+".receipt.json"), dict(id=ident))
        old["handoff"] = dict(control=str(before), plan_sha256=sha((before/"plan.json").read_bytes()),
                              worker_id=8001, callback_id=8002, segment=2)
        put(oldc/"plan.json", old)
    newc = tmp_path/"updated"
    newc.mkdir()
    plan = dict(old, config="experiments/vae/rings.json", config_sha256="d"*64,
                output_root=str(newc/"runs"), handoff=dict(control=str(oldc),
                plan_sha256=sha((oldc/"plan.json").read_bytes()), worker_id=9002, callback_id=9003, segment=segment))
    newpath = newc/"plan.json"
    if nested:
        plan["job_prefix"] = "e48-r7-roundtrip"
    put(newpath,plan)
    rows = {}
    actions = []

    def reserve(control,label,args):
        jid = max(rows,default=9000)+1
        argv = ["bsub", "-J", label, *args]
        chain.write_once(control/(label+".intent.json"), dict(argv=argv))
        chain.write_once(control/(label+".receipt.json"),dict(id=jid,argv=argv))
        rows[jid] = dict(USER="s234818",JOB_NAME=label,STAT="PSUSP" if "-H" in args else "PEND",
                         QUEUE=args[args.index("-q")+1],DEPENDENCY=args[args.index("-w")+1] if "-w" in args else "",
                         START_TIME="",RUN_TIME="00:00:00",COMMAND=shlex.join(args[args.index(chain.PYTHON):]))
        return jid
    rows[9001] = dict(USER="s234818",JOB_NAME="parent",STAT="DONE")
    monkeypatch.setattr(chain,"submit",reserve)
    chain.queue(oldc/"plan.json",old,2,sha((oldc/"plan.json").read_bytes()))
    finish_segment(oldc,old,1,1000,"allocation_budget_reached")
    if not parent_visible:
        del rows[9001]
    if segment == 3:
        rows[9002].update(STAT="RUN",START_TIME="Nov 22 22:55",RUN_TIME="00:01:13")
    monkeypatch.setattr(cut,"jobs",lambda: rows)
    def act(exe,*args):
        actions.append((exe,*args))
        jid = int(args[-1])
        if exe == "bkill":
            expected = "susp" if rollback and jid == 9004 else "pend"
            assert args==("-stat",expected,"-s","TERM",str(jid))
            assert rows[jid]["STAT"]==("PSUSP" if expected == "susp" else "PEND") and not rows[jid]["START_TIME"]
            if jid==9002 and started_race:
                rows[jid].update(STAT="RUN", START_TIME="Nov 22 22:55", RUN_TIME="00:00:01")
                return "No matching pending job"
            if purge:
                del rows[jid]
            else:
                rows[jid]["STAT"] = "EXIT"
            return f"Job <{jid}> is being terminated"
        elif exe == "bresume":
            assert rows[jid]["STAT"]=="PSUSP" and rows.get(9003,{"STAT":"EXIT"})["STAT"]=="EXIT"
            assert rows.get(9002,{"STAT":"EXIT"})["STAT"] == ("EXIT" if segment==2 else "RUN")
            rows[jid]["STAT"]="PEND"
        else: pytest.fail(exe)
    monkeypatch.setattr(cut,"command",act)
    if nested:
        staged = cut.cutover(newpath, stage_only=True)
        assert staged["stage"] == "held_and_verified" and rows[9004]["STAT"] == "PSUSP"
        assert rows[9002]["STAT"] == "RUN" and rows[9003]["STAT"] == "PEND" and not actions
        receipt = newc/(chain.name(3, plan=plan)+".receipt.json")
        receipt.unlink()
        assert cut.cutover(newpath, stage_only=True) == staged
        assert chain.read(receipt)["reconciled"] and len(rows) == 4
        if rollback:
            assert cut.rollback_staged(newpath)["old_callback"] == 9003
            assert set(rows) == {9002,9003} and rows[9002]["STAT"] == "RUN" and rows[9003]["STAT"] == "PEND"
            return
    if started_race:
        with pytest.raises(RuntimeError,match="has started; preserve it"):
            cut.cutover(newpath)
        assert rows[9004]["STAT"]=="PSUSP" and not (newc/"cutover-authorized.json").exists()
        assert [a[-1] for a in actions if a[0]=="bkill"]==["9003","9002"]
        return
    result=cut.cutover(newpath)
    assert result["worker"]==9004 and result["callback"]==9005
    assert rows[9004]["DEPENDENCY"] == ("" if segment==2 else "done(9002)")
    assert rows[9005]["DEPENDENCY"] == "ended(9004)"
    assert rows[9004]["STAT"]=="PEND" and chain.name(segment,plan=plan) == rows[9004]["JOB_NAME"]
    original=list(actions)
    removed = (2 if segment == 2 else 1) if purge else 0
    assert cut.cutover(newpath)==result and actions==original and len(rows)==4+int(parent_visible)-removed
    if segment==3:
        finish_segment(oldc,old,2,1650,"allocation_budget_reached")
    calls=[]
    monkeypatch.setattr(chain.subprocess,"run",lambda cmd,check:calls.append(cmd))
    monkeypatch.setenv("LSB_JOBID","9004")
    chain.run_worker(newpath,plan,segment)
    assert calls[0][-2:]==["--previous-config",chain.CONFIG if segment==2 else old["config"]]
    assert str(chain.output(old,segment-1)/"latest.json") in calls[0]


def test_cutover_requires_terminal_history_if_row_and_reply_vanished(tmp_path,monkeypatch):
    jid=3443
    observed=dict(STAT="PEND",START_TIME="",RUN_TIME="00:00:00")
    monkeypatch.setattr(cut,"command",lambda *argv: "accepted")
    cut.retire_pending(tmp_path,"a",jid,observed)
    (tmp_path/f"retire-{jid}.ack.json").unlink()
    class Reply:
        returncode=0
        stdout=f"Job <{jid}>\nNov 22: Exited with exit code 128."
    monkeypatch.setattr(cut.subprocess,"run",lambda *args,**kwargs:Reply())
    assert cut.retired_row(tmp_path,"a",jid)["STAT"]=="EXIT"
    for text in (f"Job <{jid}>\nNov 22: Done successfully.",
                 f"Job <{jid}>\nNov 22: Dispatched to <host>.\nNov 22: Exited with exit code 128.",
                 "Job <44>\nNov 22: Exited with exit code 128."):
        Reply.stdout=text
        with pytest.raises(RuntimeError,match="No confirmation"):
            cut.retired_row(tmp_path,"a",jid)


@pytest.mark.parametrize(("parent_status","expected_error"), [
    ("DONE","has started"), ("RUN","successful original first leg"), ("EXIT","successful original first leg")])
def test_pending_cutover_rejects_unexpected_scheduler_states(tmp_path, monkeypatch, parent_status, expected_error):
    oldc=tmp_path/"old"
    newc=tmp_path/"new"
    oldc.mkdir()
    newc.mkdir()
    original=fixture_plan(oldc)
    original.pop("handoff",None)
    handoff=dict(control=str(oldc),plan_sha256=sha((oldc/"plan.json").read_bytes()),
                 worker_id=9002,callback_id=9003,segment=2)
    plan=dict(original,output_root=str(newc/"runs"),handoff=handoff)
    path=newc/"plan.json"
    put(path,plan)
    rows={9001:dict(JOB_NAME="parent",STAT=parent_status)}
    for jid,label in ((9002,chain.name(2)),(9003,chain.name(2,after=True))):
        put(oldc/(label+".receipt.json"),dict(id=jid,argv=[chain.PYTHON,"run"]))
        rows[jid]=dict(JOB_NAME=label,USER="s234818",STAT="PEND",RUN_TIME="00:00:00",START_TIME="",
                       COMMAND=chain.PYTHON+" run",DEPENDENCY="ended(9002)" if jid==9003 else "done(9001)")
    rows[9002].update(STAT="RUN",START_TIME="Nov 22 22:55")
    monkeypatch.setattr(cut,"jobs",lambda:rows)
    monkeypatch.setattr(cut,"command",lambda *a: pytest.fail(str(a)))
    monkeypatch.setattr(chain,"submit",lambda *a:pytest.fail("unexpected submission"))
    with pytest.raises(RuntimeError,match=expected_error):
        cut.cutover(path)


def fixture_plan(tmp):
    parent = {"row": 0, "epoch": 47, "updates": chain.PARENT_UPDATES, "sha256": "a" * 64}
    put(tmp / "parent.json", parent)
    plan = dict(first_job_id=9001, first_launch_receipt=str(tmp / "first.launch.json"),
                first_directory=str(tmp / "first"), output_root=str(tmp / "runs"),
                parent_receipt=str(tmp / "parent.json"), archive=str(tmp / "source.tar.gz"),
                archive_sha256="b" * 64, config_sha256="f" * 64,
                config="experiments/vae/changed-motif.json")
    put(tmp / "plan.json", plan)
    return plan


def finish_segment(tmp, plan, seg, total_updates, reason):
    directory = chain.output(plan, seg)
    directory.mkdir(parents=True)
    updates = chain.PARENT_UPDATES + total_updates
    complete = total_updates * chain.BATCH == chain.TRAIN_SIZE
    cursor = dict(updates=updates, epoch=48 if complete else 47, real_points=total_updates * chain.BATCH,
                  row=0 if complete else total_updates * chain.BATCH)
    checkpoint = directory / f"checkpoint-{updates:09d}.pt"
    checkpoint.write_bytes(b"test checkpoint")
    latest = dict(cursor, checkpoint=str(checkpoint), sha256=sha(checkpoint.read_bytes()),
                  completed_epoch=cursor["epoch"])
    parent = chain.read(plan["parent_receipt"])
    old = seg == 1
    config_sha = chain.CONFIG_SHA if old else plan["config_sha256"]
    previous = None if old else chain.read(chain.output(plan, seg-1) / "latest.json")
    fork = dict(parent=parent, config_sha256=sha(f"{config_sha}:{parent['sha256']}:3".encode()),
                extra_rounds=3, target_updates=25_000 if old else None, resume=previous,
                target_epoch=48, wall_seconds=81_000)
    report = dict(cursor, batch_size=chain.BATCH, microbatch_size=48, training_molecules=chain.TRAIN_SIZE,
                  dataset_sha256=chain.DATA_SHA, segment_updates=updates - (previous or parent)["updates"],
                  latest=latest, status="completed" if complete else "paused", stop_reason=reason)
    put(directory / "fork.json", fork)
    put(directory / "summary.json", report)
    put(directory / "latest.json", latest)
    path = plan["first_launch_receipt"] if old else tmp / (chain.name(seg, plan=plan) + ".launch.json")
    put(path, dict(job_id=chain.job_id(tmp, plan, seg), status="worker_exited", returncode=0,
                   ready=True, source_archive_sha256=chain.FIRST_SOURCE_SHA if old else plan["archive_sha256"]))


def test_queue_resume_and_stop(tmp_path, monkeypatch):
    tmp = tmp_path
    plan = fixture_plan(tmp)
    path = tmp / "plan.json"
    submissions = []

    def submit(control, name, args):
        submissions.append(args)
        ident = 9001 + len(submissions)
        chain.write_once(control / (name + ".receipt.json"), dict(id=ident, argv=args))
        return ident

    monkeypatch.setattr(chain, "submit", submit)

    def invoke(action, seg):
        monkeypatch.setattr(sys, "argv", ["controller", action, str(path), "--segment", str(seg),
                                         "--plan-sha256", sha(path.read_bytes())])
        chain.main()

    invoke("submit", 2)
    invoke("submit", 2)
    assert len(submissions) == 2 and "done(9001)" in submissions[0] and "ended(9002)" in submissions[1]
    assert {"gpuv100", "24:00", "8"}.issubset(submissions[0])
    finish_segment(tmp, plan, 1, 25_000, "update_budget_reached")

    calls = []
    # bounded_file invokes subprocess.Popen, so mocking run still exercises the actual child I/O.
    monkeypatch.setattr(chain.subprocess, "run", lambda command, check: calls.append(command))
    monkeypatch.setenv("LSB_JOBID", "7777")
    with pytest.raises(ValueError, match="differs from assigned worker"):
        invoke("worker", 2)
    assert not calls and not (tmp / (chain.name(2) + ".claim.json")).exists()
    monkeypatch.setenv("LSB_JOBID", "9002")
    invoke("worker", 2)
    assert calls[-1][-2:] == ["--previous-config", chain.CONFIG]
    assert str(tmp / "first" / "latest.json") in calls[-1] and plan["config"] in calls[-1]
    with pytest.raises(FileExistsError):
        invoke("worker", 2)
    finish_segment(tmp, plan, 2, 80_000, "allocation_budget_reached")
    invoke("advance", 2)
    invoke("advance", 2)
    assert len(submissions) == 4 and "done(9002)" in submissions[2]
    monkeypatch.setenv("LSB_JOBID", "9004")
    invoke("worker", 3)
    assert "--previous-config" not in calls[-1]
    finish_segment(tmp, plan, 3, 250_000, "epoch_target_reached")
    invoke("advance", 3)
    assert len(submissions) == 4 and chain.read(tmp / "epoch48-completed.json")["latest"]["epoch"] == 48


def test_wall_stop_and_checkpoint_tampering(tmp_path):
    plan = fixture_plan(tmp_path)
    finish_segment(tmp_path, plan, 1, 14_131, "allocation_budget_reached")
    summary = chain.inspect(tmp_path, plan, 1)
    Path(summary["latest"]["checkpoint"]).write_bytes(b"different")
    with pytest.raises(ValueError, match="checksum"):
        chain.inspect(tmp_path, plan, 1)


def test_early_switch_resumes_only_pinned_stopped_writer_and_preserves_next_segment(tmp_path, monkeypatch):
    old_control, control = tmp_path / "rings", tmp_path / "early"
    old_control.mkdir()
    control.mkdir()
    old = fixture_plan(old_control)
    label = chain.name(2, plan=old)
    old_command = [chain.PYTHON, "ring-worker", "worker"]
    put(old_control / (label + ".receipt.json"), dict(id=9002, argv=old_command))
    put(old_control / (chain.name(2, after=True, plan=old) + ".receipt.json"), dict(id=9003))
    finish_segment(old_control, old, 1, 1000, "allocation_budget_reached")
    finish_segment(old_control, old, 2, 2700, "allocation_budget_reached")
    directory = chain.output(old, 2)
    (directory / "summary.json").unlink()
    put(old_control / (label + ".claim.json"), dict(job=9002))
    put(old_control / (label + ".launch.json"), dict(job_id="9002", status="worker_ready", nonce="nonce",
                                                     source_archive_sha256=old["archive_sha256"]))
    put(old_control / (label + ".launch.ready.json"), dict(job_id="9002", nonce="nonce",
                                                           status="models_and_inputs_ready"))
    fork = chain.read(directory / "fork.json")
    (directory / "activation").mkdir()
    put(directory / "activation" / "decision.json", dict(winner="control", protocol=fork["config_sha256"],
        predecessor=fork["resume"], arms=dict(control=dict(latest=dict(updates=fork["resume"]["updates"] + 512)))))
    handoff = dict(control=str(old_control), segment=3, worker_id=9002, callback_id=9003,
                   plan_sha256=sha((old_control / "plan.json").read_bytes()))
    plan = dict(old, handoff=handoff, early_stop={}, job_prefix="e48-r7-early",
                config_sha256="e" * 64, output_root=str(control / "runs"))
    path = control / "plan.json"
    put(path, plan)
    plan_sha = sha(path.read_bytes())
    latest = chain.read(directory / "latest.json")
    terminal = dict(JOBID="9002", STAT="EXIT", JOB_NAME=label, USER="s234818", COMMAND=shlex.join(old_command))
    assert chain.saved_checkpoint(plan) == latest
    for changed in (dict(STAT="RUN"), dict(COMMAND="another command"), dict(JOBID="9001")):
        with pytest.raises(ValueError, match="terminal LSF identity"):
            chain.stopped_checkpoint(control, plan, dict(terminal, **changed))
    stopped = chain.stopped_checkpoint(control, plan, terminal)
    assert stopped == dict(plan_sha256=plan_sha, worker_id=9002, terminal=terminal, latest=latest)

    submitted = []
    def reserve(where, label, args):
        submitted.append(args)
        jid = 9003 + len(submitted)
        put(where / (label + ".receipt.json"), dict(id=jid, argv=args))
        return jid
    monkeypatch.setattr(chain, "submit", reserve)
    chain.queue(path, plan, 3, plan_sha)
    assert "-H" in submitted[0] and "ended(9002)" in submitted[0] and "ended(9004)" in submitted[1]
    monkeypatch.setenv("LSB_JOBID", "9004")
    calls = []
    monkeypatch.setattr(chain.subprocess, "run", lambda command, check: calls.append(command))
    put(control / "cutover-authorized.json", dict(plan_sha256=plan_sha, worker=9004))
    put(control / "stopped.json", dict(stopped, latest=dict(latest, row=0)))
    with pytest.raises(ValueError, match="receipt changed"):
        chain.run_worker(path, plan, 3)
    assert not calls
    put(control / "stopped.json", stopped)
    chain.run_worker(path, plan, 3)
    assert calls[0][calls[0].index("--resume-receipt") + 1] == str(directory / "latest.json")
    assert calls[0][-2:] == ["--previous-config", old["config"]]
    assert not (directory / "summary.json").exists()
    finish_segment(control, plan, 3, 3200, "allocation_budget_reached")
    assert chain.inspect(control, plan, 3)["latest"]["updates"] == chain.PARENT_UPDATES + 3200
    chain.queue(path, plan, 4, plan_sha)
    assert "done(9004)" in submitted[2] and "-H" not in submitted[2]


def test_early_stop_scheduler_and_next_allocation(tmp_path, monkeypatch):
    ring, prior, new = [tmp_path / name for name in ("ring", "prior", "new")]
    for directory in (ring, prior, new):
        directory.mkdir()
    ring_plan = fixture_plan(ring)
    archive = ring / "source.tar.gz"
    archive.write_bytes(b"identical science")
    ring_plan["archive_sha256"] = sha(archive.read_bytes())
    put(ring / "plan.json", ring_plan)
    handoff = dict(control=str(ring), plan_sha256=early_cut.digest(ring / "plan.json"),
                   segment=3, worker_id=9002, callback_id=9003)
    original = dict(ring_plan, handoff=handoff, output_root=str(prior/"output"), job_prefix="e48-r7-original")
    put(prior/"plan.json", original)
    prior_sha = early_cut.digest(prior/"plan.json")
    put(prior/"cutover-authorized.json", dict(plan_sha256=prior_sha, worker=9004))
    plan = dict(original, early_stop=dict(superseded_control=str(prior), superseded_plan_sha256=prior_sha,
                                          superseded_worker_id=9004, superseded_callback_id=9005),
                output_root=str(new/"output"), job_prefix="e48-r7-early")
    path = new/"plan.json"
    put(path,plan)
    rows, actions = {}, []
    def reserve(control, label, args):
        jid = max(rows, default=9001) + 1
        argv = ["bsub", "-J", label, *args]
        put(control / (label + ".receipt.json"), dict(id=jid, argv=argv))
        rows[jid] = dict(JOBID=str(jid), USER="s234818", JOB_NAME=label, STAT="PSUSP" if "-H" in args else "PEND",
                         QUEUE=args[args.index("-q")+1], DEPENDENCY=args[args.index("-w")+1],
                         START_TIME="", RUN_TIME="00:00:00", COMMAND=shlex.join(args[args.index(chain.PYTHON):]))
        return jid
    monkeypatch.setattr(chain, "submit", reserve)
    chain.queue(ring/"plan.json", ring_plan, 2, handoff["plan_sha256"])
    rows[9002].update(STAT="USUSP", START_TIME="Nov 23 06:54", RUN_TIME="04:00:00")
    rows[9003].update(STAT="EXIT")
    chain.queue(prior/"plan.json", original, 3, prior_sha)
    rows[9004]["STAT"]="PEND"
    monkeypatch.setattr(cut, "jobs", lambda: rows)
    def command(exe, *args):
        actions.append((exe, *args))
        jid = int(args[-1])
        if exe == "bkill" and jid in (9004,9005):
            assert args == ("-stat", "pend", "-s", "TERM", str(jid)) and rows[jid]["STAT"] == "PEND"
            del rows[jid]
            return f"Job <{jid}> is being terminated"
        if exe == "bkill" and jid == 9002:
            assert args == ("9002",) and rows[jid]["STAT"] == "USUSP"
            rows[jid]["STAT"] = "EXIT"
            return "Job <9002> is being terminated"
        assert exe == "bresume" and jid == 9006 and rows[jid]["STAT"] == "PSUSP"
        rows[jid]["STAT"]="PEND"
        return "resumed"
    monkeypatch.setattr(cut, "command", command)
    latest = dict(updates=600116, sha256="e"*64)
    monkeypatch.setattr(chain, "saved_checkpoint", lambda plan: latest)
    monkeypatch.setattr(chain, "stopped_checkpoint", lambda control, plan, row:
                        dict(plan_sha256=early_cut.digest(path), worker_id=9002, terminal=row, latest=latest))
    assert early_cut.act(path, "stage")["worker"] == 9006
    assert rows[9006]["STAT"] == "PSUSP" and rows[9006]["DEPENDENCY"] == "ended(9002)"
    assert rows[9007]["DEPENDENCY"] == "ended(9006)"
    with pytest.raises(RuntimeError,match="still active"):
        early_cut.act(path,"stop")
    assert early_cut.act(path, "retire")["retired"] == [9005, 9004]
    assert early_cut.act(path, "stop")["verified_before_stop"] == latest
    assert early_cut.act(path, "stop")["observed_state"] == "EXIT"
    assert early_cut.act(path, "release")["latest"] == latest
    assert early_cut.act(path, "release")["scheduler"]["STAT"] == "PEND"
    assert [a[-1] for a in actions] == ["9005","9004","9002","9006"]
    next_pair = chain.queue(path, plan, 4, early_cut.digest(path))
    assert rows[next_pair["worker"]]["DEPENDENCY"] == "done(9006)"
    assert rows[next_pair["callback"]]["DEPENDENCY"] == f"ended({next_pair['worker']})"
