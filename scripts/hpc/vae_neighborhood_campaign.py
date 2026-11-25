"""Metadata-only scheduler for an exclusive, completed-epoch VAE handoff.

Queue a disposable GPU probe first. Explicit ``arm`` holds only the identified
pending old callback while its GPU job continues unchanged. A failed probe
restores that callback; a passed probe submits the new next epoch and retires
the held callback. Saved intent/receipt files prevent duplicate submissions.
No tensor/model imports or numerical work occur in this controller.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(argv, *, cwd=None):
    r = subprocess.run(argv, text=True, capture_output=True, timeout=60, cwd=cwd)
    return dict(command=argv, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)


def status(job):
    r = command(["bjobs", "-l", str(job)])
    m = re.search(r"Status\s*<([A-Z_]+)>", r["stdout"])
    if not m and re.fullmatch(r"\s*Job <"+re.escape(str(job))+r"> is not found\s*",r["stderr"]):
        return "AGED_OUT",r
    if r["returncode"] or not m:
        raise RuntimeError("Cannot establish scheduler state: " + json.dumps(r))
    if not re.search(r"Job <"+re.escape(str(job))+r">", r["stdout"]):
        raise RuntimeError("Scheduler response belongs to another job: " + json.dumps(r))
    return m.group(1), r


def terminal_history(job):
    """Exact-job terminal proof; a later restart invalidates an older DONE."""
    history = command(["bhist", "-l", str(job)])
    output = history["stdout"]
    matches = list(re.finditer(r": (Done successfully\.|Exited[^\n]*|Completed <exit>[^\n]*)", output))
    if (history["returncode"] or not re.search(r"Job <"+re.escape(str(job))+r">", output) or not matches):
        return None, history
    last = matches[-1]
    if re.search(r": (?:Requeued|Submitted|Dispatched|Starting|Running)\b", output[last.end():], re.I):
        return None, history
    return ("DONE" if last.group(1) == "Done successfully." else "EXIT"), history


def require_done(job, *, attempts=15, delay=2):
    # ended() can also mean EXIT. Never promote solely because a child saved
    # a result before a later failure. Allow brief scheduler propagation only.
    if type(attempts) is not int or not 1 <= attempts <= 30 or not 0 <= delay <= 5:
        raise ValueError("Invalid bounded scheduler retry policy")
    evidence = []
    for attempt in range(attempts):
        state, proof = status(job)
        evidence.append(dict(source="bjobs", state=state, proof=proof))
        if state == "DONE":
            return dict(proof, scheduler_evidence=evidence)
        if state == "EXIT":
            raise ValueError("Predecessor allocation did not finish DONE: " + str(job))
        if state in ("RUN", "AGED_OUT"):
            terminal, history = terminal_history(job)
            evidence.append(dict(source="bhist", state=terminal, proof=history))
            if terminal == "DONE":
                return dict(history, scheduler_evidence=evidence)
            if terminal == "EXIT":
                prefix = "No terminal DONE proof for aged-out predecessor: " if state == "AGED_OUT" else "Predecessor allocation did not finish DONE: "
                raise ValueError(prefix + str(job))
        if state not in ("RUN", "AGED_OUT") or attempt == attempts-1:
            prefix = "No terminal DONE proof for aged-out predecessor: " if state == "AGED_OUT" else "Predecessor allocation did not finish DONE: "
            raise ValueError(prefix + str(job) + "; evidence=" + json.dumps(evidence))
        time.sleep(delay)


class Campaign:
    def __init__(self, path, sha):
        self.path, self.sha = Path(path), sha
        if digest(path) != sha:
            raise ValueError("Study changed")
        self.study = read(path)
        self.root = Path(self.study["campaign"])
        recovery=self.study.get("launcher_recovery")
        self.control=Path(recovery["control_root"]) if recovery else self.root
        self.control.mkdir(parents=True,exist_ok=True)
        self.receipts = self.control / "receipts"
        self.release=Path(self.study["release"])
        self.python="/work3/s234818/latentmol/env/boltz311/bin/python"
        if recovery:
            if self.path.resolve()!=self.control/"study.json":
                raise ValueError("Recovery control must use its home-storage study")
            for name,relative in (("vae_neighborhood_campaign.py","scripts/hpc/vae_neighborhood_campaign.py"),
                                  ("hpc_launch.py","scripts/hpc/hpc_launch.py")):
                if digest(self.control/name)!=self.study["changed_sources"][relative]:
                    raise ValueError("Home control source changed")
            if digest(self.control/"base-plan.json")!=self.study["base_plan_sha256"]:
                raise ValueError("Home copy of original plan changed")
            self.base=read(self.control/"base-plan.json")
            os.environ["DBMOL_CONTROL_IO"]=str(self.control)
            # Model worker retains archive/extracted-source/asset checks.
            # Callback startup must not synchronously depend on scratch reads.
            return
        self.root.mkdir(parents=True, exist_ok=True)
        if digest(self.study["base_plan"]) != self.study["base_plan_sha256"]:
            raise ValueError("Original plan changed")
        self.base = read(self.study["base_plan"])
        if digest(str(self.release)+".tar.gz") != self.study["archive_sha256"]:
            raise ValueError("Source snapshot changed")
        for relative, expected in self.study["changed_sources"].items():
            if digest(self.release/relative) != expected:
                raise ValueError("Changed extracted study executable: " + relative)

    def scratch(self,path,operation="json"):
        if not self.study.get("launcher_recovery"):
            return read(path) if operation=="json" else digest(path)
        from hpc_launch import bounded_file
        return bounded_file(path,operation,timeout=30,spool=self.control)

    def cli(self, mode, *extra):
        if self.study.get("launcher_recovery"):
            return ["/usr/bin/python3",str(self.control/"vae_neighborhood_campaign.py"),
                    "--study",str(self.path),"--study-sha256",self.sha,"--mode",mode,*extra]
        return [self.python, str(self.release / "scripts/hpc/vae_neighborhood_campaign.py"),
                "--study", str(self.path), "--study-sha256", self.sha, "--mode", mode, *extra]

    def worker(self, mode, path, sha, epoch=None):
        args = ["env", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4",
                "CUBLAS_WORKSPACE_CONFIG=:4096:8", "PYTHONPATH="+str(self.release/"src"),
                self.python, "-m", "latentmol.vae_neighborhood_continuation",
                "--study", str(self.path), "--study-sha256", self.sha,
                "--mode", mode, "--checkpoint", str(path), "--checkpoint-sha256", sha,
                "--wall-seconds", "12600" if mode=="probe" else "82800"]
        if epoch is not None:
            args += ["--epoch", str(epoch)]
        # Preserve the working DTU Python-module/venv launch environment. The
        # wrapper defaults OMP to allocated slots, so override to4 afterwards.
        command=["env", "LATENTMOL_SOURCE_ARCHIVE="+str(self.release)+".tar.gz",
                 "bash", "scripts/hpc/boltz_run.sh", *args]
        if self.study.get("launcher_recovery"):
            label="probe" if mode=="probe" else "epoch"+str(epoch)
            return ["/usr/bin/python3",str(self.control/"hpc_launch.py"),
                "--archive",str(self.release)+".tar.gz","--archive-sha256",self.study["archive_sha256"],
                "--release",str(self.release),"--receipt",str(self.control/(label+"-launch.json")),
                "--startup-seconds","600","--run-seconds","13200" if mode=="probe" else "84600",
                "--",*command]
        return command

    def submit(self, stage, argv, *, gpu=False, dependency=None):
        receipt = self.receipts / (stage+".json")
        if receipt.exists():
            value = read(receipt)
            if value["study_sha256"] != self.sha:
                raise ValueError("Existing submission belongs to another study")
            return value["job_id"]
        name = self.study.get("job_prefix", "vae-neighborhood-")+stage
        args = ["bsub", "-J", name, "-q", "gpua100" if gpu else "hpc",
                "-n", "16" if gpu else "1", "-R", "rusage[mem=4GB]", "-R", "span[hosts=1]",
                "-W", ("4:00" if stage=="probe" else "24:00") if gpu else "0:15",
                "-oo", str(getattr(self,"control",self.root)/(stage+".%J.out")),
                "-eo", str(getattr(self,"control",self.root)/(stage+".%J.err"))]
        if gpu:
            args += ["-R", "select[gpu80gb]", "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes"]
        if dependency:
            args += ["-w", dependency]
        args += argv
        save(self.receipts/(stage+"-intent.json"), dict(study_sha256=self.sha, command=args))
        r = command(args, cwd=self.control if self.study.get("launcher_recovery") else self.release)
        save(self.receipts/(stage+"-reply.json"), r)
        match = re.fullmatch(r"Job <([0-9]+)> is submitted to queue <[^>]+>\.\s*", r["stdout"])
        if r["returncode"] or not match:
            raise RuntimeError("Unknown/failed submission; reconcile intent and scheduler before retrying")
        job = match.group(1)
        save(receipt, dict(study_sha256=self.sha, job_id=job, status="submitted_not_passed"))
        return job

    def queue_probe(self):
        p = self.study["probe_checkpoint"]
        probe = self.submit("probe", self.worker("probe", p["path"], p["sha256"]), gpu=True)
        callback = self.submit("after-probe", self.cli("after-probe"), dependency="ended("+probe+")")
        return dict(probe=probe, callback=callback)

    def arm(self):
        """Explicit user-requested next-epoch adoption; never touch the GPU job."""
        if self.study.get("launcher_recovery"):
            receipt=self.control/"handoff-armed.json"
            if receipt.exists():return read(receipt)
            source_done=require_done(self.study["source_job"])
            state,proof=status(self.study["source_callback"])
            if (state!="PSUSP" or self.study["base_plan_sha256"] not in re.sub(r"\s+","",proof["stdout"])
                    or "after-segment-003" not in re.sub(r"\s+","",proof["stdout"])):
                raise ValueError("Existing held original callback identity differs")
            result=dict(status="recovered_held_epoch43_handoff",study_sha256=self.sha,
                        held_callback=self.study["source_callback"],source_done=True,source_done_evidence=source_done)
            save(receipt,result)
            return result
        receipt = self.root/"handoff-armed.json"
        if receipt.exists():
            return read(receipt)
        if not (self.receipts/"probe.json").exists():
            raise ValueError("Queue the actual probe before holding a predecessor callback")
        intent = self.root/"callback-hold-intent.json"
        state, job = status(self.study["source_job"])
        callback_state, callback = status(self.study["source_callback"])
        recovering = callback_state == "PSUSP" and intent.exists() and read(intent).get("study_sha256") == self.sha
        if (state not in ("RUN", "PEND", "DONE") or (callback_state != "PEND" and not recovering)
                or self.study["base_plan_sha256"] not in re.sub(r"\s+", "", callback["stdout"])
                or "after-segment-003" not in re.sub(r"\s+", "", callback["stdout"])):
            raise ValueError("Original job/callback already advanced or differs; no handoff performed")
        if not recovering:
            save(intent, dict(study_sha256=self.sha, callback=self.study["source_callback"], action="hold_exact_pending_callback"))
            r = command(["bstop", self.study["source_callback"]])
            save(self.root/"callback-hold-reply.json", r)
            if r["returncode"]:
                raise RuntimeError("Could not hold original callback")
        held, proof = status(self.study["source_callback"])
        if held != "PSUSP":
            raise RuntimeError("Original pending callback is not confirmed held")
        result = dict(status="armed_next_epoch_pending_probe", study_sha256=self.sha,
                      running_job_unchanged=self.study["source_job"], held_callback=self.study["source_callback"],
                      source_status=job, callback_before=callback, callback_after=proof)
        save(receipt, result)
        return result

    def after_probe(self):
        control=getattr(self,"control",self.root)
        if (control/"after-probe-result.json").exists():
            return read(control/"after-probe-result.json")
        if self.study.get("structural_adoption"):
            return self.after_structural_probe()
        if not (control/"handoff-armed.json").exists():
            raise ValueError("Next-epoch handoff was not explicitly armed")
        probe = self.root/"probe/result.json"
        try:probe_result=self.scratch(probe)
        except FileNotFoundError:probe_result={}
        passed = probe_result.get("status")=="passed_bounded_probe"
        if passed:
            try:
                require_done(read(self.receipts/"probe.json")["job_id"])
            except ValueError:
                passed = False
        held, proof = status(self.study["source_callback"])
        if held != "PSUSP":
            raise ValueError("Exclusive held callback was lost")
        if not passed:
            if self.study.get("launcher_recovery"):
                result=dict(status="probe_failed_epoch43_preserved",study_sha256=self.sha,
                            held_callback=self.study["source_callback"],new_training_submitted=False)
                save(control/"after-probe-result.json",result)
                return result
            r = command(["bresume", self.study["source_callback"]])
            save(self.root/"callback-restore-reply.json", r)
            if r["returncode"]:
                raise RuntimeError("Probe failed and original callback restoration needs attention")
            result = dict(status="probe_failed_original_callback_restored", reply=r)
        else:
            if probe_result["study_sha256"] != self.sha:
                raise ValueError("Probe belongs to another study")
            if self.study.get("launcher_recovery") and (
                    probe_result.get("source_checkpoint_sha256")!=self.study["probe_checkpoint"]["sha256"]
                    or probe_result.get("source_completed_epoch")!=43
                    or probe_result.get("completed_updates")!=self.study["probe_updates"]
                    or probe_result.get("frozen_anchors_unchanged") is not True
                    or not probe_result.get("terminal_checks")
                    or not all(v is True for v in probe_result["terminal_checks"].values())):
                raise ValueError("Recovery probe lacks its complete epoch43/guard binding")
            dependency="done("+self.study["source_job"]+")"
            if self.study.get("launcher_recovery"):
                require_done(self.study["source_job"])
                dependency=None # Aged-out LSF IDs cannot satisfy a new dependency.
            job = self.submit("epoch44", self.cli("run-epoch", "--epoch", "44"), gpu=True,
                              dependency=dependency)
            callback = self.submit("after-epoch44", self.cli("advance", "--epoch", "44"),
                                   dependency="ended("+job+")")
            r = command(["bkill", self.study["source_callback"]])
            save(control/"callback-retire-reply.json", r)
            if r["returncode"]:
                raise RuntimeError("Replacement queued but held old callback retirement needs attention")
            result = dict(status="next_epoch_replacement_submitted_not_started", job_id=job,
                          callback_job_id=callback, retired_callback=self.study["source_callback"])
        save(control/"after-probe-result.json", result)
        return result

    def structural_source(self):
        """A new discovery policy adopts completed44, never the failed callback."""
        adoption = self.study["structural_adoption"]
        proof = require_done(self.study["source_job"])
        completion = self.scratch(self.study["source_completion"])
        if (completion.get("status") != "completed_epoch" or completion.get("epoch") != 44
                or completion.get("study_sha256") != adoption["source_study_sha256"]
                or completion.get("checkpoint") != self.study["probe_checkpoint"]["path"]
                or completion.get("checkpoint_sha256") != self.study["probe_checkpoint"]["sha256"]
                or completion.get("frozen_anchors_unchanged") is not True
                or not completion.get("terminal_checks")
                or not all(v is True for v in completion["terminal_checks"].values())):
            raise ValueError("Structural adoption needs the authentic passed epoch44 result")
        if self.scratch(self.study["source_completion"], "hash") != adoption["source_completion_sha256"]:
            raise ValueError("Predecessor completion changed")
        old_receipts = Path(adoption["source_study"]).parent/"receipts"
        if any((old_receipts/name).exists() for name in ("epoch45.json", "epoch45-intent.json")):
            raise ValueError("Old epoch45 submission exists; reconcile before new adoption")
        return completion, proof

    def after_structural_probe(self):
        probe = self.scratch(self.root/"probe/result.json")
        require_done(read(self.receipts/"probe.json")["job_id"])
        if (probe.get("status") != "passed_bounded_probe" or probe.get("study_sha256") != self.sha
                or probe.get("source_checkpoint_sha256") != self.study["probe_checkpoint"]["sha256"]
                or probe.get("source_completed_epoch") != 44
                or probe.get("completed_updates") != self.study["probe_updates"]
                or probe.get("repair_discovery") != self.study["structural_adoption"]["proposal_config"]
                or probe.get("frozen_anchors_unchanged") is not True
                or not probe.get("terminal_checks")
                or not all(v is True for v in probe["terminal_checks"].values())):
            raise ValueError("Structural GPU probe did not pass its exact checkpoint/policy checks")
        self.structural_source()
        job = self.submit("epoch45", self.cli("run-epoch", "--epoch", "45"), gpu=True)
        callback = self.submit("after-epoch45", self.cli("advance", "--epoch", "45"), dependency="ended("+job+")")
        result = dict(status="structural_epoch45_submitted", job_id=job, callback_job_id=callback,
                      source_checkpoint=self.study["probe_checkpoint"])
        save(self.control/"after-probe-result.json", result)
        return result

    def run_epoch(self, epoch):
        if epoch==44:
            require_done(self.study["source_job"])
            completion = self.scratch(self.study["source_completion"])
            if (completion.get("status")!="passed" or completion.get("completed_epoch")!=43
                    or completion.get("plan_sha256")!=self.study["base_plan_sha256"]):
                raise ValueError("Original epoch43 has not passed its complete-epoch checks")
            path, sha = completion["checkpoint"], completion["checkpoint_sha256"]
            if self.study.get("launcher_recovery") and dict(path=path,sha256=sha)!=self.study["probe_checkpoint"]:
                raise ValueError("Continuation must use the same epoch43 checkpoint as its probe")
        elif epoch==45:
            if self.study.get("structural_adoption"):
                completion, _ = self.structural_source()
            else:
                completion = self.scratch(self.root/"epoch0044/result.json")
                if completion.get("status")!="completed_epoch" or completion.get("study_sha256")!=self.sha:
                    raise ValueError("Amended epoch44 did not complete")
            path, sha = completion["checkpoint"], completion["checkpoint_sha256"]
        else:
            raise ValueError("Only epochs44/45 are authorized")
        if self.scratch(path,"hash")!=sha:
            raise ValueError("Completed input checkpoint changed")
        # A unique output folder plus scheduler dependency and the held/retired
        # old callback prevent two writers. Failed folders are never resumed.
        r = subprocess.run(self.worker("train", path, sha, epoch),
                           cwd=self.control if self.study.get("launcher_recovery") else self.release)
        if r.returncode:
            raise RuntimeError("Neighborhood epoch worker failed; do not auto-resubmit")

    def advance(self, epoch):
        require_done(read(self.receipts/f"epoch{epoch}.json")["job_id"])
        result = self.scratch(self.root/f"epoch{epoch:04d}/result.json")
        if result.get("status")!="completed_epoch" or result.get("study_sha256")!=self.sha:
            raise ValueError("Completed-epoch result required; no automatic failure restart")
        if epoch==44:
            job = self.submit("epoch45", self.cli("run-epoch", "--epoch", "45"), gpu=True)
            callback = self.submit("after-epoch45", self.cli("advance", "--epoch", "45"),
                                   dependency="ended("+job+")")
            return dict(status="epoch45_submitted", job_id=job, callback_job_id=callback)
        if epoch!=45:
            raise ValueError("Unexpected endpoint")
        control=getattr(self,"control",self.root)
        save(control/"completion.json", dict(status="completed_through45", study_sha256=self.sha,
             checkpoint=result["checkpoint"], checkpoint_sha256=result["checkpoint_sha256"]))
        return read(control/"completion.json")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True); p.add_argument("--study-sha256", required=True)
    p.add_argument("--mode", choices=("probe", "arm", "after-probe", "run-epoch", "advance"), required=True)
    p.add_argument("--epoch", type=int)
    a=p.parse_args(); campaign=None
    print(json.dumps(dict(event="controller_started",mode=a.mode,epoch=a.epoch,
                          job_id=os.environ.get("LSB_JOBID"))),file=sys.stderr,flush=True)
    try:
        campaign=Campaign(a.study,a.study_sha256)
        if a.mode=="probe": result=campaign.queue_probe()
        elif a.mode=="arm": result=campaign.arm()
        elif a.mode=="after-probe": result=campaign.after_probe()
        elif a.mode=="run-epoch": result=campaign.run_epoch(a.epoch)
        else: result=campaign.advance(a.epoch)
    except Exception as exc:
        if campaign is not None and campaign.study.get("launcher_recovery"):
            failure=campaign.control/(a.mode+"-"+str(a.epoch)+"-"+os.environ.get("LSB_JOBID","local")+"-failure.json")
            if not failure.exists():
                save(failure,dict(status="controller_failed_no_automatic_restart",study_sha256=a.study_sha256,
                    error=type(exc).__name__+": "+str(exc),mode=a.mode,epoch=a.epoch))
        raise
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
