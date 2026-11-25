"""Use a replacement held reservation when DTU restricts modifying a pending command."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


HPC = "/zhome/95/e/206121/dbmol-control"
PACKAGE = "vae-e48-r7-motifs-20261121a"
PREVIOUS = "vae-e48-paired-v100-20261120d"
PREFIX = "dbmol-e48-r7-motifs-20261121a"
FOUR, OLD_WORKER, OLD_GATE, PARENT_CALLBACK, PILOT = "30131204","30131363","30131364","30114867","30147253"
FIELDS = "jobid stat queue job_name:95 nreq_slot runtimelimit run_time dependency combined_resreq command"


def require(ok, text):
    if not ok:
        raise RuntimeError(text)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def json_file(path):
    return json.loads(path.read_bytes())


def write_identical(path, contents):
    if path.exists():
        require(path.read_bytes() == contents, f"Existing reservation file differs: {path.name}")
    else:
        with path.open("xb") as stream:
            stream.write(contents)


def scheduler():
    result = subprocess.check_output(["bjobs","-a","-u","s234818","-json","-o",FIELDS],text=True)
    return {j["JOBID"]:j for j in json.loads(result)["RECORDS"] if j.get("JOBID")}


def run_remote(action, hashes):
    package = Path(HPC)/PACKAGE
    previous = Path(HPC)/PREVIOUS
    control = Path(HPC)/(PACKAGE+"-lsf")
    for directory, names in ((package, ("source.tar.gz","plan.json")),
                             (previous, ("plan.json","vae_segment_chain.py","vae_paired_first_leg.py"))):
        for name in names:
            require(digest(directory/name)==hashes[directory.name][name], "Source differs: "+str(directory/name))
    handoff = json_file(package/"plan.json")
    previous_plan = json_file(previous/"plan.json")
    four, old = previous_plan["arms"]
    old_gate_name = previous_plan["prefix"]+"-parent-01"
    gate_name, worker_name = PREFIX+"-parent-01", PREFIX+"-r7"
    new_command = handoff["command"]
    original_command = handoff["original_command"]
    require(handoff["worker"]==OLD_WORKER and handoff["gate"]==OLD_GATE and handoff["four"]==FOUR,
            "Superseded reservation IDs differ")
    require(digest(package/"source.tar.gz")==handoff["archive_sha256"], "New source digest differs")
    old_args = old["args"]
    cut = old_args.index("/usr/bin/python3")
    require(old_args[cut:]==original_command, "Old invocation differs")
    worker = {"name":worker_name,"args":old_args[:cut]+new_command}
    plan = dict(previous_plan, prefix=PREFIX, arms=[four,worker], release_after_retirement=[int(OLD_GATE),int(OLD_WORKER)])
    require(plan["parent_receipt"]==new_command[new_command.index("--parent-receipt")+1], "Parent gate path changed")
    for name, ident in ((four["name"],FOUR),(old["name"],OLD_WORKER),(old_gate_name,OLD_GATE)):
        require(str(json_file(previous/(name+".receipt.json"))["id"])==ident,"Existing receipt points elsewhere")

    def check_original(rows, retirement_started=False):
        held,gate,control_worker = rows[OLD_WORKER],rows[OLD_GATE],rows[FOUR]
        require(shlex.split(held["COMMAND"])==original_command and held["JOB_NAME"]==old["name"],
                "Seven-round job no longer has the old command")
        require(control_worker["JOB_NAME"]==four["name"] and shlex.split(control_worker["COMMAND"])==handoff["four_command"]
                and control_worker["STAT"]=="PSUSP" and control_worker["RUN_TIME"]=="00:00:00","Four-round worker changed")
        require(gate["JOB_NAME"]==old_gate_name and gate["RUN_TIME"]=="00:00:00","Original gate changed or began")
        require(held["STAT"] in (("PSUSP","EXIT") if retirement_started else ("PSUSP",))
                and held["RUN_TIME"]=="00:00:00","Original worker started or left its hold")
        require(gate["STAT"] in (("PEND","EXIT") if retirement_started else ("PEND",)),"Original gate no longer pending")
        require(rows[PARENT_CALLBACK]["STAT"]=="PEND" and not Path(plan["parent_receipt"]).exists(),
                "Parent gate has already advanced; reconcile before attempting cutover")
        for cmd in (original_command,new_command):
            for key in ("--output","--receipt"):
                require(not Path(cmd[cmd.index(key)+1]).exists(), "Worker output already exists")

    if action=="reserve":
        check_original(scheduler())
        control.mkdir(exist_ok=True)
        for name in ("vae_segment_chain.py","vae_paired_first_leg.py",four["name"]+".receipt.json"):
            write_identical(control/name,(previous/name).read_bytes())
        write_identical(control/"plan.json",(json.dumps(plan,indent=2)+"\n").encode())
        # Both new IDs will be separately receipted by the upstream submit helper.
        rows=scheduler()
        for row in rows.values():
            if row["JOB_NAME"] in (worker_name,gate_name):
                receipt=control/(row["JOB_NAME"]+".receipt.json")
                require(receipt.is_file() and str(json_file(receipt)["id"])==row["JOBID"],
                        "Existing reservation needs scheduler receipt reconciliation")
        proc=subprocess.run(["/usr/bin/python3",str(control/"vae_paired_first_leg.py"),"submit",str(control/"plan.json")],
                            capture_output=True, text=True)
        require(proc.returncode==0,"Reservation failed: "+proc.stderr+proc.stdout)

    require(json_file(control/"plan.json")==plan,"Replacement plan differs")
    new_id=str(json_file(control/(worker_name+".receipt.json"))["id"])
    gate_id=str(json_file(control/(gate_name+".receipt.json"))["id"])
    old_dependencies=f"done({PARENT_CALLBACK})&&ended({OLD_GATE})&&ended({OLD_WORKER})"

    def inspect():
        rows=scheduler()
        check_original(rows, retirement_started=action in ("retire","verify"))
        w,g=rows[new_id], rows[gate_id]
        require(w["JOB_NAME"]==worker_name and w["STAT"]=="PSUSP" and w["RUN_TIME"]=="00:00:00"
                and shlex.split(w["COMMAND"])==new_command and w["QUEUE"]=="gpuv100"
                and w["NREQ_SLOT"]=="8" and w["RUNTIMELIMIT"]=="24:00:00"
                and "gpu32gb" in w["COMBINED_RESREQ"] and "gmem=30720" in w["COMBINED_RESREQ"],
                "Replacement worker differs from the approved held V100 allocation")
        require(g["JOB_NAME"]==gate_name and g["STAT"]=="PEND" and g["DEPENDENCY"].replace(" ","")==old_dependencies
                and shlex.split(g["COMMAND"])==["/usr/bin/python3",str(control/"vae_paired_first_leg.py"),"advance",
                                               str(control/"plan.json"),"--segment","1"],"Replacement parent gate differs")
        return rows

    rows=inspect()
    if action=="retire":
        # First disarm the previous release callback. Wait for its terminal status
        # before retiring its held worker. A retry observes any completed substep.
        for jid in (OLD_GATE, OLD_WORKER):
            rows=inspect()
            if rows[jid]["STAT"]!="EXIT":
                proc=subprocess.run(["bkill",jid],capture_output=True,text=True)
                require(proc.returncode==0,"Scheduler retirement failed: "+proc.stderr+proc.stdout)
                for _ in range(18):
                    if scheduler()[jid]["STAT"]=="EXIT":
                        break
                    time.sleep(1)
                require(scheduler()[jid]["STAT"]=="EXIT","Scheduler retirement pending; reconcile via retire")
        rows=inspect()
    if action in ("retire","verify"):
        require(all(rows[jid]["STAT"]=="EXIT" and rows[jid]["RUN_TIME"]=="00:00:00" for jid in (OLD_GATE,OLD_WORKER)),
                "Old allocation or release gate is still pending")
    ids=[new_id,gate_id,OLD_WORKER,OLD_GATE,FOUR,PARENT_CALLBACK,PILOT]
    return dict(worker=new_id, gate=gate_id, archive_sha256=handoff["archive_sha256"], parent_receipt=plan["parent_receipt"],
                run= new_command[new_command.index("--output")+1],
                scheduler={i:{k:rows[i][k] for k in ("JOB_NAME","STAT","RUN_TIME","DEPENDENCY","QUEUE")}
                           for i in ids if i in rows})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("reserve","audit","retire","verify"))
    parser.add_argument("--remote-hashes",help=argparse.SUPPRESS)
    cli=parser.parse_args()
    if cli.remote_hashes:
        print(json.dumps(run_remote(cli.action,json.loads(cli.remote_hashes))))
        return
    submissions=Path(__file__).resolve().parents[2]/"outputs/hpc-submissions"
    hashes={tag:{name:digest(submissions/tag/name) for name in files} for tag,files in (
        (PACKAGE,("source.tar.gz","plan.json")),(PREVIOUS,("plan.json","vae_paired_first_leg.py","vae_segment_chain.py")))}
    remote_cmd="python3 -I - "+cli.action+" --remote-hashes "+shlex.quote(json.dumps(hashes))
    connector=Path.home()/"DTU-HPC/helper-v1/dtu-hpc.exe"
    result=subprocess.run([str(connector),"exec","--timeout","115","bash -lc "+shlex.quote(remote_cmd)],
                          text=True,input=Path(__file__).read_text(),capture_output=True)
    require(result.returncode==0,result.stdout+result.stderr)
    rec=json.loads(result.stdout)
    target=submissions/PACKAGE/("reservation-"+cli.action+".json")
    target.write_text(json.dumps(rec,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(rec,indent=2))


if __name__=="__main__":
    main()
