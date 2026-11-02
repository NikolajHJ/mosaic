"""Bounded exploratory storage checks, with an explicit historical strict mode."""
import csv
from decimal import Decimal, ROUND_CEILING
import math
import os
from pathlib import Path
import shutil
import subprocess

GIB=1024**3
VERSION="mosaic-live-quota-remaining-writes-v2"
EXPLORATORY_VERSION="mosaic-exploratory-storage-v1"
QUOTA_BURST_BYTES=GIB
UNPROFILED_WRITER_BYTES=2*GIB
RESERVE_BYTES=4*GIB
FILE_RESERVE=100000

def require(value,message):
    if not value: raise ValueError(message)

def nonnegative(value,name):
    require(type(value) is int and value>=0,name+" must be a nonnegative integer")
    return value

def parse_pool(text,uid,name):
    """Exact-byte BeeGFS CSV, preserving every finite user pool limit."""
    require(isinstance(text,str) and text.endswith("\n"),"Truncated quota response")
    lines=text.splitlines();header="name,id,size,hard,files,hard"
    require(lines.count(header)==1,"Missing/duplicate exact quota header")
    rows=list(csv.reader(lines[lines.index(header)+1:]));rows=[r for r in rows if r]
    require(len(rows)==1 and len(rows[0])==6,"Quota must contain exactly one complete user row")
    n,i,used,hard,files,filehard=rows[0]
    require(n==name and i==str(uid),"Quota account differs")
    require(used.isdecimal() and files.isdecimal(),"Quota usage is not exact bytes/files")
    require(hard=="unlimited" or hard.isdecimal(),"Malformed quota byte limit")
    require(filehard=="unlimited" or filehard.isdecimal(),"Malformed quota file limit")
    h=None if hard=="unlimited" else int(hard);f=None if filehard=="unlimited" else int(filehard)
    require((h is None or h>0) and (f is None or f>0),"Nonpositive quota hard limit")
    return {"used_bytes":int(used),"hard_bytes":h,"used_files":int(files),"hard_files":f}

def combine_pools(pools):
    require(set(pools)==set(range(1,7)),"All six storage pools are required")
    bytes_free=[r["hard_bytes"]-r["used_bytes"] for r in pools.values() if r["hard_bytes"] is not None]
    files_free=[r["hard_files"]-r["used_files"] for r in pools.values() if r["hard_files"] is not None]
    require(bytes_free and files_free,"Finite byte and file quota evidence required")
    return {"available_bytes":min(bytes_free),"available_files":min(files_free)}

def remaining_writes(cap,current):
    nonnegative(cap,"Cap");nonnegative(current,"Current bytes")
    require(current<=cap,"Writer exceeded its declared complete output allowance")
    return cap-current

def embedding_forecast(cases,completed,observed_rates):
    """Remaining updates at max(pilot, observed rate), one 25% margin +1GiB."""
    ids={"case-001","case-002","case-003"}
    require(set(completed)==ids and set(observed_rates)==ids and {r["case"] for r in cases}==ids
            and len(cases)==3,"Exactly three authenticated embedding cases required")
    result=[];future=GIB
    for row in cases:
        name=row["case"];done=nonnegative(completed[name],"Completed updates")
        require(row["gates_passed"] is True and row["completed_updates"]==20 and 20<=done<=10000,
                "Original pilot/target or contiguous state count differs")
        a=row["profile"]["bytes_per_update"];b=observed_rates[name]
        require(type(a) in (float,int) and type(b) in (float,int) and math.isfinite(a) and math.isfinite(b)
                and a>=0 and b>=0,"Storage rate must be finite and nonnegative")
        rate=max(Decimal(str(a)),Decimal(str(b)));left=10000-done
        amount=int((rate*left*Decimal(5)/Decimal(4)).to_integral_value(rounding=ROUND_CEILING))
        future+=amount;result.append({"case":name,"completed_updates":done,"remaining_updates":left,
             "rate_bytes_per_update":str(rate),"remaining_bytes_with_25_percent_margin":amount})
    return {"remaining_bytes":future,"cases":result,"fixed_control_allowance_bytes":GIB,
            "margin_numerator":5,"margin_denominator":4,"production_settings_changed":False}

def _assess_qualified(quota,disk_bytes,inodes,scopes,external,unprofiled_writer_ids=()):
    """Existing files are already in live quota; only future growth is additive."""
    nonnegative(disk_bytes,"Filesystem free bytes")
    require(len(unprofiled_writer_ids)==len(set(unprofiled_writer_ids)),"Duplicate unprofiled writer")
    ids=set();future=files=0
    for row in scopes:
        require(row["id"] not in ids,"Duplicate future writer allowance");ids.add(row["id"])
        future+=nonnegative(row["remaining_bytes"],"Remaining bytes")
        files+=nonnegative(row["remaining_files"],"Remaining files")
    for row in external:
        require(row["id"] not in ids,"Overlapping external allowance");ids.add(row["id"])
        future+=nonnegative(row["remaining_bytes"],"Remaining external bytes")
        files+=nonnegative(row["remaining_files"],"Remaining external files")
    total,free,avail=(nonnegative(inodes[k],k) for k in ("f_files","f_ffree","f_favail"))
    require((total,free,avail)==(0,0,0) or total>0 and 0<=avail<=free<=total,"Incoherent filesystem inode evidence")
    available=min(quota["available_bytes"],disk_bytes)-QUOTA_BURST_BYTES
    available_files=quota["available_files"] if total==0 else min(quota["available_files"],avail)
    extra=UNPROFILED_WRITER_BYTES*len(unprofiled_writer_ids)
    required=RESERVE_BYTES+future+extra;required_files=FILE_RESERVE+files+10000*len(unprofiled_writer_ids)
    return {"version":VERSION,"status":"passed" if available>=required and available_files>=required_files else "failed",
        "available_bytes":available,"required_bytes":required,"byte_margin":available-required,
        "available_files":available_files,"required_files":required_files,"file_margin":available_files-required_files,
        "reserve_bytes":RESERVE_BYTES,"file_reserve":FILE_RESERVE,"single_quota_burst_allowance_bytes":QUOTA_BURST_BYTES,
        "unprofiled_writer_ids":list(unprofiled_writer_ids),"unprofiled_writer_allowance_bytes":extra,
        "future_bytes":future,"existing_bytes_recharged":0,"existing_files_recharged":0,
        "filesystem_inode_reporting":"unreported_all_zero" if total==0 else "reported_positive",
        "not_a_whole_account_reservation":True,"entry_and_periodic_checks_required":True,
        "scientific_settings_changed":False}


def assess(quota,disk_bytes,inodes,scopes,external=(),unprofiled_writer_ids=(),*,
           research_profile="exploratory",policy_version=None):
    """Admit this run's remaining writes; unrelated forecasts are advisory.

    Explicit v2 replay retains the original reserves and whole-account forecasts.
    Missing quota is reported, never invented: physical free space remains required.
    """
    require(research_profile in ("exploratory","qualified"),"Unknown research profile")
    require(policy_version in (None,VERSION,EXPLORATORY_VERSION),"Unknown storage policy version")
    if research_profile=="qualified" or policy_version==VERSION:
        result=_assess_qualified(quota,disk_bytes,inodes,scopes,external,unprofiled_writer_ids)
        result["research_profile"]="qualified"
        return result
    nonnegative(disk_bytes,"Filesystem free bytes")
    ids=set();future=files=0
    for row in scopes:
        require(row["id"] not in ids,"Duplicate future writer allowance");ids.add(row["id"])
        future+=nonnegative(row["remaining_bytes"],"Remaining bytes")
        files+=nonnegative(row["remaining_files"],"Remaining files")
    total,free,avail=(nonnegative(inodes[k],k) for k in ("f_files","f_ffree","f_favail"))
    require((total,free,avail)==(0,0,0) or total>0 and 0<=avail<=free<=total,"Incoherent filesystem inode evidence")
    warnings=[];available=disk_bytes;available_files=None if total==0 else avail
    if quota is None:
        warnings.append("Quota unavailable; admission uses measured physical free space and the own-output cap.")
    else:
        for key in ("available_bytes","available_files"):
            require(quota.get(key) is None or type(quota[key]) is int,"Malformed available quota")
        if quota.get("available_bytes") is not None:available=min(available,quota["available_bytes"])
        else:warnings.append("No finite byte quota reported; physical free space is the byte limit.")
        if quota.get("available_files") is not None:
            available_files=quota["available_files"] if available_files is None else min(available_files,quota["available_files"])
    if available_files is None:warnings.append("Filesystem/quota inode availability is unreported; the own-output file cap still applies.")
    if external:warnings.append("Other studies' future output forecasts are advisory and are not charged to this run.")
    if unprofiled_writer_ids:warnings.append("Unprofiled account writers observed; their provenance does not block this run.")
    # Ten percent, bounded to 16--256 MiB; no multi-GiB tax on small probes.
    reserve=max(16*1024**2,min(256*1024**2,(future+9)//10))
    file_reserve=max(16,min(1000,(files+9)//10))
    required=future+reserve;required_files=files+file_reserve
    file_ok=available_files is None or available_files>=required_files
    return {"version":EXPLORATORY_VERSION,"research_profile":"exploratory",
        "status":"passed" if available>=required and file_ok else "failed",
        "available_bytes":available,"required_bytes":required,"byte_margin":available-required,
        "available_files":available_files,"required_files":required_files,
        "file_margin":None if available_files is None else available_files-required_files,
        "reserve_bytes":reserve,"file_reserve":file_reserve,"single_quota_burst_allowance_bytes":0,
        "unprofiled_writer_ids":list(dict.fromkeys(unprofiled_writer_ids)),"unprofiled_writer_allowance_bytes":0,
        "external_remaining_budgets":list(external),"future_bytes":future,"future_files":files,
        "existing_bytes_recharged":0,"existing_files_recharged":0,"warnings":warnings,
        "filesystem_inode_reporting":"unreported_all_zero" if total==0 else "reported_positive",
        "quota_observation":"unavailable" if quota is None else "reported",
        "not_a_whole_account_reservation":True,"entry_and_periodic_checks_required":False,
        "scientific_settings_changed":False}


def validate_remote_paths(value):
    """Reject Windows/noncanonical paths before sealing a Linux launch plan."""
    from pathlib import PurePosixPath
    if isinstance(value,dict):
        for v in value.values():validate_remote_paths(v)
    elif isinstance(value,list):
        for v in value:validate_remote_paths(v)
    elif isinstance(value,str):
        if value.startswith(("/","\\")) or len(value)>2 and value[1]==":" and value[2] in "/\\":
            require(value.startswith("/") and "\\" not in value and str(PurePosixPath(value))==value
                    and ".." not in PurePosixPath(value).parts,"Noncanonical remote POSIX path")
    return value


def own_output(root):
    """Walk only the owned run directory, without following links outside it."""
    root=Path(root);size=count=0;pending=[root]
    while pending:
        directory=pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    require(Path(entry.path).resolve().is_relative_to(root),"Symlink escapes the bounded output target")
                    size+=entry.stat(follow_symlinks=False).st_size;count+=1
                elif entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))
                else:
                    require(entry.is_file(follow_symlinks=False),"Special file in owned output")
                    size+=entry.stat(follow_symlinks=False).st_size;count+=1
    return {"actual_bytes":size,"actual_files":count,"nofollow":True}


def exploratory_quota(root):
    """Best-effort fresh quota; missing tools/partial replies are explicit warnings."""
    if not str(root).startswith("/work3/") or shutil.which("beegfs-ctl") is None:
        return None,["BeeGFS quota tool is unavailable for this filesystem."]
    import pwd
    uid=os.getuid();name=pwd.getpwuid(uid).pw_name;pools={};notes=[]
    for index in range(1,7):
        argv=["beegfs-ctl","--getquota","--uid",str(uid),"--storagepoolid="+str(index),"--csv",
              "--cfgFile=/etc/beegfs/beegfs-client4.conf"]
        try:
            result=subprocess.run(argv,capture_output=True,text=True,timeout=2)
        except (OSError,subprocess.TimeoutExpired) as error:
            notes.append("Quota observation unavailable: "+str(error));break
        if result.returncode!=0:
            notes.append("Quota command failed: "+result.stderr.strip());break
        try:pools[index]=parse_pool(result.stdout,uid,name)
        except ValueError as error:
            notes.append("Quota response unusable: "+str(error));break
    if len(pools)!=6:notes.append("Quota pool observation is incomplete; physical capacity and any observed finite limits apply.")
    if not pools:return None,notes
    limits={}
    for key,hard,used in (("available_bytes","hard_bytes","used_bytes"),("available_files","hard_files","used_files")):
        finite=[row[hard]-row[used] for row in pools.values() if row[hard] is not None]
        limits[key]=min(finite) if finite else None
    return limits,notes


def assess_output(root, output_cap_bytes, file_cap=10000):
    """Bound the owned output tree, without loading an experiment/controller."""
    root = Path(root).absolute()
    require(root.is_dir() and root.resolve() == root and root != Path(root.anchor),
            "Canonical owned output directory required")
    measured = own_output(root)
    scopes = [{"id": "own-run", "remaining_bytes": remaining_writes(output_cap_bytes, measured["actual_bytes"]),
               "remaining_files": remaining_writes(file_cap, measured["actual_files"])}]
    quota, notes = exploratory_quota(root)
    fs = os.statvfs(root) if hasattr(os, "statvfs") else None
    inodes = {key: getattr(fs, key, 0) for key in ("f_files", "f_ffree", "f_favail")}
    result = assess(quota, shutil.disk_usage(root).free, inodes, scopes)
    result["warnings"].extend(notes)
    result.update(output_root=str(root), output_cap_bytes=output_cap_bytes, file_cap=file_cap, own_output=measured)
    return result


def main():
    """Portable own-output admission. Does not submit, clean up or load a model."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cap-mib", required=True, type=int, help="Total byte cap for the owned output tree, in MiB")
    parser.add_argument("--file-cap", default=10000, type=int)
    args = parser.parse_args()
    try:
        result = assess_output(args.output_root, nonnegative(args.cap_mib, "Output MiB cap") * 1024**2, args.file_cap)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
