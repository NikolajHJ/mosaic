"""Shared metadata-only capacity policy: live quota once, remaining writes once."""
import csv
from decimal import Decimal, ROUND_CEILING
import math

GIB=1024**3
VERSION="mosaic-live-quota-remaining-writes-v2"
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

def assess(quota,disk_bytes,inodes,scopes,external,unprofiled_writer_ids=()):
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
