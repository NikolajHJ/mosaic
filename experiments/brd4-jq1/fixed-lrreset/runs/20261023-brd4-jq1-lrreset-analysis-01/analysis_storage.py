def check_storage(parent, phase):
    """Original all3remaining projections, analysis allowance and reserves; work3 only."""
    from jq1_lrboost_report_contract import _science_ops
    ops=_science_ops();storage=ops.storage
    require(phase in ('preflight','cpu','capture','interim','final','diagnostic','diagnostic-capture','dispatch'), 'Unknown separate analysis phase')
    require(ROOT.resolve()==ROOT and Path(parent).resolve().is_relative_to(ROOT), 'Analysis receipt escaped its independent root')
    folder=Path(parent)/('storage-'+phase);folder.mkdir(exist_ok=False)
    record={'status':'failed','phase':phase,'utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Separate analysis only; original native outputs unchanged; all3remaining native projections retained',
            'reserve_bytes':4294967296,'file_reserve':100000,'full_native_projections':FULL_NATIVE}
    try:
        current={a:storage.tree_bytes(Path(ARMS[a]['output_parent'])) for a in ARMS}
        raw=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,check=True,timeout=60).stdout
        observed=storage.inode_observation(os.statvfs(ROOT))
        require(set(current)==set(FULL_NATIVE) and all(type(n) is int and 0<=n<=FULL_NATIVE[a] for a,n in current.items()),'Native output exceeded original full projection')
        remaining=sum(FULL_NATIVE[a]-current[a] for a in ARMS)
        gate=storage.gate_record('work3',raw,remaining+storage.CONCURRENT_ANALYSIS_ALLOWANCE,
            storage.stale_work_bytes(),shutil.disk_usage(ROOT).free,observed)
        require(gate['reserve_bytes']==4294967296 and gate['file_reserve']==100000
                and gate['available_files_lower']>=100000,'Mandatory native reserves/file scope differs')
        record.update(status=gate['status'],current_native_bytes=current,outstanding_native_bytes=remaining,
            analysis_allowance_bytes=storage.CONCURRENT_ANALYSIS_ALLOWANCE,work3=gate)
    except Exception as error:
        record['error']=repr(error);raise
    finally:exclusive(folder/'gate.json',record)
    require(record['status']=='passed','Whole current native/analysis byte/inode budget failed')
    return record
