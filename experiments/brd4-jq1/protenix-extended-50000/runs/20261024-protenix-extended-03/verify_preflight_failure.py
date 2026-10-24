"""Independent stdlib audit of the actual Pro03 pre-CPU capacity rejection."""
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
import re
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT=BASE+'/protenix-extended-03'
MANIFEST='da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562'
ARCHIVE='bde714a20a49308365744ab5018c06a6d68dff82c3485dc075848827a5f7a8ef'
GIB=1024**3


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())


def main():
    output=HERE/'preflight-failure-verification-01.json'
    require(not output.exists(),'Prior audit must be preserved')
    folder=REPO/'results/hpc/20261024-protenix-extended-03/failed-preflight-evidence-01'
    stdout=(folder/'stdout.bin').read_bytes();stderr=(folder/'stderr.bin').read_bytes();delivery=read(folder/'result.json')
    require(delivery['returncode']==0 and delivery['read_only'] is True
            and delivery['stdout_sha256']==sha(stdout) and delivery['stderr_sha256']==sha(stderr),
            'Read-only failure evidence did not complete authentically')
    value=json.loads(stdout)
    require(value['read_only'] is True and value['root']==ROOT and value['jobs_exists'] is False
            and value['source_manifest_sha256']==MANIFEST,'Actual early-staging/no-jobs scope differs')
    started=ROOT+'/staging-01/started.json';gate_path=ROOT+'/staging-01/storage-preflight/gate.json'
    require(set(value['files'])=={started,gate_path},'Unexpected scientific/job output in failed staging root')
    decoded={}
    for path,item in value['files'].items():
        raw=item['text'].encode();require(len(raw)==item['bytes'] and sha(raw)==item['sha256'],'Delivered raw file changed')
        decoded[path]=json.loads(raw)
    archive=REPO/'results/hpc/20261024-protenix-extended-03/source01/source.tar.gz'
    require(archive.stat().st_size==39644558 and sha(archive.read_bytes())==ARCHIVE,'Sealed package changed')
    files={}
    with tarfile.open(archive) as tar:
        for member in tar:
            require(member.isfile() and member.name.startswith('source/') and '..' not in Path(member.name).parts
                    and member.name not in files and not member.mode&0o222,'Unsafe/duplicate source member')
            files[member.name.removeprefix('source/')]=tar.extractfile(member).read()
    raw=files.pop('source_manifest.json');manifest=json.loads(raw)
    require(sha(raw)==MANIFEST and len(files)==643 and set(files)==set(manifest['files'])
            and all(sha(data)==manifest['files'][name] for name,data in files.items()),'Full source member hashes differ')
    attempt=decoded[started]
    require(attempt['source_manifest_sha256']==MANIFEST
            and attempt['old_source_manifest_sha256']=='a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00'
            and attempt['continuation_snapshot_sha256']=='edf15f06ba1b6591d62cbc3e04e19ecd93b553a0627da38c370fdeed3c1b780f'
            and len(attempt['old_failure_proofs'])==12,'Original failure/continuation provenance changed')
    old=read(REPO/'results/hpc/20261024-boltz-startup-recovery-02/peer-projection-inspection/stdout.bin')
    prefix=BASE+'/protenix-extended-01/'
    pins={path.removeprefix(prefix):{'bytes':item['bytes'],'sha256':item['sha256']} for path,item in old['files'].items()}
    require(attempt['old_failure_proofs']==pins,'Old pre-model failure proofs changed')
    gate=decoded[gate_path];work=gate['work3']
    require(gate['status']==work['status']=='failed' and gate['phase']=='preflight'
            and gate['reserve_bytes']==work['reserve_bytes']==4*GIB
            and gate['file_reserve']==work['file_reserve']==100000
            and gate['full_long_admission'] is False and gate['missing_full_projections']==[],
            'Failed admission was mislabeled or weakened')
    require(gate['own_current']=={'random':0,'continuation':0}
            and gate['full_extended']=={'random':GIB,'continuation':GIB},'Validation allowance is not a full Pro projection')
    original_full={'protenix':1493521824,'boltz2':1748021824,'boltz2_chemistry':5144021824}
    original_remaining=sum(original_full[arm]-gate['old_current'][arm] for arm in original_full)
    require(all(0<=gate['old_current'][arm]<=original_full[arm] for arm in original_full),'Original remaining projection changed')
    peers=gate['peers']
    require(set(peers)=={'baseline','dbmol'} and peers['baseline']['source_manifest_sha256']=='c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6'
            and peers['dbmol']['source_manifest_sha256']=='600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed',
            'Current peer source scope changed')
    require(all(type(p['full']) is int and 0<=p['current']<=p['full'] and p['remaining']==p['full']-p['current'] for p in peers.values()),'Peer full/remaining accounting differs')
    controls=gate['prospective_adoption_metadata_controls']
    require(controls['per_control_bytes']==GIB and controls['current']=={BASE+'/boltz-pro02-adoption-04':0,BASE+'/dbmol-pro02-adoption-06':0}
            and controls['remaining_bytes']==2*GIB,'Independent metadata allowance changed')
    inventory=gate['noisy_capsule_inventory']
    require(inventory['unknown_scopes']==[] and inventory['stale_bytes']==sum(item['bytes'] for item in inventory['capsules'])
            and all(item['scope']=='authenticated_noisy01' for item in inventory['capsules']),
            'Unknown noisy capsule was admitted or stale bytes omitted')
    noisy=gate['noisy_protenix']
    require(noisy['full']==GIB and 0<=noisy['current']<=GIB and noisy['remaining']==GIB-noisy['current']
            and noisy['scope']=='authenticated_immutable_one_GiB_cap','Known noisy bound changed')
    projected=original_remaining+2*GIB+sum(p['remaining'] for p in peers.values())+2*GIB+noisy['remaining']+2*GIB+512*1024**2
    require(gate['projected_bytes']==work['projected_bytes']==projected==37348701956,'Full prospective projection arithmetic differs')
    require(work['required_bytes']==projected+4*GIB==41643669252,'Reserve omitted or requirement changed')
    pools=work['quota']['finite_pools'];require(len(pools)==1,'Finite quota pool scope changed')
    rows=re.findall(r's234818\|306121\|\|\s*([0-9.]+)\s+GiB\|\s*([0-9.]+)\s+GiB\|\|\s*(\d+)\|\s*(\d+)',work['raw_quota'])
    require(len(rows)==1,'Raw finite quota row missing/ambiguous')
    used,hard,used_files,hard_files=rows[0]
    used_upper=int(Decimal(used).to_integral_value(rounding=ROUND_CEILING))*GIB
    hard_lower=int(Decimal(hard).to_integral_value(rounding=ROUND_FLOOR))*GIB
    require(pools[0]=={'used_bytes_upper':used_upper,'hard_bytes_lower':hard_lower,'remaining_bytes':hard_lower-used_upper,
                       'remaining_files':int(hard_files)-int(used_files)},'Conservative raw quota rounding differs')
    require(work['quota']['available_bytes_lower']==hard_lower-used_upper
            and work['available_bytes_conservative']==min(hard_lower-used_upper-work['stale_bytes_subtracted'],work['disk_free_bytes'])==27472908456,
            'Conservative available byte accounting differs')
    require(work['available_files_lower']==work['quota']['available_files_lower']==int(hard_files)-int(used_files)>=100000
            and work['filesystem_inode_observation']=={'f_files':0,'f_ffree':0,'f_favail':0}
            and work['filesystem_inode_reporting']=='unreported_all_zero','File headroom scope changed or zero FS counters overclaimed')
    shortfall=work['required_bytes']-work['available_bytes_conservative'];require(shortfall>0,'Original storage failure erased')
    result={'status':'verified_pre_cpu_storage_failure','verified_utc':datetime.now(timezone.utc).isoformat(),
            'source_manifest_sha256':MANIFEST,'source_archive_sha256':ARCHIVE,'local_source_members_verified':643,
            'actual_remote_manifest_matched':True,'remote_full_member_rehash_in_this_two_file_bundle':False,
            'jobs_file_present':False,'cpu_submitted':False,'gpu_startup_passed':False,'native_passed':False,'optimization_started':False,
            'gate_sha256':value['files'][gate_path]['sha256'],'delivery_stdout_sha256':sha(stdout),
            'required_bytes':work['required_bytes'],'available_bytes_conservative':work['available_bytes_conservative'],
            'shortfall_bytes':shortfall,'reserve_bytes':4*GIB,'file_reserve':100000,
            'peer_full_projections_retained':True,'new_pro_validation_allowance_not_full_projection':True,
            'original_failures_and_saved_results_preserved':True,'no_local_scientific_execution':True,
            'scope':'Independent raw-byte/JSON/schema/integer quota-budget audit only; staged source and failed capacity admission do not establish scientific readiness'}
    with output.open('x',encoding='utf8',newline='\n') as handle:json.dump(result,handle,indent=2,sort_keys=True);handle.write('\n')
    campaign={'status':'pre_cpu_storage_blocked','package':read(HERE/'package.json'),'root':ROOT,
              'source':BASE+'/protenix-extended-release-03/source','jobs':{},'jobs_file_present':False,
              'attempt':'recovery-attempt-02','gate':gate,'failure_verification':{'path':str(output),'sha256':sha(output.read_bytes())},
              'scientific_settings_and_saved_results_changed':False,'cpu_passed':False,'native_passed':False,'optimization_started':False}
    with (HERE/'campaign.json').open('x',encoding='utf8',newline='\n') as handle:json.dump(campaign,handle,indent=2,sort_keys=True);handle.write('\n')
    record=("Pro03 source643 was staged read-only, then its mandatory preflight rejected capacity before jobs.json/CPU creation. "
            f"Required {work['required_bytes']} bytes; conservative available {work['available_bytes_conservative']}; shortfall {shortfall}. "
            "All full peer projections, 2GiB Pro validation allowance, 2GiB adoption controls and 4GiB/100000-file reserves remain unchanged. "
            "No startup/CPU/native/optimization pass exists. This is a resource-admission failure, not a scientific failure. "
            "Original saved results and failed predecessors remain unchanged.\n")
    with (HERE/'run-record.md').open('x',encoding='utf8',newline='\n') as handle:handle.write(record)
    print(json.dumps({'status':result['status'],'receipt':str(output),'sha256':sha(output.read_bytes()),'shortfall_bytes':shortfall}))


if __name__=='__main__':main()
