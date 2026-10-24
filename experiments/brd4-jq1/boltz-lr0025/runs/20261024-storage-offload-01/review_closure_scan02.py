"""Independent stdlib review of captured scan02; proposes backups, never deletes."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
RUNS='/work3/s234818/latentmol/runs'
NAMED={'source120a004-boltz-defog-staged-24h-attempt004',
 'source-073-decoder-budget-attempt002-recovery','source-071-search-decoder-attempt001-search-study',
 'source-078-dense-100epoch-attempt002-density100-study','source-079-defog-contract-recovery-attempt001-random-study'}

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(data):return hashlib.sha256(data).hexdigest()
def rootof(path):
 rel=PurePosixPath(path).relative_to(PurePosixPath(RUNS));return RUNS+'/'+rel.parts[0]
def main():
 out=HERE/'latentmol-protection-refinement-02.json';require(not out.exists(),'Preserve prior review')
 folder=REPO/'results/hpc/20261024-storage-offload-01/shared-closure-scan-02'
 raw=(folder/'stdout.bin').read_bytes();v=json.loads(raw);result=json.loads((folder/'result.json').read_bytes())
 scanner=HERE/'scan_active_closure_02.py'
 require(result['returncode']==0 and result['script_sha256']==sha(scanner.read_bytes())=='740f5a9cb2f3b44710fcf35d0127e2da4f8bf50f0809d4cb654abc2bb089c627', 'Actual scanner execution identity differs')
 require(v['read_only'] is True and v['closed_without_true_runtime_errors'] is True
         and v['unresolved']==[] and v['queue']['returncode']==0
         and v['positive_inactivity_certified'] is False and v['delete_allowlist']==[], 'Incomplete closure cannot refine protection')
 require(sorted(x['execution_members_rehashed'] for x in v['source_seals'].values())==[577,582]
         and all(x['all_execution_members_rehashed'] for x in v['source_seals'].values())
         and len(v['runtime_code'])==152 and len(v['runtime_inputs'])==13 and len(v['hashed_proof_containers'])==45,
         'Actual source/runtime/metadata scope differs')
 require(len(v['review_items'])==3 and all(x['value']=='/work3/s234818/latentmol'
         and ('container' in x['reason'] or 'Workspace' in x['reason']) for x in v['review_items']),
         'Unresolved concrete dependency was mislabeled a workspace container')
 # Candidate absolute paths in reachable source modules are a hard exclusion.
 runtime_hits=[]
 for path,rec in v['runtime_code'].items():
  for row in rec['literal_paths']:
   value=row['path']
   if value.startswith(RUNS+'/'):
    base=PurePosixPath(value).relative_to(PurePosixPath(RUNS)).parts[0]
    if base.startswith('202609') or base in NAMED:runtime_hits.append({'owner':path,**row})
 require(not runtime_hits,'Reachable code references proposed retired scope')
 protected_trees={PurePosixPath(x) for x in v['protected_trees']};protected_files={PurePosixPath(x) for x in v['protected_files']}
 def covered(path):
  p=PurePosixPath(path)
  return p in protected_files or any(p==d or d in p.parents for d in protected_trees)
 aliases=defaultdict(dict)
 for root,row in v['top_run_inventory'].items():
  for item in row.get('hardlinks',[]):
   s=item['lstat'];aliases[(s['dev'],s['ino'])][item['path']]=item
 require(len(v['top_run_inventory'])==624,'Unexpected top-run inventory scope')
 groups=list(aliases.values())
 require(all(len(items)==next(iter(items.values()))['lstat']['nlink'] for items in groups),
         'Observed alias inventory does not close all recorded hardlink groups')
 candidates={}
 for row in v['candidate_checkpoints']:
  p=row['path'];root=rootof(p);parent=v['top_run_inventory'][root];base=PurePosixPath(root).name
  require(base.startswith('202609') or base in NAMED,'Candidate escapes explicit historical scope')
  require(p not in candidates,'Duplicate candidate path')
  if row['protected'] or covered(p) or not parent.get('root_unchanged') or parent.get('symlinks_or_specials'):continue
  require(row['lstat']['size']>=0 and row['lstat']['nlink']>=1,'Malformed inode metadata')
  candidates[p]=row
 singles=[row for row in candidates.values() if row['lstat']['nlink']==1]
 closed=[];outside=[]
 for key,items in sorted(aliases.items()):
  if not any(p in candidates for p in items):continue
  rows=list(items.values());sample=rows[0]['lstat']
  require(all(x['lstat']==sample for x in rows),'Same inode has inconsistent stat snapshot')
  rec={'device':key[0],'inode':key[1],'nlink':sample['nlink'],
       'unique_bytes':sample['size'],'unique_blocks512':sample['blocks512'],
       'aliases':rows,'fresh_alias_audit_required':True,'eligible_for_deletion':False}
  (closed if set(items)<=set(candidates) else outside).append(rec)
 require(len(singles)==1881 and len(closed)==113 and len(outside)==4,'Reviewed alias scope changed')
 full=[]
 for name in sorted(NAMED):
  root=RUNS+'/'+name;row=v['top_run_inventory'][root]
  require(row['candidate_scope'] is True and row['root_unchanged'] is True
          and not row.get('protected') and not row['hardlinks'] and not row['symlinks_or_specials']
          and not covered(root),'Named full-artifact root intersects protected inputs/links/race')
  retained=v['candidate_metadata'][root]['root_metadata']
  reason=('No selected active VAE/embedding plan, concrete runtime metadata input, reachable-module absolute literal, '
          'current all-job input/output namespace, protected tree/file or symlink/hardlink dependency references this exact historical root. '
          'Its inactive old outputs may be proposed for a complete verified backup; root metadata listed below stays remote.')
  full.append({'root':root,'reason':reason,'inventory':row,'preserve_remote_root_metadata':retained,
               'full_file_manifest_still_required':True,'backup_required_before_deletion':True,
               'no_source_env_data_cache_or_current_run_selected':True,'eligible_for_deletion':False})
 single_bytes=sum(x['lstat']['size'] for x in singles)
 group_bytes=sum(x['unique_bytes'] for x in closed)
 named_full=sum(x['inventory']['logical_bytes'] for x in full)
 named_binary=sum(x['lstat']['size'] for p,x in candidates.items() if PurePosixPath(rootof(p)).name in NAMED)
 retain_bytes=sum(x['lstat']['size'] for row in full for x in row['preserve_remote_root_metadata'])
 potential=single_bytes+group_bytes+named_full-named_binary-retain_bytes
 require((single_bytes,group_bytes,named_full,named_binary,retain_bytes,potential)==
         (6537719514,5603070529,7326572090,1419881415,116747959,17930732759),
         'Deduplicated proposal arithmetic changed')
 reviewed={'version':'latentmol-positive-current-dependency-refinement-v1',
  'status':'explicit_inactive_output_backup_proposal_parent_final_plan_required',
  'verified_utc':datetime.now(timezone.utc).isoformat(),
  'capture_utc':v['utc'],'captured_stdout_sha256':sha(raw),'captured_stdout_bytes':len(raw),
  'scanner_sha256':sha(scanner.read_bytes()),'scanner_runtime_errors':[],
  'source_execution_members_rehashed':[577,582],'reachable_runtime_members_inspected':152,
  'runtime_metadata_inputs_hashed':13,'required_historical_proof_containers_hashed':45,
  'historical_embedded_prose_is_not_adopted_as_runtime':True,
  'workspace_review_items_resolved_as_containers':v['review_items'],
  'protected_remote_trees':v['protected_trees'],'protected_exact_remote_files':v['protected_files'],
  'supersedes_blanket_latentmol_protection_only_for':
      'The exact nlink1 paths, fully named eligible inode alias groups, and five named historical output roots listed here, solely for a parent-approved complete-backup/identity-verified offload. All unlisted paths remain protected.',
  'individual_nlink1_binary_backup_candidates':singles,
  'complete_observed_inode_groups_for_fresh_audit':closed,
  'retained_groups_with_outside_scope_aliases':outside,
  'five_named_full_artifact_backup_candidates':full,
  'arithmetic':{'nlink1_paths':len(singles),'nlink1_logical_bytes':single_bytes,
   'complete_candidate_groups':len(closed),'unique_group_logical_bytes':group_bytes,
   'binary_unique_candidate_bytes':single_bytes+group_bytes,'named_full_output_logical_bytes':named_full,
   'named_binary_bytes_already_counted':named_binary,'named_root_metadata_retained_bytes':retain_bytes,
   'combined_unique_logical_proposal_bytes_after_root_metadata_retention':potential,
   'current_known_preflight_shortfall_bytes':14170760796,
   'proposal_margin_over_prior_shortfall_bytes':potential-14170760796,
   'actual_physical_or_quota_savings_established':False},
  'mandatory_before_offload':[
   'Parent authors exact regular-file allowlist and a complete per-file/path/length/SHA256 archive manifest. This review has no delete allowlist.',
   'Fresh readonly inode alias audit across runs and submissions: compare current device/inode/nlink/size/mode/mtime/ctime for every selected path, name every alias, and reject a group unless selected aliases exactly equal current nlink. Unknown/protected/outside aliases remain.',
   'Keep all four outside-scope alias groups, every listed root metadata proof, all current source/env/cache/data/context/catalog/checkpoint/current-run/proof scopes, and all unlisted paths.',
   'Download complete laptop backups, verify every regular file and every alias path with identical content hashes; preserve hardlink relationships in metadata or materialize independently verified bytes. A tar success or archive SHA alone is insufficient.',
   'Immediately before removal recheck source/current-plan hashes, actual all-job namespaces, protected-path intersections, exact file identities and alias count. Changed files or unknown outcomes stay remote.',
   'Remove only parent-approved identical backed-up files, record exact outcome, then rerun fresh quota and unchanged full campaign admission. No promised quota gain, full Pro projection pass or job start follows from this capacity proposal.'],
  'no_ssh':True,'no_local_numerical_execution':True,'no_saved_result_mutation':True,'delete_allowlist':[]}
 with out.open('x',encoding='utf8',newline='\n') as f:json.dump(reviewed,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'path':str(out),'sha256':sha(out.read_bytes()),'unique_logical_backup_proposal':potential,
                   'outside_groups_retained':len(outside),'deletion_authorized_by_this_review':False}))
if __name__=='__main__':main()
