"""Allocated stdlib controller metadata fixtures; never a native replay pass."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC=importlib.util.spec_from_file_location('allocated_replay_ops',Path(__file__).resolve().parents[1]/'scripts/replay_ops.py')
op=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(op)


class ReplayControllerTests(unittest.TestCase):
 def test_exact_root(self):
  self.assertEqual(op.ROOT,op.BASE/'chemistry-replay-01')
 def test_exact_source(self):
  self.assertEqual(op.SOURCE,op.BASE/'chemistry-replay-release-01/source')
 def test_only_three_phases(self):
  for phase in ('production','run','segment','native','resume'):
   with self.assertRaises(ValueError):op.resources(phase)
 def test_gpu_exact_resource_vector(self):
  self.assertEqual(op.resources('gpu'),['-q','gpul40s','-n','8','-R','span[hosts=1]','-R','rusage[mem=4GB]',
   '-gpu','num=1:mode=exclusive_process','-W','00:20'])
 def test_cpu_exact_resource_vector(self):
  self.assertEqual(op.resources('cpu'),['-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:40'])
 def test_callback_exact_resources(self):
  self.assertEqual(op.resources('after-cpu'),['-q','hpc','-n','1','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:20'])
 def test_gpu_independent(self):
  self.assertNotIn('-w',op.argv('gpu','a'*64))
 def test_cpu_independent(self):
  self.assertNotIn('-w',op.argv('cpu','a'*64))
 def test_callback_own_dependency(self):
  args=op.argv('after-cpu','a'*64,'ended(123)');self.assertEqual(args[args.index('-w')+1],'ended(123)')
 def test_callback_dependency_required(self):
  with self.assertRaises(ValueError):op.argv('after-cpu','a'*64)
 def test_dependency_is_not_arb_shell(self):
  for value in ('done(123)','ended(123) || ended(456)','ended(0)','ended(123);other'):
   with self.assertRaises(ValueError):op.argv('after-cpu','a'*64,value)
 def test_no_gpu_interarm_dependency(self):
  with self.assertRaises(ValueError):op.argv('gpu','a'*64,'ended(123)')
 def test_persistent_cap_is_64MiB(self):
  self.assertEqual(op.LIMIT,64*1024**2)
 def test_same_existing_group_cap(self):
  self.assertEqual(op.GROUP_LIMIT,2*1024**3)
 def test_original_guard_exact(self):
  self.assertEqual(op.GUARD_PIN,'09f759a7befb8a7b8f0e1d203e15acf939126fba493b30e6a2bf9cd2eea2216b')
 def test_no_boltz_rerun(self):
  self.assertEqual(op.PILOTS['boltz']['job_id'],'29816184')
 def test_original_failed_Pro_is_bound(self):
  self.assertEqual(op.PILOTS['protenix']['job_id'],'29816284')
 def test_lsf_success_identity(self):
  raw='Subject: Job 123: <exact> in cluster <dcc> Done\nwas executed in queue <hpc>\nSuccessfully completed.'
  op.lsf(raw,'123','exact','hpc',True)
 def test_lsf_failure_preserved(self):
  raw='Subject: Job 123: <exact> in cluster <dcc> Exited\nwas executed in queue <gpul40s>\nExited with exit code 1.'
  op.lsf(raw,'123','exact','gpul40s',False)
 def test_failure_is_not_success(self):
  raw='Subject: Job 123: <exact> in cluster <dcc> Exited\nwas executed in queue <hpc>\nExited with exit code 1.'
  with self.assertRaises(ValueError):op.lsf(raw,'123','exact','hpc',True)
  rows=[{'control':name,'repeat':repeat,'returncode':0,'assessment_sha256':'a'*64,
   'status':'diagnostic_control_completed_original_failure_preserved'}
   for name in ('plain','matrix_import','matrix_installed','combined','native_backend_import') for repeat in (0,1)]
  result={'status':'diagnostics_completed_original_failure_preserved','all_controls_completed':True,
   'original_gate_reclassified':False,'can_certify_or_restart_original_study':False,
   'optimizer_updates':0,'model_evaluations':0,'controls':rows}
  self.assertTrue(op.completed_diagnostic(result))
  for changed in ({'all_controls_completed':False},{'controls':rows[:-1]},
   {'controls':[dict(rows[0],returncode=1),*rows[1:]]},
   {'controls':[rows[0],rows[0],*rows[2:]]},
   {'controls':[dict(rows[0],status='failed'),*rows[1:]]}):
   with self.assertRaises(ValueError):op.completed_diagnostic({**result,**changed})
 def test_lsf_wrong_job_rejected(self):
  raw='Subject: Job 123: <exact> in cluster <dcc> Done\nwas executed in queue <hpc>\nSuccessfully completed.'
  with self.assertRaises(ValueError):op.lsf(raw,'124','exact','hpc',True)
 def test_lsf_wrapped_name(self):
  raw='Subject: Job 123: <ex\n act> in cluster <dcc> Done\nwas executed in queue <hpc>\nSuccessfully completed.'
  op.lsf(raw,'123','exact','hpc',True)
 def test_unknown_marker_refuses_duplicate(self):
  with tempfile.TemporaryDirectory() as directory,mock.patch.object(op,'ROOT',Path(directory)):
   (Path(directory)/'gpu').mkdir()
   with self.assertRaises(FileExistsError):op.submit('gpu','a'*64)
 def test_whole_scan_retry(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'one').write_bytes(b'123')
   original=Path.lstat;failed=[False]
   def race(path,*args,**kwargs):
    if path.name=='one' and not failed[0]:failed[0]=True;raise FileNotFoundError('atomic replacement')
    return original(path,*args,**kwargs)
   with mock.patch.object(Path,'lstat',race):self.assertEqual(op.tree_bytes((root,)),3)
 def test_whole_scan_continuing_race_fails(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'one').write_bytes(b'123');original=Path.lstat
   def race(path,*args,**kwargs):
    if path.name=='one':raise FileNotFoundError('continued replacement')
    return original(path,*args,**kwargs)
   with mock.patch.object(Path,'lstat',race):
    with self.assertRaises(FileNotFoundError):op.tree_bytes((root,))
 def test_no_double_count_nested_roots(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'inner').mkdir();(root/'inner/one').write_bytes(b'123')
   self.assertEqual(op.tree_bytes((root,root/'inner')),3)
 def test_over_own_cap_fails(self):
  with mock.patch.object(op,'tree_bytes',side_effect=[op.LIMIT+1,0]):
   with self.assertRaises(ValueError):op.caps()
 def test_over_group_cap_fails(self):
  with mock.patch.object(op,'tree_bytes',side_effect=[1,op.GROUP_LIMIT]):
   with self.assertRaises(ValueError):op.caps()
  # Actual usage can be below2GiB while the full new64MiB cannot be funded.
  with mock.patch.object(op,'tree_bytes',side_effect=[1,op.GROUP_LIMIT-op.LIMIT+2]):
   with self.assertRaises(ValueError):op.caps()
  with mock.patch.object(op,'tree_bytes',side_effect=[1,op.GROUP_LIMIT-op.LIMIT+1]):
   row=op.caps();self.assertEqual(row['reserved_group_total_bytes'],op.GROUP_LIMIT)
   self.assertEqual(row['reserved_own_remaining_bytes'],op.LIMIT-1)
 def test_source_write_not_allowed(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'source_manifest.json').write_text('{"files":{"a":"a"}}')
   (root/'a').write_bytes(b'a')
   with self.assertRaises(ValueError):op.verify(root,op.sha(root/'source_manifest.json'))
 def test_original_binding_changed_fails(self):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'one';p.write_bytes(b'old');bound=op.binding(p);p.write_bytes(b'new')
   with self.assertRaises(ValueError):op.check(bound)


if __name__=='__main__':unittest.main()
