"""Portable metadata decisions only; actual CUDA/model smoke is allocated."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'scripts/hpc'),str(Path(__file__).resolve().parents[1]/'scripts')]
import gpu_policy as gpu
import v100_migration as migration
from launch_exploratory_continuations import command_owns_output


class Policy(unittest.TestCase):
    def test_v100_is_new_default(self):
        self.assertEqual(gpu.selected({})['queue'],'gpuv100')
        self.assertIn('gmem=30G',gpu.profile()['resource'])
    def test_16gb_v100_rejected(self):
        with self.assertRaises(ValueError):gpu.validate_hardware('Tesla V100',16160)
    def test_32gb_v100_accepted(self):
        self.assertEqual(gpu.validate_hardware('Tesla V100-SXM2-32GB',32510)['profile'],'v100-32gb-v1')
    def test_family_mismatch_rejected(self):
        with self.assertRaises(ValueError):gpu.validate_hardware('NVIDIA L40S',46068)
    def test_explicit_fallback(self):
        self.assertEqual(gpu.selected({'queue':'gpul40s'})['family'],'L40S')
    def test_conflicting_queue_rejected(self):
        with self.assertRaises(ValueError):gpu.selected({'gpu_profile':'v100-32gb-v1','queue':'gpul40s'})
    def test_legacy_style_request_still_selects_32gb(self):
        self.assertIn('gmem=30G',gpu.resource({'gpu_request':'num=1:mode=exclusive_process:mps=yes'}))
        with self.assertRaises(ValueError):gpu.resource({'gpu_request':'num=1:gmem=16G'})
    def test_legacy_standalone_adapter_unchanged(self):
        with patch.dict(os.environ,{},clear=True):self.assertEqual(gpu.legacy_adapter_profile()['family'],'L40S')
        with patch.dict(os.environ,{'MOSAIC_GPU_PROFILE':'v100-32gb-v1'}):self.assertEqual(gpu.legacy_adapter_profile()['family'],'V100')


class Migration(unittest.TestCase):
    def smoke(self):
        return {'status':'passed_device_smoke','memory_headroom_MiB':2000,'summary':{
            'applied_updates':3,'frozen_before':{'model':'same'},'frozen_after':{'model':'same'},
            'update_seconds':[100,1,2],'elapsed_seconds':120}}
    def test_full_remainder_fits(self):
        self.assertTrue(migration.eligible({'steps':10000},self.smoke())['eligible'])
    def test_too_slow_keeps_l40s(self):
        self.assertFalse(migration.eligible({'steps':50000},self.smoke())['eligible'])
    def test_failed_device_keeps_original(self):
        self.assertFalse(migration.eligible({'steps':10},{'status':'failed_device_smoke'})['eligible'])
    def test_low_memory_headroom_keeps_original(self):
        value=self.smoke();value['memory_headroom_MiB']=10
        self.assertFalse(migration.eligible({'steps':10},value)['eligible'])
    def test_wrong_count_not_a_pass(self):
        value=self.smoke();value['summary']['applied_updates']=2
        with self.assertRaises(ValueError):migration.eligible({'steps':10},value)
    def test_frozen_writes_not_a_pass(self):
        value=self.smoke();value['summary']['frozen_after']={'model':'changed'}
        with self.assertRaises(ValueError):migration.eligible({'steps':10},value)


class Handoff(unittest.TestCase):
    def submission(self):return {'job_id':'123','job_name':'mosaic-v100-gpu01'}
    def footer(self):return 'Subject: Job 123: <mosaic-v100-gpu01> in cluster <dcc> Done\nJob <mosaic-v100-gpu01> was submitted\nSuccessfully completed.\n'
    def test_actual_name_header_and_numeric_subject(self):
        self.assertTrue(migration.lsf_completed(self.footer(),self.submission()))
    def test_foreign_or_failed_lsf_not_accepted(self):
        for text in (self.footer().replace('123','124'),self.footer().replace('mosaic-v100-gpu01','other'),self.footer().replace('Successfully completed.','Exited with exit code 1.')):
            self.assertFalse(migration.lsf_completed(text,self.submission()))
    def test_duplicate_subject_rejected(self):
        self.assertFalse(migration.lsf_completed(self.footer()+self.footer(),self.submission()))
    def test_ended_dependency_cli_lag_is_bounded(self):
        row={'JOBID':'123','USER':'s234818','QUEUE':'gpuv100','JOB_NAME':'mosaic-v100-gpu01','STAT':'DONE'}
        with patch.object(migration,'observed_job',side_effect=[{**row,'STAT':'RUN'},row]),patch.object(migration,'write'),patch.object(migration.time,'sleep') as sleep,patch.object(Path,'exists',return_value=True),patch.object(Path,'read_text',return_value=self.footer()):
            migration.wait_comparison(Path('fixture'),self.submission());sleep.assert_called_once_with(5)
    def test_real_terminal_failure_is_not_waited_away(self):
        row={'JOBID':'123','USER':'s234818','QUEUE':'gpuv100','JOB_NAME':'mosaic-v100-gpu01','STAT':'EXIT'}
        with patch.object(migration,'observed_job',return_value=row),patch.object(migration,'write'),patch.object(migration.time,'sleep') as sleep:
            with self.assertRaises(ValueError):migration.wait_comparison(Path('fixture'),self.submission())
            sleep.assert_not_called()


class OutputOwnership(unittest.TestCase):
    def test_neighboring_arm_is_not_a_writer(self):
        for suffix in ('_chemistry','_chemistry-plan.json','-plan.json'):
            self.assertFalse(command_owns_output('bash /production/boltz'+suffix,Path('/production/boltz')))
    def test_exact_directory_and_children_are_writers(self):
        for command in ('run /production/boltz','run --output=/production/boltz','run /production/boltz/payload/study'):
            self.assertTrue(command_owns_output(command,Path('/production/boltz')))


if __name__=='__main__':unittest.main()
