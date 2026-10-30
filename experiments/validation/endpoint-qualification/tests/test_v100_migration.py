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


if __name__=='__main__':unittest.main()
