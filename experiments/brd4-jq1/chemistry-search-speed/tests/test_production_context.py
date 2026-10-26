"""Allocated metadata-only compiler transport fixtures; no native pass."""
from pathlib import Path
import importlib.util
import unittest

SPEC = importlib.util.spec_from_file_location('production_context_metadata',
    Path(__file__).resolve().parents[1]/'scripts/production_context.py')
context = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(context)


def record(arm='protenix', overrides=None):
    return {'arm': arm, 'environment': {'XLA_FLAGS': '--xla_gpu_autotune_level=0'} if overrides is None else overrides,
            'scientific_settings_changed': False}


class ProductionContextTests(unittest.TestCase):
    def test_pro_environment_is_explicit_and_original_is_unchanged(self):
        original = {'JAX_ENABLE_X64': '0'}
        self.assertEqual(context.apply(original, record()),
                         {**original, 'XLA_FLAGS': '--xla_gpu_autotune_level=0'})
        self.assertEqual(original, {'JAX_ENABLE_X64': '0'})

    def test_boltz_receives_no_protenix_override(self):
        value = context.read_context(None, 'boltz', {})
        self.assertEqual(context.apply({'JAX_PLATFORMS': 'cpu'}, value), {'JAX_PLATFORMS': 'cpu'})
        with self.assertRaises(ValueError): context.apply({}, record('boltz'))

    def test_loss_rate_precision_or_unknown_overrides_are_rejected(self):
        for overrides in ({}, {'LR': '.03'}, {'JAX_ENABLE_X64': '1'}, {'TEMPERATURE': '.5'}, {'XLA_FLAGS': '--different'}):
            with self.assertRaises(ValueError): context.apply({}, record(overrides=overrides))

    def test_ambient_unvalidated_compiler_settings_are_rejected(self):
        with self.assertRaises(ValueError): context.apply({'XLA_FLAGS': '--unvalidated=true'}, record())
        self.assertEqual(context.apply({'XLA_FLAGS': '--xla_gpu_autotune_level=0'}, record()),
                         {'XLA_FLAGS': '--xla_gpu_autotune_level=0'})

    def test_unknown_arm_or_scientific_change_claim_is_rejected(self):
        with self.assertRaises(ValueError): context.apply({}, record('other'))
        value = record(); value['scientific_settings_changed'] = True
        with self.assertRaises(ValueError): context.apply({}, value)

    def test_read_context_rejects_unknown_arm_before_any_input_is_opened(self):
        class Fake:
            @staticmethod
            def require(value, message):
                if not value: raise ValueError(message)
        with self.assertRaises(ValueError): context.read_context(Fake(), 'dbmol', {})
