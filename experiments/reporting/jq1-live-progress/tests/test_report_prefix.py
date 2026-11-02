"""Portable branch-selection metadata tests; array collection stays on HPC."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import report_prefix as prefix


class PrefixMetadataTests(unittest.TestCase):
    def fixture(self, root):
        states=root/'states.jsonl'
        states.write_text(''.join(json.dumps({'step':i,'global_step':10+i,
                                            'snapshot_retained':i%2==0})+'\n' for i in range(5)))
        return {'global_offset':10,'states':{'path':str(states)},
                'expected_counts':{'committed_updates':4,'scalar_states':5,'retained_frames':3}}

    def test_original_segment_is_unchanged_and_cut_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            row=self.fixture(Path(d));selected=prefix.capture(row,{'through_global_update':12})
            self.assertEqual(row['expected_counts']['committed_updates'],4)
            self.assertEqual(selected['expected_counts'],{'committed_updates':2,'scalar_states':3,'retained_frames':2})
            self.assertEqual(selected['source_expected_counts'],row['expected_counts'])
            self.assertEqual(selected['report_stop_global'],12)

    def test_no_cut_keeps_existing_behavior(self):
        row={'ordinary':'unchanged'}
        self.assertIs(prefix.capture(row,{}),row)

    def test_outside_empty_boolean_or_unsaved_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            row=self.fixture(Path(d))
            for stop in (10,15,False,13):
                with self.subTest(stop=stop),self.assertRaises(ValueError):
                    prefix.capture(row,{'through_global_update':stop})


if __name__=='__main__':unittest.main()
