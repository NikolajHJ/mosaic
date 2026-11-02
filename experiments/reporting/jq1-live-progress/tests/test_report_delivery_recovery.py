"""Portable persisted-terminal metadata controls; no arrays/models."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import report_delivery_recovery as recovery


class TerminalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.value = {'state':'SUCCESS_PERSISTED','job_id':'123',
                      'live':{'stdout':'','stderr':'Job <123> is not found\n'},
                      'history':{'returncode':0,'stdout':'Job <123>, User <s234818>\nCompleted <done>\n'}}
        self.lsf = 'Subject: Job 123: fixture\nin queue <hpc>, as user <s234818>\nSuccessfully completed.\n'

    def test_real_persisted_success_does_not_require_fabricated_live_DONE(self):
        recovery.verify_terminal(self.value,self.lsf,'123')
        self.assertEqual(self.value['state'],'SUCCESS_PERSISTED')
        self.assertEqual(self.value['live']['stdout'],'')

    def test_LSF_Done_successfully_history_format(self):
        self.value['history']['stdout']='Job <123>, User <s234818>\nDone successfully. The CPU time used is 517.9 seconds;\nPost job process done successfully;'
        recovery.verify_terminal(self.value,self.lsf,'123')

    def test_wrong_live_DONE_job_rejected(self):
        self.value['live']['stdout']='Job <124>, Status <DONE>'
        with self.assertRaises(ValueError):recovery.verify_terminal(self.value,self.lsf,'123')

    def test_active_failed_or_ambiguous_live_job_rejected(self):
        for state in ('RUN','PEND','EXIT','PSUSP'):
            value=deepcopy(self.value);value['live']['stdout']='Job <123>, Status <'+state+'>'
            with self.subTest(state=state),self.assertRaises(ValueError):
                recovery.verify_terminal(value,self.lsf,'123')

    def test_unavailable_scheduler_is_not_itself_persisted_success(self):
        value=deepcopy(self.value);value['live']['stderr']='connection error'
        with self.assertRaises(ValueError):recovery.verify_terminal(value,self.lsf,'123')

    def test_wrong_job_or_failure_history_rejected(self):
        for text in ('Job <124> Completed <done>','Job <123> Completed <exit>',
                     'Job <123> Exited with exit code 1 Completed <done>'):
            value=deepcopy(self.value);value['history']['stdout']=text
            with self.subTest(text=text),self.assertRaises(ValueError):
                recovery.verify_terminal(value,self.lsf,'123')

    def test_wrong_job_queue_or_user_in_LSF_rejected(self):
        for text in (self.lsf.replace('Job 123:','Job 124:'),self.lsf.replace('<hpc>','<gpul40s>'),
                     self.lsf.replace('<s234818>','<another>')):
            with self.subTest(text=text),self.assertRaises(ValueError):
                recovery.verify_terminal(self.value,text,'123')

    def test_missing_or_duplicate_LSF_success_rejected(self):
        for text in (self.lsf.replace('Successfully completed.',''),self.lsf+'Successfully completed.'):
            with self.assertRaises(ValueError):recovery.verify_terminal(self.value,text,'123')


class ReconstructionBudgetTests(unittest.TestCase):
    def test_large_valid_delta_and_cached_copies_fit_declared_budget(self):
        import report
        self.assertEqual(report.reconstruction_budget(2*1024**3,3*1024**3,8*1024**3,6*1024**3),5*1024**3)

    def test_size_bomb_and_insufficient_local_space_rejected(self):
        import report
        for delta,omitted,cap,free in ((9*1024**3,0,8*1024**3,20*1024**3),
                                      (1,0,8*1024**3,1),
                                      (2*1024**3,3*1024**3,4*1024**3,10*1024**3)):
            with self.subTest(delta=delta,cap=cap),self.assertRaises(ValueError):
                report.reconstruction_budget(delta,omitted,cap,free)

    def test_negative_boolean_or_unbounded_budgets_rejected(self):
        import report
        for values in ((-1,0,1024**3,10*1024**3),(False,0,1024**3,10*1024**3),
                       (1,0,0,10*1024**3),(1,0,9*1024**3,10*1024**3)):
            with self.subTest(values=values),self.assertRaises(ValueError):report.reconstruction_budget(*values)


class CapFailureEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.value={'state':'EXIT_REPORT_CAP_ONLY','job_id':'123',
                    'live':{'stdout':'Job <123>, Status <EXIT>','stderr':''},
                    'history':{'returncode':0,'stdout':'Job <123> Exited with exit code 1.'}}
        self.lsf='Subject: Job 123: original\nin queue <hpc>, as user <s234818>\nExited with exit code 1.'
        self.error='Traceback (most recent call last):\n  original cap guard\nValueError: Report output exceeded its declared cap\n'

    def test_original_EXIT_is_preserved_not_relabelled_DONE(self):
        recovery.verify_cap_terminal(self.value,self.lsf,self.error,'123')
        self.assertEqual(self.value['state'],'EXIT_REPORT_CAP_ONLY')
        self.assertIn('EXIT',self.value['live']['stdout'])

    def test_other_errors_or_multiple_tracebacks_are_rejected(self):
        for error in (self.error.replace('Report output exceeded its declared cap','Nonfinite calculation'),
                      self.error+'ValueError: a later failure\n',self.error+self.error):
            with self.subTest(error=error),self.assertRaises(ValueError):
                recovery.verify_cap_terminal(self.value,self.lsf,error,'123')

    def test_active_successful_wrong_job_or_wrong_failure_is_rejected(self):
        for state in ('RUN','PEND','DONE','PSUSP'):
            value=deepcopy(self.value);value['live']['stdout']='Job <123>, Status <'+state+'>'
            with self.subTest(state=state),self.assertRaises(ValueError):
                recovery.verify_cap_terminal(value,self.lsf,self.error,'123')
        with self.assertRaises(ValueError):recovery.verify_cap_terminal(self.value,self.lsf.replace('123','124'),self.error,'123')
        with self.assertRaises(ValueError):recovery.verify_cap_terminal(self.value,self.lsf.replace('exit code 1','exit code 2'),self.error,'123')

    def test_missing_live_job_still_requires_actual_failed_history(self):
        value=deepcopy(self.value);value['live']={'stdout':'','stderr':'Job <123> is not found'}
        recovery.verify_cap_terminal(value,self.lsf,self.error,'123')
        value['history']['stdout']='Job <123> Done successfully.'
        with self.assertRaises(ValueError):recovery.verify_cap_terminal(value,self.lsf,self.error,'123')


if __name__=='__main__':unittest.main()
