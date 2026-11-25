"""Allocated CPU-only regression tests for isolated V100 checkpoint rebasing."""
import hashlib,importlib.util,json,os,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 release=Path(__file__).resolve().parents[2];root=Path('/work3/s234818/latentmol');output=root/'runs/source148-v100-rebase-test-attempt001/cpu'
 if not os.environ.get('LSB_JOBID') or release!=root/'releases/source-148-v100-rebase-test-attempt001':raise RuntimeError('Allocated isolated test required')
 spec=importlib.util.spec_from_file_location('_operator',release/'scripts/hpc/embedding_operator.py');operator=importlib.util.module_from_spec(spec);spec.loader.exec_module(operator)
 expected=os.environ['LATENTMOL_SOURCE_SHA256'];plan=sha(release/'configs/embedding_v100_handoff_audit.json');operator.authenticate(release,expected,plan,'configs/embedding_v100_handoff_audit.json')
 output.mkdir(parents=True,exist_ok=False);start=time.monotonic();result=dict(job_id=os.environ['LSB_JOBID'],source_sha256=expected,plan_sha256=plan,status='failed_not_passed')
 try:
  with (output/'tests.log').open('xb') as stream:r=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_v100_rebase_core.py','--junitxml='+str(output/'tests.xml')],stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
  suites=list(ET.parse(output/'tests.xml').getroot().iter('testsuite'));counts={k:sum(int(x.get(k,'0')) for x in suites) for k in ('tests','failures','errors','skipped')};result['junit']=counts
  if r.returncode or counts['tests']<1 or any(counts[k] for k in ('failures','errors','skipped')):raise RuntimeError('Rebase regression tests did not pass')
  result.update(status='passed_cpu_rebase_fixtures_only',junit_sha256=sha(output/'tests.xml'))
 except BaseException as error:
  result.update(error=str(error),traceback=traceback.format_exc());(output/'result.json').write_text(json.dumps(result,indent=2)+'\n');raise
 result['elapsed_seconds']=time.monotonic()-start;(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
