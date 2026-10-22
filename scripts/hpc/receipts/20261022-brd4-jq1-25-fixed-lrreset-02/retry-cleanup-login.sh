set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,shutil,os
control=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-02/cleanup-recovery-02')
require=lambda value:None if value else (_ for _ in ()).throw(ValueError('Cleanup recovery identity/prerequisite failed'))
require(control.resolve()==control and shutil.which('bjobs') and shutil.which('getquota_work3.sh'))
require((control/'attempt-01').is_dir() and not any((control/'attempt-01'/name).exists() for name in ('deletion-journal.jsonl','verified-candidates.json','receipt.json')))
for name,expected in [('cleanup-plan.json','03bbc4fa67a286b8f51b6c02f3af8b898e0ed4acd80255e9f88527f82787c8bb'),('cleanup_verified_work3.py','7f3d98cd747b114f298eea53b40514a76aa738dcd2ead807165d93001d005f52')]:
 require(hashlib.sha256((control/name).read_bytes()).hexdigest()==expected)
with (control/'attempt-02-operator-started.json').open('x') as f:
 json.dump({'utc':datetime.now(timezone.utc).isoformat(),'scope':'Same exact immutable cleanup plan/helper in login environment; attempt01 failed before verification/deletion because scheduler PATH absent','bjobs':shutil.which('bjobs'),'original_attempt01_journal_absent':True},f,indent=2);f.flush();os.fsync(f.fileno())
subprocess.run(['/work3/s234818/mosaic/envs/protenix-tiny/bin/python',str(control/'cleanup_verified_work3.py'),'--plan',str(control/'cleanup-plan.json'),'--plan-sha256','03bbc4fa67a286b8f51b6c02f3af8b898e0ed4acd80255e9f88527f82787c8bb','--attempt','attempt-02'],check=True)
PY
