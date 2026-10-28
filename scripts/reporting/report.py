"""Stable short entry point; all implementation remains with its canonical owner."""
from pathlib import Path
import runpy
import sys

OWNER_SCRIPTS = Path(__file__).resolve().parents[2] / 'experiments/reporting/jq1-live-progress/scripts'
# The canonical CLI owns its sibling metadata/runtime modules. This adapter has
# no scientific implementation and does not import a numerical library.
sys.path.insert(0, str(OWNER_SCRIPTS))

if __name__ == '__main__':
    runpy.run_path(str(OWNER_SCRIPTS / 'report.py'), run_name='__main__')
