"""Presentation-only entry point; scientific reports use report.py."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] / 'experiments/reporting/jq1-live-progress/scripts/collection.py'), run_name='__main__')
