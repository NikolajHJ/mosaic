"""Prepare a metadata-only uploader/bootstrap script; does not connect or submit."""
import ast
from pathlib import Path
here=Path(__file__).resolve().parent
old=(here.parent/'20261021-brd4-jq1-25-fixed-lrboost-01/bootstrap_hpc.py').read_text()
text=old.replace('20261021-brd4-jq1-25-fixed-lrboost-01',here.name).replace('lrboost','lrreset')
text=text.replace("ROOT = '/work3/s234818/mosaic/runs/' + TAG", "ROOT = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-01'")
text=text.replace("UPLOAD = '/work3/s234818/mosaic/' + TAG + '.tar.gz'", "UPLOAD = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/' + TAG + '.tar.gz'")
text=text.replace('==611','==623').replace('==610','==622').replace("'files':610", "'files':622")
ast.parse(text)
with (here/'bootstrap_hpc.py').open('xb') as f:f.write(text.encode())
print('Prepared exclusive single-attempt bootstrap for622source files; no execution.')
