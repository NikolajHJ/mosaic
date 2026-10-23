"""Render offline molecular trajectories from saved results, without model calls.

Example: python scripts/render_optimization_report.py --run RUN --output REPORT
Repeat --run and --label to compare initializations or control/guidance arms.
"""
from pathlib import Path
import sys

# Work both from an immutable source checkout and an installed package. The
# reporting module imports only the standard library, never torch or RDKit.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from latentmol.reporting.optimization import main


if __name__ == "__main__":
    main()
