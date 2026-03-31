"""
Run the canonical CARMA quanto pipeline end to end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "run_preprocessing.py",
    "run_marginal_carma.py",
    "run_increment_recovery.py",
    "run_levy_fit.py",
    "run_coupling.py",
    "run_pricing_validation.py",
    "run_hedging_backtest.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for script in SCRIPTS:
        path = here / script
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run([sys.executable, "-u", str(path)], check=True)


if __name__ == "__main__":
    main()
