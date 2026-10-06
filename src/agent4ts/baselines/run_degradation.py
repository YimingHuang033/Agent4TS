"""`python -m agent4ts.baselines.run_degradation` -> run_degradation()."""
import sys
from .runners import run_degradation

if __name__ == "__main__":
    sys.exit(run_degradation())
