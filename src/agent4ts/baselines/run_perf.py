"""`python -m agent4ts.baselines.run_perf` -> run_perf()."""
import sys
from .runners import run_perf

if __name__ == "__main__":
    sys.exit(run_perf())
