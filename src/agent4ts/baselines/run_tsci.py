"""`python -m agent4ts.baselines.run_tsci` -> run_tsci()."""
import sys
from .runners import run_tsci

if __name__ == "__main__":
    sys.exit(run_tsci())
