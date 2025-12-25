import sys
import os
from pathlib import Path

os.environ.setdefault("ALLOW_MISSING_ARTIFACTS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
