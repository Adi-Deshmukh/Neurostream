"""Top-level entrypoint for running the NeuroStream dashboard.

Enables:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure src/ is on python path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Run dashboard
from neurostream.dashboard.app import *

