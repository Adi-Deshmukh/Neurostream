"""Top-level entrypoint for running the NeuroStream dashboard.

Enables:
    streamlit run dashboard/app.py

Important: execute the dashboard source on every Streamlit rerun. Importing the
module normally would use Python's sys.modules cache, so widget interactions
would rerun this entrypoint without rerunning the page-rendering code.
"""

import runpy
import sys
from pathlib import Path

# Ensure src/ is on python path.
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# run_path executes the dashboard afresh on every Streamlit script rerun.
runpy.run_path(str(src_path / "neurostream" / "dashboard" / "app.py"))
