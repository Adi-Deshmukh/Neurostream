"""Download script for BNCI 2014-001 (BCI Competition IV Dataset 2a).

Usage:
    python scripts/download_dataset.py --subjects 1 2 3
    python scripts/download_dataset.py --all
"""

import argparse
import logging
import os
from pathlib import Path
import sys

# Ensure src/ is on Python path
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BNCI 2014-001 dataset via MOABB")
    parser.add_argument("--subjects", nargs="+", type=int, default=[1], help="Subject IDs to download (1-9)")
    parser.add_argument("--all", action="store_true", help="Download all 9 subjects")
    parser.add_argument("--data-dir", default=str(repo_root / "data" / "mne_data"), help="Target data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(data_dir)
    os.environ["MPLCONFIGDIR"] = str(repo_root / ".cache" / "matplotlib")

    subjects = list(range(1, 10)) if args.all else args.subjects

    logger.info("Destination directory: %s", data_dir)
    logger.info("Target subjects: %s", subjects)

    try:
        from moabb.datasets import BNCI2014_001
        dataset = BNCI2014_001()
    except ImportError as e:
        logger.error("MOABB is required to download datasets: %s", e)
        sys.exit(1)

    successful = []
    failed = []

    for s in subjects:
        logger.info("Downloading data for Subject %d...", s)
        try:
            paths = dataset.data_path(s)
            logger.info("Successfully fetched Subject %d: %s", s, paths)
            successful.append(s)
        except Exception as e:
            logger.error("Failed to download Subject %d: %s", s, e)
            failed.append((s, str(e)))

    logger.info("=== Download Summary ===")
    logger.info("Successfully downloaded: %s", successful)
    if failed:
        logger.warning("Failed downloads: %s", [f[0] for f in failed])
        logger.info(
            "\nNote: If the Graz University server (lampx.tugraz.at) is temporarily unreachable, "
            "NeuroStream automatically uses realistic offline synthetic data with true C3/C4 lateralization."
        )


if __name__ == "__main__":
    main()

