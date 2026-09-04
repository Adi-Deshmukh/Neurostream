"""Run the full subject/seed cross-session sweep for NeuroStream.

Entrypoint for reproducing all paper experiments across 9 subjects × 5 seeds.
Includes a `--quick` flag for rapid development smoke-testing (1 subject × 1 seed).
"""

import argparse
import logging
from pathlib import Path

from neurostream.evaluation.cross_session_eval import run_cross_session_sweep
from neurostream.evaluation.metrics import format_metrics_table

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroStream Full Cross-Subject Sweep")
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test: 1 subject × 1 seed, 5 epochs")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 999])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.70, help="Frozen adaptation momentum across subjects (alpha=0.30)")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--enable-consolidation", action="store_true", default=True)
    parser.add_argument("--no-consolidation", action="store_false", dest="enable_consolidation")
    parser.add_argument("--output-csv", default="results/logs/cross_session_sweep.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.quick:
        print("=== RUNNING IN QUICK SMOKE-TEST MODE (1 subject × 1 seed, 5 epochs) ===")
        subjects = [1]
        seeds = [42]
        epochs = 5
        output_csv = "results/logs/cross_session_sweep_quick.csv"
    else:
        subjects = args.subjects
        seeds = args.seeds
        epochs = args.epochs
        output_csv = args.output_csv

    sweep = run_cross_session_sweep(
        subjects=subjects,
        seeds=seeds,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        confidence_threshold=args.confidence_threshold,
        enable_consolidation=args.enable_consolidation,
        output_csv=output_csv,
    )

    metrics_list = sweep.to_evaluation_metrics()
    table = format_metrics_table(metrics_list, title="Table 5 — Multi-Subject Cross-Session Results")
    print("\n" + table)

    print(f"\nTotal Runs Completed: {len(sweep.runs)}")
    print(f"Mean Target Post-Accuracy: {sweep.mean_post_acc:.2%} (+/- {sweep.std_post_acc:.2%})")
    print(f"Mean Adaptation Gain:      {sweep.mean_gain:+.2%}")
    print(f"Mean Session-1 Retention:  {sweep.mean_retention:.2%}")
    print(f"Mean Neuromorphic Sparsity:{sweep.mean_sparsity:.2%}")
    print(f"Results CSV saved to:      {output_csv}")


if __name__ == "__main__":
    main()
