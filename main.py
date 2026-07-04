"""
main.py

Master pipeline for the Physics-Informed Heat Sink Surrogate Model.

Execution Order:
1. Generate dataset
2. Train surrogate models
3. Evaluate model performance
4. Perform sensitivity analysis
"""

import time
import sys

from src.parameter_sweep import main as generate_dataset
from src.train_models import main as train_models
from src.evaluate_models import main as evaluate_models
from src.sensitivity_analysis import main as sensitivity_analysis


def safe_run(step_name: str, func) -> None:
    """
    Safely execute a pipeline step with error handling.
    
    Parameters
    ----------
    step_name : str
        Name of the pipeline stage
    func : callable
        Function to execute
    """
    try:
        print(f"\n{step_name}")
        func()
    except Exception as e:
        print(f"\n❌ Error in {step_name}")
        print(f"Details: {e}")
        sys.exit(1)


def main() -> None:
    """Run the complete machine learning pipeline."""

    start_time = time.time()

    print("=" * 70)
    print("PHYSICS-INFORMED HEAT SINK SURROGATE MODEL")
    print("=" * 70)

    safe_run("[1/4] Generating dataset...", generate_dataset)
    safe_run("[2/4] Training surrogate models...", train_models)
    safe_run("[3/4] Evaluating models...", evaluate_models)
    safe_run("[4/4] Running sensitivity analysis...", sensitivity_analysis)

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()