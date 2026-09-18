# run.py
"""
Unified entry point for the project.

Features:
- Always runs from the project root
- Stabilises import paths (IDE / terminal / HPC behave the same)
- Automatically discovers all scripts named run_*.py under scripts/
    => run_mpgc_multi_lane.py becomes --algo=mpgc_multi_lane
- Supports arbitrary subfolder depth inside scripts/
"""

from pathlib import Path
import sys
import os
import argparse
import importlib
import pkgutil


# ============================================================
# 0. Project root directory
# ============================================================
ROOT = Path(__file__).resolve().parent


# ============================================================
# 1. Bootstrap environment
# ============================================================
def bootstrap():
    """
    Prepare a stable runtime environment:
    1) Ensure project root is in sys.path (same behavior as PyCharm)
    2) Force working directory to project root
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Make relative file paths behave consistently
    os.chdir(ROOT)


# ============================================================
# 2. Automatically discover algorithms
# ============================================================
def discover_algorithms(package_name="scripts"):
    """
    Recursively discover all Python files named run_*.py under `scripts/`.

    Example discovered structure:
        scripts/multi_lane/run_fifo.py   -> algo name: fifo
        scripts/multi_lane/run_rm.py     -> algo name: rm
        scripts/exp/run_ablation.py      -> algo name: ablation

    Returns:
        dict[str, str]:
            {
                "fifo": "scripts.multi_lane.run_fifo",
                "rm": "scripts.multi_lane.run_rm",
                "ablation": "scripts.exp.run_ablation",
            }
    """
    pkg = importlib.import_module(package_name)
    algos = {}

    for module in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        module_short_name = module.name.split(".")[-1]

        # Only accept run_*.py files (not packages)
        if module_short_name.startswith("run_") and not module.ispkg:
            algo_name = module_short_name[len("run_"):]  # strip "run_"
            algos[algo_name] = module.name

    if not algos:
        raise RuntimeError("No run_*.py files found under scripts/")

    return dict(sorted(algos.items()))


# ============================================================
# 3. Main entry
# ============================================================
def main():
    bootstrap()

    # Discover available algorithms
    algos = discover_algorithms("scripts")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Unified launcher for all algorithms in scripts/"
    )
    parser.add_argument(
        "--algo",
        required=True,
        choices=algos.keys(),
        help="Algorithm to run (auto-discovered from scripts/**/run_*.py)",
    )

    # Allow extra arguments to be forwarded to the algorithm script
    args, unknown_args = parser.parse_known_args()

    # Dynamically import selected algorithm module
    module_path = algos[args.algo]
    module = importlib.import_module(module_path)

    if not hasattr(module, "main"):
        raise AttributeError(
            f"{module_path} does not define main(args=None, root=None)"
        )

    # Call the algorithm's main function
    # Pass ROOT so all file paths can be resolved robustly
    module.main(args=unknown_args, root=ROOT)


if __name__ == "__main__":
    main()
