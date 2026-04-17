from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "artifacts" / "pipeline_runs"

NOTEBOOKS = [
    "01_process_fed_rate.ipynb",
    "02_process_economic_variables.ipynb",
    "03.1_state_identification.ipynb",
    "03_boosting_models_trained_on_economic_variables.ipynb",
    "03.2_compare_regime_feature_as_input.ipynb",
    "04_construct_calibrated_likelihood_layer.ipynb",
    "05_bayesian_update.ipynb",
]

CATBOOST_ARTIFACT_DEPENDENT_NOTEBOOKS = {
    "03_boosting_models_trained_on_economic_variables.ipynb",
    "03.2_compare_regime_feature_as_input.ipynb",
    "04_construct_calibrated_likelihood_layer.ipynb",
}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def base_environment() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    # Keep runtime caches out of the repository and make repeated runs quieter.
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    return env


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    print(f"Running: {printable}")
    print(f"Working directory: {display_path(cwd)}")
    print(f"Log: {display_path(log_path)}")

    if dry_run:
        log_path.write_text(
            "Dry run only. Command was not executed.\n"
            f"cwd: {cwd}\n"
            f"command: {printable}\n"
        )
        return

    with log_path.open("w") as log_file:
        log_file.write(f"cwd: {cwd}\n")
        log_file.write(f"command: {printable}\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}. "
            f"See {display_path(log_path)}."
        )


def notebook_command(notebook_name: str, executed_dir: Path, timeout: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        notebook_name,
        "--output",
        f"{Path(notebook_name).stem}_executed.ipynb",
        "--output-dir",
        str(executed_dir),
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
    ]


def catboost_artifact_command() -> list[str]:
    return [sys.executable, "scripts/build_catboost_artifacts.py"]


def pca_experiment_command() -> list[str]:
    return [sys.executable, "experiments/pca_tree_optimization/run_experiment.py"]


def validate_notebooks(notebooks: Iterable[str]) -> None:
    missing = [name for name in notebooks if not (ANALYSIS_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing notebooks: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the forecasting Fed rate workflow in a reproducible order. "
            "Executed notebook copies and logs are written under artifacts/pipeline_runs."
        )
    )
    parser.add_argument(
        "--run-id",
        default=timestamp(),
        help="Run folder name under artifacts/pipeline_runs.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Directory that stores pipeline run folders.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Notebook execution timeout in seconds. Use 0 for no timeout.",
    )
    parser.add_argument(
        "--skip-catboost-artifacts",
        action="store_true",
        help="Skip rebuilding CatBoost JSON artifacts before modeling notebooks.",
    )
    parser.add_argument(
        "--skip-pca-experiment",
        action="store_true",
        help="Skip the isolated PCA/tree-optimization experiment after notebooks.",
    )
    parser.add_argument(
        "--only-notebooks",
        nargs="*",
        choices=NOTEBOOKS,
        help="Run only the selected notebooks, in pipeline order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the run manifest and logs without executing commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebooks = args.only_notebooks or NOTEBOOKS
    notebooks = [name for name in NOTEBOOKS if name in set(notebooks)]
    validate_notebooks(notebooks)

    run_root = args.run_root
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    run_dir = run_root / args.run_id
    executed_dir = run_dir / "executed_notebooks"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    executed_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = base_environment()
    manifest = {
        "run_id": args.run_id,
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "dry_run": bool(args.dry_run),
        "notebooks": notebooks,
        "skip_catboost_artifacts": bool(args.skip_catboost_artifacts),
        "skip_pca_experiment": bool(args.skip_pca_experiment),
        "outputs": {
            "run_dir": str(run_dir),
            "executed_notebooks": str(executed_dir),
            "logs": str(log_dir),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Reproducible pipeline run")
    print(f"Run directory: {display_path(run_dir)}")
    print(f"Python: {sys.executable}")
    print()

    catboost_artifacts_built = False
    for notebook in notebooks:
        if (
            notebook in CATBOOST_ARTIFACT_DEPENDENT_NOTEBOOKS
            and not args.skip_catboost_artifacts
            and not catboost_artifacts_built
        ):
            run_command(
                catboost_artifact_command(),
                cwd=PROJECT_ROOT,
                env=env,
                log_path=log_dir / "catboost_artifacts.log",
                dry_run=args.dry_run,
            )
            catboost_artifacts_built = True

        run_command(
            notebook_command(notebook, executed_dir, args.timeout),
            cwd=ANALYSIS_DIR,
            env=env,
            log_path=log_dir / f"{Path(notebook).stem}.log",
            dry_run=args.dry_run,
        )

    if not args.skip_pca_experiment:
        run_command(
            pca_experiment_command(),
            cwd=PROJECT_ROOT,
            env=env,
            log_path=log_dir / "pca_tree_optimization.log",
            dry_run=args.dry_run,
        )

    print()
    print("Pipeline completed successfully.")
    print(f"Manifest: {display_path(run_dir / 'manifest.json')}")
    print(f"Executed notebooks: {display_path(executed_dir)}")
    print(f"Logs: {display_path(log_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
