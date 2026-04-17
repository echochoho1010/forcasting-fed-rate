from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling.catboost_artifacts import (
    build_03_2_artifact,
    build_03_boosting_artifact,
    build_04_artifact,
)


def main() -> None:
    project_root = PROJECT_ROOT
    print(f"Project root: {project_root}")

    print("Building CatBoost artifacts for 03 …")
    build_03_boosting_artifact(project_root)

    print("Building CatBoost artifacts for 03.2 …")
    build_03_2_artifact(project_root)

    print("Building CatBoost artifacts for 04 …")
    build_04_artifact(project_root)

    print("Done.")


if __name__ == "__main__":
    main()
