from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pandas as pd


RAW_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/echochoho1010/forecasting_fed_rate/master/data"
)


def infer_project_root(start: str | Path | None = None) -> Path:
    current = Path.cwd().resolve() if start is None else Path(start).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "data").exists() and (candidate / "analysis").exists():
            return candidate

    return current


def resolve_data_source(filename: str, project_root: str | Path | None = None) -> str:
    root = infer_project_root(project_root)
    local_path = root / "data" / filename
    if local_path.exists():
        return str(local_path)
    return f"{RAW_DATA_BASE_URL}/{quote(filename)}"


def data_output_path(filename: str, project_root: str | Path | None = None) -> Path:
    root = infer_project_root(project_root)
    output_path = root / "data" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def read_data_csv(
    filename: str,
    *,
    project_root: str | Path | None = None,
    parse_dates=None,
    **kwargs,
) -> tuple[pd.DataFrame, str]:
    source = resolve_data_source(filename, project_root=project_root)
    return pd.read_csv(source, parse_dates=parse_dates, **kwargs), source
