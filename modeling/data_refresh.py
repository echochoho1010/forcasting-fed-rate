from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import subprocess
from urllib.request import Request, urlopen

import pandas as pd

from modeling.notebook_io import data_output_path, infer_project_root, resolve_data_source


FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
NFCI_CSV_URL = "https://www.chicagofed.org/~/media/publications/nfci/nfci-data-series-csv.csv"

FED_RATE_SOURCES = {
    "Federal Funds Target Rate.csv": "DFEDTAR",
    "Federal Funds Target Range Upper Limit.csv": "DFEDTARU",
}

ECONOMIC_SOURCES = {
    "PCEPI Personal Consumption Expenditures.csv": "PCEPI",
    "Unemployment Rate UNRATE all.csv": "UNRATE",
    "Noncyclical Rate of Unemployment.csv": "NROU",
    "Real GDP Data.csv": "GDPC1",
    "Real Potential GDP.csv": "GDPPOT",
}

MONTH_NUMBERS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class RefreshResult:
    filename: str
    rows: int
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    source: str
    error: str | None = None

    def message(self) -> str:
        if self.error:
            return f"{self.filename}: kept local copy ({self.error})"
        if self.start_date is None or self.end_date is None:
            return f"{self.filename}: wrote {self.rows} rows from {self.source}"
        return (
            f"{self.filename}: wrote {self.rows} rows "
            f"({self.start_date.date()} -> {self.end_date.date()})"
        )


def _download_bytes(url: str, timeout: int = 30) -> bytes:
    try:
        completed = subprocess.run(
            ["curl", "-L", "-sS", "--max-time", str(timeout), url],
            check=True,
            capture_output=True,
        )
        if completed.stdout:
            return completed.stdout
    except Exception:
        pass

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; forecasting-fed-rate-data-refresh/1.0)"
            )
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _clean_date_column(df: pd.DataFrame, date_col: str = "observation_date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(date_col)
    df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def _write_csv(df: pd.DataFrame, filename: str, project_root: str | Path | None) -> Path:
    output_path = data_output_path(filename, project_root)
    df.to_csv(output_path, index=False)
    return output_path


def _date_span(df: pd.DataFrame, date_col: str = "observation_date") -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if df.empty or date_col not in df.columns:
        return None, None
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min(), dates.max()


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Download a full FRED CSV series and normalize the date column."""
    url = FRED_CSV_URL.format(series_id=series_id)
    df = pd.read_csv(BytesIO(_download_bytes(url)))
    if "observation_date" not in df.columns or series_id not in df.columns:
        raise ValueError(f"Unexpected FRED schema for {series_id}: {list(df.columns)}")
    return _clean_date_column(df[["observation_date", series_id]])


def refresh_fred_sources(
    sources: dict[str, str],
    project_root: str | Path | None = None,
) -> list[RefreshResult]:
    """Overwrite local FRED-backed CSVs with complete current downloads."""
    results: list[RefreshResult] = []
    for filename, series_id in sources.items():
        try:
            df = fetch_fred_series(series_id)
            _write_csv(df, filename, project_root)
            start, end = _date_span(df)
            results.append(RefreshResult(filename, len(df), start, end, "FRED"))
        except Exception as exc:  # keep notebooks runnable without internet
            results.append(RefreshResult(filename, 0, None, None, "FRED", str(exc)))
    return results


def refresh_nfci_source(project_root: str | Path | None = None) -> RefreshResult:
    """Overwrite the local NFCI CSV with the current Chicago Fed data file."""
    filename = "Chicago Fed NFCI.csv"
    try:
        raw = pd.read_csv(BytesIO(_download_bytes(NFCI_CSV_URL)))
        required = {"Friday_of_Week", "NFCI"}
        if not required.issubset(raw.columns):
            raise ValueError(f"Unexpected NFCI schema: {list(raw.columns)}")
        df = raw[["Friday_of_Week", "NFCI"]].rename(
            columns={"Friday_of_Week": "observation_date"}
        )
        df = _clean_date_column(df)
        _write_csv(df, filename, project_root)
        start, end = _date_span(df)
        return RefreshResult(filename, len(df), start, end, "Chicago Fed")
    except Exception as exc:
        return RefreshResult(filename, 0, None, None, "Chicago Fed", str(exc))


def refresh_fed_rate_sources(project_root: str | Path | None = None) -> list[RefreshResult]:
    return refresh_fred_sources(FED_RATE_SOURCES, project_root)


def refresh_economic_sources(project_root: str | Path | None = None) -> list[RefreshResult]:
    results = refresh_fred_sources(ECONOMIC_SOURCES, project_root)
    results.append(refresh_nfci_source(project_root))
    return results


def _parse_fomc_date(year: int, month_text: str, date_text: str) -> pd.Timestamp | None:
    if "notation vote" in date_text.lower():
        return None

    days = [int(match) for match in re.findall(r"\d+", date_text)]
    if not days:
        return None

    month_parts = [part.strip().lower() for part in month_text.split("/") if part.strip()]
    if not month_parts:
        return None

    # The target rate after a two-day meeting is observed after the final day.
    end_day = days[-1]
    month_key = month_parts[-1]
    month = MONTH_NUMBERS.get(month_key)
    if month is None:
        return None

    return pd.Timestamp(year=year, month=month, day=end_day)


def parse_fomc_calendar_html(html: str | bytes) -> pd.DataFrame:
    """Parse the Federal Reserve FOMC calendar page into meeting end dates."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("BeautifulSoup is required to parse the FOMC calendar page") from exc

    soup = BeautifulSoup(html, "html.parser")
    dates: list[pd.Timestamp] = []

    for panel in soup.select("div.panel.panel-default"):
        heading = panel.select_one(".panel-heading")
        if heading is None:
            continue
        year_match = re.search(r"(\d{4})\s+FOMC Meetings", heading.get_text(" ", strip=True))
        if year_match is None:
            continue
        year = int(year_match.group(1))

        for row in panel.select(".fomc-meeting"):
            month_node = row.select_one(".fomc-meeting__month")
            date_node = row.select_one(".fomc-meeting__date")
            if month_node is None or date_node is None:
                continue
            parsed = _parse_fomc_date(
                year,
                month_node.get_text(" ", strip=True),
                date_node.get_text(" ", strip=True),
            )
            if parsed is not None:
                dates.append(parsed)

    if not dates:
        raise ValueError("No FOMC meeting dates found on the calendar page")

    return (
        pd.DataFrame({"meeting_date": dates})
        .drop_duplicates()
        .sort_values("meeting_date")
        .reset_index(drop=True)
    )


def fetch_fomc_calendar() -> pd.DataFrame:
    return parse_fomc_calendar_html(_download_bytes(FOMC_CALENDAR_URL))


def _existing_meeting_dates(project_root: str | Path | None = None) -> pd.DataFrame:
    try:
        source = resolve_data_source("processed_fed_meetings.csv", project_root)
        df = pd.read_csv(source, parse_dates=["meeting_date"])
        return df[["meeting_date"]].dropna()
    except Exception:
        return pd.DataFrame({"meeting_date": pd.Series(dtype="datetime64[ns]")})


def load_fomc_meeting_dates(project_root: str | Path | None = None) -> list[str]:
    """Return the local historical calendar plus any dates currently listed by the Fed."""
    existing = _existing_meeting_dates(project_root)
    try:
        fetched = fetch_fomc_calendar()
    except Exception:
        fetched = pd.DataFrame({"meeting_date": pd.Series(dtype="datetime64[ns]")})

    if not fetched.empty:
        first_fetched_year = fetched["meeting_date"].dt.year.min()
        existing = existing[existing["meeting_date"].dt.year < first_fetched_year]

    meetings = (
        pd.concat([existing, fetched], ignore_index=True)
        .dropna(subset=["meeting_date"])
        .drop_duplicates("meeting_date")
        .sort_values("meeting_date")
        .reset_index(drop=True)
    )
    return meetings["meeting_date"].dt.strftime("%Y-%m-%d").tolist()


def _minmax_normalize(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    lo = series.expanding(min_periods=1).min()
    hi = series.expanding(min_periods=1).max()
    span = hi - lo
    scaled = (series - lo) / span
    scaled = scaled.where(span.ne(0), 0.0)
    return scaled.where(series.notna())


def _action_label(diff_val: float | None) -> str | None:
    if pd.isna(diff_val):
        return None
    if diff_val > 0:
        return "higher"
    if diff_val < 0:
        return "lower"
    return "same"


def build_processed_fed_meetings(
    meeting_dates: pd.Series | list[str],
    target_upper: pd.DataFrame,
) -> pd.DataFrame:
    """Build the clean FOMC meeting-level target-rate file."""
    meetings = pd.DataFrame({"meeting_date": pd.to_datetime(meeting_dates, errors="coerce")})
    meetings = meetings.dropna().drop_duplicates().sort_values("meeting_date").reset_index(drop=True)
    meetings["post_meeting_date"] = meetings["meeting_date"] + pd.Timedelta(days=1)

    rates = target_upper[["observation_date", "DFEDTARU"]].copy()
    rates["observation_date"] = pd.to_datetime(rates["observation_date"], errors="coerce")
    rates["DFEDTARU"] = pd.to_numeric(rates["DFEDTARU"], errors="coerce")
    rates = rates.dropna(subset=["observation_date", "DFEDTARU"]).sort_values("observation_date")

    decisions = pd.merge(
        meetings,
        rates,
        left_on="post_meeting_date",
        right_on="observation_date",
        how="left",
    )
    decisions = (
        decisions[["meeting_date", "DFEDTARU"]]
        .rename(columns={"DFEDTARU": "target_rate"})
        .dropna(subset=["target_rate"])
        .sort_values("meeting_date")
        .reset_index(drop=True)
    )
    decisions["previous_rate"] = decisions["target_rate"].shift(1)
    decisions["rate_change"] = decisions["target_rate"] - decisions["previous_rate"]
    decisions["decision"] = decisions["rate_change"].apply(_action_label)

    df_merged = decisions[["meeting_date", "target_rate", "decision"]].copy()
    df_merged["target_rate_normalized"] = _minmax_normalize(df_merged["target_rate"])
    df_merged["meeting_date"] = df_merged["meeting_date"].dt.strftime("%Y-%m-%d")
    return df_merged


def refresh_processed_fed_meetings(project_root: str | Path | None = None) -> RefreshResult:
    """Overwrite processed_fed_meetings.csv using the latest calendar and target-rate data."""
    root = infer_project_root(project_root)

    try:
        # Refresh the target-rate source first. If online refresh fails, fall back to local data.
        refresh_fred_sources({"Federal Funds Target Range Upper Limit.csv": "DFEDTARU"}, root)
        target_path = resolve_data_source("Federal Funds Target Range Upper Limit.csv", root)
        target_upper = pd.read_csv(target_path)
        meeting_dates = load_fomc_meeting_dates(root)
        df = build_processed_fed_meetings(meeting_dates, target_upper)
        _write_csv(df, "processed_fed_meetings.csv", root)
        start, end = _date_span(df, "meeting_date")
        return RefreshResult("processed_fed_meetings.csv", len(df), start, end, FOMC_CALENDAR_URL)
    except Exception as exc:
        return RefreshResult("processed_fed_meetings.csv", 0, None, None, FOMC_CALENDAR_URL, str(exc))
