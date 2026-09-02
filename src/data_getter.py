"""Data access helpers for external providers and local parquet storage.

Conventions:
- Yahoo Finance data is normalized to ``Open/High/Low/Close/Volume``.
- FRED series are stored as a single ``Close`` column by convention.
- Returned asset data uses a two-level column MultiIndex: ``(Identifier, Field)``.
- ``load_data`` caches full parquet files in memory for the current session.
"""

import os
from pathlib import Path
from typing import TypeAlias

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

FRED_API = os.getenv('FRED_API_KEY')
fred = Fred(api_key=FRED_API)
PathLike: TypeAlias = str | Path
_cache: dict[PathLike, pd.DataFrame] = {}


def fetch_data(
    ticker: str,
    source: str,
    identifier: str,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Fetch one asset series and return it in the project's standard format."""

    data: pd.DataFrame | None = None

    try:
        if source == 'FRED':
            series = fred.get_series(ticker, observation_start=start_date)
            df = series.to_frame(name='Close')
            df.columns = pd.MultiIndex.from_product([[identifier], df.columns])
            data = df

        elif source == 'yfinance':
            if start_date:
                df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
            else:
                df = yf.download(ticker, period='max', progress=False, auto_adjust=False)

            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.columns = pd.MultiIndex.from_product([[identifier], df.columns])
                data = df
        else:
            print(f"Unsupported source: {source}")

    except Exception as e:
        print(f"Error fetching data for {identifier} from {source}: {e}")

    return data


def clear_cache(path: PathLike | None = None) -> None:
    """Clear the parquet cache for one path or for the whole session."""

    if path is None:
        _cache.clear()
    else:
        _cache.pop(path, None)


def load_data(
    path: PathLike,
    interval: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load a parquet panel and optionally slice it by date.

    ``interval`` is currently kept for interface consistency with the rest
    of the project, but is not used inside this function.
    """

    if path not in _cache:
        _cache[path] = pd.read_parquet(path)

    df = _cache[path]
    return df.loc[start_date:end_date]
