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

ConfigRow: TypeAlias = dict[str, str]
CategoryConfig: TypeAlias = dict[str, ConfigRow]
WatchlistConfig: TypeAlias = dict[str, CategoryConfig]


def load_all_configs(watchlist_dir: str | Path) -> tuple[WatchlistConfig, list[str]]:
    """Load all configured watchlists from disk."""

    data_files = [
        'commodities.csv',
        'crypto.csv',
        'macro.csv',
        'equities.csv',
        'forex.csv',
        'indices.csv',
    ]

    config: WatchlistConfig = {}
    headers: list[str] = []

    for filename in data_files:
        file_path = Path(watchlist_dir) / filename
        category_name = filename.replace('.csv', '')

        try:
            df = pd.read_csv(file_path, skipinitialspace=True)
            headers = df.columns.tolist()

            if 'Identifier' not in df.columns:
                raise KeyError

            df['Identifier'] = df['Identifier'].astype(str).str.strip()

            duplicate_identifiers = df.loc[df['Identifier'].duplicated(), 'Identifier'].unique()
            if len(duplicate_identifiers) > 0:
                duplicates = ', '.join(sorted(duplicate_identifiers))
                raise ValueError(f"Duplicate Identifier values in {filename}: {duplicates}")

            indexed_df = df.set_index('Identifier')
            category_config = indexed_df.to_dict('index')
            config[category_name] = category_config

            print(f"Loaded and processed: {category_name} ({len(category_config)} items)")

        except FileNotFoundError:
            print(f"ERROR: File not found at {file_path}. Skipping.")

        except pd.errors.EmptyDataError:
            print(f"WARNING: {filename} is empty. Skipping.")

        except KeyError:
            print(f"ERROR: Missing 'Identifier' column in {filename}. Check file headers.")

        except ValueError as e:
            print(f"ERROR: {e}")

    return config, headers


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
