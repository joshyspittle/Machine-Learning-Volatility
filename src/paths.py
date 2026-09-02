"""Central path definitions for project data and watchlists.

This module keeps filesystem paths in one place and creates the main project
folders on import if they do not already exist.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = PROJECT_ROOT / "data"
WATCHLIST_DIR = PROJECT_ROOT / "watchlist"

CRYPTO_WATCHLIST_CSV = WATCHLIST_DIR / "crypto.csv"
MACRO_WATCHLIST_CSV = WATCHLIST_DIR / "macro.csv"
FOREX_WATCHLIST_CSV = WATCHLIST_DIR / "forex.csv"
COMMODITIES_WATCHLIST_CSV = WATCHLIST_DIR / "commodities.csv"
EQUITIES_WATCHLIST_CSV = WATCHLIST_DIR / "equities.csv"
INDICES_WATCHLIST_CSV = WATCHLIST_DIR / "indices.csv"

CRYPTO_DATA_PARQUET = DATA_DIR / "crypto_data.parquet"
MACRO_DATA_PARQUET = DATA_DIR / "macro_data.parquet"
FOREX_DATA_PARQUET = DATA_DIR / "forex_data.parquet"
COMMODITIES_DATA_PARQUET = DATA_DIR / "commodities_data.parquet"
EQUITIES_DATA_PARQUET = DATA_DIR / "equities_data.parquet"
INDICES_DATA_PARQUET = DATA_DIR / "indices_data.parquet"

for folder in [DATA_DIR, WATCHLIST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
