"""Incremental data update script.

Workflow:
- Load watchlists by category.
- Read the existing parquet file for each category if present.
- Fetch either full history or incremental updates per asset.
- Merge updates, replace the overlap window, and save.

Assumptions:
- Stored category files use a two-level column MultiIndex: ``(Asset, Field)``.
- Macro series use ``Close`` as their single field and are refreshed with a
  90-day overlap to catch revisions.
- Market data uses a 5-day overlap so the latest bars can be refreshed
  without depending on a single same-day fetch.
- Raw non-macro OHLCV is saved without forward-filling.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

import src.paths as paths
from src.data import load_all_configs
from src.data import clear_cache, fetch_data

MACRO_LOOKBACK_DAYS = 90
MARKET_LOOKBACK_DAYS = 5
LOGS_DIR = paths.PROJECT_ROOT / "logs"


def setup_logger() -> tuple[logging.Logger, Path]:
    """Create a timestamped file logger for detailed update output."""

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = LOGS_DIR / f"run_updates_{timestamp}.log"

    logger = logging.getLogger("run_updates")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    return logger, log_path


def get_start_date(
    category: str,
    source: str,
    asset_df: pd.DataFrame,
) -> pd.Timestamp | None:
    """Return the overlap start date for an existing asset."""

    if asset_df.isna().all().all():
        return None

    last_date = asset_df["Close"].last_valid_index()

    if category == "macro":
        return last_date - pd.Timedelta(days=MACRO_LOOKBACK_DAYS)
    if source == "yfinance":
        return last_date - pd.Timedelta(days=MARKET_LOOKBACK_DAYS)

    return last_date


def describe_empty_fetch(identifier: str, source: str, start_date: pd.Timestamp | None) -> tuple[str, str]:
    """Return a terminal warning and file-log reason for an empty fetch."""

    if source == "Scrape":
        return (
            f"warning: {identifier} -> no fetcher implemented for source '{source}'",
            "unsupported source",
        )

    if source == "yfinance" and start_date is not None:
        return (
            f"warning: {identifier} -> latest daily bar may not be available yet",
            "latest daily bar may not be available yet",
        )

    return (f"warning: {identifier} -> no data returned", "no data returned")


def merge_asset_data(
    existing_df: pd.DataFrame,
    identifier: str,
    new_data: pd.DataFrame,
    start_date: pd.Timestamp | None,
) -> tuple[pd.DataFrame, str]:
    """Merge new asset data into the category panel."""

    if not existing_df.empty and identifier in existing_df.columns.get_level_values(0):
        existing_asset = existing_df[identifier]
        new_asset = new_data[identifier]

        if start_date is not None:
            existing_asset = existing_asset.loc[existing_asset.index < start_date]

        combined = pd.concat([existing_asset, new_asset], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        combined.columns = pd.MultiIndex.from_product([[identifier], combined.columns])
        existing_without_asset = existing_df.drop(columns=identifier, level=0)
        merged_df = pd.concat([existing_without_asset, combined], axis=1, sort=True)
        return merged_df, "updated"

    merged_df = pd.concat([existing_df, new_data], axis=1, sort=True)
    return merged_df, "added"

def main():

    logger, log_path = setup_logger()
    config, headers = load_all_configs(paths.WATCHLIST_DIR)

    run_updated = 0
    run_skipped = 0
    run_failed = 0

    print("Starting update run...")
    logger.info("Starting update run")

    for category, assets in config.items():
        storage_file = os.path.join(paths.DATA_DIR, f"{category}_data.parquet")
        category_updated = 0
        category_skipped = 0
        category_failed = 0
        category_warnings: list[str] = []

        print(f"\n[{category}]")
        logger.info("[%s] start", category)

        try:
            if os.path.exists(storage_file):
                existing_df = pd.read_parquet(storage_file)
                logger.info("[%s] loaded existing parquet %s", category, storage_file)
            else:
                os.makedirs(os.path.dirname(storage_file), exist_ok=True)
                existing_df = pd.DataFrame()
                logger.info("[%s] no existing parquet found", category)
        except Exception as e:
            category_failed += len(assets)
            run_failed += len(assets)
            print(f"error: {category} -> failed to load existing data ({e})")
            logger.exception("[%s] failed to load existing data", category)
            continue

        for identifier, item in assets.items():
            source = item["Source"]
            ticker = item["Ticker"]
            full_name = item["Full_Name"]

            logger.info("[%s] %s start | source=%s ticker=%s", category, identifier, source, ticker)

            if not existing_df.empty and identifier in existing_df.columns.get_level_values(0):
                asset_df = existing_df[identifier]
                start_date = get_start_date(category, source, asset_df)

                if start_date is None:
                    logger.info("[%s] %s has no valid data; fetching full history", category, identifier)
                else:
                    logger.info("[%s] %s incremental fetch from %s", category, identifier, start_date.date())
            else:
                start_date = None
                logger.info("[%s] %s new asset; fetching full history", category, identifier)

            try:
                new_data = fetch_data(ticker, source, identifier, start_date=start_date)
            except Exception:
                category_failed += 1
                run_failed += 1
                print(f"error: {identifier} -> fetch raised an exception")
                logger.exception("[%s] %s fetch raised an exception", category, identifier)
                continue

            if new_data is None or new_data.empty:
                category_skipped += 1
                run_skipped += 1
                terminal_message, log_reason = describe_empty_fetch(identifier, source, start_date)
                category_warnings.append(terminal_message)
                print(terminal_message)
                logger.warning("[%s] %s skipped | reason=%s", category, identifier, log_reason)
                continue

            logger.info(
                "[%s] %s fetched %s rows | %s -> %s",
                category,
                identifier,
                len(new_data),
                new_data.index[0].date(),
                new_data.index[-1].date(),
            )

            try:
                existing_df, action = merge_asset_data(existing_df, identifier, new_data, start_date)
                category_updated += 1
                run_updated += 1
                logger.info("[%s] %s merge action=%s", category, identifier, action)
            except Exception:
                category_failed += 1
                run_failed += 1
                print(f"error: {identifier} -> failed during merge")
                logger.exception("[%s] %s failed during merge", category, identifier)

        if not existing_df.empty:
            try:
                existing_df = existing_df.sort_index()

                if category == "macro":
                    existing_df = existing_df.ffill()
                else:
                    existing_df = existing_df.dropna(how="all")

                existing_df = existing_df.loc[:, existing_df.columns.get_level_values(0).isin(assets.keys())]
                existing_df = existing_df.sort_index(axis=1)
                existing_df.index.name = "Date"

                existing_df.to_parquet(storage_file)
                clear_cache(storage_file)

                print(f"completed: {category_updated} updated, {category_skipped} skipped, {category_failed} failed")
                print(
                    f"saved rows={existing_df.shape[0]} cols={existing_df.shape[1]} "
                    f"range={existing_df.index[0].date()} to {existing_df.index[-1].date()}"
                )
                logger.info(
                    "[%s] saved | updated=%s skipped=%s failed=%s rows=%s cols=%s range=%s -> %s",
                    category,
                    category_updated,
                    category_skipped,
                    category_failed,
                    existing_df.shape[0],
                    existing_df.shape[1],
                    existing_df.index[0].date(),
                    existing_df.index[-1].date(),
                )
            except Exception:
                run_failed += 1
                print(f"error: {category} -> failed while saving category output")
                logger.exception("[%s] failed while saving category output", category)
        else:
            print("completed: 0 updated, 0 skipped, 0 failed")
            print("no data saved for this category")
            logger.warning("[%s] no data available after processing", category)

    print("\nRun complete")
    print(f"updated={run_updated} skipped={run_skipped} failed={run_failed}")
    print(f"log={log_path}")
    logger.info("Run complete | updated=%s skipped=%s failed=%s log=%s", run_updated, run_skipped, run_failed, log_path)


if __name__ == "__main__":
    main()