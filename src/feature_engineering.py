"""Analytics helpers for returns, indicators, and portfolio metrics.

This module groups small, mostly stateless pandas/numpy transformations:
return calculations, rolling-window helpers, technical indicators, and
performance/risk metrics.
"""

from collections.abc import Callable
from typing import Literal, TypeAlias
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import src.paths as paths
from src.data import load_data
import src.volatility as volatility

PriceData: TypeAlias = pd.Series | pd.DataFrame


FEATURES = [
    # VIX lags
    'VIX_0',
    'VIX_1',
    'VIX_2',
    'VIX_5',
    'VIX_10',
    'VIX_20',
    'VIX_60',
    'VIX_120',

    # Realised volatility lags
    'Realised_vol_0',
    'Realised_vol_1',
    'Realised_vol_2',
    'Realised_vol_5',
    'Realised_vol_10',
    'Realised_vol_20',
    'Realised_vol_60',
    'Realised_vol_120',

    # Return lags
    'Returns_0',
    'Returns_1',
    'Returns_2',
    'Returns_5',
    'Returns_10',
    'Returns_20',
    'Returns_60',
    'Returns_120',

    # Volatility moving averages
    'Vol_7_sma',
    'Vol_14_sma',
    'Vol_21_sma',
    'Vol_50_sma',
    'Vol_200_sma',

    'Vol_7_ema',
    'Vol_14_ema',
    'Vol_21_ema',
    'Vol_50_ema',
    'Vol_200_ema',

    # Price moving averages
    'Price_7_sma',
    'Price_14_sma',
    'Price_21_sma',
    'Price_50_sma',
    'Price_200_sma',

    'Price_7_ema',
    'Price_14_ema',
    'Price_21_ema',
    'Price_50_ema',
    'Price_200_ema',

    # Price relative to moving averages
    'Price_vs_sma7',
    'Price_vs_sma21',
    'Price_vs_sma50',
    'Price_vs_sma200',

    # Calendar
    'dayofweek',
    'quarter',
    'month',
    'year',
    'dayofyear',
    'dayofmonth',
    'weekofyear',
]

TARGET = 'Parkinson'

def build_ml_features(ohlcv_series: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready features and the Parkinson volatility target."""

    df = ohlcv_series.copy()
    realised_vol = volatility.parkinson_vol(df)

    df['Parkinson'] = realised_vol

    lags = [0, 1, 2, 5, 10, 20, 60, 120]

    for lag in lags:
        vix = load_data(paths.MACRO_DATA_PARQUET, '1d')['VIX']['Close']
        df[f'VIX_{lag}'] = vix.reindex(df.index).ffill().shift(lag+1)

        df[f'Realised_vol_{lag}'] = realised_vol.shift(lag+1)

        df[f'Returns_{lag}'] = get_returns(df['Close']).shift(lag+1)

    ma_days = [7, 14, 21, 50, 200]

    for moving_average in ma_days:
        df[f'Vol_{moving_average}_sma'] = sma(realised_vol, moving_average).shift(1)
        df[f'Vol_{moving_average}_ema'] = ema(realised_vol, moving_average).shift(1)
        df[f'Price_{moving_average}_sma'] = ema(df['Close'], moving_average).shift(1)
        df[f'Price_{moving_average}_ema'] = ema(df['Close'], moving_average).shift(1)

    df['Price_vs_sma7'] = df['Close'].shift(1)/df['Price_7_sma']
    df['Price_vs_sma21'] = df['Close'].shift(1)/df['Price_21_sma']
    df['Price_vs_sma50'] = df['Close'].shift(1)/df['Price_50_sma']
    df['Price_vs_sma200'] = df['Close'].shift(1)/df['Price_200_sma']

    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    return df


def get_returns(
    price_data: PriceData,
    return_type: str = 'log',
) -> PriceData:
    """Return log, simple, or arithmetic returns for a price series."""

    if return_type == 'log':
        return np.log(price_data / price_data.shift(1))
    if return_type == 'simple':
        return price_data.pct_change()
    if return_type == 'arithmetic':
        return price_data.diff()

    raise ValueError(f"Unsupported return type: {return_type}")


def mean_returns(
    close_data: pd.DataFrame,
    asset_id: str,
    interval: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> float:
    """Return annualised mean simple return for one column in a panel."""

    returns = get_returns(close_data[asset_id], return_type='simple')
    mean_return = returns.mean() * 365.25

    return float(mean_return)


def rolling(
    parameter_func: Callable[[PriceData], float],
    close_data: PriceData,
    window_size: int,
) -> pd.DataFrame:
    """Apply a parameter function over rolling windows and return the result."""

    results: list[dict[str, float | pd.Timestamp]] = []

    for i in range(window_size, len(close_data) + 1):
        window_slice = close_data.iloc[i - window_size:i]
        current_date = window_slice.index[-1]

        val = parameter_func(window_slice)
        results.append({'Date:': current_date, 'Value': val})

    return pd.DataFrame(results).set_index('Date:')


def rolling_parallel(parameter_func, close_series, window_size, n_jobs=-1):
    windows = [
        (close_series.index[i - 1], close_series.iloc[i - window_size:i])
        for i in range(window_size, len(close_series) + 1)
    ]
    results = Parallel(n_jobs=n_jobs)(
        delayed(parameter_func)(window) for _, window in windows
    )
    dates = [d for d, _ in windows]

    return pd.DataFrame({'Value': results}, index=pd.Index(dates, name='Date'))


def sma(close_data: PriceData, window: int = 21) -> PriceData:
    """Return the simple moving average over ``window`` periods."""

    return close_data.rolling(window).mean()


def ema(close_data: PriceData, window: int = 21) -> PriceData:
    """Return the exponential moving average over ``window`` periods."""

    return close_data.ewm(span=window, adjust=False).mean()


def rsi(close_data: PriceData, window: int = 14) -> PriceData:
    """Return the relative strength index over ``window`` periods."""

    delta = close_data.diff()
    gains = delta.clip(lower=0)
    losses = delta.clip(upper=0).abs()

    avg_gain = gains.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.dropna()


def macd(
    close_data: PriceData,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    """Return the moving average convergence divergence (MACD) and signal line."""

    ema_fast = ema(close_data, window=fast_window)
    ema_slow = ema(close_data, window=slow_window)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, window=signal_window)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram,
    })
