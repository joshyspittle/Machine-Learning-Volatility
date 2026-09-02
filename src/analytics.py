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

PriceData: TypeAlias = pd.Series | pd.DataFrame


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


def calculate_cagr(close_series: pd.Series) -> float:
    """Return compound annual growth rate over the supplied window."""

    initial_price = close_series.iloc[0]
    final_price = close_series.iloc[-1]

    duration_days = (close_series.index[-1] - close_series.index[0]).days
    duration_years = duration_days / 365.25

    cagr = (final_price / initial_price) ** (1 / duration_years) - 1
    return float(cagr)


def calculate_sharpe_ratio(
    close_series: pd.Series,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> float:
    """Return the annualised Sharpe ratio using ``US01Y`` as the risk-free proxy."""

    df_macro = pd.read_csv(paths.MACRO_DATA_PARQUET, index_col=0, parse_dates=True).loc[start_date:end_date]
    returns = get_returns(close_series, return_type='simple')
    risk_free_return = (df_macro['US01Y'] / (100 * 365.25)).reindex(close_series.index).ffill()

    excess_returns = returns - risk_free_return
    volatility = np.std(excess_returns, ddof=1)

    sharpe_ratio = excess_returns.mean() / volatility * np.sqrt(365.25)
    return float(sharpe_ratio)


def calculate_sortino_ratio(
    close_series: pd.Series,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> float:
    """Return the annualised Sortino ratio using downside volatility only."""

    df = pd.read_csv(paths.MACRO_DATA_PARQUET, index_col=0, parse_dates=True).loc[start_date:end_date]
    risk_free_return = df['US01Y'].mean() / 100
    risk_free_daily_return = risk_free_return / 365.25

    returns = get_returns(close_series, return_type='simple')
    mean_return = returns.mean() * 365.25
    downside_returns = returns[returns < risk_free_daily_return]
    print('risk free daily', risk_free_daily_return)
    downside_volatility = downside_returns.std(ddof=1) * np.sqrt(365.25)

    sortino_ratio = (mean_return - risk_free_return) / downside_volatility
    return float(sortino_ratio)


def max_drawdown(
    ohlcv_data: pd.Series | pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    return_type: Literal['close-to-close', 'high-to-low'] = 'close-to-close',
) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Return maximum drawdown plus the peak and trough dates."""

    ohlc_cols = {'Open', 'High', 'Low', 'Close'}
    is_ohlcv = isinstance(ohlcv_data, pd.DataFrame) and ohlc_cols.issubset(ohlcv_data.columns)

    if return_type == 'high-to-low' and not is_ohlcv:
        raise ValueError("return_type: 'high-to-low' requires full OHLCV data")

    if return_type == 'high-to-low':
        peak_series = ohlcv_data['High']
        trough_series = ohlcv_data['Low']
    else:
        if is_ohlcv:
            peak_series = ohlcv_data['Close']
        else:
            peak_series = ohlcv_data
        trough_series = peak_series

    rolling_max = peak_series.cummax()
    drawdown = (trough_series - rolling_max) / rolling_max

    max_dd = drawdown.min()
    trough = drawdown.idxmin()
    peak = peak_series[:trough].idxmax()

    return max_dd, peak, trough
