"""Volatility estimators, benchmark forecasts, and loss functions.

Conventions:
- Volatility series are expressed in daily percentage points.
- Forecasts are dated to the realised volatility observation they predict.
- Loss functions compare variance forecasts, so volatility inputs are squared.
"""

import numpy as np
import pandas as pd

import src.feature_engineering as fe

from arch import arch_model


def garch(close_series: pd.Series) -> float:
    """Return the one-step-ahead GARCH(1,1) volatility forecast.

    The model is fit on log returns scaled to percentage points so that the
    returned volatility is directly comparable with project realised-vol
    estimators such as Parkinson and absolute-return volatility.
    """

    returns = fe.get_returns(close_series).dropna()
    scaled_returns = returns * 100

    garch_model = arch_model(
        scaled_returns,
        p=1,
        q=1,
        mean='constant',
        vol='GARCH',
        dist='normal',
    )

    garch_result = garch_model.fit(update_freq=4, disp='off')

    garch_forecast = garch_result.forecast(horizon=1)

    return float(np.sqrt(garch_forecast.variance.iloc[-1, 0]))


def garch_forecast(close_series: pd.Series, window_size: int) -> pd.Series:
    """Return rolling one-step-ahead GARCH(1,1) volatility forecasts."""

    forecast = fe.rolling_parallel(garch, close_series, window_size)['Value']
    forecast = forecast.shift(1).dropna()
    forecast.name = 'GARCH(1,1)'

    return forecast


def naive_persistent_forecast(realised_vol: pd.Series) -> pd.Series:
    """Return a persistence forecast using yesterday's realised volatility."""

    forecast = realised_vol.shift(1)
    forecast.name = 'Naive (persistence)'

    return forecast


def naive_avg_forecast(realised_vol: pd.Series, window_size: int) -> pd.Series:
    """Return a rolling mean forecast of recent realised variance.

    The mean is taken over realised variance rather than realised volatility,
    then converted back to volatility units for consistency with other
    forecast series.
    """

    realised_var = np.square(realised_vol)
    forecast = np.sqrt(realised_var.rolling(window_size).mean()).shift(1).dropna()
    forecast.name = 'Naive (rolling avg)'

    return forecast


def realised_absolute_vol(close_series: pd.Series) -> pd.Series:
    """Return absolute log-return volatility in daily percentage points."""

    abs_log_returns = np.abs(fe.get_returns(close_series)) * 100
    abs_log_returns.name = 'RAV'

    return abs_log_returns


def parkinson_vol(ohlcv_data: pd.DataFrame) -> pd.Series:
    """Return the daily Parkinson volatility estimate in percentage points."""

    x = np.log(ohlcv_data['High'] / ohlcv_data['Low']) ** 2
    parkinson_var = x / (4 * np.log(2))
    parkinson_vol = np.sqrt(parkinson_var) * 100
    parkinson_vol.name = 'Parkinson'

    return parkinson_vol.dropna()


def mse_loss(forecast: pd.Series, realised: pd.Series) -> float:
    """Return mean squared error between forecast and realised variances."""

    forecast_var = np.square(forecast)
    realised_var = np.square(realised)

    mse = np.mean((realised_var - forecast_var) ** 2)

    return float(mse)


def mae_loss(forecast: pd.Series, realised: pd.Series) -> float:
    """Return mean absolute error between forecast and realised variances."""

    forecast_var = np.square(forecast)
    realised_var = np.square(realised)

    mae = np.mean(np.abs(realised_var - forecast_var))

    return float(mae)


def qlike_loss(forecast: pd.Series, realised: pd.Series) -> float:
    """Return QLIKE loss for variance forecasts."""

    forecast_var = np.square(forecast)
    realised_var = np.square(realised)

    qlike = np.mean((realised_var / forecast_var) - np.log(realised_var / forecast_var) - 1)

    return float(qlike)
