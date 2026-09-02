"""Simple return and volatility estimators for stochastic modelling.

The functions here are currently lightweight helpers for simulation work.
They assume daily data and annualise using either ``365.25`` for BTC or
``252`` for non-BTC assets. That convention is provisional and should be
revisited once trading-calendar handling is formalised.
"""

import numpy as np
import pandas as pd
from arch import arch_model

import src.analytics as analytics

MonteCarloSummary = pd.Series
MonteCarloResult = MonteCarloSummary | tuple[pd.DataFrame, MonteCarloSummary]


def mu(close_series: pd.Series) -> float:
    """Estimate drift (mu) as log drift plus half the variance."""

    drift = log_drift(close_series)
    volatility = log_volatility(close_series)
    expected_return = drift + (volatility ** 2) / 2

    return float(expected_return)


def log_volatility(close_series: pd.Series) -> float:
    """Return annualised volatility from log returns."""

    asset_id = str(close_series.name)
    log_returns = analytics.get_returns(close_series, return_type='log')
    volatility = np.std(log_returns, ddof=1)

    #if asset_id == 'BTC':
    #    annual_volatility = volatility * np.sqrt(365.25)
    #else:
    #    annual_volatility = volatility * np.sqrt(252)

    return float(volatility)


def log_drift(close_series: pd.Series) -> float:
    """Return annualised drift from mean log returns. m = log(S_t / S_{t-1}) = mu - sigma^2 / 2"""

    asset_id = str(close_series.name)
    log_returns = analytics.get_returns(close_series, return_type='log')
    drift = np.mean(log_returns)

    if asset_id == 'BTC':
        annual_drift = drift * 365.25
    else:
        annual_drift = drift * 252

    return float(annual_drift)


def price_path_monte_carlo(close_series: pd.Series, 
                           num_paths: int = 100, 
                           num_days: int = 365) -> pd.DataFrame:
    """Simulate future price paths using continuous GBM: S_t = S_0 * exp(m + sigma * W_t)"""

    S_0 = close_series.iloc[-1]
    m = log_drift(close_series)
    sigma = log_volatility(close_series)
    print(f"Simulating price paths with S_0={S_0}, m={m}, sigma={sigma}")
    t = np.arange(1, num_days + 1) / 365.25 # change this later for stocks
    dt = t[0]

    Z = np.random.normal(0, 1, size=(num_days, num_paths))
    W_t = np.cumsum(np.sqrt(dt)* Z, axis=0)

    future_prices = S_0 * np.exp(m * t[:, None] + sigma * W_t)
    start_prices = np.full((1, num_paths), S_0)
    S_t = np.vstack([start_prices, future_prices])

    dates = pd.date_range(
        start = close_series.index[-1],
        periods = num_days + 1,
        freq = 'D'
    )

    df = pd.DataFrame(S_t, index=dates, columns=range(num_paths))
    df.index.name = 'Date'

    return df


def analyse_monte_carlo(close_series: pd.Series,
                        num_paths: int = 100,
                        num_days: int = 365,
                        return_paths: bool = False) -> MonteCarloResult:
    """Run Monte Carlo simulation and return summary statistics"""

    sims = price_path_monte_carlo(close_series, num_paths=num_paths, num_days=num_days)

    terminal_prices = sims.iloc[-1]

    simulated_drawdowns = []
    for sim in sims:
        max_dd, _, _ = analytics.max_drawdown(sims[sim], return_type='close-to-close')
        simulated_drawdowns.append(max_dd)

    summary = {
        'Terminal Mean': terminal_prices.mean(),
        'Terminal Median': terminal_prices.median(),
        'Terminal Std': terminal_prices.std(),
        'Terminal High': terminal_prices.max(),
        'Terminal Low': terminal_prices.min(),
        'Terminal Range': terminal_prices.max() - terminal_prices.min(),
        'Terminal 1st percentile': terminal_prices.quantile(0.01),
        'Terminal 5th percentile': terminal_prices.quantile(0.05),
        'Terminal 25th percentile': terminal_prices.quantile(0.25),
        'Terminal 75th percentile': terminal_prices.quantile(0.75),
        'Terminal 95th percentile': terminal_prices.quantile(0.95),
        'Terminal 99th percentile': terminal_prices.quantile(0.99),
        'Probability of loss': (terminal_prices < close_series.iloc[-1]).mean(),
        'Max pain': np.min(simulated_drawdowns),
    }

    if not return_paths:
        return pd.Series(summary)
    else:
        return sims, pd.Series(summary)
