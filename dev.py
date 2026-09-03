"""Development scratchpad for interactive testing.

This file is not part of the production pipeline. It is used to load local
data, try out analysis functions, and preview charts while iterating.
"""

import src.paths as paths
import src.volatility as volatility
import src.machine_learning as ml
from data import load_data

import matplotlib.pyplot as plt
import xgboost as xgb

from arch import arch_model

asset = 'XRP'

ohlcv = load_data(paths.CRYPTO_DATA_PARQUET, '1d')#, start_date='01-01-2014')
close = ohlcv.xs('Close', axis=1, level=1)
btc_ohlcv = ohlcv[asset].dropna()
btc_close = close[asset].dropna()

per_block, comb, forecasts = ml.evaluate_walk_forward(btc_ohlcv)

print(per_block)
print(comb)

realised_vol = volatility.parkinson_vol(btc_ohlcv)
ml.plot_walk_forward(realised_vol, forecasts, per_block, asset)

ml_results = ml.evaluate(btc_ohlcv, plot=True, feature_importance=True)
print(ml_results)
