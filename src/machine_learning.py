"""Machine-learning feature engineering, training, and forecast evaluation."""

from typing import TypeAlias, TypedDict

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

import src.analytics as analytics
import src.paths as paths
import src.volatility as volatility
from src.data_getter import load_data

ForecastMap: TypeAlias = dict[str, pd.Series]
EvaluationResults: TypeAlias = dict[str, dict[str, float]]
HalvedEvaluationResults: TypeAlias = dict[str, EvaluationResults]


class TrainData(TypedDict):
    """Container for model fitting and validation data."""

    x_fit: pd.DataFrame
    y_fit: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series


class TestData(TypedDict):
    """Container for held-out model evaluation data."""

    x_test: pd.DataFrame
    y_test: pd.Series


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

        df[f'Returns_{lag}'] = analytics.get_returns(df['Close']).shift(lag+1)

    ma_days = [7, 14, 21, 50, 200]

    for moving_average in ma_days:
        df[f'Vol_{moving_average}_sma'] = analytics.sma(realised_vol, moving_average).shift(1)
        df[f'Vol_{moving_average}_ema'] = analytics.ema(realised_vol, moving_average).shift(1)
        df[f'Price_{moving_average}_sma'] = analytics.ema(df['Close'], moving_average).shift(1)
        df[f'Price_{moving_average}_ema'] = analytics.ema(df['Close'], moving_average).shift(1)

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


def split_train_test(features: pd.DataFrame, split_date: str = '2024-01-01') -> tuple[TrainData, TestData]:
    """Split features into fitting, validation, and test datasets."""

    train = features.loc[features.index < split_date].copy()
    test = features.loc[features.index >= split_date].copy()

    x_train = train[FEATURES]
    y_train = train[TARGET]

    x_test = test[FEATURES]
    y_test = test[TARGET]

    validation_start = x_train.index[int(len(x_train) * 0.8)]
    x_fit = x_train.loc[x_train.index < validation_start]
    y_fit = y_train.loc[y_train.index < validation_start]
    x_val = x_train.loc[x_train.index >= validation_start]
    y_val = y_train.loc[y_train.index >= validation_start]

    train = {'x_fit': x_fit, 'y_fit': y_fit, 
             'x_val': x_val, 'y_val': y_val}
    test = {'x_test': x_test, 'y_test': y_test}

    return train, test


def train_model(train_data: TrainData) -> xgb.XGBRegressor:
    """Train and return the XGBoost volatility model."""

    x_fit = train_data['x_fit']
    y_fit = train_data['y_fit']
    x_val = train_data['x_val']
    y_val = train_data['y_val']

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                        n_estimators=1000,
                        early_stopping_rounds=50,
                        objective='reg:squarederror',
                        max_depth=3,
                        learning_rate=0.01)
    reg.fit(x_fit, y_fit, 
            eval_set=[(x_fit, y_fit), (x_val, y_val)],
            verbose=100)

    return reg


def forecast_model(
    test_data: TestData,
    reg: xgb.XGBRegressor,
    features: pd.DataFrame,
    plot: bool = False
) -> pd.Series:
    """Forecast test-period volatility and plot predictions against realised values."""

    x_test = test_data['x_test']

    predictions = pd.Series(reg.predict(x_test), index=x_test.index, name='Prediction')
    features = features.merge(predictions, how='left', left_index=True, right_index=True)

    if plot:
        ax = features[['Parkinson']].plot(figsize=(15,5))
        features['Prediction'].plot(ax=ax, style='.')
        plt.legend(['Truth Data', 'Predictions'])
        ax.set_title('Raw Data and Predictions')
        plt.savefig('2014 forecast')
        plt.show()

    forecast = features['Prediction'].dropna()

    return forecast


def evaluate_forecasts(
    forecasts: ForecastMap,
    realised: pd.Series,
    start_date: str | pd.Timestamp = '2024-01-01',
) -> EvaluationResults:
    """Evaluate each forecast over the exact same target dates."""

    eval_index = realised.loc[start_date:].dropna().index
    for forecast in forecasts.values():
        eval_index = eval_index.intersection(forecast.dropna().index)

    aligned_realised = realised.loc[eval_index]
    results = {
        name: volatility.evaluate(forecast.loc[eval_index], aligned_realised)
        for name, forecast in forecasts.items()
    }

    print(
        f"Evaluation dates: {eval_index.min().date()} to {eval_index.max().date()} "
        f"({len(eval_index)} observations)"
    )

    return results


def evaluate_forecast_halves(
    forecasts: ForecastMap,
    realised: pd.Series,
    start_date: str | pd.Timestamp,
) -> HalvedEvaluationResults:
    """Evaluate aligned forecasts over the first and second test halves."""

    eval_index = realised.loc[start_date:].dropna().index
    for forecast in forecasts.values():
        eval_index = eval_index.intersection(forecast.dropna().index)

    midpoint = len(eval_index) // 2
    halves = {
        'First half': eval_index[:midpoint],
        'Second half': eval_index[midpoint:],
    }

    results = {}
    for half_name, half_index in halves.items():
        aligned_realised = realised.loc[half_index]
        results[half_name] = {
            name: volatility.evaluate(forecast.loc[half_index], aligned_realised)
            for name, forecast in forecasts.items()
        }
        print(
            f"{half_name} dates: {half_index.min().date()} to "
            f"{half_index.max().date()} ({len(half_index)} observations)"
        )

    return results


def evaluate(ohlcv_series: pd.DataFrame,
             plot: bool = False,
             feature_importance: bool = False) -> EvaluationResults:
    """Train the ML model and compare it with benchmark volatility forecasts."""

    close_series = ohlcv_series['Close']

    realised_vol = volatility.parkinson_vol(ohlcv_series)

    features = build_ml_features(ohlcv_series)
    train_data, test_data = split_train_test(features)
    model = train_model(train_data)
    ml_forecast = forecast_model(test_data, model, features, plot)

    forecasts = {
        'GARCH(1,1)': volatility.garch_forecast(close_series, 500),
        'Naive (persistence)': volatility.naive_persistent_forecast(realised_vol),
        'Naive (rolling avg)': volatility.naive_avg_forecast(realised_vol, 500),
        'ML Model': ml_forecast,
    }

    results = evaluate_forecasts(forecasts, realised_vol)

    if feature_importance:
        fi = pd.DataFrame(data=model.feature_importances_,
                    index=model.feature_names_in_,
                    columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.savefig('2014 fi')
        plt.show()

    return results


def walk_forward(ohlcv_series: pd.DataFrame, window_length: int = 90) -> list[pd.Series]:
    """Retrain on an expanding window every `window_length` days, forecasting the next block each time."""

    features = build_ml_features(ohlcv_series).dropna()
    n = len(features)

    forecasts = []
    importances = []
    for train_end in range(window_length, n, window_length):
        forecast_end = min(train_end + window_length, n)

        train_df = features.iloc[:train_end]
        forecast_df = features.iloc[train_end:forecast_end]

        x_train, y_train = train_df[FEATURES], train_df[TARGET]
        validation_start = x_train.index[int(len(x_train) * 0.8)]
        x_fit = x_train.loc[x_train.index < validation_start]
        y_fit = y_train.loc[y_train.index < validation_start]
        x_val = x_train.loc[x_train.index >= validation_start]
        y_val = y_train.loc[y_train.index >= validation_start]

        train_data = {'x_fit': x_fit, 'y_fit': y_fit, 'x_val': x_val, 'y_val': y_val}
        test_data = {'x_test': forecast_df[FEATURES], 'y_test': forecast_df[TARGET]}

        model = train_model(train_data)
        forecast = forecast_model(test_data, model, features, plot=False)
        forecasts.append(forecast)

        print(f"Trained rows 0-{train_end-1} ({train_end} days) -> "
              f"forecasting rows {train_end}-{forecast_end-1} ({forecast_end - train_end} days)")

    return forecasts


def evaluate_walk_forward(ohlcv_series):

    close_series = ohlcv_series['Close']
    realised_vol = volatility.parkinson_vol(ohlcv_series)

    benchmarks = {
        'GARCH(1,1)': volatility.garch_forecast(close_series, 500),
        'Naive (persistence)': volatility.naive_persistent_forecast(realised_vol),
        'Naive (rolling avg)': volatility.naive_avg_forecast(realised_vol, 500),
    }

    forecasts = walk_forward(ohlcv_series, 90)

    per_block_results = {}
    for i, forecast in enumerate(forecasts):
        block_label = f"Block {i + 1} ({forecast.index.min().date()})"
        block_forecasts = benchmarks | {'ML Model': forecast}
        per_block_results[block_label] = evaluate_forecasts(block_forecasts, realised_vol, forecast.index.min())

    combined = pd.concat(forecasts).rename('ML Model (Walk-Forward)')
    combined_forecasts = benchmarks | {'ML Model (Walk-Forward)': combined}
    combined_results = evaluate_forecasts(combined_forecasts, realised_vol, combined.index.min().date())

    return per_block_results, combined_results, combined


def plot_walk_forward(realised_vol, forecasts, per_block, asset):

    aligned_realised = realised_vol.loc[forecasts.index]

    ax = aligned_realised.plot(figsize=(15,5))
    forecasts.plot(ax=ax, style='.')
    plt.legend(['Ground Data', 'Predictions'])
    ax.set_title(f'Ground Data and Predictions: {asset}')
    plt.savefig(f'RealisedVsPredicted({asset}).png')
    plt.show()

    block_labels = list(per_block.keys())
    block_dates = [pd.Timestamp(label.split('(')[1].rstrip(')')) for label in block_labels]

    ml_qlike = [per_block[b]['ML Model']['qlike'] for b in block_labels]
    garch_qlike = [per_block[b]['GARCH(1,1)']['qlike'] for b in block_labels]

    ml_mse = [per_block[b]['ML Model']['mse'] for b in block_labels]
    garch_mse = [per_block[b]['GARCH(1,1)']['mse'] for b in block_labels]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(block_dates, ml_qlike, marker='o', label='ML Model')
    ax.plot(block_dates, garch_qlike, marker='o', label='GARCH(1,1)')
    ax.set_title(f'QLIKE per Block: {asset}')
    ax.legend()
    plt.savefig(f'MLvsGARCH(QLIKE)({asset}).png')
    plt.show()

    return None