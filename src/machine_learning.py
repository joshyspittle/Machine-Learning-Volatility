"""Machine-learning feature engineering, training, and forecast evaluation."""

from typing import TypeAlias, TypedDict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

import src.feature_engineering as fe
import src.paths as paths
import src.volatility as volatility

ForecastMap: TypeAlias = dict[str, pd.Series]
EvaluationResults: TypeAlias = dict[str, dict[str, float]]


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


def split_train_test(features_df: pd.DataFrame, split_date: str = '2024-01-01') -> tuple[TrainData, TestData]:
    """Split features into fitting, validation, and test datasets."""

    train = features_df.loc[features_df.index < split_date].copy()
    test = features_df.loc[features_df.index >= split_date].copy()

    x_train = train[fe.FEATURES]
    y_train = train[fe.TARGET]

    x_test = test[fe.FEATURES]
    y_test = test[fe.TARGET]

    validation_start = x_train.index[int(len(x_train) * 0.8)]
    x_fit = x_train.loc[x_train.index < validation_start]
    y_fit = y_train.loc[y_train.index < validation_start]
    x_val = x_train.loc[x_train.index >= validation_start]
    y_val = y_train.loc[y_train.index >= validation_start]

    train = {'x_fit': x_fit, 'y_fit': y_fit, 
             'x_val': x_val, 'y_val': y_val}
    test = {'x_test': x_test, 'y_test': y_test}

    return train, test


def train_model(train_data: TrainData,
                base_score: float = 0.5,
                n_estimators: int = 1000,
                early_stopping_rounds: int = 50,
                max_depth: int = 3,
                learning_rate: float = 0.01,
                min_child_weight: int = 1,
                verbose: int = 0) -> xgb.XGBRegressor:
    """Train and return the XGBoost volatility model."""

    x_fit = train_data['x_fit']
    y_fit = train_data['y_fit']
    x_val = train_data['x_val']
    y_val = train_data['y_val']

    model = xgb.XGBRegressor(base_score=base_score, booster='gbtree',
                        n_estimators=n_estimators,
                        early_stopping_rounds=early_stopping_rounds,
                        objective='reg:squarederror',
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        min_child_weight=min_child_weight)
    model.fit(x_fit, y_fit, 
            eval_set=[(x_fit, y_fit), (x_val, y_val)],
            verbose=verbose)

    return model


def forecast_model(
    test_data: TestData,
    model: xgb.XGBRegressor
) -> pd.Series:
    """Forecast test-period volatility and plot predictions against realised values."""

    x_test = test_data['x_test']

    forecast = pd.Series(model.predict(x_test), index=x_test.index, name='Prediction').dropna()

    return forecast


def score_forecast(forecast: pd.Series, realised: pd.Series) -> dict[str, float]:
    """Return forecast accuracy metrics after aligning forecast and realised data."""

    aligned = pd.DataFrame({'forecast': forecast, 'realised': realised}).dropna()

    return {
        'qlike': volatility.qlike_loss(aligned['forecast'], aligned['realised']),
        'mse': volatility.mse_loss(aligned['forecast'], aligned['realised']),
        'mae': volatility.mae_loss(aligned['forecast'], aligned['realised'])
    }


def compare_forecasts(
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
        name: score_forecast(forecast.loc[eval_index], aligned_realised)
        for name, forecast in forecasts.items()
    }

    # print(
    #     f"Evaluation dates: {eval_index.min().date()} to {eval_index.max().date()} "
    #     f"({len(eval_index)} observations)"
    # )

    return results


def run_static_comparison(ohlcv_series: pd.DataFrame) -> EvaluationResults:
    """Train the ML model and compare it with benchmark volatility forecasts."""

    close_series = ohlcv_series['Close']

    realised_vol = volatility.parkinson_vol(ohlcv_series)

    features_df = fe.build_ml_features(ohlcv_series)
    train_data, test_data = split_train_test(features_df)
    model = train_model(train_data)
    ml_forecast = forecast_model(test_data, model, features_df)

    forecasts = {
        'GARCH(1,1)': volatility.garch_forecast(close_series, 500),
        'Naive (persistence)': volatility.naive_persistent_forecast(realised_vol),
        'Naive (rolling avg)': volatility.naive_avg_forecast(realised_vol, 500),
        'ML Model': ml_forecast,
    }

    results = compare_forecasts(forecasts, realised_vol)

    return {'model': model, 'ml_forecast': ml_forecast, 'results': results}


def filter_forecasts(forecasts: ForecastMap, start: str, end: str) -> ForecastMap:
    """Return each forecast series restricted to the given date range."""
    return {name: forecast.loc[start:end] for name, forecast in forecasts.items()}


def plot_feature_importance(model, no_of_features=10):

    fi = pd.DataFrame(data=model.feature_importances_,
                index=model.feature_names_in_,
                columns=['importance'])
    fi.sort_values('importance').tail(no_of_features).plot(kind='barh', title='Feature Importance')
    #plt.savefig('2014 fi')
    plt.show()


def plot_predictions_vs_realised(realised: pd.Series, forecasts: ForecastMap, asset: str) -> None:
    """Plot one or more forecasts against realised volatility over their shared forecast period."""

    start_date = min(f.index.min() for f in forecasts.values())
    end_date = max(f.index.max() for f in forecasts.values())
    realised_period = realised.loc[start_date:end_date]

    ax = realised_period.plot(figsize=(15, 5), label='Realised')
    for name, forecast in forecasts.items():
        forecast.plot(ax=ax, style='.', label=name)

    ax.legend()
    ax.set_title(f'Forecasts vs Realised Volatility: {asset}')
    #plt.savefig(f'predictions_vs_realised_{asset}.png')
    plt.show()


def walk_forward(ohlcv_series: pd.DataFrame, window_length: int = 90,
                base_score = 0.5,
                n_estimators: int = 1000,
                early_stopping_rounds: int = 50,
                max_depth: int = 3,
                learning_rate: float = 0.01,
                min_child_weight: int = 1,
                verbose: int = 0) -> list[pd.Series]:
    """Retrain on an expanding window every `window_length` days, forecasting the next block each time."""

    features_df = fe.build_ml_features(ohlcv_series).dropna()
    n = len(features_df)

    forecasts = []
    for train_end in range(window_length, n, window_length):
        forecast_end = min(train_end + window_length, n)

        train_df = features_df.iloc[:train_end]
        forecast_df = features_df.iloc[train_end:forecast_end]

        x_train, y_train = train_df[fe.FEATURES], train_df[fe.TARGET]
        validation_start = x_train.index[int(len(x_train) * 0.8)]
        x_fit = x_train.loc[x_train.index < validation_start]
        y_fit = y_train.loc[y_train.index < validation_start]
        x_val = x_train.loc[x_train.index >= validation_start]
        y_val = y_train.loc[y_train.index >= validation_start]

        train_data = {'x_fit': x_fit, 'y_fit': y_fit, 'x_val': x_val, 'y_val': y_val}
        test_data = {'x_test': forecast_df[fe.FEATURES], 'y_test': forecast_df[fe.TARGET]}

        if base_score == 'mean':
            calculated_base_score = np.mean(x_fit['Realised_vol_0'])
        elif base_score == 'median':
            calculated_base_score = np.median(x_fit['Realised_vol_0'])
        else:
            calculated_base_score = 0.5

        model = train_model(train_data,
                            base_score=calculated_base_score,
                            n_estimators=n_estimators,
                            early_stopping_rounds=early_stopping_rounds,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            min_child_weight=min_child_weight,
                            verbose=verbose)
        forecast = forecast_model(test_data, model)
        forecasts.append(forecast)

        # print(f"Trained rows 0-{train_end-1} ({train_end} days) -> "
        #       f"forecasting rows {train_end}-{forecast_end-1} ({forecast_end - train_end} days)")

    return forecasts


def run_walk_forward_comparison(ohlcv_series,
                                base_score = 0.5,
                                n_estimators: int = 1000,
                                early_stopping_rounds: int = 50,
                                max_depth: int = 3,
                                learning_rate: float = 0.01,
                                min_child_weight: int = 1,
                                verbose: int = 0):

    close_series = ohlcv_series['Close']
    realised_vol = volatility.parkinson_vol(ohlcv_series)

    garch_forecast = volatility.garch_forecast(close_series, 500)

    benchmarks = {
        'GARCH(1,1)': garch_forecast,
        'Naive (persistence)': volatility.naive_persistent_forecast(realised_vol),
        'Naive (rolling avg)': volatility.naive_avg_forecast(realised_vol, 500),
    }

    forecasts = walk_forward(ohlcv_series, 90,
                            base_score=base_score,                            
                            n_estimators=n_estimators,
                            early_stopping_rounds=early_stopping_rounds,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            min_child_weight=min_child_weight,
                            verbose=verbose)

    per_block_results = {}
    for i, forecast in enumerate(forecasts):
        block_label = f"Block {i + 1} ({forecast.index.min().date()})"
        block_forecasts = benchmarks | {'ML Model': forecast}
        per_block_results[block_label] = compare_forecasts(block_forecasts, realised_vol, forecast.index.min())

    combined = pd.concat(forecasts).rename('ML Model (Walk-Forward)')
    combined_forecasts = benchmarks | {'ML Model (Walk-Forward)': combined}
    combined_results = compare_forecasts(combined_forecasts, realised_vol, combined.index.min().date())

    return per_block_results, combined_results, combined


def plot_walk_forward(realised_vol, forecasts, per_block, asset):

    aligned_realised = realised_vol.loc[forecasts.index]

    ax = aligned_realised.plot(figsize=(15,5))
    forecasts.plot(ax=ax, style='.')
    plt.legend(['Ground Data', 'Predictions'])
    ax.set_title(f'Ground Data and Predictions: {asset}')
    #plt.savefig(f'RealisedVsPredicted({asset}).png')
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
    #plt.savefig(f'MLvsGARCH(QLIKE)({asset}).png')
    plt.show()


def run_hyperparameter_sweep(asset_ohlcv: pd.DataFrame, param_name: str, values: list) -> pd.DataFrame:
    """Run a walk-forward comparison for each value of one hyperparameter, returning a summary table."""

    rows = []
    garch_row = None

    for value in values:
        results_per_block, results_combined, _ = run_walk_forward_comparison(
            asset_ohlcv, base_score='mean', **{param_name: value}
        )

        if garch_row is None:
            garch_row = results_combined['GARCH(1,1)']

        block_rows = [
            {'Block': block, 'Model': model, **metrics}
            for block, models in results_per_block.items()
            for model, metrics in models.items()
        ]
        block_df = pd.DataFrame(block_rows)
        ml_block_stats = block_df[block_df['Model'] == 'ML Model']['qlike'].agg(['median', 'std'])

        rows.append({
            param_name: value,
            'qlike': results_combined['ML Model (Walk-Forward)']['qlike'],
            'mse': results_combined['ML Model (Walk-Forward)']['mse'],
            'mae': results_combined['ML Model (Walk-Forward)']['mae'],
            'qlike_median': ml_block_stats['median'],
            'qlike_std': ml_block_stats['std'],
        })

    summary = pd.DataFrame(rows).set_index(param_name).round(4)

    print(f"GARCH(1,1) benchmark: qlike={garch_row['qlike']:.4f}, "
          f"mse={garch_row['mse']:.4f}, mae={garch_row['mae']:.4f}")
    return summary