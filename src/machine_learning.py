"""Machine-learning feature engineering, training, and forecast evaluation."""

from typing import TypeAlias, TypedDict

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

import src.feature_engineering as fe
import src.paths as paths
import src.volatility as volatility

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
    features_df: pd.DataFrame,
    plot: bool = False
) -> pd.Series:
    """Forecast test-period volatility and plot predictions against realised values."""

    x_test = test_data['x_test']

    predictions = pd.Series(reg.predict(x_test), index=x_test.index, name='Prediction')
    features_df = features_df.merge(predictions, how='left', left_index=True, right_index=True)

    if plot:
        ax = features_df[['Parkinson']].plot(figsize=(15,5))
        features_df['Prediction'].plot(ax=ax, style='.')
        plt.legend(['Truth Data', 'Predictions'])
        ax.set_title('Raw Data and Predictions')
        plt.savefig('2014 forecast')
        plt.show()

    forecast = features_df['Prediction'].dropna()

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

    features_df = fe.build_ml_features(ohlcv_series)
    train_data, test_data = split_train_test(features_df)
    model = train_model(train_data)
    ml_forecast = forecast_model(test_data, model, features_df, plot)

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

    features_df = fe.build_ml_features(ohlcv_series).dropna()
    n = len(features_df)

    forecasts = []
    importances = []
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

        model = train_model(train_data)
        forecast = forecast_model(test_data, model, features_df, plot=False)
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