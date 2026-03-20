import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import CONFIG, FEATURE_SETS
from src import (
    M4DataLoader,
    FeatureGenerator,
    BaselineModels,
    CatBoostModel,
    save_results_csv,
    create_time_series_splits,
    calculate_metrics,
    aggregate_cv_results,
)


def main():
    data_loader = M4DataLoader()
    feature_generator = FeatureGenerator(
        max_lags=CONFIG.max_lags,
        seasonal_periods=CONFIG.seasonal_periods,
        fourier_terms=CONFIG.fourier_terms,
    )
    baseline_models = BaselineModels(seasonal_period=12)

    train_raw, test_raw, info_data = data_loader.load_m4_data()

    selected_ids = data_loader.select_seasonal_series(
        train_raw,
        info_data,
        n_series=CONFIG.n_series,
        min_seasonality=CONFIG.min_seasonality_strength,
        max_seasonality=CONFIG.max_seasonality_strength,
    )

    if len(selected_ids) == 0:
        return

    train_data, test_data = data_loader.prepare_data(train_raw, test_raw, selected_ids)

    complete_dataset = pd.concat([train_data, test_data], ignore_index=True)

    all_results = {
        'config': CONFIG.__dict__,
        'selected_series': selected_ids,
        'n_series': len(selected_ids),
        'baseline_results': {},
        'feature_experiments': {},
        'model_comparison': [],
        'walkforward_validation': {},
    }

    cv_splits = create_time_series_splits(train_data, test_data, n_splits=CONFIG.cv_folds)

    for horizon in CONFIG.forecast_horizons:
        all_folds_results = {}

        for fold_idx, (fold_train, fold_test) in enumerate(cv_splits, 1):
            fold_results = {}

            horizon_test = fold_test.groupby('unique_id').head(horizon).reset_index(drop=True)

            fold_series_ids = fold_train['unique_id'].unique()

            for series_id in fold_series_ids:
                series_train = fold_train[fold_train['unique_id'] == series_id].copy()
                series_test = horizon_test[horizon_test['unique_id'] == series_id].copy()

                if len(series_train) < 20 or len(series_test) == 0:
                    continue

                baseline_predictions = baseline_models.fit_predict(series_train, len(series_test))
                predictions = {}
                for model_name, pred_df in baseline_predictions.items():
                    if 'y_pred' in pred_df.columns and len(pred_df) > 0:
                        predictions[model_name.lower()] = pred_df['y_pred'].values

                actual = series_test['y'].values

                for model_name, pred in predictions.items():
                    if len(pred) == len(actual):
                        metrics = calculate_metrics(actual, pred)

                        if model_name not in fold_results:
                            fold_results[model_name] = []

                        fold_results[model_name].append({
                            'series_id': series_id,
                            'mae': metrics['mae'],
                            'rmse': metrics['rmse'],
                            'smape': metrics['smape'],
                            'fold': fold_idx
                        })

            for model_name, results in fold_results.items():
                if model_name not in all_folds_results:
                    all_folds_results[model_name] = []
                all_folds_results[model_name].extend(results)

        for model_name, results in all_folds_results.items():
            if results:
                result_row = aggregate_cv_results(
                    results,
                    f'Simple_{model_name}',
                    horizon,
                    len(cv_splits),
                    'baseline_cv'
                )
                all_results['model_comparison'].append(result_row)

        for feature_name, feature_config in FEATURE_SETS.items():
            feature_fold_results = []

            for fold_idx, (fold_train, fold_test) in enumerate(cv_splits, 1):
                horizon_test = fold_test.groupby('unique_id').head(horizon).reset_index(drop=True)

                train_features = feature_generator.generate_features(fold_train, feature_config)
                test_features = feature_generator.generate_features(
                    pd.concat([fold_train, horizon_test], ignore_index=True),
                    feature_config
                )

                test_features = test_features[
                    test_features[['unique_id', 'ds']].apply(tuple, axis=1).isin(
                        horizon_test[['unique_id', 'ds']].apply(tuple, axis=1)
                    )
                ].reset_index(drop=True)

                if len(train_features) == 0 or len(test_features) == 0:
                    continue

                feature_columns = feature_generator.get_feature_columns(train_features)

                catboost_model = CatBoostModel(CONFIG.catboost_params)

                catboost_model.fit(train_features, feature_columns)
                predictions = catboost_model.predict(test_features)

                if isinstance(predictions, np.ndarray) and len(predictions) == len(test_features):
                    actual = test_features['y'].values
                    metrics = calculate_metrics(actual, predictions)

                    feature_fold_results.append({
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'smape': metrics['smape'],
                        'fold': fold_idx,
                        'n_features': len(feature_columns)
                    })

            if feature_fold_results:
                result_row = aggregate_cv_results(
                    feature_fold_results,
                    f'CatBoost_{feature_name}',
                    horizon,
                    len(cv_splits),
                    'ml_cv'
                )
                all_results['model_comparison'].append(result_row)

    _ = save_results_csv(all_results)

    return all_results


if __name__ == "__main__":
    main()
