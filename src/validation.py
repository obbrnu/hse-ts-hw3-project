import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

from .metrics import calculate_metrics


def aggregate_cv_results(
    results: List[Dict],
    model_name: str,
    horizon: int,
    n_folds: int,
    result_type: str = 'baseline_cv',
) -> Dict[str, Any]:
    df_results = pd.DataFrame(results)

    result_row = {
        'model': model_name,
        'horizon': horizon,
        'type': result_type,
        'avg_sMAPE': df_results['smape'].mean(),
        'std_sMAPE': df_results['smape'].std(),
        'avg_MAE': df_results['mae'].mean(),
        'avg_RMSE': df_results['rmse'].mean(),
        'n_folds': n_folds
    }

    if result_type == 'baseline_cv' and 'series_id' in df_results.columns:
        result_row['n_series'] = len(df_results['series_id'].unique())
    elif result_type == 'ml_cv' and 'n_features' in df_results.columns:
        result_row['n_features'] = df_results['n_features'].iloc[0]

    return result_row


def create_time_series_splits(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    n_splits: int = 3,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    splits = []

    all_data = pd.concat([train_data, test_data], ignore_index=True)

    series_ids = all_data['unique_id'].unique()

    for i in range(n_splits):
        train_indices = []
        test_indices = []

        for series_id in series_ids:
            series_data = all_data[all_data['unique_id'] == series_id].sort_values('ds').reset_index()
            total_len = len(series_data)

            if total_len < 30:
                continue

            test_size = max(6, total_len // 5)

            test_end = total_len - (i * test_size)
            test_start = test_end - test_size
            train_end = test_start

            if train_end < 20:
                continue

            train_idx = series_data.iloc[:train_end]['index'].tolist()
            test_idx = series_data.iloc[test_start:test_end]['index'].tolist()

            train_indices.extend(train_idx)
            test_indices.extend(test_idx)

        if len(train_indices) > 0 and len(test_indices) > 0:
            train_fold = all_data.loc[train_indices].copy()
            test_fold = all_data.loc[test_indices].copy()
            splits.append((train_fold, test_fold))

    return splits
