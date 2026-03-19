import numpy as np
from typing import Dict


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs(actual - predicted))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return 100 * np.mean(2 * np.abs(predicted - actual) / denominator)


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return 100.0 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))


def calculate_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean((actual - predicted) ** 2)


def calculate_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        'mae': calculate_mae(actual, predicted),
        'rmse': calculate_rmse(actual, predicted),
        'smape': calculate_smape(actual, predicted),
        'mape': calculate_mape(actual, predicted),
        'mse': calculate_mse(actual, predicted),
    }


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        'mae': calculate_mae(actual, predicted),
        'rmse': calculate_rmse(actual, predicted),
        'smape': calculate_smape(actual, predicted),
    }
