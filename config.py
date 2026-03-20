from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ExperimentConfig:
    n_series: int = 50
    min_seasonality_strength: float = 0.1
    max_seasonality_strength: float = 1.0
    seasonal_period: int = 12
    forecast_horizons: List[int] = field(default_factory=lambda: [6, 12, 18])
    max_lags: int = 24
    seasonal_periods: List[int] = None
    fourier_terms: List[int] = field(default_factory=lambda: [2, 4, 6])
    catboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'iterations': 200,
        'learning_rate': 0.1,
        'depth': 4,
        'random_seed': 42,
        'verbose': False,
    })
    cv_folds: int = 3
    walkforward_windows: int = 3
    validation_min_train_size: int = 50


    def __post_init__(self):
        if self.seasonal_periods is None:
            self.seasonal_periods = [self.seasonal_period, self.seasonal_period * 2]

CONFIG = ExperimentConfig()

FEATURE_SETS = {
    'lags_only': {
        'use_lags': True,
        'use_seasonal_lags': False,
        'use_calendar': False,
        'use_fourier': False
    },
    'lags_seasonal': {
        'use_lags': True,
        'use_seasonal_lags': True,
        'use_calendar': False,
        'use_fourier': False
    },
    'lags_calendar': {
        'use_lags': True,
        'use_seasonal_lags': False,
        'use_calendar': True,
        'use_fourier': False
    },
    'lags_fourier': {
        'use_lags': True,
        'use_seasonal_lags': False,
        'use_calendar': False,
        'use_fourier': True
    },
    'lags_seasonal_calendar': {
        'use_lags': True,
        'use_seasonal_lags': True,
        'use_calendar': True,
        'use_fourier': False
    },
    'lags_seasonal_fourier': {
        'use_lags': True,
        'use_seasonal_lags': True,
        'use_calendar': False,
        'use_fourier': True
    },
    'all_features': {
        'use_lags': True,
        'use_seasonal_lags': True,
        'use_calendar': True,
        'use_fourier': True
    }
}
