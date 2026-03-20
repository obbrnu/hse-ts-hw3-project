import pandas as pd
import numpy as np
from typing import List, Dict, Any


class FeatureGenerator:
    def __init__(
        self,
        max_lags: int = 24,
        seasonal_periods: List[int] = None,
        fourier_terms: List[int] = None,
    ):
        self.max_lags = max_lags
        self.seasonal_periods = seasonal_periods or [12]
        self.fourier_terms = fourier_terms or [2, 4, 6]


    def add_lags(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        df_with_lags = df.copy()

        for lag in lags:
            df_with_lags[f'lag_{lag}'] = df_with_lags.groupby('unique_id')['y'].shift(lag)

        return df_with_lags


    def add_seasonal_lags(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        df_with_seasonal = df.copy()

        for period in periods:
            for i in range(1, 4):
                lag = period * i
                df_with_seasonal[f'seasonal_lag_{period}_{i}'] = (
                    df_with_seasonal.groupby('unique_id')['y'].shift(lag)
                )

        return df_with_seasonal


    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_calendar = df.copy()

        df_with_calendar['month'] = ((df_with_calendar['ds'] - 1) % 12) + 1

        df_with_calendar['month_sin'] = np.sin(2 * np.pi * df_with_calendar['month'] / 12)
        df_with_calendar['month_cos'] = np.cos(2 * np.pi * df_with_calendar['month'] / 12)

        df_with_calendar['quarter'] = ((df_with_calendar['month'] - 1) // 3) + 1
        df_with_calendar['quarter_sin'] = np.sin(2 * np.pi * df_with_calendar['quarter'] / 4)
        df_with_calendar['quarter_cos'] = np.cos(2 * np.pi * df_with_calendar['quarter'] / 4)

        return df_with_calendar


    def add_fourier_features(
        self,
        df: pd.DataFrame,
        period: int = 12,
        fourier_terms: List[int] = None,
    ) -> pd.DataFrame:
        if fourier_terms is None:
            fourier_terms = self.fourier_terms

        df_with_fourier = df.copy()

        for k in fourier_terms:
            for i in range(1, k + 1):
                df_with_fourier[f'fourier_sin_{period}_{i}'] = np.sin(
                    2 * np.pi * i * df_with_fourier['ds'] / period
                )
                df_with_fourier[f'fourier_cos_{period}_{i}'] = np.cos(
                    2 * np.pi * i * df_with_fourier['ds'] / period
                )

        return df_with_fourier


    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = None,
    ) -> pd.DataFrame:
        if windows is None:
            windows = [3, 6, 12]

        df_with_rolling = df.copy()

        for window in windows:
            df_with_rolling[f'rolling_mean_{window}'] = (
                df_with_rolling.groupby('unique_id')['y']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

            df_with_rolling[f'rolling_std_{window}'] = (
                df_with_rolling.groupby('unique_id')['y']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )

        return df_with_rolling


    def generate_features(
        self,
        df: pd.DataFrame,
        feature_config: Dict[str, Any],
    ) -> pd.DataFrame:
        df_features = df.copy()

        if feature_config.get('use_lags', False):
            lags = list(range(1, min(13, self.max_lags + 1)))
            df_features = self.add_lags(df_features, lags)

        if feature_config.get('use_seasonal_lags', False):
            df_features = self.add_seasonal_lags(df_features, self.seasonal_periods)

        if feature_config.get('use_calendar', False):
            df_features = self.add_calendar_features(df_features)

        if feature_config.get('use_fourier', False):
            for period in self.seasonal_periods:
                df_features = self.add_fourier_features(df_features, period)

        df_features = self.add_rolling_features(df_features)

        df_features['trend'] = df_features.groupby('unique_id')['ds'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
        )

        return df_features


    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude_cols = {'unique_id', 'ds', 'y'}
        return [col for col in df.columns if col not in exclude_cols]
