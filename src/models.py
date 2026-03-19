import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoTheta, AutoETS
from typing import Dict, Any, List, Tuple


class BaselineModels:
    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = seasonal_period
        self.models = {}


    def fit_predict(
        self,
        train_data: pd.DataFrame,
        forecast_horizon: int,
    ) -> Dict[str, pd.DataFrame]:
        results = {}

        models = [
            Naive(),
            SeasonalNaive(season_length=self.seasonal_period),
            AutoTheta(season_length=self.seasonal_period),
            AutoETS(season_length=self.seasonal_period)
        ]

        sf = StatsForecast(
            models=models,
            freq=1,
            n_jobs=1,
        )

        forecasts = sf.forecast(
            df=train_data,
            h=forecast_horizon,
            level=[95],
        )

        for model in models:
            model_name = model.__class__.__name__
            if model_name in forecasts.columns:
                model_forecasts = forecasts[['unique_id', 'ds', model_name]].copy()
                model_forecasts.rename(columns={model_name: 'y_pred'}, inplace=True)
                results[model_name] = model_forecasts

        return results


class CatBoostModel:
    def __init__(self, catboost_params: Dict[str, Any] = None):
        self.catboost_params = catboost_params or {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'random_seed': 42,
            'verbose': False,
        }
        self.model = None
        self.feature_columns = None


    def prepare_training_data(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        df_clean = df.dropna(subset=['y'] + feature_columns)

        X = df_clean[feature_columns].values
        y = df_clean['y'].values

        return X, y


    def fit(
        self,
        train_data: pd.DataFrame,
        feature_columns: List[str],
        validation_split: float = 0.2,
    ) -> None:
        feature_columns_with_id = ['unique_id'] + feature_columns
        self.feature_columns = feature_columns_with_id

        df_clean = train_data.dropna(subset=['y'] + feature_columns_with_id)

        if len(df_clean) == 0:
            raise ValueError("No valid training data after removing NaN values")

        X = df_clean[feature_columns_with_id]
        y = df_clean['y'].values

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.model = CatBoostRegressor(**self.catboost_params)

        cat_features = [0]

        if len(X_val) > 0:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features,
                early_stopping_rounds=100,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, cat_features=cat_features, verbose=False)


    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")

        test_clean = test_data.dropna(subset=self.feature_columns)

        if len(test_clean) == 0:
            return np.array([])

        X_test = test_clean[self.feature_columns]
        predictions = self.model.predict(X_test)

        return np.asarray(predictions, dtype=float)
