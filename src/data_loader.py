import os
import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.seasonal import STL
from typing import Tuple, List


class M4DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.base_url = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/"

    def download_m4_data(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)

        files_to_download = [
            "Train/Monthly-train.csv",
            "Test/Monthly-test.csv",
            "M4-info.csv"
        ]

        for file_path in files_to_download:
            url = self.base_url + file_path
            local_path = os.path.join(self.data_dir, os.path.basename(file_path))

            if not os.path.exists(local_path):
                response = requests.get(url)
                response.raise_for_status()

                with open(local_path, 'wb') as f:
                    f.write(response.content)

    def load_m4_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.download_m4_data()

        train_path = os.path.join(self.data_dir, "Monthly-train.csv")
        test_path = os.path.join(self.data_dir, "Monthly-test.csv")
        info_path = os.path.join(self.data_dir, "M4-info.csv")

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        info_data = pd.read_csv(info_path)

        info_data['StartingDate'] = pd.to_datetime(info_data['StartingDate'], format='mixed', dayfirst=True, errors='coerce')

        return train_data, test_data, info_data

    def select_seasonal_series(
        self,
        train_data: pd.DataFrame,
        info_data: pd.DataFrame,
        n_series: int = 150,
        min_seasonality: float = 0.3,
        max_seasonality: float = 1.0,
    ) -> List[str]:
        monthly_info = info_data[info_data['SP'] == 'Monthly'].copy()

        valid_series = []

        for idx, row in monthly_info.head(n_series * 3).iterrows():
            series_id = row['M4id']

            series_row = train_data[train_data['V1'] == series_id]
            if len(series_row) == 0:
                continue

            values = series_row.iloc[0, 1:].dropna().values

            if len(values) >= 36:
                seasonality_strength = self._calculate_seasonality_strength(values)
                if min_seasonality <= seasonality_strength <= max_seasonality:
                    valid_series.append(series_id)

            if len(valid_series) >= n_series:
                break

        return valid_series
    

    def _calculate_seasonality_strength(self, series: np.ndarray, period: int = 12) -> float:
        if len(series) < period * 3:
            return 0.0
            
        series_float = pd.to_numeric(series, errors='coerce')
        series_clean = series_float[pd.notna(series_float)]
        
        series_clean = series_clean.values if hasattr(series_clean, 'values') else series_clean
        
        if np.std(series_clean) < 1e-8:
            return 0.0
        
        stl = STL(series_clean, period=period, seasonal=13)
        result = stl.fit()
        
        seasonal = result.seasonal
        remainder = result.resid
        
        var_remainder = np.var(remainder)
        var_seasonal_plus_remainder = np.var(seasonal + remainder)
        
        if var_seasonal_plus_remainder < 1e-8:
            return 0.0
        
        seasonality_strength = max(0, 1 - var_remainder / var_seasonal_plus_remainder)
        return float(seasonality_strength)


    def prepare_data(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        selected_ids: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        def to_long_format(data: pd.DataFrame, selected_ids: List[str], is_test: bool = False) -> pd.DataFrame:
            long_data = []

            for series_id in selected_ids:
                series_row = data[data['V1'] == series_id]
                if len(series_row) == 0:
                    continue

                values = series_row.iloc[0, 1:].dropna().values

                if is_test:
                    train_row = train_data[train_data['V1'] == series_id]
                    if len(train_row) > 0:
                        train_values = train_row.iloc[0, 1:].dropna().values
                        start_idx = len(train_values)
                    else:
                        start_idx = 0
                else:
                    start_idx = 0

                for i, value in enumerate(values):
                    long_data.append({
                        'unique_id': series_id,
                        'ds': start_idx + i + 1,
                        'y': float(value),
                    })

            return pd.DataFrame(long_data)

        train_long = to_long_format(train_data, selected_ids, is_test=False)
        test_long = to_long_format(test_data, selected_ids, is_test=True)

        return train_long, test_long
