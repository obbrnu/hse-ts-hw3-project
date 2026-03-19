import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any


def save_results_csv(
    results: Dict[str, Any],
    results_dir: str = "results",
    filename_prefix: str = None,
) -> Dict[str, str]:
    os.makedirs(results_dir, exist_ok=True)

    if filename_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"experiment_results_{timestamp}"

    saved_files = {}

    if 'model_comparison' in results:
        main_results_df = pd.DataFrame(results['model_comparison'])
        main_file = os.path.join(results_dir, f"{filename_prefix}_main.csv")
        main_results_df.to_csv(main_file, index=False, encoding='utf-8')
        saved_files['main_results'] = main_file

    if 'config' in results:
        config_data = []
        config = results['config']
        for key, value in config.items():
            if isinstance(value, (list, dict)):
                value = str(value)
            config_data.append({'parameter': key, 'value': value})

        config_df = pd.DataFrame(config_data)
        config_file = os.path.join(results_dir, f"{filename_prefix}_config.csv")
        config_df.to_csv(config_file, index=False, encoding='utf-8')
        saved_files['config'] = config_file

    return saved_files

