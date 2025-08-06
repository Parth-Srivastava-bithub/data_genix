import pandas as pd
import numpy as np
from faker import Faker
from typing import List, Optional, Dict, Any, Callable, Tuple
import datetime as dt
from sklearn.preprocessing import minmax_scale
import warnings

# Suppress SettingWithCopyWarning as we handle it correctly
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)


class DataGenerator:
    """
    An advanced class to generate realistic synthetic datasets for machine learning.
    Refactored for stability, reliability, and clarity.
    """

    def __init__(self, seed: Optional[int] = None):
        self._faker = Faker()
        if seed is not None:
            np.random.seed(seed)
            Faker.seed(seed)

    def _get_faker_method(self, provider_name: str) -> Callable[[], Any]:
        """Fetches a provider method from the Faker instance."""
        try:
            return getattr(self._faker, provider_name)
        except AttributeError:
            raise AttributeError(f"'{provider_name}' is not a valid Faker provider.")

    def _add_missing_values(self, df: pd.DataFrame, col_name: str, fraction: float):
        """Injects missing values (NaN) into a specified column in place."""
        if fraction > 0:
            n_missing = int(len(df) * fraction)
            if n_missing > 0:
                missing_indices = np.random.choice(df.index, n_missing, replace=False)
                df.loc[missing_indices, col_name] = np.nan

    def _inject_outliers(self, df: pd.DataFrame, col_name: str, fraction: float):
        """Injects extreme outliers into a numeric column in place."""
        series = df[col_name]
        # Guard clause: Only run on numeric columns that are not boolean
        if fraction <= 0 or not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return

        n_outliers = int(len(series) * fraction)
        if n_outliers == 0:
            return

        outlier_indices = np.random.choice(series.dropna().index, n_outliers, replace=False)

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1

        # Handle cases with zero IQR to prevent division by zero or non-sensical bounds
        if iqr == 0:
            iqr = series.std() if series.std() != 0 else 1

        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        is_integer_col = pd.api.types.is_integer_dtype(series)

        for idx in outlier_indices:
            if np.random.rand() > 0.5:
                outlier_value = upper_bound * (1 + np.random.uniform(0.5, 2.0))
            else:
                outlier_value = lower_bound * (1 - np.random.uniform(0.5, 2.0))

            # Cast outlier to the original column type to avoid FutureWarning
            df.loc[idx, col_name] = int(outlier_value) if is_integer_col else outlier_value

    def _generate_features(self, df: pd.DataFrame, num_rows: int, configs: dict):
        """Generates all primary features and adds them to the DataFrame."""
        # Numerical Whole
        low_w, high_w = configs.get("numerical_whole_range", (0, 1000))
        for i in range(configs.get("numerical_whole", 0)):
            df[f"numerical_whole_{i}"] = np.random.randint(low_w, high_w, size=num_rows)

        # Decimal
        low_d, high_d = configs.get("decimal_range", (0.0, 100.0))
        for i in range(configs.get("decimal", 0)):
            raw_data = np.random.uniform(low_d, high_d, size=num_rows)
            df[f"decimal_{i}"] = np.round(raw_data, configs.get("custom_configs", {}).get("decimal", {}).get("decimals", 4))

        # Categorical
        cats = configs.get("custom_configs", {}).get("categorical", {}).get("categories", ['Alpha', 'Beta', 'Gamma', 'Delta'])
        for i in range(configs.get("categorical", 0)):
            df[f"categorical_{i}"] = np.random.choice(cats, size=num_rows)

        # Boolean
        for i in range(configs.get("boolean", 0)):
            df[f"boolean_{i}"] = np.random.choice([True, False], size=num_rows)

        # Datetime
        start_date_config = configs.get("custom_configs", {}).get("datetime", {}).get("start_date", "-30y")
        for i in range(configs.get("datetime", 0)):
            df[f"datetime_{i}"] = [self._faker.date_time_between(start_date=start_date_config) for _ in range(num_rows)]

        # Text
        text_style = configs.get("text_style", "sentence")
        for i in range(configs.get("text", 0)):
            if text_style == 'review':
                df[f"text_{i}"] = [self._faker.paragraph(nb_sentences=3) for _ in range(num_rows)]
            elif text_style == 'tweet':
                df[f"text_{i}"] = [f"{self._faker.sentence(nb_words=8)} #{self._faker.word()}" for _ in range(num_rows)]
            else:
                df[f"text_{i}"] = [self._faker.sentence() for _ in range(num_rows)]

        # Other types
        for i in range(configs.get("uuid", 0)): df[f"uuid_{i}"] = [self._faker.uuid4() for _ in range(num_rows)]
        for i in range(configs.get("coordinates", 0)):
            df[f"latitude_{i}"] = [self._faker.latitude() for _ in range(num_rows)]
            df[f"longitude_{i}"] = [self._faker.longitude() for _ in range(num_rows)]

        if configs.get("object_types"):
            for obj_type in configs["object_types"]:
                df[obj_type] = [self._get_faker_method(obj_type)() for _ in range(num_rows)]

    def _apply_correlation(self, df: pd.DataFrame, strength: Optional[float]):
        """Applies correlation to numerical columns if specified."""
        if strength is None or strength < 0 or strength > 1:
            return

        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_cols) < 2:
            return

        base_col_name = numerical_cols[0]
        base_col_scaled = minmax_scale(df[base_col_name])

        for col_name in numerical_cols[1:]:
            noise = minmax_scale(np.random.normal(size=len(df)))
            correlated_data = strength * base_col_scaled + (1 - strength) * noise

            original_min, original_max = df[col_name].min(), df[col_name].max()
            df[col_name] = correlated_data * (original_max - original_min) + original_min

            if pd.api.types.is_integer_dtype(df[col_name].dtype):
                df[col_name] = df[col_name].astype(int)

    def _apply_postprocessing(self, df: pd.DataFrame, configs: dict):
        """Applies missing data and outlier injection."""
        missing_map = {
            "numerical": configs.get("missing_numerical", 0.0),
            "decimal": configs.get("missing_numerical", 0.0),
            "categorical": configs.get("missing_categorical", 0.0),
            "boolean": configs.get("missing_boolean", 0.0),
            "datetime": configs.get("missing_datetime", 0.0),
            "text": configs.get("missing_text", 0.0),
        }

        for col in df.columns:
            # Inject Outliers first, before some values become NaN
            if configs.get("add_outliers", False):
                self._inject_outliers(df, col, configs.get("outlier_fraction", 0.01))

            # Inject Missing Values
            col_type = next((k for k in missing_map if k in col), None)
            if col_type:
                self._add_missing_values(df, col, missing_map[col_type])

    def _generate_target(self, df: pd.DataFrame, target_type: Optional[str]):
        """Generates a target column based on numeric features."""
        if not target_type:
            return

        numeric_features = df.select_dtypes(include=np.number).dropna()
        if numeric_features.empty:
            # Fallback if no numeric data is available for target generation
            if target_type == 'binary': df['target'] = np.random.choice([0, 1], size=len(df))
            elif target_type == 'multi': df['target'] = np.random.choice([0, 1, 2], size=len(df))
            else: df['target'] = np.random.rand(len(df)) * 100
            return

        weights = np.random.uniform(-1, 1, size=numeric_features.shape[1])
        latent_variable = np.dot(numeric_features, weights) + np.random.normal(0, 0.1, size=len(numeric_features))

        target_series = pd.Series(index=df.index, dtype=float)

        if target_type == 'regression':
            target_series.loc[numeric_features.index] = latent_variable
        elif target_type == 'binary':
            # Clip values to prevent overflow with np.exp
            clipped_latent_variable = np.clip(latent_variable, -20, 20) # Adjust clip range as needed
            prob = 1 / (1 + np.exp(-clipped_latent_variable))
            target_series.loc[numeric_features.index] = (prob > 0.5).astype(int)
        elif target_type == 'multi':
            # Use qcut on the series directly to get labels
            target_series.loc[numeric_features.index] = pd.qcut(latent_variable, q=3, labels=[0, 1, 2], duplicates='drop')

        # Fill any missing target values (from rows with NaN in numeric features)
        df['target'] = target_series
        if df['target'].isnull().any():
            fill_value = df['target'].mode()[0] if not df['target'].mode().empty else 0
            # Use .loc for assignment to avoid FutureWarning
            df.loc[df['target'].isnull(), 'target'] = fill_value
            if target_type in ['binary', 'multi']:
                df['target'] = df['target'].astype(int)

    def generate(self, num_rows: int, **kwargs) -> pd.DataFrame:
        """
        Generates a highly customizable Pandas DataFrame for ML tasks.

        Args:
            num_rows: The number of rows to generate.
            **kwargs: A dictionary of configuration options.

        Returns:
            A pandas DataFrame with the generated synthetic data.
        """
        if not isinstance(num_rows, int) or num_rows <= 0:
            raise ValueError("`num_rows` must be a positive integer.")

        df = pd.DataFrame(index=range(num_rows))

        # --- Time Series and Grouping ---
        if kwargs.get('time_series', False):
            start_date = dt.datetime.now() - dt.timedelta(days=num_rows)
            df['timestamp'] = pd.to_datetime(pd.date_range(start=start_date, periods=num_rows, freq='D'))

        if kwargs.get('group_by'):
            group_ids = [self._faker.uuid4() for _ in range(kwargs.get('num_groups', 10))]
            df[kwargs['group_by']] = np.random.choice(group_ids, size=num_rows)

        # --- Generation Steps ---
        self._generate_features(df, num_rows, kwargs)
        self._apply_correlation(df, kwargs.get('correlation_strength'))
        self._apply_postprocessing(df, kwargs)
        self._generate_target(df, kwargs.get('target_type'))

        return df
