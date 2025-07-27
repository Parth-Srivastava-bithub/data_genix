import pandas as pd
import numpy as np
from faker import Faker
from typing import List, Optional, Dict, Any, Callable, Tuple
import json
import datetime as dt
from sklearn.preprocessing import minmax_scale

class DataGenerator:
    """
    An advanced class to generate realistic synthetic datasets for machine learning.
    """

    def __init__(self, seed: Optional[int] = None):
        self._faker = Faker()
        if seed is not None:
            np.random.seed(seed)
            Faker.seed(seed)

    def _get_faker_method(self, provider_name: str) -> Callable[[], Any]:
        try:
            return getattr(self._faker, provider_name)
        except AttributeError:
            raise AttributeError(f"'{provider_name}' is not a valid Faker provider.")

    def _add_missing_values(self, series: pd.Series, fraction: float) -> pd.Series:
        if fraction > 0:
            n_missing = int(len(series) * fraction)
            missing_indices = np.random.choice(series.index, n_missing, replace=False)
            series.loc[missing_indices] = np.nan
        return series

    def _inject_outliers(self, series: pd.Series, fraction: float) -> pd.Series:
        if fraction > 0 and pd.api.types.is_numeric_dtype(series):
            n_outliers = int(len(series) * fraction)
            outlier_indices = np.random.choice(series.index, n_outliers, replace=False)
            
            # Generate extreme outliers
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            for idx in outlier_indices:
                if np.random.rand() > 0.5:
                    series.loc[idx] = upper_bound * (1 + np.random.uniform(0.5, 2.0))
                else:
                    series.loc[idx] = lower_bound * (1 - np.random.uniform(0.5, 2.0))
        return series

    def generate(
        self,
        num_rows: int,
        # Basic feature types
        numerical_whole: int = 0,
        decimal: int = 0,
        categorical: int = 0,
        ordinal: int = 0,
        boolean: int = 0,
        datetime: int = 0,
        text: int = 0,
        uuid: int = 0,
        coordinates: int = 0,
        object_types: Optional[List[str]] = None,
        # Advanced features
        target_type: Optional[str] = None,
        correlation_strength: Optional[float] = None,
        group_by: Optional[str] = None,
        num_groups: int = 10,
        time_series: bool = False,
        add_outliers: bool = False,
        outlier_fraction: float = 0.01,
        # Missing data control
        missing_numerical: float = 0.0,
        missing_categorical: float = 0.0,
        missing_boolean: float = 0.0,
        missing_datetime: float = 0.0,
        missing_text: float = 0.0,
        # Customization
        numerical_whole_range: Optional[Tuple[int, int]] = None,
        decimal_range: Optional[Tuple[float, float]] = None,
        text_style: str = 'sentence',
        custom_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Generates a highly customizable Pandas DataFrame for ML tasks.
        """
        if not isinstance(num_rows, int) or num_rows <= 0:
            raise ValueError("`num_rows` must be a positive integer.")

        data = {}
        if custom_configs is None: custom_configs = {}

        # --- Time Series Mode ---
        if time_series:
            start_date = dt.datetime.now() - dt.timedelta(days=num_rows)
            data['timestamp'] = pd.to_datetime(pd.date_range(start=start_date, periods=num_rows, freq='D'))
        
        # --- Grouped Data Simulation ---
        if group_by:
            group_ids = [self._faker.uuid4() for _ in range(num_groups)]
            data[group_by] = np.random.choice(group_ids, size=num_rows)

        # --- Feature Generation ---
        num_config = custom_configs.get("numerical_whole", {})
        low_w, high_w = numerical_whole_range if numerical_whole_range else (num_config.get("low", 0), num_config.get("high", 1000))
        for i in range(numerical_whole):
            data[f"numerical_whole_{i}"] = np.random.randint(low_w, high_w, size=num_rows)

        dec_config = custom_configs.get("decimal", {})
        low_d, high_d = decimal_range if decimal_range else (dec_config.get("low", 0.0), dec_config.get("high", 100.0))
        for i in range(decimal):
            raw_data = np.random.uniform(low_d, high_d, size=num_rows)
            data[f"decimal_{i}"] = np.round(raw_data, dec_config.get("decimals", 4))
            
        # --- Feature Correlation ---
        numerical_cols = [col for col in data if "numerical" in col or "decimal" in col]
        if correlation_strength and len(numerical_cols) > 1:
            base_col_name = numerical_cols[0]
            base_col_scaled = minmax_scale(data[base_col_name])
            for col_name in numerical_cols[1:]:
                noise = minmax_scale(np.random.normal(size=num_rows))
                correlated_data = correlation_strength * base_col_scaled + (1 - correlation_strength) * noise
                # Rescale to original column's range
                original_min, original_max = data[col_name].min(), data[col_name].max()
                data[col_name] = correlated_data * (original_max - original_min) + original_min
                if "whole" in col_name: data[col_name] = data[col_name].astype(int)

        # --- Other Feature Types ---
        cat_config = custom_configs.get("categorical", {})
        cats = cat_config.get("categories", ['Alpha', 'Beta', 'Gamma', 'Delta'])
        for i in range(categorical): data[f"categorical_{i}"] = np.random.choice(cats, size=num_rows)
        
        for i in range(boolean): data[f"boolean_{i}"] = np.random.choice([True, False], size=num_rows)
        
        dt_config = custom_configs.get("datetime", {})
        for i in range(datetime): data[f"datetime_{i}"] = [self._faker.date_time_between(start_date=dt_config.get("start_date", "-30y")) for _ in range(num_rows)]

        for i in range(text):
            if text_style == 'review':
                data[f"text_{i}"] = [self._faker.paragraph(nb_sentences=3) for _ in range(num_rows)]
            elif text_style == 'tweet':
                data[f"text_{i}"] = [f"{self._faker.sentence(nb_words=8)} #{self._faker.word()}" for _ in range(num_rows)]
            else: # sentence
                data[f"text_{i}"] = [self._faker.sentence() for _ in range(num_rows)]

        for i in range(uuid): data[f"uuid_{i}"] = [self._faker.uuid4() for _ in range(num_rows)]
        for i in range(coordinates):
            data[f"latitude_{i}"] = [self._faker.latitude() for _ in range(num_rows)]
            data[f"longitude_{i}"] = [self._faker.longitude() for _ in range(num_rows)]

        if object_types:
            for obj_type in object_types:
                data[obj_type] = [self._get_faker_method(obj_type)() for _ in range(num_rows)]
        
        df = pd.DataFrame(data)

        # --- Post-processing: Outliers and Missing Data ---
        for col in df.columns:
            if add_outliers: df[col] = self._inject_outliers(df[col], outlier_fraction)
            if "numerical" in col or "decimal" in col: df[col] = self._add_missing_values(df[col], missing_numerical)
            if "categorical" in col: df[col] = self._add_missing_values(df[col], missing_categorical)
            if "boolean" in col: df[col] = self._add_missing_values(df[col], missing_boolean)
            if "datetime" in col: df[col] = self._add_missing_values(df[col], missing_datetime)
            if "text" in col: df[col] = self._add_missing_values(df[col], missing_text)

        # --- Target Column Generation ---
        if target_type:
            numeric_features = df.select_dtypes(include=np.number).dropna()
            if not numeric_features.empty:
                # Create a latent variable from a weighted sum of numeric features
                weights = np.random.uniform(-1, 1, size=numeric_features.shape[1])
                latent_variable = np.dot(numeric_features, weights) + np.random.normal(0, 0.1, size=len(numeric_features))
                
                if target_type == 'regression':
                    df['target'] = latent_variable
                elif target_type == 'binary':
                    prob = 1 / (1 + np.exp(-latent_variable)) # Sigmoid
                    df['target'] = (prob > 0.5).astype(int)
                elif target_type == 'multi':
                    prob = pd.qcut(latent_variable, q=3, labels=[0, 1, 2], duplicates='drop')
                    df['target'] = prob
                # Handle rows that had NaNs in numeric features
                if df['target'].isnull().any():
                    df['target'].fillna(df['target'].mode()[0], inplace=True)

        return df

# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == '__main__':
    print("--- The Ultimate DataGenix Example ---")
    generator = DataGenerator(seed=42)
    
    try:
        df = generator.generate(
            num_rows=1000,
            numerical_whole=3,
            decimal=2,
            categorical=2,
            boolean=1,
            text=1,
            uuid=1,
            object_types=['name', 'email'],
            target_type='binary',
            missing_numerical=0.05,
            missing_categorical=0.1,
            correlation_strength=0.7,
            group_by='customer_id',
            num_groups=50,
            time_series=True,
            numerical_whole_range=(100, 999),
            add_outliers=True,
            outlier_fraction=0.02,
            text_style='review'
        )
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)

        print(df.head())
        print("\n--- DataFrame Info ---")
        df.info()
        
        print("\n--- Data Quality Checks ---")
        print(f"\nMissing values injected:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        
        numeric_cols_for_corr = [c for c in df.columns if 'numerical' in c or 'decimal' in c]
        if len(numeric_cols_for_corr) > 1:
            print(f"\nCorrelation matrix for numerical features:\n{df[numeric_cols_for_corr].corr()}")
            
        if 'target' in df.columns:
            print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True)}")

    except Exception as e:
        print(f"\nAn error occurred during generation: {e}")
