import pandas as pd
import numpy as np
from faker import Faker
from typing import List, Optional, Dict, Any, Callable
import json
import datetime as dt

class DataGenerator:
    """
    A class to generate synthetic datasets for machine learning and data science tasks.
    It can create columns with numerical, categorical, ordinal, and text-based data.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the DataGenerator.

        Args:
            seed (Optional[int]): A seed for the random number generators to ensure
                                  reproducibility. If None, results will be random.
        """
        self._faker = Faker()
        if seed is not None:
            np.random.seed(seed)
            Faker.seed(seed)

    def _get_faker_method(self, provider_name: str) -> Callable[[], Any]:
        """
        Safely retrieves a provider method from the Faker instance.

        Args:
            provider_name (str): The name of the Faker provider (e.g., 'name', 'address').

        Returns:
            Callable[[], Any]: The corresponding Faker method.

        Raises:
            AttributeError: If the provider name is not a valid Faker provider.
        """
        try:
            return getattr(self._faker, provider_name)
        except AttributeError:
            raise AttributeError(
                f"'{provider_name}' is not a valid Faker provider. "
                "Please check the Faker documentation for available providers."
            )

    def generate(
        self,
        num_rows: int,
        numerical_whole: int = 0,
        decimal: int = 0,
        categorical: int = 0,
        ordinal: int = 0,
        boolean: int = 0,
        datetime: int = 0,
        text: int = 0,
        uuid: int = 0,
        coordinates: int = 0,
        ip_address: int = 0,
        phone_number: int = 0,
        url: int = 0,
        json_string: int = 0,
        object_types: Optional[List[str]] = None,
        custom_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Generates a Pandas DataFrame with specified column types and quantities.

        Args:
            num_rows (int): The number of rows (samples) in the dataset.
            numerical_whole (int): Number of integer columns.
            decimal (int): Number of float columns.
            categorical (int): Number of categorical columns.
            ordinal (int): Number of ordered categorical columns.
            boolean (int): Number of True/False columns.
            datetime (int): Number of datetime columns.
            text (int): Number of sentence-based text columns.
            uuid (int): Number of UUID columns.
            coordinates (int): Number of latitude/longitude column pairs.
            ip_address (int): Number of IPv4 address columns.
            phone_number (int): Number of phone number columns.
            url (int): Number of URL columns.
            json_string (int): Number of JSON-formatted string columns.
            object_types (Optional[List[str]]): List of strings for Faker-based columns.
            custom_configs (Optional[Dict[str, Dict[str, Any]]]): Advanced configuration.

        Returns:
            pd.DataFrame: The generated synthetic dataset.
        """
        if not isinstance(num_rows, int) or num_rows <= 0:
            raise ValueError("`num_rows` must be a positive integer.")

        if object_types is None: object_types = []
        if custom_configs is None: custom_configs = {}

        data = {}
        all_col_types = [
            numerical_whole, decimal, categorical, ordinal, boolean, datetime,
            text, uuid, coordinates, ip_address, phone_number, url, json_string,
            len(object_types)
        ]
        if sum(all_col_types) == 0:
            print("Warning: No columns specified to generate. Returning an empty DataFrame.")
            return pd.DataFrame(index=range(num_rows))

        # --- Generate Numerical (Whole) Columns ---
        config = custom_configs.get("numerical_whole", {})
        for i in range(numerical_whole):
            data[f"numerical_whole_{i}"] = np.random.randint(
                config.get("low", 0), config.get("high", 1000), size=num_rows
            )

        # --- Generate Numerical (Decimal) Columns ---
        config = custom_configs.get("decimal", {})
        for i in range(decimal):
            raw_data = np.random.uniform(
                config.get("low", 0.0), config.get("high", 100.0), size=num_rows
            )
            data[f"decimal_{i}"] = np.round(raw_data, config.get("decimals", 4))

        # --- Generate Categorical Columns ---
        config = custom_configs.get("categorical", {})
        cats = config.get("categories", ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'])
        for i in range(categorical):
            data[f"categorical_{i}"] = np.random.choice(cats, size=num_rows)

        # --- Generate Ordinal Columns ---
        config = custom_configs.get("ordinal", {})
        ord_cats = config.get("categories", ['Low', 'Medium', 'High', 'Critical'])
        cat_type = pd.api.types.CategoricalDtype(categories=ord_cats, ordered=True)
        for i in range(ordinal):
            series = pd.Series(np.random.choice(ord_cats, size=num_rows))
            data[f"ordinal_{i}"] = series.astype(cat_type)

        # --- Generate Boolean Columns ---
        for i in range(boolean):
            data[f"boolean_{i}"] = np.random.choice([True, False], size=num_rows)

        # --- Generate Datetime Columns ---
        config = custom_configs.get("datetime", {})
        start_date = config.get("start_date", "-30y")
        end_date = config.get("end_date", "now")
        for i in range(datetime):
            data[f"datetime_{i}"] = [
                self._faker.date_time_between(start_date=start_date, end_date=end_date)
                for _ in range(num_rows)
            ]

        # --- Generate Text Columns ---
        config = custom_configs.get("text", {})
        nb_words = config.get("nb_words", 10)
        for i in range(text):
            data[f"text_{i}"] = [self._faker.sentence(nb_words=nb_words) for _ in range(num_rows)]

        # --- Generate UUID Columns ---
        for i in range(uuid):
            data[f"uuid_{i}"] = [self._faker.uuid4() for _ in range(num_rows)]

        # --- Generate Coordinate Columns ---
        for i in range(coordinates):
            data[f"latitude_{i}"] = [self._faker.latitude() for _ in range(num_rows)]
            data[f"longitude_{i}"] = [self._faker.longitude() for _ in range(num_rows)]

        # --- Generate IP Address Columns ---
        for i in range(ip_address):
            data[f"ip_address_{i}"] = [self._faker.ipv4() for _ in range(num_rows)]
            
        # --- Generate Phone Number Columns ---
        for i in range(phone_number):
            data[f"phone_number_{i}"] = [self._faker.phone_number() for _ in range(num_rows)]

        # --- Generate URL Columns ---
        for i in range(url):
            data[f"url_{i}"] = [self._faker.url() for _ in range(num_rows)]
            
        # --- Generate JSON String Columns ---
        for i in range(json_string):
            json_data = []
            for _ in range(num_rows):
                nested_dict = {
                    "user_id": self._faker.uuid4(),
                    "status": np.random.choice(['active', 'pending', 'inactive']),
                    "timestamp": dt.datetime.now().isoformat()
                }
                json_data.append(json.dumps(nested_dict))
            data[f"json_string_{i}"] = json_data

        # --- Generate Object (Faker) Columns ---
        for obj_type in object_types:
            if obj_type in data:
                print(f"Warning: Column name '{obj_type}' already exists. Skipping this object type.")
                continue
            faker_method = self._get_faker_method(obj_type)
            data[obj_type] = [faker_method() for _ in range(num_rows)]

        return pd.DataFrame(data)

# ==============================================================================
# Example Usage (you can put this in a separate test file, e.g., test_generator.py)
# ==============================================================================

if __name__ == '__main__':
    # This block runs only when the script is executed directly

    print("--- Basic Usage Example ---")
    generator = DataGenerator(seed=42)
    try:
        df_basic = generator.generate(
            num_rows=10,
            numerical_whole=1,
            decimal=1,
            categorical=1,
            object_types=['name', 'job']
        )
        print(df_basic.head())

        print("\n--- NEW: Comprehensive Data Types Example ---")
        df_full = generator.generate(
            num_rows=8,
            numerical_whole=1,
            decimal=1,
            categorical=1,
            ordinal=1,
            boolean=1,
            datetime=1,
            text=1,
            uuid=1,
            coordinates=1,
            ip_address=1,
            phone_number=1,
            url=1,
            json_string=1,
            object_types=['company', 'country']
        )
        pd.set_option('display.max_columns', None) # Show all columns
        print(df_full)
        print("\nDataFrame Info:")
        df_full.info()

    except (ValueError, AttributeError) as e:
        print(f"\nAn error occurred: {e}")
