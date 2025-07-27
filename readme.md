# DataGenix

An advanced and robust library for generating synthetic datasets for machine learning and deep learning projects. Go from idea to prototype in seconds without data acquisition bottlenecks.

## Installation

Install from PyPI (once published):
```bash
pip install datagenix
```

Or install directly from the repository:
```bash
git clone [https://github.com/yourusername/datagenix.git](https://github.com/yourusername/datagenix.git)
cd datagenix
pip install .
```

## Ultimate Usage Example

Generate a complex, realistic dataset for a binary classification task with a single, intuitive command:

```python
from datagenix import DataGenerator

generator = DataGenerator(seed=42)

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

print(df.head())
print(df.info())
```

## Advanced Features

- **Target Generation**: Automatically create a `target` column for `binary`, `multi-class`, or `regression` tasks that is logically correlated with the features.
- **Missing Data**: Inject missing values (`NaN`) into any feature type with precise fractional control (e.g., `missing_numerical=0.1`).
- **Feature Correlation**: Create linear dependencies between numerical features with adjustable `correlation_strength`.
- **Grouped Data**: Simulate real-world scenarios like customer data by grouping rows with a common ID using `group_by` and `num_groups`.
- **Time Series**: Generate a chronologically sorted `timestamp` column for time-dependent modeling.
- **Outlier Injection**: Introduce extreme values into numerical columns to test model robustness using `add_outliers` and `outlier_fraction`.
- **Custom Ranges**: Define exact `(min, max)` ranges for numerical columns.
- **Text Styles**: Generate varied text content like `review`, `tweet`, or standard `sentence`.