# DataGenix

A robust and simple library for generating synthetic datasets for machine learning and deep learning projects. Avoid the hassle of downloading and managing data files for testing and prototyping.

## Installation

Clone the repository and install using pip:

```bash
git clone [https://github.com/yourusername/datagenix.git](https://github.com/yourusername/datagenix.git)
cd datagenix
pip install .
```

## Quick Start

Generate a DataFrame with a variety of data types with a single function call.

```python
from datagenix import DataGenerator

# Initialize the generator
generator = DataGenerator()

# Generate a dataset with 1000 rows
df = generator.generate(
    num_rows=1000,
    numerical_whole=3,
    decimal=2,
    categorical=2,
    ordinal=1,
    boolean=1,
    datetime=1,
    text=1,
    uuid=1,
    object_types=['name', 'country', 'email', 'job']
)

print(df.head())
print(df.info())
```

## Features

- **Numerical Data**: Generate columns of whole numbers (integers) or decimals (floats).
- **Categorical Data**: Generate columns with a predefined set of unordered categories.
- **Ordinal Data**: Generate columns with a predefined set of *ordered* categories.
- **Boolean Data**: Generate columns of `True`/`False` values.
- **Datetime Data**: Generate columns with `datetime` objects.
- **Text Data**: Generate columns with random sentences.
- **ID Data**: Generate columns with unique identifiers (UUIDs).
- **Coordinates**: Generate paired latitude and longitude columns.
- **Web Data**: Generate columns for IP addresses, URLs, and phone numbers.
- **Nested Data**: Generate columns containing JSON-formatted strings.
- **Object/Text Data**: Leverage the power of the `Faker` library to generate realistic text data like names, addresses, emails, and much more.

### Supported `object_types`

You can use any standard `Faker` provider method name as a string in the `object_types` list. Common examples include:

- `name`
- `email`
- `address`
- `country`
- `city`
- `job`
- `text`
- `datetime`
- `phone_number`
- `company`
- `url`
- `credit_card_number`

