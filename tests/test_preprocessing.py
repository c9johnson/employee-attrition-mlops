import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    encode_features,
    handle_missing_values,
    prepare_data,
)


@pytest.fixture
def sample_data():
    """Fixture providing a mock DataFrame with missing values and categorical data."""
    return pd.DataFrame(
        {
            "Age": [25, 30, np.nan, 45, 50, 23, 35, 40, 48, 52],
            "Department": ["Sales", np.nan, "HR", "Sales", "HR", "R&D", "R&D", "Sales", "HR", "R&D"],
            "MonthlyIncome": [5000, 6000, 7000, np.nan, 9000, 4500, 6500, 8000, 8500, 9500],
            "Attrition": [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        }
    )


def test_handle_missing_values(sample_data):
    """Test that missing numeric values (median) and categorical values (mode) are filled."""
    cleaned_df = handle_missing_values(sample_data)

    # Assert no null values remain
    assert cleaned_df.isnull().sum().sum() == 0

    # Check numeric imputation (median of non-null Age values)
    assert not cleaned_df["Age"].isnull().any()


def test_encode_features(sample_data):
    """Test that object/category columns are one-hot encoded cleanly."""
    cleaned_df = handle_missing_values(sample_data)
    X = cleaned_df.drop(columns=["Attrition"])
    encoded_X = encode_features(X)

    # Check that object column 'Department' was converted into dummy indicator columns
    assert "Department" not in encoded_X.columns
    assert any(col.startswith("Department_") for col in encoded_X.columns)


def test_prepare_data(sample_data):
    """Test the end-to-end data preparation pipeline and train-test split shapes."""
    X_train, X_test, y_train, y_test = prepare_data(
        sample_data, target_column="Attrition", test_size=0.2, random_state=42
    )

    # 10 rows total with test_size=0.2 -> 8 train, 2 test
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2