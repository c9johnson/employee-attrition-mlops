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


# Unit Test 1: Missing values handling
def test_handle_missing_values(sample_data):
    cleaned_df = handle_missing_values(sample_data)
    assert cleaned_df.isnull().sum().sum() == 0


# Unit Test 2: Categorical encoding
def test_encode_features(sample_data):
    cleaned_df = handle_missing_values(sample_data)
    X = cleaned_df.drop(columns=["Attrition"])
    encoded_X = encode_features(X)
    assert "Department" not in encoded_X.columns
    assert any(col.startswith("Department_") for col in encoded_X.columns)


# Unit Test 3: Non-mutation of raw dataframes
def test_non_mutation_raw_dataframe(sample_data):
    original_copy = sample_data.copy()
    _ = handle_missing_values(sample_data)
    pd.testing.assert_frame_equal(sample_data, original_copy)


# Unit Test 4: Raising invalid input errors (missing target column)
def test_invalid_input_missing_target_col(sample_data):
    with pytest.raises(KeyError):
        prepare_data(sample_data, target_column="MissingTargetCol")


# Unit Test 5: Raising invalid input errors (non-dataframe input)
def test_invalid_input_type():
    with pytest.raises((TypeError, AttributeError)):
        handle_missing_values("invalid_data_type_string")


# Unit Test 6: End-to-end split pipeline shapes
def test_prepare_data_shapes(sample_data):
    X_train, X_test, y_train, y_test = prepare_data(
        sample_data, target_column="Attrition", test_size=0.2, random_state=42
    )
    assert len(X_train) == 8
    assert len(X_test) == 2