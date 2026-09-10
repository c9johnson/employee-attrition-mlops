import pandas as pd
import pytest
import yaml


@pytest.fixture
def config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def ref_df(config):
    ref_path = config["data"]["reference_path"]
    return pd.read_csv(ref_path)


# Data Test 1: Column existence
def test_data_column_existence(config, ref_df):
    target_col = config["data"].get("target_column", "Attrition")
    assert target_col in ref_df.columns, f"Required column '{target_col}' missing"


# Data Test 2: Valid target values
def test_data_valid_target_values(config, ref_df):
    target_col = config["data"].get("target_column", "Attrition")
    target_vals = set(ref_df[target_col].dropna().unique())
    assert target_vals.issubset({0, 1}) or target_vals.issubset({"Yes", "No"})


# Data Test 3: Feature ranges
def test_data_feature_ranges(ref_df):
    if "Age" in ref_df.columns:
        assert (ref_df["Age"].dropna() > 0).all(), "Age values must be greater than zero"
    if "MonthlyIncome" in ref_df.columns:
        assert (ref_df["MonthlyIncome"].dropna() >= 0).all(), "Income cannot be negative"