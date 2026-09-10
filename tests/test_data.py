import os
import pandas as pd
import pytest
import yaml


@pytest.fixture
def config():
    """Fixture to load configuration setting for tests."""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_reference_file_exists(config):
    """Verify that the reference data file exists at the path specified in config."""
    ref_path = config["data"]["reference_path"]
    assert os.path.exists(ref_path), f"Reference file not found at {ref_path}"


def test_data_quality(config):
    """Verify that the reference dataset is non-empty and contains the target column."""
    ref_path = config["data"]["reference_path"]
    df = pd.read_csv(ref_path)
    target_col = config["data"].get("target_column", "Attrition")

    assert not df.empty, "Reference dataset should not be empty"
    assert target_col in df.columns, f"Target column '{target_col}' missing from reference dataset"