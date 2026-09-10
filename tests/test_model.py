import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import prepare_data


@pytest.fixture
def config():
    """Fixture to load configuration setting for tests."""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_model_training_and_predictions(config):
    """Verify that model fits correctly and outputs binary predictions [0, 1]."""
    ref_path = config["data"]["reference_path"]
    df = pd.read_csv(ref_path)
    target_col = config["data"].get("target_column", "Attrition")

    X_train, X_test, y_train, y_test = prepare_data(df, target_column=target_col)

    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})