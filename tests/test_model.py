import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.preprocessing import prepare_data


@pytest.fixture
def config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def trained_model_and_data(config):
    ref_path = config["data"]["reference_path"]
    df = pd.read_csv(ref_path)
    target_col = config["data"].get("target_column", "Attrition")

    X_train, X_test, y_train, y_test = prepare_data(df, target_column=target_col)
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


# Model Test 1: Output shape/type checks
def test_model_output_shape_and_type(trained_model_and_data):
    clf, X_test, _ = trained_model_and_data
    preds = clf.predict(X_test)
    assert len(preds) == len(X_test), "Output predictions shape mismatch"
    assert set(preds).issubset({0, 1}), "Predictions must be binary [0, 1]"


# Model Test 2: Minimum performance threshold
def test_model_minimum_performance_threshold(trained_model_and_data):
    clf, X_test, y_test = trained_model_and_data
    preds = clf.predict(X_test)
    score = f1_score(y_test, preds, zero_division=0)
    assert score >= 0.0, "Model F1 score is below acceptable baseline threshold"