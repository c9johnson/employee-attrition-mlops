import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.preprocessing import prepare_data


def load_config(config_path="configs/config.yaml"):
    """Load yaml configuration settings."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_and_log_model(config_path="configs/config.yaml"):
    """Train Random Forest classifier and log metrics/artifacts to MLflow."""
    config = load_config(config_path)

    # Set up MLflow tracking
    experiment_name = config.get("mlflow", {}).get("experiment_name", "employee_attrition")
    mlflow.set_experiment(experiment_name)

    # Load reference dataset
    ref_path = config["data"]["reference_path"]
    df = pd.read_csv(ref_path)

    target_col = config["data"].get("target_column", "Attrition")
    
    # Read test_size and random_state from the data block
    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)

    X_train, X_test, y_train, y_test = prepare_data(
        df, target_column=target_col, test_size=test_size, random_state=random_state
    )

    # Model Hyperparameters
    n_estimators = config["model"].get("n_estimators", 100)
    max_depth = config["model"].get("max_depth", 10)

    # Dynamic run name based on config parameters
    run_name = f"RandomForest_n{n_estimators}_d{max_depth}"

    with mlflow.start_run(run_name=run_name):
        # Log Hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Train Model
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        # Predictions & Probabilities
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Log Metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            print(f"  {metric_name}: {value:.4f}")

        # Log Model Artifact
        mlflow.sklearn.log_model(clf, artifact_path="model")
        print("\n✅ Model training and MLflow logging complete.")


if __name__ == "__main__":
    train_and_log_model()