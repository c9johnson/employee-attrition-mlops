import mlflow
import pandas as pd
import yaml


def load_config(config_path="configs/config.yaml"):
    """Load configuration settings."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_best_run(metric="metrics.f1_score"):
    """Query MLflow experiment runs and identify the best run based on the given metric."""
    config = load_config()
    experiment_name = config.get("mlflow", {}).get("experiment_name", "employee_attrition_monitoring")

    # Retrieve experiment details
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Experiment '{experiment_name}' not found.")
        return

    # Search for all runs in the experiment, sorted by f1_score descending
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{metric} DESC"]
    )

    if runs.empty:
        print("❌ No runs found in MLflow.")
        return

    # Select best run (top row)
    best_run = runs.iloc[0]

    print("==================================================")
    print("      MLFLOW EXPERIMENT COMPARISON SUMMARY        ")
    print("==================================================")
    print(f"Total Runs Analyzed: {len(runs)}")
    print(f"Primary Selection Metric: {metric}\n")

    print(f"🏆 BEST RUN DETAILS:")
    print(f"  Run ID:       {best_run['run_id']}")
    print(f"  Run Name:     {best_run.get('tags.mlflow.runName', 'N/A')}")
    print(f"  F1 Score:     {best_run.get('metrics.f1_score', 0):.4f}")
    print(f"  Precision:    {best_run.get('metrics.precision', 0):.4f}")
    print(f"  Recall:       {best_run.get('metrics.recall', 0):.4f}")
    print(f"  Accuracy:     {best_run.get('metrics.accuracy', 0):.4f}")
    print(f"  ROC AUC:      {best_run.get('metrics.roc_auc', 0):.4f}")
    
    print("\n⚙️ BEST HYPERPARAMETERS:")
    print(f"  n_estimators: {best_run.get('params.n_estimators', 'N/A')}")
    print(f"  max_depth:    {best_run.get('params.max_depth', 'N/A')}")
    print("==================================================")

    return best_run['run_id']


if __name__ == "__main__":
    find_best_run()