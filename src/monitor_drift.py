import os
import pandas as pd
import yaml

# Imports for Evidently 0.7.x legacy pipeline
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report


def load_config(config_path="configs/config.yaml"):
    """Load monitoring configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_drift_report():
    """Compare reference and production datasets to generate an Evidently Data Drift HTML report."""
    config = load_config()

    ref_path = config["data"]["reference_path"]
    prod_path = config["data"]["production_path"]
    report_output_path = config["monitoring"]["report_output_path"]

    print(f"Loading reference data from: {ref_path}")
    reference_data = pd.read_csv(ref_path)

    print(f"Loading production data from: {prod_path}")
    production_data = pd.read_csv(prod_path)

    print("Generating Data Drift Report using Evidently AI...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=production_data)

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)

    report.save_html(report_output_path)
    print(f"Data drift report saved to: {report_output_path}")


if __name__ == "__main__":
    generate_drift_report()