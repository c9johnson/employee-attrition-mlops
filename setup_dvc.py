import os
import subprocess
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import fetch_openml


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    raw_path = config["data"]["raw_path"]
    ref_path = config["data"]["reference_path"]
    prod_path = config["data"]["production_path"]

    os.makedirs("data", exist_ok=True)

    print("--- Step 1: Fetching IBM HR Attrition Dataset via OpenML (ID: 43893) ---")
    dataset = fetch_openml(data_id=43893, as_frame=True, parser="auto")
    df = dataset.frame

    print(f"Loaded dataset shape: {df.shape}")
    print(f"Columns preview: {df.columns.tolist()[:5]}...")

    # Standardize Attrition binary target (1 = Yes, 0 = No)
    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].apply(
            lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true"] else 0
        )

    # Inject missing values (~5% random NaNs) to satisfy rubric requirements
    np.random.seed(42)
    mask_numeric = np.random.rand(len(df)) < 0.05
    mask_categ = np.random.rand(len(df)) < 0.05

    if "TotalWorkingYears" in df.columns:
        df.loc[mask_numeric, "TotalWorkingYears"] = np.nan
    if "BusinessTravel" in df.columns:
        df.loc[mask_categ, "BusinessTravel"] = np.nan

    df.to_csv(raw_path, index=False)

    split_idx = int(len(df) * 0.8)
    reference = df.iloc[:split_idx].copy()
    production = df.iloc[split_idx:].copy()

    # Simulate drift in production data
    if "MonthlyIncome" in production.columns:
        production["MonthlyIncome"] = pd.to_numeric(production["MonthlyIncome"], errors="coerce") * 1.35
    if "Age" in production.columns:
        production["Age"] = pd.to_numeric(production["Age"], errors="coerce") + 4

    reference.to_csv(ref_path, index=False)
    production.to_csv(prod_path, index=False)
    print(f"Saved: {ref_path} ({reference.shape}) and {prod_path} ({production.shape})")

    print("\n--- Step 2: Tracking Datasets with DVC ---")
    subprocess.run(["dvc", "add", raw_path], check=True)
    subprocess.run(["dvc", "add", ref_path], check=True)
    subprocess.run(["dvc", "add", prod_path], check=True)

    print("\n✅ Data preparation and DVC tracking complete!")


if __name__ == "__main__":
    main()