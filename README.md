# Employee Attrition MLOps Pipeline

## Overview
An end-to-end MLOps pipeline for predicting employee attrition. Includes automated data preprocessing, unit testing with `pytest`, data tracking via `DVC`, and continuous drift monitoring using `Evidently AI`.

## Project Structure
* **configs/** — YAML configuration settings
* **data/** — DVC-tracked dataset files
* **reports/** — Generated Evidently AI data drift reports (`drift_report.html`)
* **src/** — Pipeline source code (`preprocessing.py`, `train_model.py`, `monitor_drift.py`)
* **tests/** — Pytest test suite (11 unit, data, and model validation tests)

## Setup & Execution

### 1. Activate Environment
source ../.venv/bin/activate

### 2. Run Training
python -m src.train_model

### 3. Run Test Suite (11 tests)
python -m pytest -v

### 4. Generate Data Drift Monitoring Report
python -m src.monitor_drift
