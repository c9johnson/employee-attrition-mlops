# Data Drift Analysis Report

## 1. Which features showed drift and why?
Feature drift was primarily detected in continuous numerical variables such as `MonthlyIncome` and `TotalWorkingYears`. This drift occurred due to distribution shifts between the historical reference dataset and production month 1 data, reflecting changes in compensation structures and employee demographics over time.

## 2. Would this drift likely affect model performance?
Yes. Significant drift in high-importance predictors like `MonthlyIncome` directly impacts the decision boundaries of the attrition model, leading to potential misclassifications and degraded prediction confidence in production.

## 3. Recommended Action
**Investigate and Retrain:** Investigate the root cause of demographic shifts in the incoming data. Trigger a model retraining pipeline using the newly collected production data to align feature distributions and restore model accuracy.
