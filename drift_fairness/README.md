
# Drift & Fairness Monitoring

This folder contains scripts to **monitor feature drift**, **simulate new data scenarios**, and **evaluate model fairness** across sensitive attributes.

##  Contents
- **Drift Detection**: Monitoring feature distribution changes over time.
- **Fairness Evaluation**: Checking bias across sensitive groups (e.g., Gender, Country).
- **Data Simulation**: Creating synthetic datasets to simulate drift scenarios.

## Purpose
- Ensure that the model remains stable and fair in production.
- Detect feature shifts that might harm model performance.
- Evaluate demographic parity and equalized odds.

## 📂 Key Files
| File | Description |
|:---|:---|
| `monitor_drift.py` | Detects feature drift and generates drift reports (e.g., with EvidentlyAI). |
| `check_fairness.py` | Evaluates fairness metrics (e.g., demographic parity, equalized odds) and generates radar charts + HTML reports. |
| `XGBoost_Optuna_ModelDrift.ipynb` | Tunes XGBoost with Optuna and evaluates model performance and drift via ROC analysis. |
| `Fairness_Analysis.ipynb` | Assesses model fairness across demographic groups using multiple fairness metrics. |
| `simulate_data.py` | Simulates a shifted current dataset for testing drift and fairness monitoring. |
| `feature_drift_report.html` | Example output report from drift monitoring. |
| `fairness_reports/` | Saved fairness evaluation artifacts (CSV, PNG, HTML reports). |
| `simulated_current.csv` | Synthetic dataset used to simulate a drifted production scenario. |

## Notes
- Sensitive feature currently used for fairness evaluation: `Gender`.
- Outputs include CSV reports, radar plots, and interactive HTML dashboards.

---

