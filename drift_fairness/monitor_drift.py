# drift_fairness/monitor_drift.py

import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def monitor_feature_drift(reference_data_path, current_data_path, output_html_path):
    """
    Compare feature distributions between training and current datasets using EvidentlyAI,
    and generate a clean HTML report for PM presentation.

    Args:
        reference_data_path (str): Path to the training data (e.g., X_train_final.csv)
        current_data_path (str): Path to the current/new data (e.g., simulated_current.csv)
        output_html_path (str): Path where the HTML report will be saved
    """

    # Load data
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)

    # Only compare feature distributions, no model involved
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])

    drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Save as HTML
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    drift_report.save_html(output_html_path)
    print(f"✅ Feature drift report saved to {output_html_path}")

if __name__ == "__main__":
    monitor_feature_drift(
        reference_data_path="/Users/qianzhao/Desktop/Enterprise/formal version/modeling/X_train_final.csv",
        current_data_path="/Users/qianzhao/Desktop/Enterprise/formal version/drift_fairness/simulated_current.csv",
        output_html_path="/Users/qianzhao/Desktop/Enterprise/formal version/drift_fairness/feature_drift_report.html"
    )
