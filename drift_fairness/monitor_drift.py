"""
Script for monitoring feature distribution drift between training and current datasets.

Uses EvidentlyAI's DataDriftPreset to:
- Compare feature distributions without requiring model outputs.
- Generate a visual HTML report for PMs, stakeholders, or model maintenance teams.

"""
import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def monitor_feature_drift(reference_data_path: str, current_data_path: str, output_html_path: str) -> None:
    """
    Compare feature distributions between a reference (training) dataset and a current dataset,
    and save an EvidentlyAI HTML drift report.

    Args:
        reference_data_path (str): Path to the reference (training) data CSV.
        current_data_path (str): Path to the current (new) data CSV.
        output_html_path (str): Path where the generated HTML drift report will be saved.

    Returns:
        None
    """
    # Load datasets
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)

    # Initialize Evidently report for feature drift
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])

    # Run drift detection
    drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Save the report
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    drift_report.save_html(output_html_path)
    print(f"✅ Feature drift report saved to {output_html_path}")

if __name__ == "__main__":
    monitor_feature_drift(
        reference_data_path="/Users/qianzhao/Desktop/Enterprise/formal version/modeling/X_train_final.csv",
        current_data_path="/Users/qianzhao/Desktop/Enterprise/formal version/drift_fairness/simulated_current.csv",
        output_html_path="/Users/qianzhao/Desktop/Enterprise/formal version/drift_fairness/feature_drift_report.html"
    )
