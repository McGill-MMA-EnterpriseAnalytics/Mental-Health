"""
Script for evaluating fairness of a trained model on a test dataset.

This script:
- Loads a trained model and test data.
- Predicts test labels.
- Calculates group-based fairness metrics (Demographic Parity Difference, Equalized Odds Difference).
- Saves group accuracy as CSV.
- Generates a radar chart to visualize group accuracies.
- Produces an HTML report summarizing overall performance and fairness results.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score

# Paths
model_path = "/Users/qianzhao/Desktop/Enterprise/formal version/modeling/best_xgb_pipeline.pkl"
X_test_path = "/Users/qianzhao/Desktop/Enterprise/formal version/modeling/X_test_final.csv"
y_test_path = "/Users/qianzhao/Desktop/Enterprise/formal version/modeling/y_test_final.csv"

# Verify paths exist
assert os.path.exists(model_path), "Model file not found."
assert os.path.exists(X_test_path), "X_test file not found."
assert os.path.exists(y_test_path), "y_test file not found."

# Load model and data
model = joblib.load(model_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

# Predict
y_pred = model.predict(X_test)

# Fairness Evaluation
sensitive_attr = 'Gender'
metric_frame = MetricFrame(
    metrics={"accuracy": accuracy_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test[sensitive_attr]
)

# Calculate fairness metrics
dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=X_test[sensitive_attr])
eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=X_test[sensitive_attr])

dp_diff_value = dp_diff.max() if hasattr(dp_diff, 'max') else dp_diff
eo_diff_value = eo_diff.max() if hasattr(eo_diff, 'max') else eo_diff

dp_color = "green" if abs(dp_diff_value) < 0.1 else "red"
eo_color = "green" if abs(eo_diff_value) < 0.1 else "red"

# Print evaluation results
print("\n===== Fairness Evaluation =====")
print(f"Overall Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Accuracy by {sensitive_attr} group:\n{metric_frame.by_group}\n")
print(f"Demographic Parity Difference (max group diff): {dp_diff_value:.4f}")
print(f"Equalized Odds Difference (max group diff): {eo_diff_value:.4f}")

# Save group accuracies
output_dir = "/Users/qianzhao/Desktop/Enterprise/formal version/drift_fairness/fairness_reports"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "accuracy_by_group.csv")
metric_frame.by_group.to_csv(csv_path)
print(f"✅ Accuracy by group saved to {csv_path}")

# Radar chart plotting function
def plot_fairness_radar(data: pd.DataFrame, title: str, save_path: str) -> None:
    """
    Generate and save a radar chart of group accuracies.

    Args:
        data (pd.DataFrame): Metric values by group.
        title (str): Title of the chart.
        save_path (str): Path to save the chart image.
    """
    labels = list(data.index)
    values = data.values.flatten().tolist()
    values += values[:1]
    labels += labels[:1]

    angles = [n / float(len(labels)) * 2 * 3.14159 for n in range(len(labels))]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels[:-1], color='grey', size=12)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    plt.title(title, size=20, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Save radar chart
radar_path = os.path.join(output_dir, "fairness_radar.png")
plot_fairness_radar(metric_frame.by_group, f"Accuracy by {sensitive_attr}", radar_path)
print(f"✅ Fairness radar chart saved to {radar_path}")

# Generate HTML report
overall_accuracy = accuracy_score(y_test, y_pred)
html_report_path = os.path.join(output_dir, "fairness_report.html")

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fairness Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f8f9fa;
        }}
        h1 {{
            color: #343a40;
        }}
        p {{
            font-size: 16px;
            color: #495057;
        }}
        img {{
            width: 600px;
            height: auto;
            margin-top: 20px;
            border: 1px solid #dee2e6;
        }}
        table {{
            width: 50%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        table, th, td {{
            border: 1px solid #dee2e6;
            text-align: center;
            padding: 8px;
        }}
        th {{
            background-color: #e9ecef;
        }}
    </style>
</head>
<body>

<h1>Fairness Evaluation Report</h1>

<p><strong>Overall Test Accuracy:</strong> {overall_accuracy:.4f}</p>

<p><strong>Demographic Parity Difference (max diff):</strong> 
<span style="color: {dp_color};">{dp_diff_value:.4f}</span></p>

<p><strong>Equalized Odds Difference (max diff):</strong> 
<span style="color: {eo_color};">{eo_diff_value:.4f}</span></p>

<h2>Accuracy by Group ({sensitive_attr})</h2>
<table>
    <tr>
        <th>Group</th>
        <th>Accuracy</th>
    </tr>
"""

for group, acc in metric_frame.by_group.iterrows():
    acc_value = acc['accuracy']
    html_content += f"""
    <tr>
        <td>{group}</td>
        <td>{acc_value:.4f}</td>
    </tr>
    """

html_content += f"""
</table>

<h2>Fairness Radar Chart</h2>
<img src="fairness_radar.png" alt="Fairness Radar Chart">

</body>
</html>
"""

# Write HTML file
with open(html_report_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ HTML fairness report generated: {html_report_path}")
