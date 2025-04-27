# drift_fairness/check_fairness.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score

# 1. 加载模型和测试数据
model_path = "/Users/qianzhao/Desktop/Enterprise/formal version/modeling/best_xgb_pipeline.pkl"
X_test_path = "/Users/qianzhao/Desktop/Enterprise/formal version/modeling/X_test_final.csv"
y_test_path = "/Users/qianzhao/Desktop/Enterprise/formal version/modeling/y_test_final.csv"

assert os.path.exists(model_path), "❌ 找不到模型文件，请检查路径！"
assert os.path.exists(X_test_path), "❌ 找不到X_test文件，请检查路径！"
assert os.path.exists(y_test_path), "❌ 找不到y_test文件，请检查路径！"

model = joblib.load(model_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()

# 2. 模型预测
y_pred = model.predict(X_test)

# 3. 创建 MetricFrame
sensitive_attr = 'Gender'  # 可以换成 Country_grouped 等其他 sensitive features
metric_frame = MetricFrame(
    metrics={"accuracy": accuracy_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test[sensitive_attr]
)

# 4. 计算公平性指标
dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=X_test[sensitive_attr])
eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=X_test[sensitive_attr])

# 确保公平性指标是 float（处理Series问题）
dp_diff_value = dp_diff.max() if hasattr(dp_diff, 'max') else dp_diff
eo_diff_value = eo_diff.max() if hasattr(eo_diff, 'max') else eo_diff

# 颜色判断
dp_color = "green" if abs(dp_diff_value) < 0.1 else "red"
eo_color = "green" if abs(eo_diff_value) < 0.1 else "red"

# 5. 打印结果
print("\n===== Fairness Evaluation =====")
print(f"Overall Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Accuracy by {sensitive_attr} group:\n{metric_frame.by_group}\n")
print(f"Demographic Parity Difference (max group diff): {dp_diff_value:.4f}")
print(f"Equalized Odds Difference (max group diff): {eo_diff_value:.4f}")

# 6. 保存分组Accuracy为 CSV
output_dir = "/Users/qianzhao/Desktop/Enterprise/formal version/drift_fairness/fairness_reports"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "accuracy_by_group.csv")
metric_frame.by_group.to_csv(csv_path)
print(f"✅ Accuracy by group saved to {csv_path}")

# 7. 绘制公平性雷达图
def plot_fairness_radar(data, title, save_path):
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

radar_path = os.path.join(output_dir, "fairness_radar.png")
plot_fairness_radar(metric_frame.by_group, f"Accuracy by {sensitive_attr}", radar_path)
print(f"✅ Fairness radar chart saved to {radar_path}")

# 8. 生成 HTML 报告
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

# 插入分组 Accuracy

for group, acc in metric_frame.by_group.iterrows():
    acc_value = acc['accuracy']  # 直接取出 accuracy这列的数
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

# 保存 HTML 文件
with open(html_report_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ HTML fairness report generated: {html_report_path}")
