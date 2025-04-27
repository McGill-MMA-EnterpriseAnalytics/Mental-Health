# drift_fairness/simulate_data.py

import pandas as pd
import numpy as np
import os

def simulate_data(train_data_path, output_path, random_state=42):
    np.random.seed(random_state)
    
    # 读取原始 X_train_final
    df = pd.read_csv(train_data_path)

    # 创建副本
    df_simulated = df.copy()

    # ------- 1. 类别特征 Drift 示例：改变 Gender 分布 -------
    if 'Gender' in df_simulated.columns:
        male_idx = df_simulated[df_simulated['Gender'] == 'Male'].index
        female_idx = df_simulated[df_simulated['Gender'] == 'Female'].index
        
        # 让男性样本比例上升，女性样本比例下降
        new_male_idx = np.random.choice(male_idx, size=int(len(male_idx) * 1.2), replace=True)  # 放大20%
        new_female_idx = np.random.choice(female_idx, size=int(len(female_idx) * 0.7), replace=False)  # 缩小30%
        
        selected_idx = np.concatenate([new_male_idx, new_female_idx])
        df_simulated = df_simulated.loc[selected_idx]
    
    # ------- 2. 连续特征 Drift 示例：Age 加一点噪声 -------
    if 'Age' in df_simulated.columns:
        noise = np.random.normal(0, 2, size=len(df_simulated))  # 均值0，标准差2的小噪声
        df_simulated['Age'] = df_simulated['Age'] + noise
        df_simulated['Age'] = df_simulated['Age'].clip(lower=0)  # 防止出现负数年龄

    # ------- 3. 重新 shuffle 一下 -------
    df_simulated = df_simulated.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_simulated.to_csv(output_path, index=False)
    print(f"✅ Simulated data saved to {output_path}")

if __name__ == "__main__":
    simulate_data(
        train_data_path="/Users/qianzhao/Desktop/Enterprise/formal version/modeling/X_train_final.csv",
        output_path="/Users/qianzhao/Desktop/Enterprise/formal version/drift_fairness/simulated_current.csv"
    )
