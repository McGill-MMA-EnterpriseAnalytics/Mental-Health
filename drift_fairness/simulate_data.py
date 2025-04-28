# drift_fairness/simulate_data.py

import pandas as pd
import numpy as np
import os

def simulate_data(train_data_path, output_path, random_state=42):
    np.random.seed(random_state)
    
    df = pd.read_csv(train_data_path)

    df_simulated = df.copy()

    if 'Gender' in df_simulated.columns:
        male_idx = df_simulated[df_simulated['Gender'] == 'Male'].index
        female_idx = df_simulated[df_simulated['Gender'] == 'Female'].index
        
        new_male_idx = np.random.choice(male_idx, size=int(len(male_idx) * 1.2), replace=True)  
        new_female_idx = np.random.choice(female_idx, size=int(len(female_idx) * 0.7), replace=False)  
        
        selected_idx = np.concatenate([new_male_idx, new_female_idx])
        df_simulated = df_simulated.loc[selected_idx]
    
    if 'Age' in df_simulated.columns:
        noise = np.random.normal(0, 2, size=len(df_simulated)) 
        df_simulated['Age'] = df_simulated['Age'] + noise
        df_simulated['Age'] = df_simulated['Age'].clip(lower=0) 

    df_simulated = df_simulated.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_simulated.to_csv(output_path, index=False)
    print(f"✅ Simulated data saved to {output_path}")

if __name__ == "__main__":
    simulate_data(
        train_data_path="/Users/qianzhao/Desktop/Enterprise/formal version/modeling/X_train_final.csv",
        output_path="/Users/qianzhao/Desktop/Enterprise/formal version/drift_fairness/simulated_current.csv"
    )
