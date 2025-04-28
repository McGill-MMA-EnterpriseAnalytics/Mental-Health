"""
Script for simulating data drift from training data.

This script:
- Simulates demographic shift by oversampling males and undersampling females.
- Adds slight noise to the 'Age' feature.
- Outputs a modified dataset for drift monitoring experiments.

Useful for evaluating how a model reacts to shifted real-world distributions.

Author: Your Name
Date: YYYY-MM-DD
"""

import pandas as pd
import numpy as np
import os

def simulate_data(train_data_path: str, output_path: str, random_state: int = 42) -> None:
    """
    Simulate drifted data based on the training dataset.

    Steps:
    - Oversample 'Male' group and undersample 'Female' group in 'Gender' column.
    - Add Gaussian noise to 'Age' column.
    - Shuffle the dataset and save to output path.

    Args:
        train_data_path (str): Path to the original training dataset (CSV).
        output_path (str): Path where the simulated dataset will be saved.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        None
    """
    np.random.seed(random_state)
    
    # Load original training data
    df = pd.read_csv(train_data_path)
    df_simulated = df.copy()

    # Simulate demographic shift (Gender)
    if 'Gender' in df_simulated.columns:
        male_idx = df_simulated[df_simulated['Gender'] == 'Male'].index
        female_idx = df_simulated[df_simulated['Gender'] == 'Female'].index
        
        new_male_idx = np.random.choice(male_idx, size=int(len(male_idx) * 1.2), replace=True)  # Oversample males
        new_female_idx = np.random.choice(female_idx, size=int(len(female_idx) * 0.7), replace=False)  # Undersample females
        
        selected_idx = np.concatenate([new_male_idx, new_female_idx])
        df_simulated = df_simulated.loc[selected_idx]

    # Simulate feature noise (Age)
    if 'Age' in df_simulated.columns:
        noise = np.random.normal(0, 2, size=len(df_simulated))  # Small Gaussian noise
        df_simulated['Age'] = df_simulated['Age'] + noise
        df_simulated['Age'] = df_simulated['Age'].clip(lower=0)  # Ensure Age is non-negative

    # Shuffle dataset
    df_simulated = df_simulated.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Save simulated dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_simulated.to_csv(output_path, index=False)
    print(f"✅ Simulated data saved to {output_path}")

if __name__ == "__main__":
    simulate_data(
        train_data_path="/Users/qianzhao/Desktop/Enterprise/formal version/modeling/X_train_final.csv",
        output_path="/Users/qianzhao/Desktop/Enterprise/formal version/drift_fairness/simulated_current.csv"
    )
