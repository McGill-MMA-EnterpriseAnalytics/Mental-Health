# data_preprocessing/preprocess.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

from data_preprocessing.cleaning_utils import standardize_gender, clean_age

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

from data_preprocessing.cleaning_utils import standardize_gender, clean_age


def initial_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['comments', 'Timestamp', 'state', 'no_employees', 'anonymity'], errors='ignore')

    df['Age'] = df['Age'].apply(clean_age)
    df = df.dropna(subset=['Age'])

    df['Gender'] = df['Gender'].apply(standardize_gender)

    df['self_employed'] = df['self_employed'].fillna('No')
    df['work_interfere'] = df['work_interfere'].fillna('Not applicable')

    df = df.drop_duplicates()

    return df


def group_rare_categories(df: pd.DataFrame, column: str, threshold: float = 0.01) -> pd.DataFrame:

    vc = df[column].value_counts(normalize=True)
    rare_labels = vc[vc < threshold].index.tolist()
    df[f"{column}_grouped"] = df[column].replace(rare_labels, 'Other')
    return df

def save_clean_data(input_path: str, output_dir: str = "/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing"):
    
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    df = initial_preprocessing(df)

    df = group_rare_categories(df, 'Country')

    X = df.drop(columns=["treatment"])
    y = df["treatment"].map({"Yes": 1, "No": 0})

    X.to_csv(os.path.join(output_dir, "X_clean.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "y_clean.csv"), index=False)

    print(f"✅ Cleaned X saved to {output_dir}/X_clean.csv")
    print(f"✅ Cleaned y saved to {output_dir}/y_clean.csv")


if __name__ == "__main__":
    save_clean_data("/Users/qianzhao/Desktop/Enterprise/survey.csv")
