"""
Preprocessing script for cleaning and preparing mental health survey data.

Includes:
- Standardizing gender labels.
- Cleaning and validating age values.
- Handling missing values for specific columns.
- Grouping rare categories for specified categorical variables.
- Saving cleaned feature matrix (X) and target vector (y) to CSV files.

"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Adjust path to allow importing from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing.cleaning_utils import standardize_gender, clean_age


def initial_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply initial cleaning operations on the raw dataframe.

    Steps:
    - Drop irrelevant columns.
    - Clean 'Age' values and drop rows with invalid ages.
    - Standardize 'Gender' values.
    - Fill missing values for 'self_employed' and 'work_interfere'.
    - Remove duplicate rows.

    Args:
        df (pd.DataFrame): Raw input dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.drop(columns=['comments', 'Timestamp', 'state', 'no_employees', 'anonymity'], errors='ignore')

    df['Age'] = df['Age'].apply(clean_age)
    df = df.dropna(subset=['Age'])

    df['Gender'] = df['Gender'].apply(standardize_gender)

    df['self_employed'] = df['self_employed'].fillna('No')
    df['work_interfere'] = df['work_interfere'].fillna('Not applicable')

    df = df.drop_duplicates()

    return df


def group_rare_categories(df: pd.DataFrame, column: str, threshold: float = 0.01) -> pd.DataFrame:
    """
    Group rare categories in a specified column into 'Other' if below a frequency threshold.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name to process.
        threshold (float): Minimum frequency for a category to not be grouped.

    Returns:
        pd.DataFrame: Dataframe with an additional column '[column]_grouped'.
    """
    vc = df[column].value_counts(normalize=True)
    rare_labels = vc[vc < threshold].index.tolist()
    df[f"{column}_grouped"] = df[column].replace(rare_labels, 'Other')
    return df


def save_clean_data(input_path: str, output_dir: str = "/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing") -> None:
    """
    Read raw survey data, clean it, and save processed feature matrix and labels as CSV files.

    Args:
        input_path (str): Path to the raw input CSV file.
        output_dir (str, optional): Directory to save cleaned outputs. Defaults to '/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing'.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load raw data
    df = pd.read_csv(input_path)

    # Initial preprocessing
    df = initial_preprocessing(df)

    # Group rare countries
    df = group_rare_categories(df, 'Country')

    # Split into features and target
    X = df.drop(columns=["treatment"])
    y = df["treatment"].map({"Yes": 1, "No": 0})

    # Save processed data
    X.to_csv(os.path.join(output_dir, "X_clean.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "y_clean.csv"), index=False)

    print(f"✅ Cleaned X saved to {output_dir}/X_clean.csv")
    print(f"✅ Cleaned y saved to {output_dir}/y_clean.csv")


if __name__ == "__main__":
    # Example usage: clean the raw survey data
    save_clean_data("/Users/qianzhao/Desktop/Enterprise/survey.csv")
