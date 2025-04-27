# train_label_propagation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.semi_supervised import LabelPropagation

# 1. Load Data
X = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/X_clean.csv") 
y = pd.read_csv("/Users/qianzhao/Desktop/Enterprise/formal version/data_preprocessing/y_clean.csv").squeeze()  

# 2. Feature Columns
cat_cols = [
    'Gender', 'Country_grouped', 'self_employed', 'family_history',
    'work_interfere', 'remote_work', 'tech_company', 'benefits',
    'care_options', 'wellness_program', 'seek_help', 'leave',
    'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview',
    'phys_health_interview', 'mental_vs_physical', 'obs_consequence'
]
num_cols = ['Age']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 4. Define Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

# 5. Encode Features
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

# 6. Create Semi-supervised Labels
rng = np.random.RandomState(42)
mask = rng.rand(len(y_train)) < 0.5  
y_train_semi = y_train.copy()
y_train_semi[mask] = -1 

# 7. Define Label Propagation Model
label_prop = LabelPropagation(
    kernel='knn',         
    n_neighbors=7,
    max_iter=1000
)

# 8. Train Label Propagation Model
label_prop.fit(X_train_enc, y_train_semi)

# 9. Predict on Test Set
y_pred = label_prop.predict(X_test_enc)

# 10. Evaluation
print("===== Label Propagation Results =====")
print(classification_report(y_test, y_pred))
