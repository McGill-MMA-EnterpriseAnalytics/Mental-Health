"""
Script for applying pseudo-labeling to enhance a labeled training dataset
using confident predictions from an unlabeled dataset.

Pseudo-labeling steps:
- Train a base model on labeled data.
- Predict probabilities on unlabeled data.
- Select high-confidence predictions (above a threshold).
- Augment the training set with pseudo-labeled examples.
"""

import numpy as np
import pandas as pd

def apply_pseudo_labeling(
    base_model,
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    X_unlabeled: pd.DataFrame,
    preprocessor=None,
    threshold: float = 0.9
):
    """
    Apply pseudo-labeling to augment labeled data with high-confidence unlabeled samples.

    Args:
        base_model: A scikit-learn compatible model with `fit` and `predict_proba` methods.
        X_labeled (pd.DataFrame): Features of the labeled dataset.
        y_labeled (pd.Series): Labels corresponding to X_labeled.
        X_unlabeled (pd.DataFrame): Features of the unlabeled dataset.
        preprocessor (optional): Preprocessing pipeline (e.g., scaler, encoder) to apply before model training.
        threshold (float, optional): Probability threshold above which pseudo-labels are accepted. Defaults to 0.9.

    Returns:
        tuple:
            X_augmented (pd.DataFrame): Combined labeled + pseudo-labeled features.
            y_augmented (pd.Series): Combined true + pseudo labels.
    """
    # Preprocess if preprocessor is provided
    if preprocessor is not None:
        preprocessor.fit(X_labeled)
        X_labeled_encoded = preprocessor.transform(X_labeled)
        X_unlabeled_encoded = preprocessor.transform(X_unlabeled)
    else:
        X_labeled_encoded = X_labeled
        X_unlabeled_encoded = X_unlabeled

    # Train base model on labeled data
    base_model.fit(X_labeled_encoded, y_labeled)

    # Predict probabilities on unlabeled data
    proba_unlabeled = base_model.predict_proba(X_unlabeled_encoded)
    preds_unlabeled = np.argmax(proba_unlabeled, axis=1)
    max_proba = np.max(proba_unlabeled, axis=1)

    # Select pseudo-labels with high confidence
    mask = max_proba >= threshold
    X_pseudo = X_unlabeled.iloc[mask]
    y_pseudo = preds_unlabeled[mask]

    # Combine original labeled data with pseudo-labeled data
    X_augmented = pd.concat([X_labeled, X_pseudo], axis=0)
    y_augmented = pd.concat([y_labeled, pd.Series(y_pseudo, index=X_pseudo.index)], axis=0)

    return X_augmented, y_augmented
