**Best\_XGBoost\_with\_Tuning\_(1).ipynb** --- *Model Explainability Report*

This notebook focuses on **interpreting the final tuned XGBoost model** using both **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-Agnostic Explanations). It aims to provide both **global** and **local** interpretability insights, helping to explain why the model makes certain predictions.

### Key Components:

* Loads the final XGBoost pipeline trained via Random Search
* Computes **global SHAP values** to identify overall feature importance
* Generates **local SHAP force plots** for individual predictions
* Applies **LIME** to selected samples for comparison and validation
* Highlights important **token-level contributions** (if applicable for text inputs)
* Aims to support fairness and transparency in model deployment

This file is dedicated entirely to **explainability**, and complements the main training pipeline by enabling model introspection and stakeholder communication.

