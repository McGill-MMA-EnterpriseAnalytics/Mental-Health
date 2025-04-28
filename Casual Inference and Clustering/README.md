This repository contains two notebooks that analyse a mental health survey dataset using causal inference and clustering techniques.
The project aims to uncover factors influencing mental health benefits allocation and to explore the underlying structure of the dataset.

Table of Contents
1.Project Overview
2.Files
3.Causal Inference Notebook
4.Clustering Notebook
5.Key Findings
Project Overview
The analysis focuses on a mental health survey dataset containing demographic information, workplace environment details, and mental health indicators.
The project explores:
- Causal Inference: Understanding what factors causally impact access to mental health benefits.
- Clustering: Investigating whether meaningful groups exist in the dataset based on encoded features.

Files
- Causal Inference.ipynb — Causal inference analysis using machine learning models and feature importance techniques.
- Clustering.ipynb — Clustering analysis using K-Means, GMM, Agglomerative Clustering, and HDBSCAN.

Causal Inference Notebook
1. Objective - Identify causal factors associated with receiving mental health benefits.
2. Methods
   - Propensity score modelling
   - Meta-learners (S-Learner, T-Learner)
   - SHAP (SHapley Additive exPlanations) analysis
   - Feature importance ranking
3. Main Observations
   - Psychological risk indicators (e.g., family history, work interference) strongly influence benefit allocation.
   - Gender and age show significant causal effects.
   - Regional disparities exist (e.g., UK employees less likely to receive benefits).
   - Organisational support (wellness programs) impacts treatment uptake.

4. Business Recommendations
   - Improve targeted mental health initiatives.
   - Address regional inconsistencies.
   - Monitor mental health proactively through early work stress indicators.

Clustering Notebook
1. Objective - Explore natural groupings in the survey data without supervision.
2. Methods
   - Dimensionality reduction via PCA (Principal Component Analysis)
   - Clustering techniques:
       - K-Means
       - Gaussian Mixture Model (GMM)
       - Agglomerative Clustering (Single and Average linkage)
       - HDBSCAN (Density-based clustering)

3. Main Observations
   - Traditional clustering methods (K-Means, GMM, Agglomerative) achieved low silhouette scores (~0.07–0.24).
   - HDBSCAN classified ~90% of data as noise, detecting only two small clusters.
   - The dataset is highly sparse and high-dimensional after encoding, making strong clustering difficult.

4. Challenges Identified
   - Categorical survey responses lose relational meaning after encoding.
   - The dataset exhibits high sparsity, low variance, and overlapping distributions.
   - Survey semantics are rich, but geometric clustering structures are weak.

Key Findings
- Causal Inference provides actionable insights: demographic and psychological factors meaningfully affect mental health benefit distribution.
- Clustering reveals that the dataset does not naturally form strong, distinct groups, suggesting cautious interpretation for unsupervised segmentation.

