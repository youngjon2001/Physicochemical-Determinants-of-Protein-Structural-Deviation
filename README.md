# Physicochemical Determinants of Protein Structural Deviation 
## A Machine Learning Approach to Predict RMSD in Protein Tertiary Structures

---

## Overview

This project implements a supervised regression pipeline to predict **Root Mean Square Deviation (RMSD)** of protein tertiary structures using physicochemical descriptors (F1–F9).

RMSD quantifies structural deviation between predicted and experimentally validated protein conformations. Reliable RMSD estimation supports:

- Structural validation in computational biology  
- Protein engineering optimization  
- Drug discovery and molecular docking workflows  
- Computational screening before laboratory validation  

The notebook evaluates linear and nonlinear models under controlled preprocessing to determine the most robust predictive solution.

---

## Problem Statement

**Objective:**  
Predict continuous RMSD values from nine structural descriptors.

**Prediction Type:**  
Supervised regression.

**Domain Context:**  
Lower RMSD indicates higher structural fidelity. Predicting RMSD enables rapid structural quality assessment and reduces experimental validation cost in protein science workflows.

---
## 🔬 Biological Motivation

- Protein folding is governed by:

- Hydrophobic collapse

- Surface-solvent interactions

- Secondary structure formation

- Spatial packing constraints

Misfolded or structurally unstable proteins are associated with:

- Neurodegenerative disorders (e.g., aggregation diseases)

- Loss of enzymatic function

- Structural instability from mutations

Predicting structural deviation computationally reduces reliance on expensive wet-lab validation and accelerates:

- Drug discovery pipelines

- Mutation impact assessment

- Protein engineering workflows

  ---
  
## Dataset Description

### Target Variable

- `RMSD` (continuous)
  - Log-transformed for linear modeling due to strong positive skewness.

### Feature Set

Nine numerical descriptors:

- `F1` – `F9`
- Represent physicochemical and tertiary structure properties of proteins.

### Statistical Characteristics

- Strong positive skew across most variables  
- Extreme skew in `F7`  
- High-value outliers present  
- Non-normal distributions confirmed via Q-Q analysis  

These properties directly influenced preprocessing and model selection strategy.

---

## Exploratory Data Analysis (EDA)

### Analyses Performed

- Distribution inspection (Histogram + KDE)
- Skewness quantification
- Outlier detection (Boxplots)
- Normality assessment (Q-Q plots)
- Correlation structure evaluation

### Key Insights

- Majority of features are right-skewed.
- RMSD distribution is positively skewed.
- Linear modeling assumptions are violated without transformation.
- Feature interactions demonstrate nonlinear structure.

Conclusion: Nonlinear modeling approaches are necessary to adequately capture structural behavior.

---

## Modeling Strategy

### Train/Test Split

- 80% training
- 20% testing
- `random_state=42` for reproducibility

Separate preprocessing pipelines were maintained for linear and nonlinear models to prevent leakage and ensure fair comparison.

---

## Models Evaluated

### 1. Linear Regression

**Role:** Baseline model.

**Preprocessing:**
- Log transformation of RMSD
- Log transformation of highly skewed features
- Standard scaling

**Strengths:**
- Interpretability
- Minimal computational overhead

**Limitations:**
- Assumes linear relationships
- High bias under nonlinear structural behavior

Observed outcome: Underfitting relative to nonlinear alternatives.

---

### 2. Support Vector Regressor (SVR – RBF Kernel)

**Role:** Nonlinear benchmark.

**Rationale:**  
Protein structural relationships are not linearly separable. The RBF kernel enables nonlinear mapping into higher-dimensional space.

**Strengths:**
- Captures nonlinear interactions
- Strong regularization control
- Effective on moderate-sized structured datasets

**Limitations:**
- Sensitive to hyperparameters
- Requires feature scaling
- Less interpretable

SVR demonstrated clear performance gains over linear regression.

---

### 3. Random Forest Regressor

**Role:** Ensemble benchmark.

**Strengths:**
- Robust to skewness and outliers
- Captures nonlinear feature interactions
- Provides feature importance metrics

**Limitations:**
- Potential overfitting without tuning
- Larger model footprint

Performance: Strong generalization with moderate variance.

---

### 4. XGBoost Regressor

**Role:** Final candidate model.

**Rationale:**  
Gradient boosting sequentially optimizes residuals, making it highly effective for skewed continuous targets.

**Strengths:**
- Strong bias–variance balance
- Built-in regularization
- Handles non-normal data effectively
- Production-scalable

**Limitations:**
- Requires careful hyperparameter tuning
- Reduced interpretability compared to linear models

---

## Final Model Selection

**Selected Model:** XGBoost Regressor

### Selection Criteria

Models were compared across:

- RMSE
- MAE
- Train/Test generalization gap
- Stability of predictions
- Bias–variance characteristics

### Justification

- Linear Regression: High bias  
- SVR: Improved nonlinear fit  
- Random Forest: Strong but slightly higher variance  
- XGBoost: Lowest test RMSE and best generalization balance  

XGBoost provided the optimal trade-off between predictive accuracy, stability, and deployment readiness.

---

## Bias–Variance Assessment

- Linear Regression: High bias
- SVR: Moderate bias, controlled variance
- Random Forest: Low bias, moderate variance
- XGBoost: Low bias with regularized variance

The dataset’s nonlinear structure makes tree-based boosting the most appropriate modeling strategy.

---

## Model Evaluation

Primary Metrics:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Predicted vs actual RMSD analysis confirmed:

- Linear regression underfits
- Nonlinear models capture structural interactions
- XGBoost achieves strongest alignment with true RMSD values

Generalization confirmed through controlled holdout evaluation.

---

## 🧠 Key Insights

- Hydrophobic exposure metrics strongly influence RMSD.

- Structural penalty features correlate with higher deviation.

- Extreme skew (F7) represents rare but biologically significant structural distortions.

- Nonlinear models outperform linear models due to complex structural interactions.

  ---

## Model Explainability

Feature importance extracted from tree-based models.

Explainability is critical in structural bioinformatics because:

- Identifies dominant structural determinants of deviation
- Supports biological interpretation
- Guides protein engineering hypotheses

Feature importance implementation is included in the notebook.

---

## Inference Strategy

For unseen protein samples:

1. Apply identical preprocessing pipeline
2. Preserve feature ordering
3. Generate predictions
4. Apply inverse transformation if log target used

Predictions can be used for ranking, screening, or validation workflows.

---

## Model Persistence

Models serialized using:

```python
import joblib
joblib.dump(model, "xgb_rmsd_model.pkl")
# Physicochemical-Determinants-of-Protein-Structural-Deviation
