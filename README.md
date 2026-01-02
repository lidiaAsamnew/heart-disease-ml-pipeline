# Heart Disease Classification - ML Pipeline

End-to-end ML pipeline for heart disease classification using Logistic Regression and Decision Tree models.

## Overview

Trains and evaluates two classification models on a heart disease dataset (1025 samples, 13 features). Includes data cleaning, feature scaling, model training, evaluation, and model export using joblib.

## Usage

Run `model_training/heart_disease_training.ipynb` in Jupyter Notebook or Google Colab. The notebook:
1. Loads and cleans `heart.csv`
2. Trains Logistic Regression and Decision Tree models
3. Evaluates performance with metrics and 5-fold cross-validation
4. Exports models to `.joblib` files

## Results

| Model | Test Accuracy | CV Accuracy |
|-------|--------------|-------------|
| Logistic Regression | 77.6% | 83.5% |
| Decision Tree | 82.9% | 87.3% |

Decision Tree performs better on this dataset.

## Project Structure

```
model_training/
├── heart_disease_training.ipynb    # Main training notebook
├── heart.csv                        # Dataset
├── logistic_regression_model.joblib
├── decision_tree_model.joblib
├── scaler.joblib
└── feature_names.joblib
```

## Technologies

Python, pandas, scikit-learn, joblib, matplotlib, seaborn
