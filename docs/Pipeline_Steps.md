# Step-by-Step Pipeline Details

## Step 0: Preprocessing

Purpose:

- Drop user-specified non-predictor columns.
- Convert binary categorical columns with label encoding.
- Convert multi-class categorical columns with one-hot encoding.
- Save all encoding logs.

Inputs:

- `<TRAINING_DATASET.csv>`
- Optional: `<EXTERNAL_VALIDATION_DATASET.csv>`

Outputs:

- `prestep/encoded_training_dataset.csv`
- `prestep/encoded_external_dataset.csv`
- `prestep/encoding_summary.txt`

## Step 1: Diagnostics

Purpose:

- Check dimensions, missingness, feature types, and target availability.

Outputs:

- `STEP1_Diagnostics/training_shape.txt`
- `STEP1_Diagnostics/missingness_training.csv`
- `STEP1_Diagnostics/feature_type_summary.csv`

## Step 2: Data split

Purpose:

- Split the training dataset into 80% training and 20% validation.

Outputs:

- `STEP2_Split/train80.csv`
- `STEP2_Split/validation20.csv`

## Step 3: Regression model training

Purpose:

- Train regression models for numeric targets.

Models:

- Linear Regression
- Ridge
- Elastic Net
- SVR
- Decision Tree
- Random Forest
- Gradient Boosting
- HistGradientBoosting
- XGBoost, if installed
- KNN

Outputs:

- `STEP3_Regression/regression_metrics.csv`
- `STEP3_Regression/regression_predictions_by_model.csv`
- model `.joblib` files
- R² and RMSE barplots

## Step 4: Classification model training

Purpose:

- Train classification models for categorical targets.

Models:

- Logistic Regression
- Gaussian Naive Bayes
- SVC
- KNN
- Decision Tree
- Random Forest
- XGBoost, if installed

Outputs:

- `STEP4_Classification/classification_metrics.csv`
- ROC curves
- Precision-recall curves
- Confusion matrices

## Step 5: Model selection

Purpose:

- Select top 3 regression models by validation R².
- Select top 3 classification models by validation ROC-AUC or F1.

Output:

- `STEP5_Model_Selection/TOP3_selected_models.csv`

## Step 6: Feature importance on training data

Purpose:

- Estimate the most influential predictors from top models.

Outputs:

- `top50_overall.csv`
- `top30_SNP.csv`
- `top30_gene_expression.csv`
- importance plots

## Step 7: Internal validation

Purpose:

- Apply selected models to the 20% validation dataset.

Outputs:

- observed-vs-predicted plots for regression
- confusion matrices for classification
- internal validation metrics

## Step 8: Feature importance on validation data

Purpose:

- Check whether feature importance is stable in held-out samples.

## Step 9: Feature importance comparison

Purpose:

- Compare top features across top models.

Outputs:

- overlap tables
- Venn diagrams, when available

## Step 10: External validation

Purpose:

- Test model transferability to an independent dataset.
- Retrain training data using only features common between training and external validation datasets.

Outputs:

- `common_features_by_group.csv`
- `external_validation_metrics.csv`
- `external_validation_predictions.csv`
- external validation plots
