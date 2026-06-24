# ML-DL-Predictive-Modeling-Since-2026-
ML/DL phenotype prediction in rice pipeline

# Machine Learning / Deep Learning Pipeline for Phenotype Prediction

## Overview

This repository provides an end-to-end machine learning and deep learning framework for predicting complex phenotypes from multi-modal biological datasets.

The pipeline supports:

* Regression traits (e.g., yield, fecundity, grain number)
* Classification traits (e.g., flowering success, disease status)
* Feature importance analysis (SHAP)
* Internal validation
* External validation
* Biological interpretation of predictive features

Applications include:

* Evolutionary Biology
* Plant Breeding
* Functional Genomics
* Precision Agriculture
* Biomedical Prediction

---

# Pipeline Overview

The workflow consists of 10 sequential steps:

1. Dataset Diagnostics
2. Train/Validation Split
3. Regression Model Training
4. Classification Model Training
5. Top Model Selection
6. SHAP Analysis (Training Set)
7. Internal Validation
8. SHAP Analysis (Validation Set)
9. SHAP Comparison
10. External Validation

---

# Quick Start

## 1. Clone Repository

git clone <repository_url>

## 2. Prepare Input Files

Place your files inside:

input/

Required files:

training_dataset.csv
external_validation_dataset.csv

## 3. Submit Pipeline

bash submit_all_steps.sh

or submit manually:

qsub step1_diagnostics.pbs
qsub step2_split.pbs
...
qsub step10_external_validation_common_features.pbs

---

# Supported Data Types

The pipeline can use:

* Phenotypes
* Genotypes (SNPs)
* Gene expression
* Environmental variables

individually or in combination.

---

# Outputs

The pipeline automatically generates:

* Model performance tables
* Feature importance results
* SHAP summaries
* Confusion matrices
* Regression plots
* External validation reports
* Publication-ready figures

---

# Citation

If you use this pipeline in your research, please cite:

Kim M., Franks S.J. et al.

Machine Learning Framework for Phenotype Prediction in Rice.

Evolution 2026.
