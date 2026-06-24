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

# Input File Format Guide

## Overview

This pipeline is designed for phenotype prediction using machine learning (ML) and deep learning (DL).

Supported data types include:

* Phenotypes
* Genotypes (SNPs)
* Gene expression
* Environmental variables
* Population labels
* Experimental metadata

All input files must be provided as CSV files with samples as rows and features as columns.

---

# Example Input Dataset

Each row represents one biological sample (individual plant, animal, or patient).

| IRGC_ID  | FilledGrain# | Flowering_Success | 1_5039 | 1_8264 | OS01G0100100 | OS01G0100200 | BIO1 | Cultivation_Type | Treatment |
| -------- | ------------ | ----------------- | ------ | ------ | ------------ | ------------ | ---- | ---------------- | --------- |
| IRGC1001 | 430          | 1                 | 0      | 2      | 8.12         | 3.44         | 21.3 | Improved         | Wet       |
| IRGC1002 | 210          | 0                 | 1      | 1      | 5.34         | 4.82         | 19.8 | Landrace         | Dry       |
| IRGC1003 | 520          | 1                 | 2      | 0      | 9.21         | 2.77         | 22.1 | Improved         | Wet       |

---

# Data Type Descriptions

## 1. Sample Identifiers

Unique identifiers used for merging datasets.

Examples:

| IRGC_ID  |
| -------- |
| IRGC1001 |
| IRGC1002 |
| IRGC1003 |

Typical identifier columns:

* IRGC_ID
* Universal_ID
* Sample_ID
* Accession_ID

These columns are not used as predictors.

---

## 2. Phenotypes

Phenotypes are observed traits that serve as prediction targets or predictors.

Examples:

| FilledGrain# | PlantHeight | FloweringTime |
| ------------ | ----------- | ------------- |
| 430          | 120.4       | 98            |
| 210          | 95.3        | 110           |
| 520          | 134.1       | 92            |

Examples of phenotype traits:

* FilledGrain#
* Flowering_Success
* PlantHeight
* GrainYield
* FloweringTime
* ChlorophyllContent

---

## 3. SNP Genotypes

SNP columns must use:

CHR_POSITION

format.

Examples:

| 1_5039 | 1_8264 | 2_15024 |
| ------ | ------ | ------- |
| 0      | 2      | 1       |
| 1      | 1      | 0       |
| 2      | 0      | 2       |

Genotype encoding:

| Value | Meaning              |
| ----- | -------------------- |
| 0     | Homozygous Reference |
| 1     | Heterozygous         |
| 2     | Homozygous Alternate |

Missing values should be imputed before modeling.

---

## 4. Gene Expression

Gene expression columns should use gene IDs.

Examples:

| OS01G0100100 | OS01G0100200 | OS03G0214500 |
| ------------ | ------------ | ------------ |
| 8.12         | 3.44         | 5.76         |
| 5.34         | 4.82         | 6.91         |
| 9.21         | 2.77         | 4.55         |

Recommended preprocessing:

* TMM normalization
* CPM normalization
* TPM normalization
* log2 transformation
* z-score normalization

---

## 5. Environmental Variables

Environmental predictors describe climate and soil conditions.

Examples:

| BIO1 | BIO12 | Soil_pH |
| ---- | ----- | ------- |
| 21.3 | 980   | 5.8     |
| 19.8 | 1102  | 6.1     |
| 22.1 | 875   | 5.4     |

Typical variables:

Climate

* BIO1 Annual Mean Temperature
* BIO12 Annual Precipitation
* BIO15 Precipitation Seasonality

Soil

* Soil_pH
* Clay
* Sand
* Organic_Carbon
* Nitrogen

---

# Encoding Non-Numeric Variables

Many ML and DL algorithms require numeric input.

Therefore categorical variables must be encoded.

---

## Label Encoding

Used when categories have only a few levels.

Example:

Original

| Treatment |
| --------- |
| Wet       |
| Dry       |
| Wet       |
| Dry       |

Encoded

| Treatment |
| --------- |
| 1         |
| 0         |
| 1         |
| 0         |

Encoding scheme:

| Category | Encoded Value |
| -------- | ------------- |
| Dry      | 0             |
| Wet      | 1             |

Another example:

| Subspecies |
| ---------- |
| japonica   |
| indica     |

Encoded

| Subspecies |
| ---------- |
| 0          |
| 1          |

---

## One-Hot Encoding

Used when categories contain multiple classes.

Example:

Original

| Cultivation_Type |
| ---------------- |
| Landrace         |
| Improved         |
| Modern           |

One-hot encoded

| Cultivation_Type_Landrace | Cultivation_Type_Improved | Cultivation_Type_Modern |
| ------------------------- | ------------------------- | ----------------------- |
| 1                         | 0                         | 0                       |
| 0                         | 1                         | 0                       |
| 0                         | 0                         | 1                       |

Another example:

Original

| Population |
| ---------- |
| Tropical   |
| Temperate  |
| Aromatic   |

Encoded

| Population_Tropical | Population_Temperate | Population_Aromatic |
| ------------------- | -------------------- | ------------------- |
| 1                   | 0                    | 0                   |
| 0                   | 1                    | 0                   |
| 0                   | 0                    | 1                   |

Recommended variables for one-hot encoding:

* Population
* Cultivation_Type
* Country
* Experimental_Group
* Variety_Class

---

# Missing Value Handling

Recommended imputation methods:

| Data Type       | Method |
| --------------- | ------ |
| SNPs            | Mode   |
| Gene Expression | Mean   |
| Phenotypes      | Median |
| Environment     | Median |

Examples:

SNP

0, 2, NA, 2, 1

↓

0, 2, 2, 2, 1

Gene expression

5.3, 6.1, NA, 7.4

↓

5.3, 6.1, 6.27, 7.4

---

# Output File Requirements

The final modeling file should contain:

* One sample per row
* One feature per column
* Numeric predictors
* Encoded categorical variables
* No duplicate samples
* Missing values imputed

Example:

| IRGC_ID  | FilledGrain# | Treatment | Cultivation_Type_Modern | 1_5039 | OS01G0100100 | BIO1 |
| -------- | ------------ | --------- | ----------------------- | ------ | ------------ | ---- |
| IRGC1001 | 430          | 1         | 1                       | 0      | 8.12         | 21.3 |
| IRGC1002 | 210          | 0         | 0                       | 1      | 5.34         | 19.8 |
| IRGC1003 | 520          | 1         | 0                       | 2      | 9.21         | 22.1 |

This format is directly compatible with the ML/DL phenotype prediction pipeline.




# Citation

If you use this pipeline in your research, please cite:

Minju Kim

Integrativa Machine Learning Framework for Phenotype Prediction in Rice.

Evolution 2026.
