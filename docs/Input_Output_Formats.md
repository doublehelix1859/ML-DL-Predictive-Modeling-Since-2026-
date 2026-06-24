# Input and Output File Format Guide

## 1. General rule

Input files must be comma-separated CSV files. Each row represents one biological sample, accession, individual, plot, patient, or experimental unit. Each column represents either an identifier, target trait, predictor feature, or metadata variable.

## 2. Training dataset format

Required file name is defined in `config.env`:

```bash
export TRAIN_FILE="<TRAINING_DATASET_FILENAME.csv>"
```

Example training dataset:

| Sample_ID | FilledGrain# | Flowering_Success | Treatment | Cultivation_Type | 1_5039 | 1_8810 | OS01T0133400-01 | OS03T0289100-01 | BIO1 | BIO12 |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| IRGC_001 | 120 | 1 | Wet | Lowland | 0 | 1 | 4.23 | 1.12 | 26.3 | 1430 |
| IRGC_002 | 15 | 0 | Drought | Upland | 2 | 0 | 2.91 | 3.44 | 24.8 | 1102 |
| IRGC_003 | 85 | 1 | Wet | Lowland | 1 | 1 | 5.02 | 2.05 | 25.7 | 1344 |

## 3. External validation dataset format

Required file name is defined in `config.env`:

```bash
export EXT_FILE="<EXTERNAL_VALIDATION_DATASET_FILENAME.csv>"
```

Example external validation dataset:

| Sample_ID | FilledGrain# | Flowering_Success | Treatment | 1_5039 | 1_8810 | OS01T0133400-01 | OS03T0289100-01 |
|---|---:|---:|---|---:|---:|---:|---:|
| EXT_001 | 70 | 1 | Salt | 0 | 1 | 3.89 | 1.76 |
| EXT_002 | 8 | 0 | Salt | 2 | 0 | 2.21 | 3.02 |

External validation uses only features common to both training and external datasets.

## 4. Data type examples

### Phenotype predictors

These are measured biological traits that can be used to predict another phenotype.

Examples:

| Feature | Meaning | Type |
|---|---|---|
| DTF_50 | days to 50% flowering | numeric |
| Plant_height_cm | plant height | numeric |
| Spikelet_number | number of spikelets | numeric |
| CCI | chlorophyll concentration index | numeric |

### Regression target

A continuous numeric trait.

Examples:

| Target | Meaning |
|---|---|
| FilledGrain# | number of filled grains |
| Plant_height_cm | plant height |
| Disease_severity_score | continuous disease score |

### Classification target

A categorical trait or binary outcome.

Examples:

| Target | Values |
|---|---|
| Flowering_Success | 0 / 1 |
| Disease_Status | healthy / diseased |
| Subspecies | indica / japonica |
| Treatment_Response | responder / nonresponder |

### SNP predictors

SNP columns should be encoded as 0, 1, or 2 and named using chromosome-position format.

Examples:

| SNP column | Meaning |
|---|---|
| 1_5039 | chromosome 1, position 5039 |
| 2_25864639 | chromosome 2, position 25864639 |
| 11_9469340 | chromosome 11, position 9469340 |

SNP encoding example:

| Genotype | Code |
|---|---:|
| homozygous reference | 0 |
| heterozygous | 1 |
| homozygous alternate | 2 |

### Gene-expression predictors

Gene-expression columns should be numeric, often log-transformed expression values.

Examples:

| Gene-expression column | Example value |
|---|---:|
| OS01T0133400-01 | 4.23 |
| OS03T0289100-01 | 1.12 |
| OS11T0159000-01 | 0.88 |

### Environmental predictors

Environmental columns may include climate, soil, or treatment-level variables.

Examples:

| Environmental column | Meaning |
|---|---|
| BIO1 | annual mean temperature |
| BIO12 | annual precipitation |
| Soil_pH | soil pH |
| Annual_solar_radiation | solar radiation |

## 5. Output file formats

### Metrics CSV

Regression metrics:

| target | model | R2_train | R2_test | RMSE_train | RMSE_test |
|---|---|---:|---:|---:|---:|

Classification metrics:

| target | model | Accuracy_test | AUC_test | F1_test | Precision_test | Recall_test |
|---|---|---:|---:|---:|---:|---:|

### Prediction CSV

Regression prediction output:

| Sample_ID | target | model | y_true | y_pred |
|---|---|---|---:|---:|

Classification prediction output:

| Sample_ID | target | model | y_true | y_pred | predicted_probability |
|---|---|---|---|---|---:|

### Feature-importance CSV

| feature | importance | feature_type |
|---|---:|---|
| Spikelet_number | 0.152 | Phenotype |
| 2_25864639 | 0.034 | SNP |
| OS01T0133400-01 | 0.018 | GeneExpression |
