# Encoding Guide for Non-Numeric Variables

Most machine learning and deep learning models require numeric input. Therefore, non-numeric variables must be converted into numeric representations before model training.

## 1. Label encoding

Label encoding converts each class into one integer.

Use label encoding when:

- The variable is binary, such as Wet/Drought or Control/Treated.
- The target variable is a classification label.
- The categories are mutually exclusive and you want one column.

### Example: Treatment

Original:

| Sample_ID | Treatment |
|---|---|
| S1 | Wet |
| S2 | Drought |
| S3 | Wet |

Label encoded:

| Sample_ID | Treatment_encoded |
|---|---:|
| S1 | 1 |
| S2 | 0 |
| S3 | 1 |

Encoding log example:

```text
Treatment label encoding:
Drought = 0
Wet = 1
```

In the pipeline, add this to `config.env`:

```bash
export LABEL_COLS="Treatment"
```

## 2. One-hot encoding

One-hot encoding creates one column per category.

Use one-hot encoding when:

- The variable has more than two categories.
- The categories are not naturally ordered.
- You want to avoid implying that one category is numerically larger than another.

### Example: Population label

Original:

| Sample_ID | Population_Label |
|---|---|
| S1 | indica |
| S2 | japonica |
| S3 | aus |

One-hot encoded:

| Sample_ID | Population_Label_indica | Population_Label_japonica | Population_Label_aus |
|---|---:|---:|---:|
| S1 | 1 | 0 | 0 |
| S2 | 0 | 1 | 0 |
| S3 | 0 | 0 | 1 |

In the pipeline, add this to `config.env`:

```bash
export ONEHOT_COLS="Population_Label"
```

## 3. Example: Cultivation type

Original:

| Sample_ID | Cultivation_Type |
|---|---|
| S1 | Lowland |
| S2 | Upland |
| S3 | Irrigated |

One-hot encoded:

| Sample_ID | Cultivation_Type_Lowland | Cultivation_Type_Upland | Cultivation_Type_Irrigated |
|---|---:|---:|---:|
| S1 | 1 | 0 | 0 |
| S2 | 0 | 1 | 0 |
| S3 | 0 | 0 | 1 |

Recommended config:

```bash
export ONEHOT_COLS="Cultivation_Type,Population_Label"
```

## 4. What the pipeline saves

The preprocessing step saves:

```text
prestep/encoding_summary.txt
prestep/label_encoding_map_<COLUMN>.txt
prestep/onehot_encoding_columns_<COLUMN>.txt
prestep/encoded_training_dataset.csv
prestep/encoded_external_dataset.csv
```

These files allow users to verify exactly how categorical variables were transformed.
