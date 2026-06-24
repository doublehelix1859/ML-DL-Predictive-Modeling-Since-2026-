# Optional Deep Learning Extension

This repository is primarily PBS-ready for classical ML models. For deep learning, the same input/output structure can be used with MLP or 1D-CNN models.

Recommended design for small-N/high-dimensional biological data:

- Median imputation
- MaxAbs or Standard scaling
- Feature selection or dimensionality reduction before DL
- MLP with dropout and batch normalization
- Early stopping
- Weight decay
- Validation monitoring

Suggested MLP architecture:

```text
Input features
→ Dense(512) + BatchNorm + ReLU + Dropout(0.3)
→ Dense(256) + BatchNorm + ReLU + Dropout(0.3)
→ Dense(64) + ReLU
→ Output
```

Suggested CNN1D architecture:

```text
Input vector
→ Conv1D filters
→ BatchNorm
→ GlobalAveragePooling
→ Dense layers
→ Output
```

The ML pipeline outputs can be used as the reporting template for DL metrics and plots.
