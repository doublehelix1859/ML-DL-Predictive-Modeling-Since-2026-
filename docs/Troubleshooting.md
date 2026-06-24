# Troubleshooting

## PYTHONPATH: unbound variable

Cause:

The PBS script uses `set -u`, but `PYTHONPATH` is undefined.

Fix:

Use:

```bash
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
```

instead of:

```bash
export PYTHONPATH="$ROOT:$PYTHONPATH"
```

## Job killed without Python traceback

Likely cause:

- Memory limit exceeded.

Solutions:

- Use `FAST_MODE=1`.
- Reduce `SELECT_K`.
- Reduce number of estimators for Random Forest or XGBoost.
- Run external validation models one at a time.

## External validation dimension mismatch

Cause:

Training and external datasets do not share identical feature columns.

Fix:

Step 10 automatically uses common features only:

```text
training_features ∩ external_features
```

## XGBoost not installed

The pipeline skips XGBoost automatically if it is unavailable.

Install if needed:

```bash
conda install -c conda-forge xgboost
```
