from pipeline_common import *
OUT=ensure_dir(os.path.join(ROOT,"STEP1_Diagnostics"))
# Prefer encoded file if it exists.
encoded=os.path.join(ROOT,"prestep","encoded_training_dataset.csv")
df=pd.read_csv(encoded, low_memory=False) if os.path.exists(encoded) else load_train()

pd.Series(df.columns, name="column").to_csv(os.path.join(OUT,"training_columns.csv"), index=False)
miss=pd.DataFrame({"feature":df.columns,"missing_count":df.isna().sum().values,"missing_rate":df.isna().mean().values,"feature_type":[feature_type(c) for c in df.columns]})
miss.to_csv(os.path.join(OUT,"missingness_training.csv"), index=False)
feat_summary=miss.groupby("feature_type").size().reset_index(name="n_features")
feat_summary.to_csv(os.path.join(OUT,"feature_type_summary.csv"), index=False)
with open(os.path.join(OUT,"training_shape.txt"),"w") as f:
    f.write(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\n")
    f.write(f"Regression targets: {REG_TARGETS}\nClassification targets: {CLF_TARGETS}\n")
    for t in ALL_TARGETS:
        f.write(f"Target {t} present: {t in df.columns}\n")
print("Step1 complete")
