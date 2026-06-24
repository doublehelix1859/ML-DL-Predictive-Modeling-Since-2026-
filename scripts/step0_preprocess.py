from pipeline_common import *
from sklearn.preprocessing import LabelEncoder

OUT = ensure_dir(os.path.join(ROOT, "prestep"))
label_cols=[x.strip() for x in os.environ.get("LABEL_COLS","").split(",") if x.strip() and not x.startswith("<")]
onehot_cols=[x.strip() for x in os.environ.get("ONEHOT_COLS","").split(",") if x.strip() and not x.startswith("<")]

def preprocess(df, dataset_name):
    df=df.copy()
    log=[]
    # Drop user-defined columns if they exist, except targets and ID.
    drop=[c for c in DROP_ALWAYS if c in df.columns and c not in ALL_TARGETS and c != SAMPLE_ID_COL]
    if drop:
        df=df.drop(columns=drop)
    log.append(f"Dropped columns: {drop}")
    # Label encoding for binary variables.
    for col in label_cols:
        if col in df.columns:
            le=LabelEncoder()
            mask=df[col].notna()
            df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
            with open(os.path.join(OUT, f"label_encoding_map_{dataset_name}_{col}.txt"), "w") as f:
                for cls, val in zip(le.classes_, range(len(le.classes_))):
                    f.write(f"{cls} = {val}\n")
            log.append(f"Label encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
    # One-hot encoding for multi-class variables.
    for col in onehot_cols:
        if col in df.columns:
            dummies=pd.get_dummies(df[col].astype(str), prefix=col, dummy_na=False)
            dummies=dummies.astype(int)
            df=df.drop(columns=[col]).join(dummies)
            with open(os.path.join(OUT, f"onehot_encoding_columns_{dataset_name}_{col}.txt"), "w") as f:
                for c in dummies.columns: f.write(c+"\n")
            log.append(f"One-hot encoded {col}: {list(dummies.columns)}")
    return df, log

train=load_train()
train2, log_train=preprocess(train, "training")
train2.to_csv(os.path.join(OUT, "encoded_training_dataset.csv"), index=False)

ext=load_external()
log_ext=[]
if ext is not None:
    ext2, log_ext=preprocess(ext, "external")
    ext2.to_csv(os.path.join(OUT, "encoded_external_dataset.csv"), index=False)

with open(os.path.join(OUT, "encoding_summary.txt"), "w") as f:
    f.write("Preprocessing and encoding summary\n")
    f.write("==================================\n\n")
    f.write(f"Training input: {train_path()}\n")
    f.write(f"Training output: {os.path.join(OUT,'encoded_training_dataset.csv')}\n")
    f.write(f"Training shape before: {train.shape}\n")
    f.write(f"Training shape after: {train2.shape}\n\n")
    for x in log_train: f.write(x+"\n")
    if ext is not None:
        f.write("\nExternal validation preprocessing\n")
        for x in log_ext: f.write(x+"\n")
print("Step0 complete")
