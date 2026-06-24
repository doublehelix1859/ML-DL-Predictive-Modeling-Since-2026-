from pipeline_common import *
from sklearn.model_selection import train_test_split
OUT=ensure_dir(os.path.join(ROOT,"STEP2_Split"))
encoded=os.path.join(ROOT,"prestep","encoded_training_dataset.csv")
df=pd.read_csv(encoded, low_memory=False) if os.path.exists(encoded) else load_train()
stratify=None
# Use first classification target for stratification if possible.
for t in CLF_TARGETS:
    if t in df.columns and df[t].nunique(dropna=True) > 1:
        stratify=df[t].astype(str); break
train, val = train_test_split(df, test_size=0.2, random_state=SEED, stratify=stratify)
train.to_csv(os.path.join(OUT,"train80.csv"), index=False)
val.to_csv(os.path.join(OUT,"validation20.csv"), index=False)
with open(os.path.join(OUT,"split_info.txt"),"w") as f:
    f.write(f"Train rows: {train.shape[0]}\nValidation rows: {val.shape[0]}\nColumns: {df.shape[1]}\n")
print("Step2 complete")
