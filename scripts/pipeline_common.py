"""
Shared helper functions for the phenotype prediction pipeline.
Users should normally edit config.env, not this file.
"""
import os, re, math, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = int(os.environ.get("SEED", "42"))
ROOT = os.environ.get("ROOT", os.getcwd())
INPUT_DIR = os.environ.get("INPUT_DIR", os.path.join(ROOT, "input"))
TRAIN_FILE = os.environ.get("TRAIN_FILE", "training_dataset.csv")
EXT_FILE = os.environ.get("EXT_FILE", "")
SAMPLE_ID_COL = os.environ.get("SAMPLE_ID_COL", "Sample_ID")
REG_TARGETS = [x.strip() for x in os.environ.get("REG_TARGETS", "").split(",") if x.strip() and not x.startswith("<")]
CLF_TARGETS = [x.strip() for x in os.environ.get("CLF_TARGETS", "").split(",") if x.strip() and not x.startswith("<")]
ALL_TARGETS = REG_TARGETS + CLF_TARGETS
DROP_ALWAYS = [x.strip() for x in os.environ.get("DROP_ALWAYS", "").split(",") if x.strip() and not x.startswith("<")]
SNP_REGEX = os.environ.get("SNP_REGEX", r"^\d+_\d+$")
GENE_PREFIX = os.environ.get("GENE_PREFIX", "OS")
ENV_PREFIX = os.environ.get("ENV_PREFIX", "BIO")
N_JOBS = int(os.environ.get("N_JOBS", "2"))
FAST_MODE = os.environ.get("FAST_MODE", "0") == "1"
SELECT_K = int(os.environ.get("SELECT_K", "12000"))

SNP_RE = re.compile(SNP_REGEX)

MODEL_COLORS={
 'LinearRegression':'#1f77b4','Ridge':'#aec7e8','ElasticNet':'#ff7f0e','SVR':'#2ca02c',
 'DecisionTreeRegressor':'#98df8a','RandomForestRegressor':'#8c564b','GradientBoostingRegressor':'#e377c2',
 'HistGradientBoostingRegressor':'#7f7f7f','XGBoostRegressor':'#bcbd22','KNeighborsRegressor':'#d62728',
 'LogisticRegression':'#1f77b4','GaussianNB':'#17becf','SVC':'#2ca02c','KNN':'#d62728',
 'DecisionTreeClassifier':'#98df8a','RandomForestClassifier':'#8c564b','GradientBoostingClassifier':'#e377c2',
 'HistGradientBoostingClassifier':'#7f7f7f','XGBoostClassifier':'#bcbd22'
}
MODEL_CATEGORY={
 'LinearRegression':'Linear','Ridge':'Linear','ElasticNet':'Linear','LogisticRegression':'Linear',
 'SVR':'Kernel','SVC':'Kernel','KNeighborsRegressor':'Distance','KNN':'Distance','GaussianNB':'Bayesian',
 'DecisionTreeRegressor':'Tree','DecisionTreeClassifier':'Tree','RandomForestRegressor':'TreeEnsemble','RandomForestClassifier':'TreeEnsemble',
 'GradientBoostingRegressor':'TreeEnsemble','GradientBoostingClassifier':'TreeEnsemble','HistGradientBoostingRegressor':'TreeEnsemble',
 'HistGradientBoostingClassifier':'TreeEnsemble','XGBoostRegressor':'TreeEnsemble','XGBoostClassifier':'TreeEnsemble'
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def train_path():
    return os.path.join(INPUT_DIR, TRAIN_FILE)

def ext_path():
    return os.path.join(INPUT_DIR, EXT_FILE) if EXT_FILE else ""

def load_train():
    return pd.read_csv(train_path(), low_memory=False)

def load_external():
    p = ext_path()
    if not p or not os.path.exists(p):
        return None
    return pd.read_csv(p, low_memory=False)

def is_snp(col):
    return bool(SNP_RE.match(str(col)))

def is_gene(col):
    return str(col).upper().startswith(GENE_PREFIX.upper())

def is_env(col):
    return str(col).upper().startswith(ENV_PREFIX.upper())

def feature_type(col):
    if is_snp(col): return "SNP"
    if is_gene(col): return "GeneExpression"
    if is_env(col): return "Environment"
    if col in DROP_ALWAYS or col in ALL_TARGETS or col == SAMPLE_ID_COL: return "Metadata_or_Target"
    return "Phenotype_or_Other"

def predictor_cols(df, target):
    drop = set(DROP_ALWAYS + ALL_TARGETS + [SAMPLE_ID_COL])
    return [c for c in df.columns if c not in drop]

def numeric_predictor_cols(df, target):
    cols = predictor_cols(df, target)
    out=[]
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]): out.append(c)
        else:
            converted = pd.to_numeric(df[c], errors='coerce')
            if converted.notna().sum() > 0: out.append(c)
    return out

def clean_xy(df, target, features, task):
    X = df[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    y = df[target]
    if task == "regression":
        y = pd.to_numeric(y, errors="coerce")
        keep = y.notna()
    else:
        y = y.astype(str).str.strip()
        keep = y.notna() & (y != "") & (y.str.lower() != "nan")
    return X.loc[keep], y.loc[keep]

def rmse(y, pred):
    y=np.asarray(y,dtype=float); pred=np.asarray(pred,dtype=float)
    return math.sqrt(np.mean((y-pred)**2))

def plot_regression(y, pred, out, title):
    y=np.asarray(y,dtype=float); pred=np.asarray(pred,dtype=float)
    plt.figure(figsize=(6,6))
    plt.scatter(y, pred, s=18, alpha=.75)
    lo=float(np.nanmin([np.nanmin(y),np.nanmin(pred)])); hi=float(np.nanmax([np.nanmax(y),np.nanmax(pred)]))
    plt.plot([lo,hi],[lo,hi], 'r--', lw=1.5, label='y = x')
    if len(y) > 1:
        z=np.polyfit(y, pred, 1); xs=np.linspace(lo,hi,100)
        plt.plot(xs,z[0]*xs+z[1], 'r:', lw=1.5, label='best-fit')
    plt.xlabel('Observed', fontweight='bold'); plt.ylabel('Predicted', fontweight='bold')
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()

def plot_metric_bar(df, metric, out, title, positive_only=False):
    d=df.copy()
    d[metric]=pd.to_numeric(d[metric], errors='coerce')
    if positive_only: d=d[d[metric] >= 0]
    if d.empty: return
    d=d.sort_values(metric, ascending=True)
    colors=[MODEL_COLORS.get(m,'#333333') for m in d['model']]
    plt.figure(figsize=(10, max(4, .45*len(d))))
    bars=plt.barh(d['model'], d[metric], color=colors)
    plt.xlabel(metric, fontweight='bold'); plt.title(title)
    for b,v in zip(bars,d[metric]):
        if pd.notna(v): plt.text(float(v), b.get_y()+b.get_height()/2, f' {float(v):.2f}', va='center')
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
