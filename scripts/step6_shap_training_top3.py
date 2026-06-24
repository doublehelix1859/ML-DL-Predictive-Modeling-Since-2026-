from pipeline_common import *
from importance_tools import raw_importance_from_model, save_imp_outputs
import joblib
OUT=ensure_dir(os.path.join(ROOT,"STEP6_SHAP_Training"))
sel=pd.read_csv(os.path.join(ROOT,"STEP5_Model_Selection","TOP3_selected_models.csv"))
train=pd.read_csv(os.path.join(ROOT,"STEP2_Split","train80.csv"), low_memory=False)
for _,r in sel.iterrows():
    task,target,model=r['task'],r['target'],r['model']
    feats=numeric_predictor_cols(train,target)
    if task=='regression': path=os.path.join(ROOT,'STEP3_Regression',target,model,f'{target}__{model}.joblib')
    else: path=os.path.join(ROOT,'STEP4_Classification',target,model,f'{target}__{model}.joblib')
    if os.path.exists(path):
        pipe=joblib.load(path); imp=raw_importance_from_model(pipe,feats); save_imp_outputs(imp,os.path.join(OUT,task,target,model),f'{task} {target} {model}')
print("Step6 complete")
