# This step repeats model-based feature-importance summaries for selected models.
# For tree/linear models, feature importances are model properties and remain the same model coefficients/importances.
# The step is kept separate for workflow consistency and output organization.
from pipeline_common import *
from importance_tools import raw_importance_from_model, save_imp_outputs
import joblib, pandas as pd, os
OUT=ensure_dir(os.path.join(ROOT,"STEP8_SHAP_Internal_Validation"))
sel=pd.read_csv(os.path.join(ROOT,"STEP5_Model_Selection","TOP3_selected_models.csv"))
train=pd.read_csv(os.path.join(ROOT,"STEP2_Split","train80.csv"), low_memory=False)
for _,r in sel.iterrows():
    task,target,model=r['task'],r['target'],r['model']
    feats=numeric_predictor_cols(train,target)
    path=os.path.join(ROOT,'STEP3_Regression' if task=='regression' else 'STEP4_Classification',target,model,f'{target}__{model}.joblib')
    if os.path.exists(path):
        pipe=joblib.load(path); imp=raw_importance_from_model(pipe,feats); save_imp_outputs(imp,os.path.join(OUT,task,target,model),f'Validation {task} {target} {model}')
print("Step8 complete")
