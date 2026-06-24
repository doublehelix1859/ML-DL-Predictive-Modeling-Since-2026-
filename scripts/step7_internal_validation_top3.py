from pipeline_common import *
import joblib, pandas as pd, os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
OUT=ensure_dir(os.path.join(ROOT,"STEP7_Internal_Validation"))
sel=pd.read_csv(os.path.join(ROOT,"STEP5_Model_Selection","TOP3_selected_models.csv"))
train=pd.read_csv(os.path.join(ROOT,"STEP2_Split","train80.csv"), low_memory=False)
val=pd.read_csv(os.path.join(ROOT,"STEP2_Split","validation20.csv"), low_memory=False)
rows=[]; preds=[]
for _,r in sel.iterrows():
    task,target,model=r['task'],r['target'],r['model']
    feats=numeric_predictor_cols(train,target)
    outdir=ensure_dir(os.path.join(OUT,task,target,model))
    if task=='regression':
        path=os.path.join(ROOT,'STEP3_Regression',target,model,f'{target}__{model}.joblib')
        pipe=joblib.load(path); Xv,yv=clean_xy(val,target,feats,'regression'); pred=pipe.predict(Xv)
        rows.append({'task':task,'target':target,'model':model,'R2_internal':r2_score(yv,pred),'RMSE_internal':rmse(yv,pred)})
        pd.DataFrame({'y_true':yv.values,'y_pred':pred}).to_csv(os.path.join(outdir,'internal_validation_predictions.csv'),index=False)
        plot_regression(yv,pred,os.path.join(outdir,'internal_validation_dotplot.png'),f'Internal {target} | {model}')
    else:
        path=os.path.join(ROOT,'STEP4_Classification',target,model,f'{target}__{model}.joblib')
        pipe=joblib.load(path); Xtr,ytr_raw=clean_xy(train,target,feats,'classification'); Xv,yv_raw=clean_xy(val,target,feats,'classification')
        le=LabelEncoder(); ytr=le.fit_transform(ytr_raw.astype(str)); keep=yv_raw.astype(str).isin(le.classes_); Xv=Xv.loc[keep]; yv=le.transform(yv_raw.loc[keep].astype(str))
        pred=pipe.predict(Xv); prob=None; auc=np.nan
        if hasattr(pipe,'predict_proba'):
            prob=pipe.predict_proba(Xv)
            if len(le.classes_)==2 and len(set(yv))==2: auc=roc_auc_score(yv,prob[:,1])
        rows.append({'task':task,'target':target,'model':model,'Accuracy_internal':accuracy_score(yv,pred),'AUC_internal':auc,'F1_internal':f1_score(yv,pred,average='weighted',zero_division=0),'Precision_internal':precision_score(yv,pred,average='weighted',zero_division=0),'Recall_internal':recall_score(yv,pred,average='weighted',zero_division=0)})
        pd.DataFrame({'y_true':yv,'y_pred':pred,'y_true_label':le.inverse_transform(yv),'y_pred_label':le.inverse_transform(pred)}).to_csv(os.path.join(outdir,'internal_validation_predictions.csv'),index=False)
pd.DataFrame(rows).to_csv(os.path.join(OUT,'internal_validation_metrics.csv'),index=False)
print("Step7 complete")
