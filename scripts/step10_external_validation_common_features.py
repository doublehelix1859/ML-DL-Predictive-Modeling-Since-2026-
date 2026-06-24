from pipeline_common import *
# Memory-safe external validation: retrain on common features only using selected top models if possible.
import os, pandas as pd, numpy as np, joblib, gc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.metrics import r2_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAVE_XGB=True
except Exception: HAVE_XGB=False
OUT=ensure_dir(os.path.join(ROOT,"STEP10_External_Validation"))
train=pd.read_csv(os.path.join(ROOT,'prestep','encoded_training_dataset.csv'), low_memory=False) if os.path.exists(os.path.join(ROOT,'prestep','encoded_training_dataset.csv')) else load_train()
ext=pd.read_csv(os.path.join(ROOT,'prestep','encoded_external_dataset.csv'), low_memory=False) if os.path.exists(os.path.join(ROOT,'prestep','encoded_external_dataset.csv')) else load_external()
if ext is None: raise SystemExit('No external validation dataset provided. Set EXT_FILE in config.env.')
reg_models={'LinearRegression':LinearRegression(),'Ridge':Ridge(),'ElasticNet':ElasticNet(alpha=.001,l1_ratio=.5,max_iter=20000,random_state=SEED),'SVR':SVR(C=10),'DecisionTreeRegressor':DecisionTreeRegressor(random_state=SEED),'RandomForestRegressor':RandomForestRegressor(n_estimators=150,random_state=SEED,n_jobs=N_JOBS),'GradientBoostingRegressor':GradientBoostingRegressor(random_state=SEED),'HistGradientBoostingRegressor':HistGradientBoostingRegressor(random_state=SEED),'KNeighborsRegressor':KNeighborsRegressor(n_neighbors=15,weights='distance')}
clf_models={'LogisticRegression':LogisticRegression(max_iter=10000,solver='saga',n_jobs=N_JOBS,random_state=SEED),'GaussianNB':GaussianNB(),'SVC':SVC(C=10,probability=True,random_state=SEED),'KNN':KNeighborsClassifier(n_neighbors=15,weights='distance'),'DecisionTreeClassifier':DecisionTreeClassifier(random_state=SEED),'RandomForestClassifier':RandomForestClassifier(n_estimators=150,random_state=SEED,n_jobs=N_JOBS),'GradientBoostingClassifier':GradientBoostingClassifier(random_state=SEED),'HistGradientBoostingClassifier':HistGradientBoostingClassifier(random_state=SEED)}
if HAVE_XGB:
    reg_models['XGBoostRegressor']=XGBRegressor(n_estimators=150,learning_rate=.05,max_depth=4,subsample=.8,colsample_bytree=.8,random_state=SEED,n_jobs=N_JOBS,objective='reg:squarederror')
    clf_models['XGBoostClassifier']=XGBClassifier(n_estimators=150,learning_rate=.05,max_depth=4,subsample=.8,colsample_bytree=.8,random_state=SEED,n_jobs=N_JOBS,eval_metric='logloss')
metric_rows=[]; common_rows=[]
for target in REG_TARGETS:
    if target not in train.columns or target not in ext.columns: continue
    feats=sorted(list(set(numeric_predictor_cols(train,target)).intersection(set(numeric_predictor_cols(ext,target)))))
    common_rows.append({'task':'regression','target':target,'n_common_features':len(feats),'n_common_SNP':sum(is_snp(c) for c in feats),'n_common_gene':sum(is_gene(c) for c in feats)})
    pd.Series(feats).to_csv(os.path.join(OUT,f'{target}_common_features.csv'),index=False,header=['feature'])
    Xtr,ytr=clean_xy(train,target,feats,'regression'); Xex,yex=clean_xy(ext,target,feats,'regression')
    for name,model in reg_models.items():
        steps=[('imputer',SimpleImputer(strategy='median')),('scaler',MaxAbsScaler())]
        if len(feats)>SELECT_K: steps.append(('select',SelectKBest(f_regression,k=SELECT_K)))
        steps.append(('model',model)); pipe=Pipeline(steps)
        try:
            pipe.fit(Xtr,ytr); pred=pipe.predict(Xex); outdir=ensure_dir(os.path.join(OUT,'regression',target,name))
            pd.DataFrame({'y_true':yex.values,'y_pred':pred}).to_csv(os.path.join(outdir,'external_validation_predictions.csv'),index=False)
            plot_regression(yex,pred,os.path.join(outdir,'observed_vs_predicted.png'),f'External {target} | {name}')
            metric_rows.append({'task':'regression','target':target,'model':name,'R2_external':r2_score(yex,pred),'RMSE_external':rmse(yex,pred),'n_common_features':len(feats)})
        except Exception as e: metric_rows.append({'task':'regression','target':target,'model':name,'error':str(e),'n_common_features':len(feats)})
        gc.collect()
for target in CLF_TARGETS:
    if target not in train.columns or target not in ext.columns: continue
    feats=sorted(list(set(numeric_predictor_cols(train,target)).intersection(set(numeric_predictor_cols(ext,target)))))
    common_rows.append({'task':'classification','target':target,'n_common_features':len(feats),'n_common_SNP':sum(is_snp(c) for c in feats),'n_common_gene':sum(is_gene(c) for c in feats)})
    Xtr,ytr_raw=clean_xy(train,target,feats,'classification'); Xex,yex_raw=clean_xy(ext,target,feats,'classification')
    le=LabelEncoder(); ytr=le.fit_transform(ytr_raw.astype(str)); keep=yex_raw.astype(str).isin(le.classes_); Xex=Xex.loc[keep]; yex=le.transform(yex_raw.loc[keep].astype(str))
    for name,model in clf_models.items():
        steps=[('imputer',SimpleImputer(strategy='median')),('scaler',MaxAbsScaler())]
        if len(feats)>SELECT_K: steps.append(('select',SelectKBest(f_classif,k=SELECT_K)))
        steps.append(('model',model)); pipe=Pipeline(steps)
        try:
            pipe.fit(Xtr,ytr); pred=pipe.predict(Xex); prob=None; auc=np.nan
            if hasattr(pipe,'predict_proba'):
                prob=pipe.predict_proba(Xex)
                if len(le.classes_)==2 and len(set(yex))==2: auc=roc_auc_score(yex,prob[:,1])
            outdir=ensure_dir(os.path.join(OUT,'classification',target,name))
            pd.DataFrame({'y_true':yex,'y_pred':pred,'y_true_label':le.inverse_transform(yex),'y_pred_label':le.inverse_transform(pred)}).to_csv(os.path.join(outdir,'external_validation_predictions.csv'),index=False)
            metric_rows.append({'task':'classification','target':target,'model':name,'Accuracy_external':accuracy_score(yex,pred),'AUC_external':auc,'F1_external':f1_score(yex,pred,average='weighted',zero_division=0),'Precision_external':precision_score(yex,pred,average='weighted',zero_division=0),'Recall_external':recall_score(yex,pred,average='weighted',zero_division=0),'n_common_features':len(feats)})
        except Exception as e: metric_rows.append({'task':'classification','target':target,'model':name,'error':str(e),'n_common_features':len(feats)})
        gc.collect()
pd.DataFrame(common_rows).to_csv(os.path.join(OUT,'common_features_by_group.csv'),index=False)
pd.DataFrame(metric_rows).to_csv(os.path.join(OUT,'external_validation_metrics.csv'),index=False)
print("Step10 complete")
