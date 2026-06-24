from pipeline_common import *
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
try:
    from xgboost import XGBClassifier
    HAVE_XGB=True
except Exception:
    HAVE_XGB=False
OUT=ensure_dir(os.path.join(ROOT,"STEP4_Classification"))
train=pd.read_csv(os.path.join(ROOT,"STEP2_Split","train80.csv"), low_memory=False)
val=pd.read_csv(os.path.join(ROOT,"STEP2_Split","validation20.csv"), low_memory=False)
models={'LogisticRegression':LogisticRegression(max_iter=10000,solver='saga',n_jobs=N_JOBS,random_state=SEED),'GaussianNB':GaussianNB(),'SVC':SVC(C=10,probability=True,random_state=SEED),'KNN':KNeighborsClassifier(n_neighbors=15,weights='distance'),'DecisionTreeClassifier':DecisionTreeClassifier(random_state=SEED),'RandomForestClassifier':RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=N_JOBS),'GradientBoostingClassifier':GradientBoostingClassifier(random_state=SEED),'HistGradientBoostingClassifier':HistGradientBoostingClassifier(random_state=SEED)}
if HAVE_XGB: models['XGBoostClassifier']=XGBClassifier(n_estimators=300,learning_rate=.05,max_depth=5,subsample=.8,colsample_bytree=.8,random_state=SEED,n_jobs=N_JOBS,eval_metric='logloss')
rows=[]; pred_rows=[]
for target in CLF_TARGETS:
    if target not in train.columns: continue
    feats=numeric_predictor_cols(train,target)
    Xtr,ytr_raw=clean_xy(train,target,feats,'classification'); Xv,yv_raw=clean_xy(val,target,feats,'classification')
    le=LabelEncoder(); ytr=le.fit_transform(ytr_raw.astype(str)); keep=yv_raw.astype(str).isin(le.classes_); Xv=Xv.loc[keep]; yv=le.transform(yv_raw.loc[keep].astype(str))
    for name,model in models.items():
        steps=[('imputer',SimpleImputer(strategy='median')),('scaler',MaxAbsScaler())]
        if FAST_MODE and len(feats)>SELECT_K: steps.append(('select',SelectKBest(f_classif,k=SELECT_K)))
        steps.append(('model',model)); pipe=Pipeline(steps)
        try:
            pipe.fit(Xtr,ytr); pred=pipe.predict(Xv); prob=None; auc=np.nan
            if hasattr(pipe,'predict_proba'):
                prob=pipe.predict_proba(Xv)
                if len(le.classes_)==2 and len(set(yv))==2: auc=roc_auc_score(yv, prob[:,1])
            outdir=ensure_dir(os.path.join(OUT,target,name)); joblib.dump(pipe, os.path.join(outdir,f'{target}__{name}.joblib'))
            pd.DataFrame({'class_label':le.classes_,'class_id':range(len(le.classes_))}).to_csv(os.path.join(outdir,'class_map.csv'),index=False)
            if prob is not None and len(le.classes_)==2 and len(set(yv))==2:
                RocCurveDisplay.from_predictions(yv, prob[:,1]); plt.title(f'{target} | {name} ROC'); plt.tight_layout(); plt.savefig(os.path.join(outdir,'ROC_curve.png'),dpi=300); plt.close()
                PrecisionRecallDisplay.from_predictions(yv, prob[:,1]); plt.title(f'{target} | {name} PR'); plt.tight_layout(); plt.savefig(os.path.join(outdir,'Precision_recall_curve.png'),dpi=300); plt.close()
            met={'task':'classification','target':target,'model':name,'category':MODEL_CATEGORY.get(name,'Other'),'Accuracy_test':accuracy_score(yv,pred),'AUC_test':auc,'F1_test':f1_score(yv,pred,average='weighted',zero_division=0),'Precision_test':precision_score(yv,pred,average='weighted',zero_division=0),'Recall_test':recall_score(yv,pred,average='weighted',zero_division=0),'n_features':len(feats)}
            rows.append(met)
            for yt,yp in zip(yv,pred): pred_rows.append({'target':target,'model':name,'y_true':yt,'y_pred':yp,'y_true_label':le.inverse_transform([yt])[0],'y_pred_label':le.inverse_transform([yp])[0]})
        except Exception as e:
            rows.append({'task':'classification','target':target,'model':name,'category':MODEL_CATEGORY.get(name,'Other'),'error':str(e),'n_features':len(feats)})
metrics=pd.DataFrame(rows); metrics.to_csv(os.path.join(OUT,'classification_metrics.csv'),index=False)
pd.DataFrame(pred_rows).to_csv(os.path.join(OUT,'classification_predictions_by_model.csv'),index=False)
if not metrics.empty and 'AUC_test' in metrics.columns: plot_metric_bar(metrics.dropna(subset=['AUC_test']),'AUC_test',os.path.join(OUT,'classification_AUC_barplot.png'),'Classification AUC')
print("Step4 complete")
