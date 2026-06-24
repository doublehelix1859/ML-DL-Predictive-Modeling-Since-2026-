from pipeline_common import *
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
try:
    from xgboost import XGBRegressor
    HAVE_XGB=True
except Exception:
    HAVE_XGB=False
OUT=ensure_dir(os.path.join(ROOT,"STEP3_Regression"))
train=pd.read_csv(os.path.join(ROOT,"STEP2_Split","train80.csv"), low_memory=False)
val=pd.read_csv(os.path.join(ROOT,"STEP2_Split","validation20.csv"), low_memory=False)
models={
 'LinearRegression':LinearRegression(), 'Ridge':Ridge(), 'ElasticNet':ElasticNet(alpha=.001,l1_ratio=.5,max_iter=20000,random_state=SEED),
 'SVR':SVR(C=10), 'DecisionTreeRegressor':DecisionTreeRegressor(random_state=SEED),
 'RandomForestRegressor':RandomForestRegressor(n_estimators=300,random_state=SEED,n_jobs=N_JOBS),
 'GradientBoostingRegressor':GradientBoostingRegressor(random_state=SEED),
 'HistGradientBoostingRegressor':HistGradientBoostingRegressor(random_state=SEED),
 'KNeighborsRegressor':KNeighborsRegressor(n_neighbors=15,weights='distance')}
if HAVE_XGB:
    models['XGBoostRegressor']=XGBRegressor(n_estimators=300,learning_rate=.05,max_depth=5,subsample=.8,colsample_bytree=.8,random_state=SEED,n_jobs=N_JOBS,objective='reg:squarederror')
rows=[]; pred_rows=[]
for target in REG_TARGETS:
    if target not in train.columns: continue
    feats=numeric_predictor_cols(train,target)
    Xtr,ytr=clean_xy(train,target,feats,'regression'); Xv,yv=clean_xy(val,target,feats,'regression')
    for name,model in models.items():
        steps=[('imputer',SimpleImputer(strategy='median')),('scaler',MaxAbsScaler())]
        if FAST_MODE and len(feats)>SELECT_K:
            steps.append(('select',SelectKBest(f_regression,k=SELECT_K)))
        steps.append(('model',model)); pipe=Pipeline(steps)
        try:
            pipe.fit(Xtr,ytr); ptr=pipe.predict(Xtr); pv=pipe.predict(Xv)
            outdir=ensure_dir(os.path.join(OUT,target,name)); joblib.dump(pipe, os.path.join(outdir,f'{target}__{name}.joblib'))
            plot_regression(yv,pv,os.path.join(outdir,'observed_vs_predicted.png'),f'{target} | {name}')
            met={'task':'regression','target':target,'model':name,'category':MODEL_CATEGORY.get(name,'Other'),'R2_train':r2_score(ytr,ptr),'R2_test':r2_score(yv,pv),'RMSE_train':rmse(ytr,ptr),'RMSE_test':rmse(yv,pv),'n_features':len(feats)}
            rows.append(met)
            for yt,yp in zip(yv,pv): pred_rows.append({'target':target,'model':name,'y_true':yt,'y_pred':yp})
        except Exception as e:
            rows.append({'task':'regression','target':target,'model':name,'category':MODEL_CATEGORY.get(name,'Other'),'error':str(e),'n_features':len(feats)})
metrics=pd.DataFrame(rows); metrics.to_csv(os.path.join(OUT,'regression_metrics.csv'),index=False)
pd.DataFrame(pred_rows).to_csv(os.path.join(OUT,'regression_predictions_by_model.csv'),index=False)
if not metrics.empty and 'R2_test' in metrics.columns: plot_metric_bar(metrics.dropna(subset=['R2_test']),'R2_test',os.path.join(OUT,'regression_R2_barplot.png'),'Regression R2')
print("Step3 complete")
