from pipeline_common import *
OUT=ensure_dir(os.path.join(ROOT,"STEP5_Model_Selection"))
rows=[]
reg_path=os.path.join(ROOT,"STEP3_Regression","regression_metrics.csv")
if os.path.exists(reg_path):
    df=pd.read_csv(reg_path)
    if 'R2_test' in df.columns:
        df=df[pd.to_numeric(df['R2_test'],errors='coerce').notna()].copy()
        for target,g in df.groupby('target'):
            g=g.sort_values(['R2_test','RMSE_test'],ascending=[False,True]).head(3)
            for rank,(_,r) in enumerate(g.iterrows(),1): rows.append({'task':'regression','target':target,'model':r['model'],'rank':rank,'primary_metric':'R2_test','primary_value':r['R2_test']})
clf_path=os.path.join(ROOT,"STEP4_Classification","classification_metrics.csv")
if os.path.exists(clf_path):
    df=pd.read_csv(clf_path)
    metric='AUC_test' if 'AUC_test' in df.columns and df['AUC_test'].notna().any() else 'F1_test'
    for target,g in df.groupby('target'):
        g=g[pd.to_numeric(g[metric],errors='coerce').notna()].sort_values([metric,'F1_test'],ascending=[False,False]).head(3)
        for rank,(_,r) in enumerate(g.iterrows(),1): rows.append({'task':'classification','target':target,'model':r['model'],'rank':rank,'primary_metric':metric,'primary_value':r[metric]})
out=pd.DataFrame(rows)
out.to_csv(os.path.join(OUT,'TOP3_selected_models.csv'),index=False)
with open(os.path.join(OUT,'TOP3_selected_models.txt'),'w') as f: f.write(out.to_string(index=False))
print("Step5 complete")
