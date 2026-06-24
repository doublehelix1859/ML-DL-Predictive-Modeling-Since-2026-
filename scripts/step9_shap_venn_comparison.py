from pipeline_common import *
import pandas as pd, os
OUT=ensure_dir(os.path.join(ROOT,"STEP9_SHAP_Comparison"))
sel=pd.read_csv(os.path.join(ROOT,"STEP5_Model_Selection","TOP3_selected_models.csv"))
rows=[]
for (task,target),g in sel.groupby(['task','target']):
    feature_sets={}
    for _,r in g.iterrows():
        model=r['model']; p=os.path.join(ROOT,'STEP6_SHAP_Training',task,target,model,'top50_overall.csv')
        if os.path.exists(p): feature_sets[model]=set(pd.read_csv(p)['feature'].astype(str))
    all_features=sorted(set().union(*feature_sets.values())) if feature_sets else []
    for feat in all_features:
        rows.append({'task':task,'target':target,'feature':feat,**{m:int(feat in s) for m,s in feature_sets.items()},'n_models_present':sum(feat in s for s in feature_sets.values())})
pd.DataFrame(rows).to_csv(os.path.join(OUT,'SHAP_overlap_feature_lists.csv'),index=False)
print("Step9 complete")
