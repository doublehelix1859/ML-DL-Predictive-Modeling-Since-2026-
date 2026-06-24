import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pipeline_common import feature_type

def raw_importance_from_model(pipe, features):
    model=pipe.named_steps.get('model')
    names=list(features)
    if 'select' in pipe.named_steps:
        mask=pipe.named_steps['select'].get_support()
        names=[f for f,m in zip(names,mask) if m]
    if hasattr(model,'feature_importances_'):
        vals=np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model,'coef_'):
        vals=np.asarray(model.coef_)
        vals=np.mean(np.abs(vals),axis=0) if vals.ndim>1 else np.abs(vals)
    else:
        vals=np.zeros(len(names))
    if len(vals) != len(names): vals=np.resize(vals,len(names))
    out=pd.DataFrame({'feature':names,'importance':np.abs(vals)})
    out['feature_type']=out['feature'].map(feature_type)
    return out.sort_values('importance',ascending=False)

def save_imp_outputs(imp,outdir,title):
    os.makedirs(outdir,exist_ok=True)
    imp.to_csv(os.path.join(outdir,'feature_importance_full.csv'),index=False)
    top50=imp.head(50); snp=imp[imp.feature_type=='SNP'].head(30); gene=imp[imp.feature_type=='GeneExpression'].head(30)
    top50.to_csv(os.path.join(outdir,'top50_overall.csv'),index=False)
    snp.to_csv(os.path.join(outdir,'top30_SNP.csv'),index=False)
    gene.to_csv(os.path.join(outdir,'top30_gene_expression.csv'),index=False)
    cnt=top50.feature_type.value_counts().reset_index(); cnt.columns=['feature_type','count']; cnt.to_csv(os.path.join(outdir,'top50_feature_makeup.csv'),index=False)
    if not cnt.empty:
        plt.figure(figsize=(6,4)); plt.bar(cnt.feature_type,cnt['count']); plt.title(title+' top50 feature makeup'); plt.tight_layout(); plt.savefig(os.path.join(outdir,'feature_makeup_top50.png'),dpi=300); plt.close()
    for name,df in [('top50_overall',top50),('top30_SNP',snp),('top30_gene_expression',gene)]:
        if df.empty: continue
        d=df.iloc[::-1]
        plt.figure(figsize=(10,max(4,.25*len(d)))); plt.barh(d.feature,d.importance); plt.title(title+' '+name); plt.tight_layout(); plt.savefig(os.path.join(outdir,name+'_barplot.png'),dpi=300); plt.close()
