# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import pickle,numpy as np,pandas as pd,glob,os
from sklearn.metrics import roc_auc_score, average_precision_score
def build_gt(a_s,a_e,down,length):
    len_=min(int(a_e[-1]/down)+1,length); gt=np.zeros(len_)
    for s,e in zip(a_s,a_e):
        s,e=int(s)//down,int(e)//down+1; gt[max(0,min(s,len_)):max(0,min(e,len_))]=1
    return gt,len_
def metric(pkl,csv,down=1):
    if not os.path.exists(pkl) or not os.path.exists(csv): return None
    try:
        loss=np.nan_to_num(np.asarray(pickle.load(open(pkl,"rb"))["test_global_errror"]).astype(float))
        a=pd.read_csv(csv).iloc[:,:2].values
        if len(a)==0: return None
        gt,l=build_gt(a[:,0],a[:,1],down,len(loss))
        if not(0<gt.sum()<l): return None
        return roc_auc_score(gt,loss[:l]), average_precision_score(gt,loss[:l])
    except Exception as e: return None
res={}
for head in ["pred","vae"]:
    au=[];ap=[]
    for d in sorted(glob.glob("data/SMD/machine-*")):
        m=os.path.basename(d)
        r=metric(f"output/SMD/{m}/s42{head}/epoch_0.1_lr_baseline/anomaly_scores.pkl", f"data/SMD/{m}/label.csv",1)
        if r: au.append(r[0]); ap.append(r[1])
    res[f"SMD_{head}"]=(np.mean(au),np.std(au),np.mean(ap),np.std(ap),len(au))
    r=metric(f"output/PSM/s42{head}/epoch_0.1_lr_baseline/anomaly_scores.pkl","data/PSM/PSM_test_anomaly_intervals.csv",1)
    res[f"PSM_{head}"]=(r[0],0,r[1],0,1) if r else None
print("== SMD (28机 mean±std) ==")
for h in ["pred","vae"]:
    a,sa,p,sp,n=res[f"SMD_{h}"]; print(f"  {h:5s}: AUROC {a:.4f}±{sa:.4f}  AUPR {p:.4f}±{sp:.4f}  (n={n})")
print("== PSM ==")
for h in ["pred","vae"]:
    r=res[f"PSM_{h}"]
    if r: print(f"  {h:5s}: AUROC {r[0]:.4f}  AUPR {r[2]:.4f}")
sb=max(res["SMD_pred"][0],res["SMD_vae"][0]); sbh="pred" if res["SMD_pred"][0]>=res["SMD_vae"][0] else "vae"
pb=max(res["PSM_pred"][0],res["PSM_vae"][0]); pbh="pred" if res["PSM_pred"][0]>=res["PSM_vae"][0] else "vae"
print(f"\n== 最优头 == SMD→{sbh}(AUROC {sb:.4f})  PSM→{pbh}(AUROC {pb:.4f})")
