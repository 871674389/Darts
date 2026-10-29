# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import pickle,numpy as np,pandas as pd,os
from sklearn.metrics import roc_auc_score
def build_gt(a_s,a_e,down,length):
    len_=min(int(a_e[-1]/down)+1,length); gt=np.zeros(len_)
    for s,e in zip(a_s,a_e):
        s,e=int(s)//down,int(e)//down+1; gt[max(0,min(s,len_)):max(0,min(e,len_))]=1
    return gt,len_
def au(pkl,csv):
    if not os.path.exists(pkl): return None
    try:
        loss=np.nan_to_num(np.asarray(pickle.load(open(pkl,"rb"))["test_global_errror"]).astype(float))
        a=pd.read_csv(csv).iloc[:,:2].values; gt,l=build_gt(a[:,0],a[:,1],1,len(loss))
        return roc_auc_score(gt,loss[:l]) if 0<gt.sum()<l else None
    except: return None
seeds=[42,43,44]
print("== 跨seed稳定性(mean±std over seeds 42/43/44) ==")
for head in ["pred","vae"]:
    # PSM
    v=[au(f"output/PSM/s{S}{head}/epoch_0.1_lr_baseline/anomaly_scores.pkl","data/PSM/PSM_test_anomaly_intervals.csv") for S in seeds]
    v=[x for x in v if x is not None]
    print(f"  PSM {head}: {np.mean(v):.4f}±{np.std(v):.4f}  seeds={[round(x,4) for x in v]}")
    # SMD 6机: 每机跨seed波动, 再报6机的均值波动
    perm=[]
    for m in ["machine-1-1","machine-1-5","machine-2-1","machine-2-5","machine-3-1","machine-3-5"]:
        vm=[au(f"output/SMD/{m}/s{S}{head}/epoch_0.1_lr_baseline/anomaly_scores.pkl",f"data/SMD/{m}/label.csv") for S in seeds]
        vm=[x for x in vm if x is not None]
        if len(vm)==3: perm.append(np.std(vm))
    print(f"  SMD {head} (6机): 每机跨seed std 中位={np.median(perm):.4f} 最大={np.max(perm):.4f}")
