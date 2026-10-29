# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import pickle,numpy as np,pandas as pd,glob,re
from sklearn.metrics import roc_auc_score
def build_gt(a_s,a_e,down,length):
    len_=min(int(a_e[-1]/down)+1,length); gt=np.zeros(len_)
    for s,e in zip(a_s,a_e):
        s,e=int(s)//down,int(e)//down+1; gt[max(0,min(s,len_)):max(0,min(e,len_))]=1
    return gt,len_
a=pd.read_csv("data/MSL/MSL_test_anomaly_intervals.csv").iloc[:,:2].values
fs=sorted(glob.glob("output/MSL_noise/XCHK_MSL/epoch_*/anomaly_scores.pkl"), key=lambda p:float(re.search(r"epoch_([0-9.]+)",p).group(1)) if re.search(r"epoch_([0-9.]+)",p) else -1)
res=[]
for f in fs:
    m=re.search(r"epoch_([0-9.]+)",f)
    try:
        loss=np.nan_to_num(np.asarray(pickle.load(open(f,"rb"))["test_global_errror"]).astype(float))
        gt,l=build_gt(a[:,0],a[:,1],1,len(loss))
        if 0<gt.sum()<l: res.append((m.group(1),roc_auc_score(gt,loss[:l])))
    except Exception as e: pass
vals=[r[1] for r in res]
print(f"共{len(res)}个epoch有分数")
for e,v in res[-15:]: print(f"  epoch {e}: {v:.4f}")
if vals:
    print(f"末15: min={min(vals[-15:]):.4f} max={max(vals[-15:]):.4f} 极差={max(vals[-15:])-min(vals[-15:]):.4f}")
    print(f"全程: min={min(vals):.4f} max={max(vals):.4f} 极差={max(vals)-min(vals):.4f}")
