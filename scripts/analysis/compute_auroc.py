# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import pickle, numpy as np, pandas as pd, sys
from sklearn.metrics import roc_auc_score, average_precision_score
def build_gt(a_s,a_e,down,length):
    len_=min(int(a_e[-1]/down)+1,length); gt=np.zeros(len_)
    for s,e in zip(a_s,a_e):
        s,e=int(s)//down,int(e)//down+1; gt[max(0,min(s,len_)):max(0,min(e,len_))]=1
    return gt,len_
def auroc(pkl, anom_csv, down=1):
    loss=np.nan_to_num(np.asarray(pickle.load(open(pkl,"rb"))["test_global_errror"]).astype(float))
    a=pd.read_csv(anom_csv).iloc[:,:2].values
    gt,len_=build_gt(a[:,0],a[:,1],down,len(loss))
    if not (0<gt.sum()<len_): return None,None,len_,gt.sum()
    return roc_auc_score(gt,loss[:len_]), average_precision_score(gt,loss[:len_]), len_, int(gt.sum())
if __name__=="__main__":
    pkl,csv=sys.argv[1],sys.argv[2]; down=int(sys.argv[3]) if len(sys.argv)>3 else 1
    r,ap,n,pos=auroc(pkl,csv,down)
    print(f"AUROC={r:.4f} AUPR={ap:.4f} (len={n}, pos={pos})" if r is not None else f"退化(len={n},pos={pos})")
