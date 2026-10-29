# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import pickle,numpy as np,pandas as pd
m="machine-1-6"
loss=np.nan_to_num(np.asarray(pickle.load(open(f"output/SMD/{m}/s42pred/epoch_0.1_lr_baseline/anomaly_scores.pkl","rb"))["test_global_errror"]).astype(float))
te=np.load(f"data/SMD/{m}/test.npy")
lab=pd.read_csv(f"data/SMD/{m}/label.csv")
print("test长度",len(te),"| score长度",len(loss),"| 差",len(te)-len(loss))
print("label列名",list(lab.columns),"| 区间数",len(lab))
print("label前3:",lab.iloc[:3].values.tolist(),"| End最大",lab.iloc[:,1].max())
# 逐时刻gt(不做downsampling缩放)
a=lab.iloc[:,:2].values; gt=np.zeros(len(loss))
for s,e in zip(a[:,0],a[:,1]):
    s,e=int(s),int(e)+1; gt[max(0,min(s,len(loss))):max(0,min(e,len(loss)))]=1
print("gt异常点",int(gt.sum()),"/",len(gt),"占比",gt.mean().round(3))
from sklearn.metrics import roc_auc_score
if 0<gt.sum()<len(gt):
    print("AUROC(正)",round(roc_auc_score(gt,loss[:len(gt)]),4),"| AUROC(反相loss)",round(roc_auc_score(gt,-loss[:len(gt)]),4))
# 分数分布
print("loss统计 min/med/max",loss.min().round(4),np.median(loss).round(4),loss.max().round(4))
