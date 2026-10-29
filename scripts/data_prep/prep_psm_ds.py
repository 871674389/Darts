# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import pickle,numpy as np,csv
src=_os.environ.get("DARTS_IMD_MACHINE_DIR", "/path/to/IMDiffusion-master/data/Machine"); K=18
tr=np.nan_to_num(pickle.load(open(f"{src}/PSM_train.pkl","rb")).astype(np.float32))
te=np.nan_to_num(pickle.load(open(f"{src}/PSM_test.pkl","rb")).astype(np.float32))
lb=pickle.load(open(f"{src}/PSM_test_label.pkl","rb")).astype(int).ravel()
def pm(x,k): n=(len(x)//k)*k; return x[:n].reshape(-1,k,x.shape[1]).mean(1)
def pl(l,k): n=(len(l)//k)*k; return (l[:n].reshape(-1,k).max(1)>0).astype(int)
trd,ted,lbd=pm(tr,K),pm(te,K),pl(lb,K); L=min(len(ted),len(lbd)); ted,lbd=ted[:L],lbd[:L]
np.save("data/PSM/PSM_train.npy",trd); np.save("data/PSM/PSM_test.npy",ted)
iv=[];i=0;n=len(lbd)
while i<n:
    if lbd[i]==1:
        j=i
        while j<n and lbd[j]==1:j+=1
        iv.append((i,j-1));i=j
    else:i+=1
with open("data/PSM/PSM_test_anomaly_intervals.csv","w",newline="") as f:
    w=csv.writer(f);w.writerow(["Start","End"])
    for s,e in iv:w.writerow([s,e])
print(f"PSM下采样K={K}: train{trd.shape} test{ted.shape} seg{len(iv)} pos{lbd.mean():.3f}")
