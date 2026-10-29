# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import pickle,numpy as np,csv,os,glob
src=_os.environ.get("DARTS_IMD_MACHINE_DIR", "/path/to/IMDiffusion-master/data/Machine")
K=4  # 下采样倍率, SMD 23k-28k -> ~6-7k (与MSL/SwaT同量级)
ms=sorted(set(os.path.basename(f).split("_test")[0] for f in glob.glob(f"{src}/machine-*_test.pkl")))
def pool_mean(x,k):
    n=(len(x)//k)*k; return x[:n].reshape(-1,k,x.shape[1]).mean(1)
def pool_max_lab(lb,k):
    n=(len(lb)//k)*k; return (lb[:n].reshape(-1,k).max(1)>0).astype(int)
def to_iv(lb):
    iv=[];i=0;n=len(lb)
    while i<n:
        if lb[i]==1:
            j=i
            while j<n and lb[j]==1: j+=1
            iv.append((i,j-1)); i=j
        else: i+=1
    return iv
os.makedirs("data/SMD",exist_ok=True)
lens=[]
for m in ms:
    tr=np.nan_to_num(np.asarray(pickle.load(open(f"{src}/{m}_train.pkl","rb"))).astype(np.float32))
    te=np.nan_to_num(np.asarray(pickle.load(open(f"{src}/{m}_test.pkl","rb"))).astype(np.float32))
    lb=np.asarray(pickle.load(open(f"{src}/{m}_test_label.pkl","rb"))).astype(int).ravel()
    trd=pool_mean(tr,K); ted=pool_mean(te,K); lbd=pool_max_lab(lb,K)
    L=min(len(ted),len(lbd)); ted=ted[:L]; lbd=lbd[:L]
    d=f"data/SMD/{m}"; os.makedirs(d,exist_ok=True)
    np.save(f"{d}/train.npy",trd); np.save(f"{d}/test.npy",ted)
    with open(f"{d}/label.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["Start","End"])
        for s,e in to_iv(lbd): w.writerow([s,e])
    lens.append(len(ted))
print(f"下采样K={K}, {len(ms)}台: test长度 {min(lens)}~{max(lens)} 中位{int(np.median(lens))}")
# 验证 machine-1-1
import pandas as pd
te=np.load("data/SMD/machine-1-1/test.npy"); lb=pd.read_csv("data/SMD/machine-1-1/label.csv")
print(f"machine-1-1: test{te.shape} End最大{int(lb.iloc[:,1].max()) if len(lb) else 0} 对齐{(int(lb.iloc[:,1].max())<len(te)) if len(lb) else True}")
