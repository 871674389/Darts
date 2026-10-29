# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import pickle,numpy as np,csv,os,glob
src=_os.environ.get("DARTS_IMD_MACHINE_DIR", "/path/to/IMDiffusion-master/data/Machine")
ms=sorted(set(os.path.basename(f).split("_test")[0] for f in glob.glob(f"{src}/machine-*_test.pkl")))
def to_iv(lb):
    lb=np.asarray(lb).astype(int).ravel(); iv=[];i=0;n=len(lb)
    while i<n:
        if lb[i]==1:
            j=i
            while j<n and lb[j]==1: j+=1
            iv.append((i,j-1)); i=j
        else: i+=1
    return iv
ok=0
for m in ms:
    tr=np.nan_to_num(np.asarray(pickle.load(open(f"{src}/{m}_train.pkl","rb"))).astype(np.float32))
    te=np.nan_to_num(np.asarray(pickle.load(open(f"{src}/{m}_test.pkl","rb"))).astype(np.float32))
    lb=np.asarray(pickle.load(open(f"{src}/{m}_test_label.pkl","rb"))).astype(int).ravel()
    assert len(te)==len(lb), f"{m} 长度不匹配"
    d=f"data/SMD/{m}"; os.makedirs(d,exist_ok=True)
    np.save(f"{d}/train.npy",tr); np.save(f"{d}/test.npy",te)
    iv=to_iv(lb)
    with open(f"{d}/label.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["Start","End"])
        for s,e in iv: w.writerow([s,e])
    ok+=1
print(f"重建{ok}台clean SMD, 示例{ms[0]}: train{tr.shape} test{te.shape} seg{len(iv)}")
