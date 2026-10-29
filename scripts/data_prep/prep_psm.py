# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import pickle, numpy as np, csv
src=_os.environ.get("DARTS_IMD_MACHINE_DIR", "/path/to/IMDiffusion-master/data/Machine")
tr=np.nan_to_num(pickle.load(open(f"{src}/PSM_train.pkl","rb")).astype(np.float32))
te=np.nan_to_num(pickle.load(open(f"{src}/PSM_test.pkl","rb")).astype(np.float32))
lb=pickle.load(open(f"{src}/PSM_test_label.pkl","rb")).astype(int).ravel()
np.save("data/PSM/PSM_train.npy", tr)
np.save("data/PSM/PSM_test.npy", te)
iv=[]; i=0; n=len(lb)
while i<n:
    if lb[i]==1:
        j=i
        while j<n and lb[j]==1: j+=1
        iv.append((i,j-1)); i=j
    else: i+=1
with open("data/PSM/PSM_test_anomaly_intervals.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["Start","End"])
    for s,e in iv: w.writerow([s,e])
print(f"train {tr.shape} test {te.shape} | seg {len(iv)} | pos {lb.mean():.3f}")
print("head:", iv[:3])
