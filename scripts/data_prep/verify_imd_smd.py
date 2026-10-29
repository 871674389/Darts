# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import pickle,numpy as np,glob,os
src=_os.environ.get("DARTS_IMD_MACHINE_DIR", "/path/to/IMDiffusion-master/data/Machine")
ms=sorted(set(os.path.basename(f).split("_test")[0] for f in glob.glob(f"{src}/machine-*_test.pkl")))
print(f"共{len(ms)}台机器")
for m in ms[:3]:
    te=pickle.load(open(f"{src}/{m}_test.pkl","rb")); lb=pickle.load(open(f"{src}/{m}_test_label.pkl","rb"))
    print(f"  {m}: test{np.asarray(te).shape} label{np.asarray(lb).shape} 长度匹配={len(te)==len(lb)} pos占比={np.asarray(lb).mean():.3f}")
