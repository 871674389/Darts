# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
# 方案B合成验证集生成(配置驱动,支持SMD/PSM)。从train注入合成异常,输出到train同目录。
import numpy as np, os
from config_files.config_general import Config
c=Config()
tf=c.train_filename; outdir=os.path.dirname(tf)
data=np.load(tf).astype(np.float64).copy(); T,F=data.shape
std=data.std(axis=0)+1e-6; label=np.zeros(T,dtype=np.int64)
rng=np.random.default_rng(0); n_seg=max(40,T//150); guard=min(500,T//4)
for _ in range(n_seg):
    L=int(rng.integers(3,40)); start=int(rng.integers(guard,max(guard+1,T-L)))
    nch=int(rng.integers(max(1,F//10),max(2,F//2))); chans=rng.choice(F,nch,replace=False)
    seg=data[start:start+L,chans].copy(); sc=std[chans]; typ=int(rng.integers(0,5))
    if typ==0: seg+=rng.choice([-1.0,1.0])*rng.uniform(4,8)*sc
    elif typ==1: seg+=rng.choice([-1.0,1.0])*rng.uniform(3,6)*sc
    elif typ==2: seg*=rng.uniform(3,6)
    elif typ==3: seg+=rng.normal(0,rng.uniform(2,4),seg.shape)*sc
    else: seg[:]=data[start,chans]
    data[start:start+L,chans]=seg; label[start:start+L]=1
np.save(os.path.join(outdir,"synth_val.npy"),data.astype(np.float32))
np.save(os.path.join(outdir,"synth_val_label.npy"),label)
print(f"synth_val -> {outdir} T={T} F={F} pos={label.mean():.3f}")
