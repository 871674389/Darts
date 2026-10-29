# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""② 嵌套协议: val池(前60%)3-fold选Darts配置 -> 留出test'(后40%)报; Darts vs 基线同test'段对齐。
Darts分数在 output/<ds>/<tag>/; TranAD仓库基线分数需先从sungroup取来放 baselines_scores/<method>_<ds>.npy(per-timestamp)。"""
import pickle, glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

BASE = _ROOT
CFG = {'MSL': ('MSL_noise', 'data/MSL/MSL_test_anomaly_intervals.csv', 1),
       'SWAT': ('swat_noise', 'data/SWAT/SWAT_Time.csv', 60),
       'WADI': ('wadi', 'data/WADI/labels.csv', 300)}
KEEP = ['baseline', 'A2_w10L300', 'A2_w15L300', 'A2_w60L300', 'B6_vae', 'B7_joint',
        'C10_noKL', 'C4_softgraph', 'C5_diff0', 'C8_nodotgraph',
        'CX_MSL', 'CX_SWAT', 'CX_WADI_w15', 'CX_WADI_vae']
POOL_FRAC = 0.6  # 前60%为val池, 后40%为test'

def infer_ds(p):
    for k, v in [('MSL_noise','MSL'), ('swat','SWAT'), ('wadi','WADI')]:
        if k in p.lower() or k in p: return v
    return None
def build_gt(a_s, a_e, down, length):
    len_ = min(int(a_e[-1]/down)+1, length); gt = np.zeros(len_)
    for s, e in zip(a_s, a_e):
        s, e = int(s)//down, int(e)//down+1
        gt[max(0,min(s,len_)):max(0,min(e,len_))] = 1
    return gt, len_
def auc(g, s):
    return roc_auc_score(g, s) if (0 < g.sum() < len(g)) else np.nan

anom = {}
darts = {ds: {} for ds in CFG}
for f in glob.glob(os.path.join(BASE, 'output/**/epoch_0.1_lr_baseline/anomaly_scores.pkl'), recursive=True):
    ds = infer_ds(f)
    if ds is None: continue
    parts = f.split('/'); i = parts.index('epoch_0.1_lr_baseline')
    tag = parts[i-1] if parts[i-1] != CFG[ds][0] else 'baseline'
    if tag not in KEEP: continue
    try: loss = np.nan_to_num(np.asarray(pickle.load(open(f,'rb'))['test_global_errror']).astype(float))
    except Exception: continue
    if ds not in anom:
        a = pd.read_csv(os.path.join(BASE, CFG[ds][1])).iloc[:,:2].values; anom[ds] = (a[:,0], a[:,1])
    gt, len_ = build_gt(*anom[ds], CFG[ds][2], len(loss))
    darts[ds][tag] = (gt, loss[:len_])

for ds in CFG:
    tags = list(darts[ds]); n = len(next(iter(darts[ds].values()))[0])
    npool = int(n * POOL_FRAC); fold = npool // 3
    # 3-fold CV 在 val池(前npool) 选配置
    def cv_score(t):
        g, l = darts[ds][t]
        a = [auc(g[k*fold:(k+1)*fold], l[k*fold:(k+1)*fold]) for k in range(3)]
        a = [x for x in a if not np.isnan(x)]; return np.mean(a) if a else -1
    sel = max(tags, key=cv_score)
    # 报 test'(后40%)
    g, l = darts[ds][sel]; gt_te, l_te = g[npool:], l[npool:]
    darts_te = auc(gt_te, l_te)
    base = darts[ds].get('baseline'); base_te = auc(base[0][npool:], base[1][npool:]) if base else np.nan
    print(f'\n===== {ds}  val池选中={sel} =====')
    print(f'  Darts(val选) test\'-AUC = {darts_te:.4f}   |  Darts基线 test\'-AUC = {base_te:.4f}   (+{darts_te-base_te:.4f})')
    # 基线(TranAD仓库,若已放分数)
    bdir = os.path.join(BASE, 'baselines_scores')
    if os.path.isdir(bdir):
        for bf in sorted(glob.glob(os.path.join(bdir, f'*_{ds}.npy'))):
            if 'label' in bf: continue
            m = os.path.basename(bf).replace(f'_{ds}.npy', '')
            sc = np.nan_to_num(np.load(bf).astype(float))
            lb = np.load(bf.replace('.npy', '_label.npy')) if os.path.exists(bf.replace('.npy','_label.npy')) else gt_te
            L = min(len(sc), len(l)); sc = sc[:L]
            gt_full = darts[ds][sel][0][:L]
            print(f'  {m:14s} test\'-AUC = {auc(gt_full[npool:], sc[npool:]):.4f}')
