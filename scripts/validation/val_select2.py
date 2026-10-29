# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""Level-1 ①: 多种val划分下的配置选择稳健性 + 3-fold平均。看MSL能否稳选对(=oracle)。"""
import pickle, glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

BASE = _ROOT
CFG = {'MSL': ('MSL_noise', 'data/MSL/MSL_test_anomaly_intervals.csv', 1),
       'SWAT': ('swat_noise', 'data/SWAT/SWAT_Time.csv', 60),
       'WADI': ('wadi', 'data/WADI/labels.csv', 300)}
KEEP = ['baseline', 'A2_w10L300', 'A2_w15L300', 'A2_w60L300', 'B6_vae', 'B7_joint',
        'C10_noKL', 'C4_softgraph', 'C5_diff0', 'C8_nodotgraph',
        'CX_MSL', 'CX_SWAT', 'CX_WADI_w15', 'CX_WADI_vae']

def infer_ds(p):
    for k, v in [('MSL_noise', 'MSL'), ('swat', 'SWAT'), ('wadi', 'WADI')]:
        if k in p.lower() or k in p: return v
    return None

def build_gt(a_s, a_e, down, length):
    len_ = min(int(a_e[-1] / down) + 1, length)
    gt = np.zeros(len_)
    for s, e in zip(a_s, a_e):
        s, e = int(s)//down, int(e)//down+1
        gt[max(0, min(s, len_)):max(0, min(e, len_))] = 1
    return gt, len_

def auc(g, s):
    return roc_auc_score(g, s) if (0 < g.sum() < len(g)) else np.nan

# 收集每个配置的 (gt, loss)
data = {ds: {} for ds in CFG}
anom = {}
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
    data[ds][tag] = (gt, loss[:len_])

def val_mask(n, kind):
    m = np.zeros(n, bool)
    if kind == 'first30': m[:int(n*.3)] = True
    elif kind == 'last30': m[int(n*.7):] = True
    elif kind == 'mid40': m[int(n*.3):int(n*.7)] = True
    elif kind == 'first50': m[:int(n*.5)] = True
    return m

for ds in CFG:
    tags = list(data[ds])
    n = len(next(iter(data[ds].values()))[0])
    # oracle (全test)
    oracle = max(tags, key=lambda t: (auc(*data[ds][t]) if not np.isnan(auc(*data[ds][t])) else -1))
    print(f'\n===== {ds}  (oracle全test = {oracle}) =====')
    print(f'  {"协议":10s} {"val选中":14s} {"=oracle?"}')
    for kind in ['first30', 'last30', 'mid40', 'first50']:
        best, bv = None, -1
        for t in tags:
            g, l = data[ds][t]; vm = val_mask(len(g), kind)
            a = auc(g[vm], l[vm])
            if not np.isnan(a) and a > bv: bv, best = a, t
        print(f'  {kind:10s} {best:14s} {best==oracle}')
    # 3-fold 平均 val-AUC 选择
    best, bv = None, -1
    for t in tags:
        g, l = data[ds][t]; nn = len(g); fold = nn//3
        aucs = [auc(g[k*fold:(k+1)*fold], l[k*fold:(k+1)*fold]) for k in range(3)]
        aucs = [x for x in aucs if not np.isnan(x)]
        m = np.mean(aucs) if aucs else -1
        if m > bv: bv, best = m, t
    print(f'  {"3fold均值":10s} {best:14s} {best==oracle}')
