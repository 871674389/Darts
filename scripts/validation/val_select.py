# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""Level-1 验证集配置选择(精简版): 复用已存 test 分数, 把 test 切 val(前30%)+test'(后70%),
在 val 上按 AUC-ROC 选配置, 看是否= oracle(test'最优), 并报选中配置 test' AUC vs 基线。证明配置非test调出。"""
import pickle, glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

BASE = _ROOT
CFG = {'MSL': ('MSL_noise', 'data/MSL/MSL_test_anomaly_intervals.csv', 1),
       'SWAT': ('swat_noise', 'data/SWAT/SWAT_Time.csv', 60),
       'WADI': ('wadi', 'data/WADI/labels.csv', 300)}
VAL_FRAC = 0.3
# 只看"自适应旋钮"相关配置(w/头/KL), 排除纯L扫描等
KEEP = ['baseline', 'A2_w10L300', 'A2_w15L300', 'A2_w60L300', 'B6_vae', 'B7_joint',
        'C10_noKL', 'C4_softgraph', 'C5_diff0', 'C8_nodotgraph',
        'CX_MSL', 'CX_SWAT', 'CX_WADI_w15', 'CX_WADI_vae']

def infer_ds(p):
    if 'MSL_noise' in p: return 'MSL'
    if 'swat' in p.lower(): return 'SWAT'
    if 'wadi' in p.lower(): return 'WADI'
    return None

def build_gt(a_s, a_e, down, length):
    len_ = min(int(a_e[-1] / down) + 1, length)
    gt = np.zeros(len_)
    for s, e in zip(a_s, a_e):
        s = int(s) // down; e = int(e) // down + 1
        gt[max(0, min(s, len_)):max(0, min(e, len_))] = 1
    return gt, len_

anom = {}
res = {ds: [] for ds in CFG}
for f in glob.glob(os.path.join(BASE, 'output/**/epoch_0.1_lr_baseline/anomaly_scores.pkl'), recursive=True):
    ds = infer_ds(f)
    if ds is None: continue
    parts = f.split('/'); i = parts.index('epoch_0.1_lr_baseline')
    tag = parts[i-1] if parts[i-1] != CFG[ds][0] else 'baseline'
    if tag not in KEEP: continue
    try: loss = np.nan_to_num(np.asarray(pickle.load(open(f, 'rb'))['test_global_errror']).astype(float))
    except Exception: continue
    if ds not in anom:
        a = pd.read_csv(os.path.join(BASE, CFG[ds][1])).iloc[:, :2].values
        anom[ds] = (a[:, 0], a[:, 1])
    gt, len_ = build_gt(*anom[ds], CFG[ds][2], len(loss))
    l = loss[:len_]
    nval = int(len_ * VAL_FRAC)
    gv, lv, gt2, lt = gt[:nval], l[:nval], gt[nval:], l[nval:]
    def safe(g, s):
        try: return roc_auc_score(g, s) if g.sum() > 0 and g.sum() < len(g) else np.nan
        except Exception: return np.nan
    res[ds].append((tag, safe(gv, lv), safe(gt2, lt)))

for ds in CFG:
    rows = [r for r in res[ds] if not np.isnan(r[1]) and not np.isnan(r[2])]
    if not rows: print(f'== {ds}: 无有效配置(val/test类别不全) =='); continue
    val_pick = max(rows, key=lambda x: x[1])
    oracle = max(rows, key=lambda x: x[2])
    base = next((r for r in rows if r[0] == 'baseline'), None)
    print(f'\n===== {ds} (val前30% 选 / test后70% 报) =====')
    print(f'  {"config":14s} {"val-AUC":>8s} {"test-AUC":>9s}')
    for t, av, at in sorted(rows, key=lambda x: -x[1]):
        mark = ' <val选' if t == val_pick[0] else ('  (oracle)' if t == oracle[0] else '')
        print(f'  {t:14s} {av:8.4f} {at:9.4f}{mark}')
    print(f'  --> val选中: {val_pick[0]} (test-AUC={val_pick[2]:.4f})')
    print(f'  --> oracle : {oracle[0]} (test-AUC={oracle[2]:.4f})  | val选==oracle? {val_pick[0]==oracle[0]}')
    if base: print(f'  --> 基线   : baseline (test-AUC={base[2]:.4f})  | val选比基线 +{val_pick[2]-base[2]:.4f}')
