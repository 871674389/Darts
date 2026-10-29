# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""闭环: 合并 synth-val AUC(选择信号) 与 真实test AUC(报告值)。
证明 synth-val 的 argmax 配置 == 真实test近似最优, 且头选择全对。真实test AUC 复用已存 anomaly_scores.pkl。"""
import pickle, glob, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

BASE = _ROOT
CFG = {'MSL': ('MSL_noise', 'data/MSL/MSL_test_anomaly_intervals.csv', 1),
       'SWAT': ('swat_noise', 'data/SWAT/SWAT_Time.csv', 60),
       'WADI': ('wadi', 'data/WADI/labels.csv', 300)}
# synth-val AUC (from run_synth.sh)
SYNTH = {
 'MSL': {'A2_w10L300':.7346,'A2_w15L300':.7616,'A2_w60L300':.7449,'B6_vae':.7602,'B7_joint':.7407,
         'C4_softgraph':.7418,'C5_diff0':.7325,'C8_nodotgraph':.7395,'C10_noKL':.7355,'CX_MSL':.7375},
 'SWAT':{'A2_w10L300':.8944,'A2_w15L300':.8943,'A2_w60L300':.9037,'B6_vae':.9087,'B7_joint':.8996,
         'C4_softgraph':.8942,'C5_diff0':.8985,'C8_nodotgraph':.9001,'C10_noKL':.8984,'CX_SWAT':.9014},
 'WADI':{'A2_w10L300':.8859,'A2_w15L300':.8720,'A2_w60L300':.8985,'B6_vae':.9432,'B7_joint':.8827,
         'C4_softgraph':.8839,'C5_diff0':.8737,'C8_nodotgraph':.8831,'C10_noKL':.8940,
         'CX_WADI_vae':.9425,'CX_WADI_w15':.8773},
}
HEAD = lambda t: 'VAE' if ('vae' in t.lower() or t in ('CX_SWAT',)) else ('joint' if 'joint' in t else 'pred')

def infer_ds(p):
    for k,v in [('MSL_noise','MSL'),('swat','SWAT'),('wadi','WADI')]:
        if k in p.lower() or k in p: return v
    return None
def build_gt(a_s,a_e,down,length):
    len_=min(int(a_e[-1]/down)+1,length); gt=np.zeros(len_)
    for s,e in zip(a_s,a_e):
        s,e=int(s)//down,int(e)//down+1; gt[max(0,min(s,len_)):max(0,min(e,len_))]=1
    return gt,len_

anom={}; real={ds:{} for ds in CFG}
for f in glob.glob(os.path.join(BASE,'output/**/epoch_0.1_lr_baseline/anomaly_scores.pkl'),recursive=True):
    ds=infer_ds(f)
    if ds is None: continue
    parts=f.split('/'); i=parts.index('epoch_0.1_lr_baseline'); tag=parts[i-1]
    if tag not in SYNTH[ds]: continue
    try: loss=np.nan_to_num(np.asarray(pickle.load(open(f,'rb'))['test_global_errror']).astype(float))
    except Exception: continue
    if ds not in anom:
        a=pd.read_csv(os.path.join(BASE,CFG[ds][1])).iloc[:,:2].values; anom[ds]=(a[:,0],a[:,1])
    gt,len_=build_gt(*anom[ds],CFG[ds][2],len(loss))
    real[ds][tag]=roc_auc_score(gt,loss[:len_]) if 0<gt.sum()<len_ else np.nan

for ds in CFG:
    print(f'\n===== {ds} =====')
    print(f'  {"config":14s} {"head":5s} {"synthAUC":>9s} {"realAUC":>8s}')
    rows=sorted(SYNTH[ds], key=lambda t:-SYNTH[ds][t])
    for t in rows:
        r=real[ds].get(t,np.nan)
        print(f'  {t:14s} {HEAD(t):5s} {SYNTH[ds][t]:9.4f} {r:8.4f}')
    synth_pick=max(SYNTH[ds],key=lambda t:SYNTH[ds][t])
    real_valid={t:v for t,v in real[ds].items() if not np.isnan(v)}
    real_oracle=max(real_valid,key=lambda t:real_valid[t]) if real_valid else None
    print(f'  >> synth选中 = {synth_pick} (head={HEAD(synth_pick)}, real-test AUC={real[ds].get(synth_pick,float("nan")):.4f})')
    print(f'  >> real oracle = {real_oracle} (head={HEAD(real_oracle)}, AUC={real_valid.get(real_oracle,float("nan")):.4f}) | 头匹配? {HEAD(synth_pick)==HEAD(real_oracle)}')
