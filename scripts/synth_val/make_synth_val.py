# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""方案B: 从训练数据(全正常)注入合成异常, 造无监督AD验证集. 完全不碰 test.
每个数据集输出 data/<DS>/synth_val.npy (含异常) + synth_val_label.npy (逐时刻0/1)。
注入类型: spike/mean-shift/scale/noise/flatten, 部分通道(多元), 段长随机。种子固定可复现。"""
import numpy as np, os

TRAIN = {'MSL': 'data/MSL/MSL_train_noisy.npy',
         'SWAT': 'data/SWAT/train_downsampled_noisy.npy',
         'WADI': 'data/WADI/train.npy'}
OUT = {'MSL': 'data/MSL', 'SWAT': 'data/SWAT', 'WADI': 'data/WADI'}
BASE = _ROOT

for ds, tf in TRAIN.items():
    data = np.load(os.path.join(BASE, tf)).astype(np.float64).copy()
    T, F = data.shape
    std = data.std(axis=0) + 1e-6
    label = np.zeros(T, dtype=np.int64)
    rng = np.random.default_rng(0)                 # 固定种子
    n_seg = max(40, T // 150)
    guard = 500                                    # 前guard步留作previous_steps上下文, 不注入
    for _ in range(n_seg):
        L = int(rng.integers(3, 40))
        start = int(rng.integers(guard, max(guard + 1, T - L)))
        nch = int(rng.integers(max(1, F // 10), max(2, F // 2)))
        chans = rng.choice(F, nch, replace=False)
        seg = data[start:start + L, chans].copy()
        sc = std[chans]
        typ = int(rng.integers(0, 5))
        if typ == 0:   seg += rng.choice([-1.0, 1.0]) * rng.uniform(4, 8) * sc        # spike
        elif typ == 1: seg += rng.choice([-1.0, 1.0]) * rng.uniform(3, 6) * sc        # mean-shift
        elif typ == 2: seg *= rng.uniform(3, 6)                                       # scale
        elif typ == 3: seg += rng.normal(0, 1, seg.shape) * rng.uniform(3, 6) * sc    # noise burst
        else:          seg[:] = data[start, chans]                                    # flatten/freeze
        data[start:start + L, chans] = seg
        label[start:start + L] = 1
    np.save(os.path.join(BASE, OUT[ds], 'synth_val.npy'), data.astype(np.float32))
    np.save(os.path.join(BASE, OUT[ds], 'synth_val_label.npy'), label)
    print(f'{ds}: synth_val {data.shape}  pos={label.mean():.3f}  segs={n_seg}')
