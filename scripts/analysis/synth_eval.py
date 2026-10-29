# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""方案B: 在合成异常验证集上评估某个已训练配置(由环境变量决定架构+TAG定位params)。
输出该配置的 synth-val AUC-ROC。打分路径与 train/test 完全一致(nll_gaussian -> 逐时刻global_error)。
用法(环境变量选配置): DUALV4_DS=SWAT DUALV4_VAE=1 DUALV4_WINDOW=15 python synth_eval.py --tag CX_SWAT"""
import torch, numpy as np, glob, os, argparse, pickle
from lib.utils import *
from config_files.config_general import Config
from model.DualGraphSTNet_dualshortpath import DualGraphSTNet
from sklearn.metrics import roc_auc_score

DS_DIR = {'MSL': 'MSL_noise', 'SWAT': 'swat_noise', 'WADI': 'wadi'}
SYNTH = {'MSL': 'data/MSL', 'SWAT': 'data/SWAT', 'WADI': 'data/WADI'}

def newest(pat):
    fs = glob.glob(pat); return max(fs, key=os.path.getmtime) if fs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')          # '' = baseline(无子目录)
    args = ap.parse_args()
    ds = os.environ.get('DUALV4_DS', 'WADI').upper()
    config = Config()
    device = config.device
    import os as _os
    synth_dir = _os.path.dirname(config.train_filename)
    base = config.save_dir
    save_dir = os.path.join(base, args.tag) if args.tag else base
    params = newest(os.path.join(save_dir, '*.params'))
    if not params:
        print(f'RESULT\t{args.tag or "baseline"}\t{ds}\tNA\tno-params'); return

    dpath = os.path.join(synth_dir, 'synth_val.npy')
    ts_label = np.load(os.path.join(synth_dir, 'synth_val_label.npy'))
    loader = load_data_test(dpath, config, device)
    model = DualGraphSTNet(
        matrix_decay=config.matrix_decay, num_features=config.num_features,
        window_size=config.window_size, out_dim=config.out_dim, target_size=config.target_size,
        gru_dim=config.gru_dim, Graph_learner_n_hid=config.Graph_learner_n_hid,
        Graph_learner_n_head_dim=config.Graph_learner_n_head_dim, num_heads=config.num_heads,
        num_nodes=config.num_nodes, temperature=config.temperature,
        max_diffusion_step=config.max_diffusion_step, num_rnn_layers=config.num_rnn_layers,
        device=device).to(device)
    with torch.no_grad():
        iw, ps, _ = next(iter(loader)); model(iw.to(device), ps.to(device))
    try:
        model.load_state_dict(torch.load(params, map_location=device))
    except Exception as e:
        print(f'RESULT\t{args.tag or "baseline"}\t{ds}\tNA\tARCH-MISMATCH:{str(e)[:80]}'); return
    model.eval()

    err = []
    with torch.no_grad():
        for iw, ps, tw in loader:
            _, out = model(iw.to(device), ps.to(device))
            _, ch, _, _ = nll_gaussian_with_intermediate(out, tw.to(device), 1)
            err.append(ch.detach().cpu().numpy())
    ch = np.concatenate(err, axis=0); ch = np.concatenate(ch, axis=0)   # (Teff, F)
    score = np.nan_to_num(np.mean(ch, axis=1))                          # 逐时刻 global_error

    # 对齐标签: 复刻 create_windows 的 target 展平顺序
    win, tgt, step = config.window_size, config.target_size, config.step_size
    T = np.load(dpath).shape[0]
    num_samples = (T - win - tgt) // step + 1
    lab = np.concatenate([ts_label[i*step+win : i*step+win+tgt] for i in range(num_samples)])
    L = min(len(lab), len(score)); lab, score = lab[:L], score[:L]
    auc = roc_auc_score(lab, score) if 0 < lab.sum() < len(lab) else float('nan')
    print(f'RESULT\t{args.tag or "baseline"}\t{ds}\t{auc:.4f}\tpos={lab.mean():.3f}\tn={L}')

if __name__ == '__main__':
    main()
