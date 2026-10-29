# -*- coding: utf-8 -*-
"""
与 train_dualpath.py 配套的测评脚本。
修复 test_DualGraph.py 的两个接线问题：
  1) 使用与训练相同的模型 DualGraphSTNet_dualshortpath（而非 base DualGraphSTNet）
  2) 自动定位 config.save_dir 下最新的 .params 与 config.score_dir 下最新的 train 分数 pkl，
     不再依赖 config 里写死的旧 epoch 路径。
协议：200x200 双阈值网格搜索取最优 point-adjust F1（与原 test 一致）。
"""
import torch, numpy as np, glob, os, json, pickle, sys, argparse
import pandas as pd
from lib.utils import *
from lib.evaluate import *
from config_files.config_general import Config
from model.DualGraphSTNet_dualshortpath import DualGraphSTNet   # ← 与训练同模型
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

def newest(pattern):
    fs = glob.glob(pattern)
    if not fs:
        return None
    return max(fs, key=os.path.getmtime)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None, help="模型权重路径；默认自动取 save_dir 下最新 .params")
    ap.add_argument("--scores", default=None, help="train 分数 pkl；默认自动取 score_dir 下最新")
    ap.add_argument("--grid", type=int, default=200, help="阈值网格大小(默认200，可调小加速调试)")
    args = ap.parse_args()

    config = Config()
    device = config.device

    # --- 自动定位权重与分数文件 ---
    params_path = args.params or newest(os.path.join(config.save_dir, "*.params"))
    scores_path = args.scores or newest(os.path.join(config.score_dir, "*", "anomaly_scores.pkl"))
    assert params_path and os.path.exists(params_path), f"找不到权重: {config.save_dir}/*.params"
    assert scores_path and os.path.exists(scores_path), f"找不到train分数: {config.score_dir}/*/anomaly_scores.pkl"
    print(f"[wire] 数据集 train={config.train_filename}")
    print(f"[wire] 权重  = {params_path}")
    print(f"[wire] train分数 = {scores_path}")

    # --- 数据与模型 ---
    test_loader = load_data_test(config.test_filename_data, config, device)
    model = DualGraphSTNet(
        matrix_decay=config.matrix_decay, num_features=config.num_features,
        window_size=config.window_size, out_dim=config.out_dim, target_size=config.target_size,
        gru_dim=config.gru_dim, Graph_learner_n_hid=config.Graph_learner_n_hid,
        Graph_learner_n_head_dim=config.Graph_learner_n_head_dim, num_heads=config.num_heads,
        num_nodes=config.num_nodes, temperature=config.temperature,
        max_diffusion_step=config.max_diffusion_step, num_rnn_layers=config.num_rnn_layers,
        device=device).to(device)
    # LazyLinear 需先跑一次 forward 才能 load_state_dict：用一个批次做 lazy init
    with torch.no_grad():
        iw, ps, _ = next(iter(test_loader))
        model(iw.to(device), ps.to(device))
    model.load_state_dict(torch.load(params_path, map_location=device))
    model.eval()
    print("[ok] 模型已加载")

    # --- 前向收集预测误差 ---
    all_channel_error = []
    with torch.no_grad():
        for input_window, previous_steps, target_window in test_loader:
            input_window = input_window.to(device); previous_steps = previous_steps.to(device)
            target_window = target_window.to(device)
            _, out = model(input_window, previous_steps)
            _, loss_nll_channel, _, _ = nll_gaussian_with_intermediate(out, target_window, 1)
            all_channel_error.append(loss_nll_channel.detach().cpu().numpy())

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)
    train_channel_error = scores.get("train_channel_error")

    test_channel_error, test_global_error, _, _ = get_test_score(
        train_channel_error=train_channel_error, test_channel_error=all_channel_error,
        epoch=0.1, lr="baseline", config=config)

    # --- 阈值网格搜索取最优 point-adjust F1 ---
    at = pd.read_csv(config.anomaly_file); at = np.array(at.iloc[:, :2])
    a_start, a_end = at[:, 0], at[:, 1]
    G = args.grid
    best = {"f1": -1}
    for i in range(G):
        for j in range(G):
            thr1, thr2 = 0.0005 * i, -0.0005 * j
            anomaly, gt, _ = point_adjust_eval(a_start, a_end, config.downsampling_fre, test_global_error, thr1, thr2)
            f1 = f1_score(gt, anomaly)
            if f1 > best["f1"]:
                best = {"f1": f1, "i": i, "j": j}
    anomaly, gt, _ = point_adjust_eval(a_start, a_end, config.downsampling_fre, test_global_error,
                                       best["i"] * 0.0005, -best["j"] * 0.0005)
    P = precision_score(gt, anomaly, zero_division=0)
    R = recall_score(gt, anomaly, zero_division=0)
    F1 = f1_score(gt, anomaly, zero_division=0)
    try:
        auc = roc_auc_score(gt, anomaly); ap = average_precision_score(gt, anomaly)
    except Exception:
        auc = ap = float("nan")
    cm = confusion_matrix(gt, anomaly).tolist()

    ds = os.path.basename(os.path.dirname(config.train_filename)) or config.train_filename
    res = {"dataset_train": config.train_filename, "params": params_path,
           "best_thr1": best["i"] * 0.0005, "best_thr2": -best["j"] * 0.0005,
           "Precision": float(P), "Recall": float(R), "F1": float(F1),
           "ROC_AUC": float(auc), "AP": float(ap), "confusion_matrix": cm,
           "protocol": "best-F1 grid + point-adjust"}
    os.makedirs(config.results_dir, exist_ok=True)
    out_json = os.path.join(config.results_dir, "baseline_metrics.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print("=" * 60)
    print(f"[RESULT] {config.train_filename}")
    print(f"  P={P:.4f}  R={R:.4f}  F1={F1:.4f}  AUC={auc:.4f}  AP={ap:.4f}")
    print(f"  best thr1={best['i']*0.0005:.4f} thr2={-best['j']*0.0005:.4f}  saved -> {out_json}")
    print("=" * 60)

if __name__ == "__main__":
    main()
