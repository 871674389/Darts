import numpy as np
from lib.spot import SPOT  # 假设您已经定义了 SPOT 类
from lib.utils import *
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import logging
import pickle
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import signal
from config_files.config_general import Config  # 假设配置文件中定义了Config类
config = Config()
os.makedirs(config.log_dir, exist_ok=True)
log_file_path = os.path.join(config.log_dir, "training.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    使用异常分数、阈值和真实标签调整预测结果，返回调整后的预测标签和平均延迟。

    参数：
    - score (np.ndarray): 异常分数
    - label (np.ndarray): 实际标签（真实情况，异常为 True）
    - threshold (float): 异常分数的阈值
    - pred (np.ndarray or None): 如果已提供预测标签，将忽略 score 和 threshold
    - calc_latency (bool): 是否计算平均延迟

    返回：
    - predict (np.ndarray): 调整后的预测标签，True 表示异常
    - latency (float or None): 如果 calc_latency 为 True，则返回平均延迟
    """
    # 如果未提供预测标签 pred，则使用 score 和 threshold 生成初始预测标签
    if pred is None:
        if len(score) != len(label):
            raise ValueError("score 和 label 必须具有相同的长度")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    # 遍历每个时间步，调整预测结果
    for i in range(len(predict)):
        # 检测到真实异常开始，并首次预测异常
        if any(actual[max(i - 1, 0):i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            # 倒序检查之前的时间步，修正预测为异常
            for j in range(i, 0, -1):
                if not actual[j]:  # 直到遇到正常点
                    break
                elif not predict[j]:  # 修正先前预测为正常的时间步
                    predict[j] = True
                    latency += 1

        elif not actual[i]:  # 当前为正常点
            anomaly_state = False

        if anomaly_state:
            predict[i] = True  # 标记异常状态

    # 计算平均延迟
    if calc_latency:
        latency = latency / (anomaly_count + 1e-4)
        return predict, latency
    else:
        return predict, None

def calc_point2point(predict, actual):
    """
    使用预测标签和实际标签计算 F1 分数、精度、召回率和混淆矩阵指标。

    参数：
    - predict (np.ndarray): 预测标签（True 表示异常）
    - actual (np.ndarray): 实际标签（True 表示异常）

    返回：
    - f1 (float): F1 分数
    - precision (float): 精度
    - recall (float): 召回率
    - TP (int): 真阳性数量
    - TN (int): 真阴性数量
    - FP (int): 假阳性数量
    - FN (int): 假阴性数量
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return f1, precision, recall, TP, TN, FP, FN

def find_epsilon(errors, reg_level=2):
    """
    Threshold method to find the best epsilon for anomaly detection.

    Parameters:
    - errors (np.ndarray): Anomaly scores for a single channel.
    - reg_level (int): Regularization level to control the penalty on anomaly detection.

    Returns:
    - float: The best threshold (epsilon) for detecting anomalies.
    """
    e_s = errors
    best_epsilon = None
    max_score = -float("inf")
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):  # Iterate over possible z values for thresholding
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        # Identify anomaly points
        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        # Calculate score for this epsilon
        if len(i_anom) > 0:
            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            # Update best epsilon
            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon

def epsilon_channel_eval(init_channel_score, channel_score, channel_label, reg_level=2):
    """
    Evaluate anomaly detection performance using epsilon threshold on each channel.

    Parameters:
    - init_channel_score (np.ndarray): Anomaly scores from the training set to determine thresholds.
                                       Shape: (num_train_samples, num_channels).
    - channel_score (np.ndarray): Anomaly scores from the test set.
                                  Shape: (num_test_samples, num_channels).
    - channel_label (np.ndarray): Boolean array indicating true anomalies for each channel.
                                  Shape: (num_test_samples, num_channels).
    - reg_level (int): Regularization level for threshold determination.

    Returns:
    - dict: A dictionary containing evaluation metrics for each channel and mean metrics across channels.
    """
    print(f"Running Epsilon Evaluation with reg_level={reg_level} for each channel:")

    n_channels = init_channel_score.shape[1]  # Number of channels
    channel_thresholds = []
    all_channel_preds = []
    all_channel_latencies = []
    all_channel_metrics = []

    for i in range(n_channels):
        print(f"Processing channel {i + 1}/{n_channels}")
        
        # Find the best threshold for this channel
        best_epsilon = find_epsilon(init_channel_score[:, i], reg_level=reg_level)
        channel_thresholds.append(best_epsilon)
        
        # Predict anomalies and calculate latency
        pred, latency = adjust_predicts(channel_score[:, i], channel_label[:, i], best_epsilon, calc_latency=True)
        all_channel_preds.append(pred)
        all_channel_latencies.append(latency)
        
        # Calculate metrics for this channel
        if channel_label is not None:
            f1, precision, recall, TP, TN, FP, FN = calc_point2point(pred, channel_label[:, i])
            channel_metrics = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "threshold": best_epsilon,
                "latency": latency,
            }
            all_channel_metrics.append(channel_metrics)

    # Calculate mean metrics across all channels
    if channel_label is not None:
        mean_metrics = {
            "mean_f1": np.mean([m["f1"] for m in all_channel_metrics]),
            "mean_precision": np.mean([m["precision"] for m in all_channel_metrics]),
            "mean_recall": np.mean([m["recall"] for m in all_channel_metrics]),
            "mean_TP": np.mean([m["TP"] for m in all_channel_metrics]),
            "mean_TN": np.mean([m["TN"] for m in all_channel_metrics]),
            "mean_FP": np.mean([m["FP"] for m in all_channel_metrics]),
            "mean_FN": np.mean([m["FN"] for m in all_channel_metrics]),
            "mean_latency": np.mean(all_channel_latencies),
        }
        return {
            "channel_thresholds": np.array(channel_thresholds),
            "all_channel_metrics": all_channel_metrics,
            **mean_metrics,
        }
    else:
        return {
            "channel_thresholds": np.array(channel_thresholds),
            "mean_latency": np.mean(all_channel_latencies),
        }

def pot_channel_eval(init_channel_score, channel_score, channel_label, q=1e-3, level=0.99, dynamic=True, device="cuda:0"):
    """
    GPU-accelerated POT method for anomaly detection performance metrics.

    Parameters:
    - init_channel_score (torch.Tensor): Training anomaly scores (on GPU). Shape: (num_train_samples, num_channels).
    - channel_score (torch.Tensor): Test anomaly scores (on GPU). Shape: (num_test_samples, num_channels).
    - channel_label (torch.Tensor): True anomalies (on GPU). Shape: (num_test_samples, num_channels).
    - q (float): Detection level (risk).
    - level (float): Probability for threshold initialization.
    - dynamic (bool): Whether to use dynamic threshold.
    - device (str): Device to run the calculations ("cuda" or "cpu").

    Returns:
    - dict: Evaluation metrics for each channel and mean metrics across channels.
    """
    print(f"Running GPU-accelerated POT with q={q}, level={level} for each channel:")

    n_channels = init_channel_score.shape[1]  # Number of channels
    channel_thresholds = []
    all_channel_preds = []
    all_channel_latencies = []
    all_channel_metrics = []

    for i in range(n_channels):
        print(f"Processing channel {i + 1}/{n_channels}")
        
        # Use a modified GPU-compatible SPOT implementation
        s = SPOT(q)
        s.fit(init_channel_score[:, i].cpu().numpy(), channel_score[:, i].cpu().numpy())  # Still requires NumPy here
        s.initialize(level=level, min_extrema=False)
        ret = s.run(dynamic=dynamic)
        
        # Store each channel's threshold
        channel_threshold = np.mean(ret["thresholds"])
        channel_thresholds.append(channel_threshold)
        
        # Adjust predictions and evaluate latency for each channel
        pred, latency = adjust_predicts(
            channel_score[:, i].cpu().numpy(),
            channel_label[:, i].cpu().numpy(),
            channel_threshold,
            calc_latency=True
        )
        all_channel_preds.append(pred)
        all_channel_latencies.append(latency)
        
        # Calculate metrics for this channel
        if channel_label is not None:
            f1, precision, recall, TP, TN, FP, FN = calc_point2point(pred, channel_label[:, i].cpu().numpy())
            channel_metrics = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "threshold": channel_threshold,
                "latency": latency,
            }
            all_channel_metrics.append(channel_metrics)

    # Calculate mean metrics across all channels
    if channel_label is not None:
        mean_metrics = {
            "mean_f1": np.mean([m["f1"] for m in all_channel_metrics]),
            "mean_precision": np.mean([m["precision"] for m in all_channel_metrics]),
            "mean_recall": np.mean([m["recall"] for m in all_channel_metrics]),
            "mean_TP": np.mean([m["TP"] for m in all_channel_metrics]),
            "mean_TN": np.mean([m["TN"] for m in all_channel_metrics]),
            "mean_FP": np.mean([m["FP"] for m in all_channel_metrics]),
            "mean_FN": np.mean([m["FN"] for m in all_channel_metrics]),
            "mean_latency": np.mean(all_channel_latencies),
        }
        return {
            "channel_thresholds": np.array(channel_thresholds),
            "all_channel_metrics": all_channel_metrics,
            **mean_metrics,
        }
    else:
        return {
            "channel_thresholds": np.array(channel_thresholds),
            "mean_latency": np.mean(all_channel_latencies),
        }
    
def moving_average(score_tc, window=3):
    """
    计算二维时间序列数据的移动平均值。

    参数：
    - score_tc: numpy.ndarray，形状为 (T, C)，其中 T 是时间步数，C 是通道数。
    - window: int，移动窗口的大小。

    返回：
    - moving_avg: numpy.ndarray，形状为 (T - window + 1, C)，每个通道的移动平均值。
    """
    if not isinstance(score_tc, np.ndarray):
        raise TypeError("score_tc must be a numpy.ndarray")
    if score_tc.ndim != 2:
        raise ValueError("score_tc must be a 2D array with shape (T, C)")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    if window > score_tc.shape[0]:
        raise ValueError("window size cannot exceed the number of time steps")

    # 计算累积和，沿时间步方向（axis=0）
    ret = np.cumsum(score_tc, dtype=float, axis=0)

    # 调整累积和以获得每个窗口的总和
    ret[window:] = ret[window:] - ret[:-window]

    # 计算移动平均值
    moving_avg = ret[window - 1:] / window

    return moving_avg

def get_dynamic_score_t(error_t_train, error_t_test, long_window, short_window, constant_std=1e-6):
    """
    计算动态异常分数，适用于二维训练和测试误差数据。

    参数：
    - error_t_train: numpy.ndarray 或 None，形状为 (T_train, C)。
    - error_t_test: numpy.ndarray，形状为 (T_test, C)。
    - long_window: int，长期移动平均窗口大小。
    - short_window: int，短期移动平均窗口大小。
    - constant_std: float，防止标准差为零的常数。

    返回：
    - score_t_test_dyn: numpy.ndarray，形状为 (T_test, C)，每个通道的动态异常分数。
    """
    if not isinstance(error_t_test, np.ndarray) or error_t_test.ndim != 2:
        raise ValueError("error_t_test must be a 2D numpy.ndarray with shape (T_test, C)")
    if error_t_train is not None:
        if not isinstance(error_t_train, np.ndarray) or error_t_train.ndim != 2:
            raise ValueError("error_t_train must be a 2D numpy.ndarray with shape (T_train, C) or None")
        if error_t_train.shape[1] != error_t_test.shape[1]:
            raise ValueError("error_t_train and error_t_test must have the same number of channels (C)")

    T_test, C = error_t_test.shape
    n_t = T_test

    # 计算短期移动平均
    short_term_means = np.concatenate(
        (error_t_test[:short_window - 1], moving_average(error_t_test, short_window)), axis=0
    )
    # short_term_means.shape = (T_test, C)

    # 确定长期窗口大小
    if long_window >= n_t:
        long_win = n_t - 1
    else:
        long_win = long_window

    if error_t_train is None:
        # 初始化长期窗口
        init_score_t_test = np.zeros((long_win - 1, C))

        # 计算长期移动平均
        means_test_t = moving_average(error_t_test, long_win)  # shape: (T_test - long_win + 1, C)

        # 计算长期标准差
        df_test = pd.DataFrame(error_t_test)
        stds_test_t = df_test.rolling(window=long_win).std().values[long_win - 1:, :]  # shape: (T_test - long_win + 1, C)

        # 防止标准差为零
        stds_test_t[stds_test_t == 0] = constant_std

        # 计算标准化分数
        standardized_scores = (short_term_means[(long_win - 1):] - means_test_t) / stds_test_t  # shape: (T_test - long_win + 1, C)

        # 计算动态异常分数
        distribution = norm(0, 1)
        score_t_test_dyn_partial = -distribution.logsf(standardized_scores)  # shape: (T_test - long_win + 1, C)

        # 拼接初始零值和计算的分数
        score_t_test_dyn = np.concatenate([init_score_t_test, score_t_test_dyn_partial], axis=0)  # shape: (T_test, C)
    else:
        T_train, C_train = error_t_train.shape
        if C_train != C:
            raise ValueError("error_t_train and error_t_test must have the same number of channels (C)")

        # 处理训练数据与测试数据的拼接
        if T_train < long_win - 1:
            # 如果训练数据不足长期窗口大小，前面补零
            padding = np.zeros((long_win - 1 - T_train, C))
            full_ts = np.concatenate([padding, error_t_train, error_t_test], axis=0)  # shape: (long_win - 1 + T_test, C)
        else:
            # 如果训练数据足够，从训练数据末尾截取
            full_ts = np.concatenate([error_t_train[-long_win + 1:], error_t_test], axis=0)  # shape: (long_win - 1 + T_test, C)

        # 计算长期移动平均
        means_test_t = moving_average(full_ts, long_win)  # shape: (T_test, C)

        # 计算长期标准差
        df_full = pd.DataFrame(full_ts)
        stds_test_t = df_full.rolling(window=long_win).std().values[long_win - 1:, :]  # shape: (T_test, C)

        # 防止标准差为零
        stds_test_t[stds_test_t == 0] = constant_std

        # 计算标准化分数
        standardized_scores = (short_term_means - means_test_t) / stds_test_t  # shape: (T_test, C)

        # 计算动态异常分数
        distribution = norm(0, 1)
        score_t_test_dyn = -distribution.logsf(standardized_scores)  # shape: (T_test, C)

    return score_t_test_dyn

def get_gaussian_kernel_scores(score_t_dyn, score_tc_dyn, kernel_sigma):
    """
    使用高斯核对动态异常分数进行平滑处理。

    参数：
    - score_t_dyn: numpy.ndarray 或 None，形状为 (T,) 或 None。
    - score_tc_dyn: numpy.ndarray 或 None，形状为 (T, C) 或 None。
    - kernel_sigma: float，高斯核的标准差。

    返回：
    - score_t_dyn_gauss_conv: numpy.ndarray 或 None，形状为 (T,) 或 (T, C)。
    - score_tc_dyn_gauss_conv: numpy.ndarray 或 None，形状为 None 或 (T, C)。
    """
    # 创建高斯核，长度为 kernel_sigma * 8
    gaussian_kernel = signal.windows.gaussian(kernel_sigma * 8, std=kernel_sigma)
    # 归一化高斯核，使其和为1
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    if score_tc_dyn is None:
        if score_t_dyn is None:
            raise ValueError("At least one of score_t_dyn or score_tc_dyn must be provided.")
        # 对一维分数进行卷积
        score_tc_dyn_gauss_conv = None
        score_t_dyn_gauss_conv = signal.convolve(score_t_dyn, gaussian_kernel, mode="same")
    else:
        n_cols = score_tc_dyn.shape[1]
        # 对每个通道进行卷积
        score_tc_dyn_gauss_conv = np.stack([
            signal.convolve(score_tc_dyn[:, col], gaussian_kernel, mode="same")
            for col in range(n_cols)
        ], axis=-1)
        # 将所有通道的卷积结果求和，得到总的动态异常分数
        score_t_dyn_gauss_conv = np.sum(score_tc_dyn_gauss_conv, axis=1)

    return score_t_dyn_gauss_conv, score_tc_dyn_gauss_conv


def get_train_score(all_pred_per_channel, epoch, lr, config):
    """
    计算通道级和全局异常分数，并将结果保存以供后续分析。

    参数：
    - all_pred_per_channel: np.ndarray，形状为 (num_batches, batch_size, window_size, num_features)，
      所有批次中每个通道的预测误差。
    - epoch: int，当前计算分数的epoch编号。
    - lr: float，当前epoch使用的学习率。

    返回：
    - channel_scores: np.ndarray，形状为 (num_samples, num_features)，通道级别的异常分数。
    - global_scores: np.ndarray，形状为 (num_samples,)，全局异常分数。
    """
    # 将所有批次的数据拼接起来，以便恢复到全局时间步 T 的维度
    all_pred_per_channel = np.concatenate(all_pred_per_channel, axis=0)  # (bs, window_size, num_features)
    #print(f"all_pred_per_channel shape (num_samples, window_size, num_features): {all_pred_per_channel.shape}") # (79485, 5, 51) (89980, 5, 51)

    prediction_channel_error = np.concatenate(all_pred_per_channel, axis=0)
    #print(f"prediction_channel_error shape 不确定长度 可能不是连续时间戳: {prediction_channel_error.shape}") # (397425=496800*0.8-windowsize_15, 51) (449900, 51)


    # Step 1: 计算通道级别的动态异常分数
    global_errror = np.mean(prediction_channel_error, axis=1)
    channel_scores = np.sqrt(prediction_channel_error**2)
    global_scores = np.mean(channel_scores, axis=1)
    '''
    score_t_dyn = get_dynamic_score_t(
        error_t_train=None,
        error_t_test=prediction_channel_error,
        long_window=config.long_window, 
        short_window=config.short_window, 
        constant_std=config.constant_std
    )  # shape: (T_test, C)

    # Step 2: 应用高斯卷积进行平滑
    global_scores, channel_scores = get_gaussian_kernel_scores(
        score_t_dyn=None,  # 不使用一维分数
        score_tc_dyn=score_t_dyn,  # 使用二维分数
        kernel_sigma=config.kernel_sigma
    )
    '''

    # 定义保存结果的输出目录，包含epoch和学习率作为子目录
    output_dir = os.path.join(config.score_dir, f"epoch_{epoch}_lr_{lr}")
    os.makedirs(output_dir, exist_ok=True)

    # 将通道分数和全局分数保存为字典，存储为 .pkl 文件
    scores = {"train_channel_error": prediction_channel_error, "train_global_errror": global_errror, "channel_scores": channel_scores, "global_scores": global_scores}
    with open(os.path.join(output_dir, "anomaly_scores.pkl"), "wb") as f:
        pickle.dump(scores, f)
    logging.info(f'Save anomaly scores to file: {output_dir}')
    print(f'Save anomaly scores to file: {output_dir}')
    # 输出异常分数的维度信息
    # print(f"Channel-level anomaly scores shape: {channel_scores.shape}")
    # print(f"Anomaly scores have been saved to {output_dir}/anomaly_scores.pkl")

    return channel_scores, global_scores

def get_test_score(train_channel_error, test_channel_error, epoch, lr, config):
    """
    计算通道级和全局异常分数，并将结果保存以供后续分析。

    参数：
    - all_pred_per_channel: np.ndarray，形状为 (num_batches, batch_size, window_size, num_features)，
      所有批次中每个通道的预测误差。
    - epoch: int，当前计算分数的epoch编号。
    - lr: float，当前epoch使用的学习率。

    返回：
    - channel_scores: np.ndarray，形状为 (num_samples, num_features)，通道级别的异常分数。
    - global_scores: np.ndarray，形状为 (num_samples,)，全局异常分数。
    """
    # 将所有批次的数据拼接起来，以便恢复到全局时间步 T 的维度
    test_channel_error = np.concatenate(test_channel_error, axis=0)  # (bs, window_size, num_features)
    #print(f"test_channel_error shape (num_samples, window_size, num_features): {test_channel_error.shape}") # (79485, 5, 51) (89980, 5, 51)

    test_channel_error = np.concatenate(test_channel_error, axis=0)
    #print(f"test_channel_error shape 不确定长度 可能不是连续时间戳: {test_channel_error.shape}") # (397425=496800*0.8-windowsize_15, 51) (449900, 51)

    # Step 1: 计算通道级别的动态异常分数
    test_global_error = np.mean(test_channel_error, axis=1)
    channel_scores = np.sqrt(test_channel_error**2)
    global_scores = np.mean(channel_scores, axis=1)
    '''
    score_t_dyn = get_dynamic_score_t(
        error_t_train=train_channel_error,
        error_t_test=test_channel_error,
        long_window=config.long_window, 
        short_window=config.short_window, 
        constant_std=config.constant_std
    )  # shape: (T_test, C)

    # Step 2: 应用高斯卷积进行平滑
    global_scores, channel_scores = get_gaussian_kernel_scores(
        score_t_dyn=None,  # 不使用一维分数
        score_tc_dyn=score_t_dyn,  # 使用二维分数
        kernel_sigma=config.kernel_sigma
    )
    '''
    # 定义保存结果的输出目录，包含epoch和学习率作为子目录
    output_dir = os.path.join(config.score_dir, f"epoch_{epoch}_lr_{lr}")
    os.makedirs(output_dir, exist_ok=True)

    # 将通道分数和全局分数保存为字典，存储为 .pkl 文件
    scores = {"test_channel_error": test_channel_error, "test_global_errror": test_global_error, "channel_scores": channel_scores, "global_scores": global_scores}
    with open(os.path.join(output_dir, "anomaly_scores.pkl"), "wb") as f:
        pickle.dump(scores, f)

    # 输出异常分数的维度信息
    # print(f"Channel-level anomaly scores shape: {channel_scores.shape}")
    # print(f"Anomaly scores have been saved to {output_dir}/anomaly_scores.pkl")

    return test_channel_error, test_global_error, channel_scores, global_scores


def compute_anomaly_labels(pred_window, target_window, init_data_length=100, pot_level=0.01):
    """
    计算异常分数、应用POT方法并生成通道级和样本级的异常标签。

    参数：
    - pred_window: numpy array，形状为 (batch_size, window_size, num_features)，模型的预测值。
    - target_window: numpy array，形状为 (batch_size, window_size, num_features)，实际的目标值。
    - init_data_length: int，用于 POT 初始化的初始数据长度，默认为 100。
    - pot_level: float，POT 方法的检测水平，默认为 0.01。

    返回：
    - sample_labels: numpy array，形状为 (batch_size,)，表示样本级别的预测标签。
    - channel_labels: numpy array，形状为 (batch_size, num_features)，表示通道级别的预测标签。
    - thresholds: list，包含每个通道的 POT 动态阈值。
    - anomaly_scores: numpy array，形状为 (batch_size, num_features)，每个通道的异常分数。
    """
    

    # Step 1: 计算每个通道的均绝对误差（MAE）异常分数
    anomaly_scores = np.mean(np.abs(pred_window - target_window), axis=1)  # (batch_size, num_features)
    
    # Step 2: 应用峰值超调（POT）方法确定每个通道的动态阈值
    thresholds = []
    pot_models = []
    for channel in range(anomaly_scores.shape[1]):
        channel_scores = anomaly_scores[:, channel]
        
        init_data_length = min(init_data_length, channel_scores.shape[0] // 2)
        
        # 初始化 POT
        pot = SPOT(q=pot_level)  # 设置 q 为检测水平
        
        if np.isnan(channel_scores).any() or np.isinf(channel_scores).any():
            raise ValueError(f"Invalid values in channel_scores for channel {channel}")


        pot.fit(init_data=channel_scores[:init_data_length], data=channel_scores[init_data_length:])
        
        try:
            pot.initialize()
        except ValueError as e:
            print(f"Error in POT initialization for channel {channel}: {e}")
            thresholds.append(np.nan)  # 将阈值设为 NaN 以标记异常
            continue

        pot_models.append(pot)
        
        # 获取阈值
        thresholds.append(pot.extreme_quantile)
    
    # Step 3: 根据阈值生成通道级别的异常标签
    channel_labels = (anomaly_scores >= np.array(thresholds)).astype(int)  # (batch_size, num_features)
    
    # Step 4: 转化为样本级的预测标签
    sample_labels = (channel_labels.sum(axis=1) > 0).astype(int)  # (batch_size,)
    
    return sample_labels, channel_labels, thresholds, anomaly_scores


def compute_anomaly_labels(pred_window, target_window, init_data_length=100, threshold_factor=2):
    """
    计算异常分数、生成样本级的异常标签（基于样本的 MAE）。

    参数：
    - pred_window: numpy array，形状为 (batch_size, window_size, num_features)，模型的预测值。
    - target_window: numpy array，形状为 (batch_size, window_size, num_features)，实际的目标值。
    - init_data_length: int，用于初始化的初始数据长度，默认为 100。
    - threshold_factor: float，阈值因子，用于计算阈值。

    返回：
    - sample_labels: numpy array，形状为 (batch_size,)，表示样本级别的预测标签。
    - thresholds: float，样本级别的阈值。
    - anomaly_scores: numpy array，形状为 (batch_size,)，每个样本的异常分数。
    """

    # Step 1: 计算每个样本的均绝对误差（MAE）异常分数
    anomaly_scores = np.mean(np.abs(pred_window - target_window), axis=(1, 2))  # (batch_size,)

    # Step 2: 基于统计方法确定样本级的动态阈值
    init_data_length = min(init_data_length, len(anomaly_scores) // 2)
    init_scores = anomaly_scores[:init_data_length]

    mean_score = np.mean(init_scores)
    std_score = np.std(init_scores)
    threshold = mean_score + threshold_factor * std_score

    # Step 3: 根据阈值生成样本级别的异常标签
    sample_labels = (anomaly_scores >= threshold).astype(int)  # (batch_size,)
    anomaly_ratio = np.mean(sample_labels) * 100  # 转换为百分比形式

    # 输出异常比率
    print(f"当前的异常比率为：{anomaly_ratio:.2f}%")
    
    return sample_labels, threshold, anomaly_scores


'''
def compute_anomaly_labels(pred_window, target_window, init_data_length=100, quantile=91.7):
    """
    计算异常分数、生成样本级的异常标签（基于样本的 MAE 分位数阈值）。

    参数：
    - pred_window: numpy array，形状为 (batch_size, window_size, num_features)，模型的预测值。
    - target_window: numpy array，形状为 (batch_size, window_size, num_features)，实际的目标值。
    - init_data_length: int，用于初始化的初始数据长度，默认为 100。
    - quantile: float，分位数百分比，用于确定异常检测的阈值，默认为 91.7。

    返回：
    - sample_labels: numpy array，形状为 (batch_size,)，表示样本级别的预测标签。
    - threshold: float，样本级别的阈值（基于分位数）。
    - anomaly_scores: numpy array，形状为 (batch_size,)，每个样本的异常分数。
    """

    # Step 1: 计算每个样本的均绝对误差（MAE）异常分数
    anomaly_scores = np.mean(np.abs(pred_window - target_window), axis=(1, 2))  # (batch_size,)

    # Step 2: 基于分位数确定样本级的动态阈值
    init_data_length = min(init_data_length, len(anomaly_scores) // 2)
    init_scores = anomaly_scores[:init_data_length]

    # 使用分位数计算阈值，确保异常比例接近真实比例
    threshold = np.percentile(init_scores, quantile)

    # Step 3: 根据阈值生成样本级别的异常标签
    sample_labels = (anomaly_scores >= threshold).astype(int)  # (batch_size,)
    anomaly_ratio = np.mean(sample_labels) * 100  # 转换为百分比形式

    # 输出异常比率
    print(f"当前的异常比率为：{anomaly_ratio:.2f}%")
    
    return sample_labels, threshold, anomaly_scores
'''
'''
def compute_anomaly_labels(pred_window, target_window, init_data_length=100, quantile=91.1):
    """
    计算异常分数、生成通道级和样本级的异常标签（使用分位数方法）。

    参数：
    - pred_window: numpy array，形状为 (batch_size, window_size, num_features)，模型的预测值。
    - target_window: numpy array，形状为 (batch_size, window_size, num_features)，实际的目标值。
    - init_data_length: int，用于初始化的初始数据长度，默认为 100。
    - quantile: float，分位数百分比（默认91.1），用于确定阈值。

    返回：
    - sample_labels: numpy array，形状为 (batch_size,)，表示样本级别的预测标签。
    - channel_labels: numpy array，形状为 (batch_size, num_features)，表示通道级别的预测标签。
    - thresholds: list，包含每个通道的阈值。
    - anomaly_scores: numpy array，形状为 (batch_size, num_features)，每个通道的异常分数。
    """
    
    # Step 1: 计算每个通道的均绝对误差（MAE）异常分数
    anomaly_scores = np.mean(np.abs(pred_window - target_window), axis=1)  # (batch_size, num_features)
    
    # Step 2: 基于分位数确定每个通道的动态阈值
    thresholds = []
    for channel in range(anomaly_scores.shape[1]):
        channel_scores = anomaly_scores[:, channel]
        
        # 使用初始化数据的分位数确定阈值
        init_data_length = min(init_data_length, len(channel_scores) // 2)
        init_scores = channel_scores[:init_data_length]
        
        threshold = np.percentile(init_scores, quantile)
        thresholds.append(threshold)
    
    # Step 3: 根据阈值生成通道级别的异常标签
    channel_labels = (anomaly_scores >= np.array(thresholds)).astype(int)  # (batch_size, num_features)
    
    # Step 4: 转化为样本级的预测标签
    sample_labels = (channel_labels.sum(axis=1) > 0).astype(int)  # (batch_size,)
    
    return sample_labels, channel_labels, thresholds, anomaly_scores
'''

def prediction_adjust(prediction, labels):
    """
    对样本级别的异常标签进行调整，使得在连续的异常段中所有相关样本都被标记为异常。

    参数:
    - prediction: numpy array, 样本级别的预测标签 (0 表示正常，1 表示异常)。
    - labels: numpy array, 样本级别的真实标签 (0 表示正常，1 表示异常)。

    返回:
    - adjusted_prediction: numpy array, 调整后的预测标签。
    """
    # 确保 prediction 和 labels 长度一致
    prediction = prediction[:len(labels)]
    adjusted_prediction = prediction.copy()
    
    i = 0
    length = len(labels)
    
    while i < length:
        if labels[i] == 1:  # 找到一个异常的开始
            j = i
            adjust_flag = False
            
            # 寻找连续异常段的结束
            while j < length and labels[j] == 1:
                if prediction[j] == 1:
                    adjust_flag = True
                j += 1
            
            # 如果在该段中检测到预测异常，则将整段标记为异常
            if adjust_flag:
                adjusted_prediction[i:j] = 1
            
            i = j  # 跳到下一个段的起始位置
        else:
            i += 1
    
    return adjusted_prediction



def calculate_metrics(true_labels, predicted_labels):
    """
    计算召回率、准确率和 F1 值。

    参数:
    - true_labels: numpy array, 真实标签，形状为 (batch,)，0 表示正常，1 表示异常。
    - predicted_labels: numpy array, 预测标签，形状为 (batch,)，0 表示正常，1 表示异常。

    返回:
    - metrics: dict, 包含召回率、准确率和 F1 值。
    """
    # 计算准确率、召回率和 F1 值
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    # 将结果存储在字典中
    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    # 输出结果
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    return metrics

def test_prediction_adjust(true_labels, predicted_labels):
    """
    测试 prediction_adjust 函数的效果。
    计算调整前后指标值，分别包含召回率、准确率和 F1 值。

    参数:
    - true_labels: numpy array, 真实标签，形状为 (batch,)。
    - predicted_labels: numpy array, 原始预测标签，形状为 (batch,)。

    返回:
    - original_metrics: dict, 调整前的指标值。
    - adjusted_metrics: dict, 调整后的指标值。
    """
    # 调整前的指标值
    original_metrics = calculate_metrics(true_labels, predicted_labels)
    print("调整前的指标值:")
    print(f"Precision: {original_metrics['Precision']:.2f}")
    print(f"Recall: {original_metrics['Recall']:.2f}")
    print(f"F1 Score: {original_metrics['F1 Score']:.2f}")

    # 使用 prediction_adjust 函数对预测标签进行调整
    adjusted_predictions = prediction_adjust(predicted_labels, true_labels)

    # 调整后的指标值
    adjusted_metrics = calculate_metrics(true_labels, adjusted_predictions)
    print("\n调整后的指标值:")
    print(f"Precision: {adjusted_metrics['Precision']:.2f}")
    print(f"Recall: {adjusted_metrics['Recall']:.2f}")
    print(f"F1 Score: {adjusted_metrics['F1 Score']:.2f}")

    return original_metrics, adjusted_metrics