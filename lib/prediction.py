import numpy as np

def calculate_anomaly_score(predictions, actuals, recons, pi, gamma):
    """
    计算异常分数。

    参数：
    predictions : np.ndarray
        模型预测值，形状为 (batch, windowsize, channel)
    actuals : np.ndarray
        实际值，形状为 (batch, windowsize, channel)
    recons : np.ndarray
        重构值，形状为 (batch, windowsize, channel)
    pi : np.ndarray
        重构概率，形状为 (batch, windowsize, channel)
    gamma : float
        权重系数

    返回：
    scores : np.ndarray
        异常分数，形状为 (batch, windowsize, channel)
    global_scores : np.ndarray
        全局异常分数，形状为 (batch, windowsize)
    """
    # 计算绝对误差
    abs_error_predictions = np.abs(predictions - actuals)  # (batch, windowsize, channel)
    abs_error_weights = np.abs(recons - actuals)           # (batch, windowsize, channel)

    # 计算通道级别的异常分数
    channel_anomaly_scores = (abs_error_predictions + abs_error_weights) / 2 + gamma * (1 - pi[:, :, np.newaxis])

    # 计算全局异常分数
    global_scores = np.mean(channel_anomaly_scores, axis=2)  # (batch, windowsize)

    return channel_anomaly_scores, global_scores

# 示例用法
if __name__ == "__main__":
    # 假设我们有以下数据
    predictions = np.array([[[0.5, 0.2], [0.4, 0.1]], 
                            [[0.6, 0.3], [0.5, 0.2]], 
                            [[0.7, 0.4], [0.8, 0.5]]])  # 形状 (3, 2, 2)
    
    actuals = np.array([[[0.6, 0.3], [0.5, 0.2]], 
                        [[0.7, 0.4], [0.6, 0.3]], 
                        [[0.8, 0.5], [0.9, 0.6]]])  # 形状 (3, 2, 2)
    
    weights = np.array([[[1, 1], [1, 1]], 
                        [[1, 1], [1, 1]], 
                        [[1, 1], [1, 1]]])  # 形状 (3, 2, 2)
    
    pi = np.array([[[0.9, 0.9], [0.9, 0.9]], 
                   [[0.8, 0.8], [0.8, 0.8]], 
                   [[0.85, 0.85], [0.85, 0.85]]])  # 形状 (3, 2, 2)
    gamma = 0.5

    channel_scores, global_scores = calculate_anomaly_score(predictions, actuals, weights, pi, gamma)
    print("Channel Anomaly Scores:", channel_scores)
    print("Global Anomaly Scores:", global_scores)
