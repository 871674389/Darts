import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config_files.config_general import Config
from torch.autograd import Variable
import torch.nn.functional as F

# 配置参数
config = Config()

def nll_gaussian_with_intermediate(pred_window, target_window, variance, add_const=False):
    neg_log_p = ((pred_window - target_window) ** 2 / (2 * variance))  # (batch_size, target_window, features_num)
    
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const

    per_feature_loss = neg_log_p  # (batch_size, target_window, features_num)
    per_window_loss = per_feature_loss.mean(dim=2)  # (batch_size, target_window)
    per_sample_loss = per_window_loss.mean(dim=1)  # (batch_size,)
    total_loss = per_sample_loss.mean()  # 标量

    return total_loss, per_feature_loss, per_window_loss, per_sample_loss

def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))

def point_adjust_eval(anomaly_start, anomaly_end, down, loss, thr1, thr2):
    len_ = int(anomaly_end[-1] / down) + 1
    anomaly = np.zeros((len_))
    loss = loss[:len_]
    anomaly[np.where(loss > thr1)] = 1
    anomaly[np.where(loss < thr2)] = 1
    ground_truth = np.zeros((len_))
    for i in range(len(anomaly_start)):
        start_idx = int(anomaly_start[i] / down)
        end_idx = int(anomaly_end[i] / down) + 1
        ground_truth[start_idx:end_idx] = 1
        if np.sum(anomaly[start_idx:end_idx]) > 0:
            anomaly[start_idx:end_idx] = 1
        anomaly[start_idx] = ground_truth[start_idx]
        anomaly[end_idx - 1] = ground_truth[end_idx - 1]

    # 返回 anomaly, ground_truth 和 loss
    return anomaly, ground_truth, loss


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def create_windows(data, window_size, target_size, step_size, max_previous_steps):
    """
    使用重叠窗口划分时间序列数据，每个窗口的目标窗口为下一个连续窗口，previous_steps 采用“重复data[0]若干次+当前数据拼接”机制。
    
    参数:
    - data: 时间序列数据，形状为 (T, features)
    - window_size: input_window 的大小
    - target_size: target_window 的大小
    - step_size: 窗口滑动的步长（通常设为1，确保目标窗口为下一个连续窗口）
    - max_previous_steps: previous_steps 的最大长度，若不足则在前面重复data[0]若干次

    返回:
    - input_windows: 形状为 (num_samples, window_size, features)
    - previous_steps: 形状为 (num_samples, max_previous_steps, features)，包含“重复data[0]若干次+当前数据拼接”
    - target_windows: 形状为 (num_samples, target_size, features)，即下一个连续窗口
    """
    num_samples = (data.shape[0] - window_size - target_size) // step_size + 1
    input_windows = []
    previous_steps = []
    target_windows = []
    
    for i in range(num_samples):
        start_idx = i * step_size
        input_window = data[start_idx:start_idx + window_size]
        target_window = data[start_idx + window_size:start_idx + window_size + target_size]
        
        # previous_steps: 不包含 input_window 在内的所有之前时间步
        prev_steps = data[:start_idx]
        
        # 如果长度超出，进行截断
        if max_previous_steps and len(prev_steps) > max_previous_steps:
            prev_steps = prev_steps[-max_previous_steps:]
        # 如果长度不足，重复data[0]若干次，和当前数据拼接，刚好长度为max_previous_steps
        elif max_previous_steps and len(prev_steps) < max_previous_steps:
            # 计算需要填充的数据长度
            num_fill = max_previous_steps - len(prev_steps)
            
            # 填充：重复 data[0]，直到长度达到 num_fill，并且每次重复的长度为 features
            fill_data = np.tile(data[0], (num_fill, 1))  # 将 data[0] 重复 num_fill 次，形成 (num_fill, features) 形状
            
            # 将填充的数据与现有的 prev_steps 拼接，形成完整的 prev_steps
            prev_steps = np.concatenate([fill_data, prev_steps], axis=0)


        input_windows.append(input_window)
        previous_steps.append(prev_steps)
        target_windows.append(target_window)
        

    return np.array(input_windows), np.array(previous_steps), np.array(target_windows)

def create_labels(labels, window_size, target_size, step_size):
    """
    使用重叠窗口划分时间序列数据，每个窗口的目标窗口为下一个连续窗口。
    
    参数:
    - labels: 时间序列数据标签，形状为 (T, features)
    - window_size: input_window 的大小
    - target_size: target_window 的大小
    - step_size: 窗口滑动的步长（通常设为1，确保目标窗口为下一个连续窗口）
    
    返回:
    - target_labels: 形状为 (num_samples, target_size, features_labels)，即下一个连续窗口
    """
    num_samples = (labels.shape[0] - window_size - target_size) // step_size + 1
    target_labels = []
    
    for i in range(num_samples):
        start_idx = i * step_size
        target_label = labels[start_idx + window_size:start_idx + window_size + target_size]

        target_labels.append(target_label)
        
    return np.array(target_labels)
    
def create_labels_voting(labels, window_size, target_size, step_size):
    """
    使用多数投票法对标签进行窗口划分。若窗口内超过一半的标签为异常，则该窗口标记为异常。
    
    参数:
    - labels: 原始标签数据，形状为 (num_samples,)
    - window_size: input_window 的大小
    - target_size: target_window 的大小
    - step_size: 窗口滑动的步长
    
    返回:
    - labels_windows: 经过多数投票法划分后的窗口标签数组，形状为 (num_windows,)
    """
    # 计算窗口数量
    num_windows = (len(labels) - window_size - target_size) // step_size + 1
    labels_windows = []

    for i in range(num_windows):
        # 获取当前窗口的标签范围
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # 使用多数投票法：超过一半的时间步为异常则标记为异常
        window_labels = labels[start_idx:end_idx]
        window_label = 1 if np.sum(window_labels) > (window_size // 2) else 0
        labels_windows.append(window_label)
    
    return np.array(labels_windows)
    
def load_data_train(filename, config, DEVICE, shuffle=True):
    """
    加载训练集数据，生成训练集和验证集DataLoader
    参数:
    - filename: 数据文件路径
    - config: 模型配置
    - DEVICE: 设备(CPU或GPU)
    - shuffle: 是否打乱数据
    
    返回:
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    """
    data = np.load(filename)  # 加载数据文件
    
    # 获取超参数
    window_size = config.window_size
    target_size = config.target_size
    step_size = config.step_size
    val_split = config.val_split
    previous_steps_length = config.previous_steps_length

    # 划分训练数据和验证数据
    num_val_samples = int(len(data) * val_split)
    train_data, val_data = data[:-num_val_samples], data[-num_val_samples:]
    
    # 生成窗口
    train_input, train_prev, train_target = create_windows(train_data, window_size, target_size, step_size, previous_steps_length)
    val_input, val_prev, val_target = create_windows(val_data, window_size, target_size, step_size, previous_steps_length)
    
    # 转换为Tensor
    train_input_tensor = torch.from_numpy(train_input).float().to(DEVICE)  # (num_samples, window_size, features)
    train_prev_tensor = torch.from_numpy(train_prev).float().to(DEVICE)  # (num_samples, previous_steps_length, features)
    train_target_tensor = torch.from_numpy(train_target).float().to(DEVICE)  # (num_samples, target_size, features)
    
    val_input_tensor = torch.from_numpy(val_input).float().to(DEVICE)
    val_prev_tensor = torch.from_numpy(val_prev).float().to(DEVICE)
    val_target_tensor = torch.from_numpy(val_target).float().to(DEVICE)
    
    # 构建DataLoader
    train_dataset = TensorDataset(train_input_tensor, train_prev_tensor, train_target_tensor)
    val_dataset = TensorDataset(val_input_tensor, val_prev_tensor, val_target_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False) # 批次（batch）之间的顺序
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader
    

def load_data_test(filename_data, config, DEVICE):
    """
    加载测试集数据，生成测试集 DataLoader
    参数:
    - filename: 数据文件路径
    - config: 模型配置
    - DEVICE: 设备(CPU或GPU)
    
    返回:
    - test_loader: 测试数据加载器
    """
    # 加载测试数据文件
    data = np.load(filename_data)  # 加载数据文件
    
    # 获取超参数
    window_size = config.window_size
    target_size = config.target_size
    step_size = config.step_size
    previous_steps_length = config.previous_steps_length

    # 生成窗口数据
    test_input, test_prev, test_target = create_windows(data, window_size, target_size, step_size, previous_steps_length)
    #test_target_labels = create_labels(labels, window_size, target_size, step_size)
    
    # 转换为 Tensor
    test_input_tensor = torch.from_numpy(test_input).float().to(DEVICE)  # (num_samples, window_size, features)
    test_prev_tensor = torch.from_numpy(test_prev).float().to(DEVICE)  # (num_samples, previous_steps_length, features)
    test_target_tensor = torch.from_numpy(test_target).float().to(DEVICE)  # (num_samples, target_size, features)
    #test_target_labels_tensor = torch.from_numpy(test_target_labels).float().to(DEVICE)  # (num_samples, target_size, features)
    
    # 构建 DataLoader
    test_dataset = TensorDataset(test_input_tensor, test_prev_tensor, test_target_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return test_loader

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float().to(config.device)
    return - torch.log(eps - torch.log(U + eps))

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps).to(config.device)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.to(logits.device) 
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dim=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape).to(config.device)
        if y_soft.is_cuda:
            y_hard = y_hard.to(logits.device) 
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0).to(config.device)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def combine_windows(windows, window_size, step_size):
    """
    将窗口数据还原回时间序列，计算原始T。

    参数：
    - windows: numpy 数组，形状为 (num_samples, target_size, num_features)
    - window_size: 每个目标窗口的大小
    - step_size: 每个窗口之间的步长

    返回：
    - combined_data: numpy 数组，形状为 (T, num_features)
    """
    num_samples, _, num_features = windows.shape
    # 根据窗口信息计算 T 的实际长度
    T_estimated = (num_samples - 1) * step_size + window_size

    # 初始化组合数据和重叠计数数组
    combined_data = np.zeros((T_estimated, num_features))
    overlap_counts = np.zeros(T_estimated)

    # 累加窗口数据并计数重叠次数
    for i in range(num_samples):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        combined_data[start_idx:end_idx] += windows[i]
        overlap_counts[start_idx:end_idx] += 1

    # 对重叠部分取平均值
    combined_data /= np.maximum(overlap_counts[:, np.newaxis], 1)
    # print(f"shape of combined_window_data: {combined_data.shape}")
    return combined_data


def combine_batches_windows(batched_data, window_size, step_size):
    """
    将批次的窗口数据还原回时间序列，计算原始T。

    参数：
    - batched_data: numpy 数组，形状为 (num_batches, batch_size, window_size)
    - window_size: 每个窗口的大小
    - step_size: 每个窗口之间的步长

    返回：
    - combined_data: numpy 数组，形状为 (T, )
    """
    num_batches, batch_size, _ = batched_data.shape
    # 计算实际的 T
    num_samples = num_batches * batch_size
    T_estimated = (num_samples - 1) * step_size + window_size

    # 初始化组合数据和重叠计数数组
    combined_data = np.zeros(T_estimated)
    overlap_counts = np.zeros(T_estimated)

    for batch_idx in range(num_batches):
        for sample_idx in range(batch_size):
            start_idx = (batch_idx * batch_size + sample_idx) * step_size
            end_idx = start_idx + window_size
            combined_data[start_idx:end_idx] += batched_data[batch_idx, sample_idx]
            overlap_counts[start_idx:end_idx] += 1

    # 对重叠部分取平均值
    combined_data /= np.maximum(overlap_counts, 1)
    print(f"shape of combined_batch_data: {combined_data.shape}")
    return combined_data
