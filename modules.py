import torch.nn as nn
import sys
sys.path.append('..')
from lib.utils import *
import torch.nn.utils.prune as prune
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import torch as tr
import math

class PredictionModule(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout=0.2):
        super(PredictionModule, self).__init__()
        
        # 创建一个列表 layers 包含输入层和隐藏层的线性变换
        layers = [nn.Linear(in_dim, hid_dim)]  # 输入层
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))  # 隐藏层
        
        # 最后一层是输出层，确保输出维度为 out_dim
        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入形状为 (batch_size, window_size, in_dim)

        # 逐时间步处理输入
        pred_window = []
        for t in range(x.size(1)):  # 对于每个时间步
            h_t = x[:, t, :]  # 取出时间步 t 的输入，形状为 (batch_size, in_dim)
            for i in range(len(self.layers) - 1):
                h_t = self.relu(self.layers[i](h_t))
                h_t = self.dropout(h_t)
            pred_window.append(self.layers[-1](h_t))  # 最后一层输出
        
        # 将输出堆叠为张量，形状为 (batch_size, window_size, out_dim)
        pred_window = torch.stack(pred_window, dim=1)
        
        # 确保输出在 [0, 1] 范围内
        pred_window = torch.sigmoid(pred_window)  # 通过Sigmoid函数将输出限制在[0, 1]范围内
        
        return pred_window

    def calculate_pred_loss(self, pred_window, target_window):
        """
        计算预测损失。

        参数：
        pred_window : torch.Tensor
            预测值，形状为 (batch_size, window_size, out_dim)
        target_window : torch.Tensor
            实际值，形状为 (batch_size, window_size, out_dim)

        返回：
        pred_loss : torch.Tensor
            预测损失，标量
        """
        
        # 计算均方损失
        squared_loss_per_channel = (pred_window - target_window) ** 2
        mse_per_timestamp = squared_loss_per_channel.mean(dim=2)  # (batch_size, window_size)
        pred_loss = mse_per_timestamp.mean()  # 得到标量损失

        return squared_loss_per_channel, mse_per_timestamp, pred_loss
    
    def nll_gaussian_with_intermediate(self, pred_window, target_window, variance, add_const=False):
        neg_log_p = ((pred_window - target_window) ** 2 / (2 * variance))  # (batch_size, window_size, features_num)
        
        if add_const:
            const = 0.5 * np.log(2 * np.pi * variance)
            neg_log_p += const
    
        per_feature_loss = neg_log_p  # (batch_size, window_size, features_num)
        per_window_loss = per_feature_loss.mean(dim=2)  # (batch_size, window_size)
        per_sample_loss = per_window_loss.mean(dim=1)  # (batch_size,)
        total_loss = per_sample_loss.mean()  # 标量
    
        return total_loss, per_feature_loss, per_window_loss, per_sample_loss
    


class ReconstructionModel(nn.Module):
    def __init__(self, in_dim, latent_dim, output_dim):
        super(ReconstructionModel, self).__init__()
        
        # 编码器部分
        self.fc_mu = nn.Linear(in_dim, latent_dim)  # 均值
        self.fc_log_var = nn.Linear(in_dim, latent_dim)  # 对数方差
        
        # 解码部分
        self.fc_out = nn.Linear(latent_dim, in_dim)  # 重构输出
        self.output_map = nn.Linear(in_dim, output_dim)  # 映射到输出维度

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std  # 重参数化

    def forward(self, x):
        # 输入 x 的形状为 (batch, windowsize, in_dim)
        mu = self.fc_mu(x)  # 计算均值
        log_var = self.fc_log_var(x)  # 计算对数方差
        z = self.reparameterize(mu, log_var)  # 重参数化，得到潜在向量

        reconstructed_x = torch.sigmoid(self.fc_out(z))  # 重构输出
        reconstructed_x = torch.sigmoid(self.output_map(reconstructed_x))  # 映射到输出维度
        
        # 通道级别的重构概率可以直接使用重构值，假设重构值是概率
        reconstruction_probability = reconstructed_x  # 已在 [0, 1] 范围内
        
        return reconstructed_x, reconstruction_probability, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        # 计算每个时间戳、每个通道的均方误差
        mse_per_channel = (recon_x - x) ** 2  # 形状为 (batch, windowsize, channel)
        
        # 计算每个时间戳的均方误差
        mse_per_timestamp = mse_per_channel.mean(dim=2)  # 在通道维度上求均值，形状为 (batch, windowsize)
        
        # 计算最终的重构误差
        reconstruction_loss = mse_per_timestamp.mean()  # 对时间戳求均值，得到标量误差

        # 计算 KLD 误差
        kld_per_channel = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())  # 形状为 (batch, windowsize, channel)
        
        # 计算每个时间戳的 KLD
        kld_per_timestamp = kld_per_channel.mean(dim=2)  # 在通道维度上求均值，形状为 (batch, windowsize)
        
        # 计算最终的 KLD 误差
        kld_loss = kld_per_timestamp.mean()  # 对时间戳求均值，得到标量误差
        
        return mse_per_channel, mse_per_timestamp, reconstruction_loss, kld_per_channel, kld_per_timestamp, kld_loss

