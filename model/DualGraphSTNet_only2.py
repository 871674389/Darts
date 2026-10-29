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
import torch.nn.functional as F


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        # 第一层：输入维度为 n_in，输出维度为 n_hid
        self.fc1 = nn.Linear(n_in, n_hid)
        # 第二层：隐藏层维度为 n_hid，输出维度为 n_out
        self.fc2 = nn.Linear(n_hid, n_out)
        # 批归一化层（针对第二层输出）
        self.bn = nn.BatchNorm1d(n_out)
        # Dropout 概率
        self.dropout_prob = do_prob
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, input_data):
        # 将数据转换为批归一化的格式
        x = input_data.view(-1, input_data.size(-1))  # (batch_size * num_things, num_features)
        x = self.bn(x)
        return x.view(input_data.size(0), input_data.size(1), -1)  # 转回原始形状

    def forward(self, input_data):
        # 输入形状： [batch_size, num_features=51, window_size=30]
        # 第一层全连接+ELU激活
        x = F.elu(self.fc1(input_data))
        # Dropout
        x = F.dropout(x, self.dropout_prob, training=self.training)
        # 第二层全连接+ELU激活
        x = F.elu(self.fc2(x))
        # 批归一化
        x = self.batch_norm(x)
        
        return x

        
class Graph_learner(nn.Module):
    def __init__(self, n_in, n_hid, n_head_dim, head, num_nodes, temperature, device, do_prob=0.):
        super(Graph_learner, self).__init__()
        self.n_hid = n_hid
        self.head = head
        self.n_in = n_in
        self.n_head_dim = n_head_dim
        self.num_nodes = num_nodes  # 定义 num_nodes
        self.temperature = temperature
        self.device = device  # 定义 device

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.Wq = nn.Linear(n_hid, n_head_dim * head)
        self.Wk = nn.Linear(n_hid, n_head_dim * head)
        for m in [self.Wq, self.Wk]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs):  # inputs: [B, input_window, N(features)]
        X = self.mlp1(inputs.permute(0, 2, 1)) # [128, 51, 64]
        Xq = self.Wq(X)  # [B, 51, n_hid_subspace]
        Xk = self.Wk(X)
        B, N, n_hid = Xq.shape
        Xq = Xq.view(B, N, self.head, self.n_head_dim)  # [B, N(features), head, head_dim]
        Xk = Xk.view(B, N, self.head, self.n_head_dim)
        Xq = Xq.permute(0, 2, 1, 3)  # [B, head, N, head_dim=32]
        Xk = Xk.permute(0, 2, 1, 3)
        
        # 计算注意力分数
        probs = torch.matmul(Xq, Xk.transpose(-1, -2))  # [B, head, N, N]
        # print(f"probs shape[B, head, N, N]: {probs.shape}") # probs shape: torch.Size([128, 4, 51, 51])
        mask_loc = torch.eye(self.num_nodes, dtype=bool).to(self.device) # [51, 51]
        #print(f"mask_loc shape[N, N]: {mask_loc.shape}") # [51, 51]
        probs_reshaped = probs.masked_select(~mask_loc).view(B, self.head, self.num_nodes * (self.num_nodes - 1)) # [B, head, N * (N - 1)]
        #print(f"probs_reshaped shape[B, head, N * (N - 1)]: {probs_reshaped.shape}") 
        probs_reshaped = probs_reshaped.permute(0, 2, 1)  # [B, N * (N-1), head]
        #print(f"probs_reshaped shape[B, N * (N-1), head]: {probs_reshaped.shape}") 
        prob = F.softmax(probs_reshaped, -1) # [B, 51 * 50, head]
        
        # 使用 gumbel-softmax 进行采样
        edges = gumbel_softmax(torch.log(prob + 1e-5), tau=self.temperature, hard=True) # [B, N * (N-1), head]
        # print(f"edges shape: {edges.shape}") # [B, 51 * 50, head]
        
        # 初始化全 1 邻接矩阵
        adj_list = torch.ones(self.head, B, self.num_nodes, self.num_nodes).to(self.device) # 初始化一个全 1 的邻接矩阵列表 adj_list，形状为 [head, B, N, N]
        mask = ~torch.eye(self.num_nodes, dtype=bool).unsqueeze(0).unsqueeze(0).to(self.device) # 构建一个形状为 [1, 1, N, N] 的掩码矩阵 mask，用于屏蔽对角线元素
        mask = mask.repeat(self.head, B, 1, 1).to(self.device) # mask 的形状更新为 [head, B, N, N]，匹配 adj_list 的形状
        adj_list[mask] = edges.permute(2, 0, 1).flatten() # 将 edges 的形状由 [B, N * (N - 1), head] 转换为 [head, B, N * (N - 1)]，并展平
        return prob, adj_list  # [B, N * (N-1), head], 形状为 [head, B, N, N]
        # 返回完整的邻接矩阵列表 [head, B, N, N]
        # 提取第一个头的邻接矩阵，形状为 [B, N, N]：adj_head_0 = adj_list[0, ...]


# DCGRU单元能够接受从上一步多头注意力生成的邻接矩阵作为输入，然后在此图结构上进行时空卷积操作，输出每个传感器的特征。
class DCGRUCell_(torch.nn.Module):
    def __init__(self, device, num_units, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self.device = device
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        self._gconv_0 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2)
        self._gconv_1 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units * 2)
        self._gconv_c_0 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units)
        self._gconv_c_1 = nn.Linear(self._num_units * 2 * (self._max_diffusion_step + 1), self._num_units)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def _calculate_random_walk_matrix(self, adj_mx):
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(self.device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def _calculate_random_walk0(self, adj_mx, B):  # adj_mx是tensor形式
        # print("Shape of adj_mx:", adj_mx.shape)
        # print("Shape of torch.eye(int(adj_mx.shape[1])):", torch.eye(int(adj_mx.shape[1])).unsqueeze(0).repeat(B, 1, 1).shape)
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).repeat(B, 1, 1).to(self.device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)
        d_mat_inv = torch.diag_embed(d_inv)
        random_walk_mx = torch.matmul(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hx, adj):
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, adj, hx, output_size, bias_start=1.0))

        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv_c(inputs, adj, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases
        return value

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        B = inputs.shape[0]
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state  # [B, N, 2 * C]
        x0_0 = torch.unsqueeze(x, 0)
        x1_0 = torch.unsqueeze(x, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            x0_1 = torch.matmul(adj_mx0, x0_0)
            x1_1 = torch.matmul(adj_mx1, x1_0)
            # print('x0_1', x0_1.shape)

            x0_0 = torch.cat([x0_0, x0_1], dim=0)
            x1_0 = torch.cat([x1_0, x1_1], dim=0)

            for k in range(2, self._max_diffusion_step + 1):
                x0_2 = torch.matmul(adj_mx0, x0_1)
                x1_2 = torch.matmul(adj_mx1, x1_1)
                x0_0 = torch.cat([x0_0, x0_1], dim=0)
                x1_0 = torch.cat([x1_0, x1_1], dim=0)
                x0_1 = x0_2
                x1_1 = x1_2
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.

        x0_0 = x0_0.permute(1, 2, 3, 0)  # [3, 90, 128]
        x1_0 = x1_0.permute(1, 2, 3, 0)  # [3, 90, 128]

        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        x0_0 = self._gconv_0(x0_0)
        x1_0 = self._gconv_1(x1_0)

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_nodes * output_size])


    def _gconv_c(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        B = inputs.shape[0]
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        adj_mx0 = self._calculate_random_walk0(adj_mx, B)
        adj_mx1 = self._calculate_random_walk0(adj_mx.permute(0, 2, 1), B)

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state  # [B, N, 2 * C]
        x0_0 = torch.unsqueeze(x, 0)
        x1_0 = torch.unsqueeze(x, 0)


        if self._max_diffusion_step == 0:
            pass
        else:
            x0_1 = torch.matmul(adj_mx0, x0_0)
            x1_1 = torch.matmul(adj_mx1, x1_0)

            x0_0 = torch.cat([x0_0, x0_1], dim=0)
            x1_0 = torch.cat([x1_0, x1_1], dim=0)

            for k in range(2, self._max_diffusion_step + 1):
                x0_2 = torch.matmul(adj_mx0, x0_1)
                x1_2 = torch.matmul(adj_mx1, x1_1)
                x0_0 = torch.cat([x0_0, x0_1], dim=0)
                x1_0 = torch.cat([x1_0, x1_1], dim=0)
                x0_1 = x0_2
                x1_1 = x1_2
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.

        x0_0 = x0_0.permute(1, 2, 3, 0)
        x1_0 = x1_0.permute(1, 2, 3, 0)

        x0_0 = torch.reshape(x0_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x1_0 = torch.reshape(x1_0, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        x0_0 = self._gconv_c_0(x0_0)
        x1_0 = self._gconv_c_1(x1_0)

        return torch.reshape(x0_0 + x1_0, [batch_size, self._num_nodes * output_size])

# 编码器可以是单层或多层的DCGRU模型，具体取决于在实例化时传入的num_rnn_layers参数。
class EncoderModel(nn.Module):
    def __init__(self, device, n_dim, n_hid, max_diffusion_step, num_nodes, num_rnn_layers):
        super(EncoderModel, self).__init__()
        self.device = device
        self.input_dim = n_dim
        self.rnn_units = n_hid
        self.max_diffusion_step = max_diffusion_step
        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell_(self.device, self.rnn_units, self.max_diffusion_step, self.num_nodes
                     ) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, adj, hidden_state=None):
        # Input: [B, N, GRU_n_dim, T]
        batch_size = inputs.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size)).to(self.device)
        hidden_states = []
        
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv1d, self).__init__()
        
        # 深度卷积：每个输入通道使用独立的卷积核
        self.depthwise = nn.Conv1d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, groups=input_channels, bias=bias)
        
        # 逐点卷积：对深度卷积的输出进行1x1卷积，混合通道信息
        self.pointwise = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
        # BatchNorm和激活函数
        self.batchnorm = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

'''
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dropout=0):
        super(FeatureExtractor, self).__init__()

        # 第一层卷积：将输入通道从 input_channels 变成 output_channels
        self.conv_block1 = nn.Sequential(
            DepthwiseSeparableConv1d(input_channels, output_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.Dropout(dropout)
        )

        # 第二层卷积：将通道数从 output_channels 变成 output_channels // 2
        self.conv_block2 = nn.Sequential(
            DepthwiseSeparableConv1d(output_channels, output_channels // 2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.Dropout(dropout)
        )

        # 第三层卷积：将通道数从 output_channels // 2 变成 output_channels // 4
        self.conv_block3 = nn.Sequential(
            DepthwiseSeparableConv1d(output_channels // 2, output_channels // 4, kernel_size=kernel_size, padding=kernel_size//2),
            nn.Dropout(dropout)
        )

        self.nonlin_map = nn.Sequential(
            nn.Linear(output_channels // 4, output_channels // 2),
            nn.BatchNorm1d(output_channels // 2)
        )

    def forward(self, x_in):
        # 输入数据形状为 [batch_size * N, T, out_dim]，转置为 [batch_size * N, out_dim, T]
        x = x_in.transpose(1, 2)  # [batch_size * N, out_dim, T]
        
        # 经过第一个卷积块
        x = self.conv_block1(x)
        
        # 经过第二个卷积块
        x = self.conv_block2(x)
        
        # 经过第三个卷积块
        x = self.conv_block3(x)

        batchN, outdim, T = x.shape
        # Flatten the tensor before passing to Linear layer: [batchN * T, outdim]
        x = self.nonlin_map(x.view(batchN * T, outdim))  # [batchN * T, outdim]
        
        # Reshape back to [batchN, T, -1]
        x = x.view(batchN, T, -1)
        
        return x
'''

class PositionalEncoding(nn.Module):
    def __init__(self, device, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        # 创建一个 (1, max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(1, max_len, d_model).to(self.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(self.device)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(self.device)  # [d_model//2]

        # 偶数维度使用 sin 函数
        pe[0, :, 0::2] = torch.sin(position * div_term)  # [1, max_len, d_model//2]
        # 奇数维度使用 cos 函数
        pe[0, :, 1::2] = torch.cos(position * div_term)  # [1, max_len, d_model//2]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: 输入的特征图，形状为 [batch_size, T, N, out_dim]
        :return: 带有位置编码的特征图，形状不变为 [batch_size, T, N, out_dim]
        """
        # 获取输入数据的时间维度大小 T
        time_length = x.size(1)  # T
        # 生成适用于当前输入的时间维度的位置编码
        pe = self.pe[:, :time_length, :]  # 取出与时间维度 T 匹配的位置编码

        # 对输入数据应用位置编码
        # x的形状为 [batch_size, T, N, out_dim]
        # 我们只在时间维度（T）上加上位置编码，不改变 N 和 out_dim 维度
        x = x + pe.unsqueeze(2)  # 广播到 [batch_size, T, N, out_dim]

        return self.dropout(x)



def Conv_GraphST(input, time_window_size, stride): 
    """
    将输入时间序列划分成多个不重叠的时间窗口。
    
    参数:
    - input: 输入数据, 形状为 (batch_size, T, N, out_dim)
    - time_window_size: 每个时间窗口的大小
    - stride: 时间步的滑动步幅
    
    返回:
    - 窗口划分后的数据，形状为 (batch_size, T, N, time_window_size, out_dim)
    """
    # 输入数据的形状
    batch_size, time_length, num_sensors, out_dim = input.shape
    
    # 计算可以整除的时间窗口数量
    num_windows = (time_length - time_window_size) // stride + 1
    
    # 对输入进行调整，以便按时间维度进行窗口划分
    input_unf = input.permute(0, 1, 3, 2)  # 转换为 (batch_size, T, out_dim, N)
    
    # 使用 unfold 来滑动窗口划分
    # unfold 操作沿时间维度进行滑动窗口操作
    input_windows = input_unf.unfold(dimension=1, size=time_window_size, step=stride)
    
    # 将窗口数据的形状从 (batch_size, num_windows, out_dim, time_window_size, num_sensors) 转换为
    # (batch_size, time_length, num_sensors, time_window_size, out_dim)
    input_windows = input_windows.permute(0, 1, 3, 4, 2)
    
    return input_windows


def window_pooling(input_windows, kernel_size, stride):
    """
    对每个时间窗口进行平均池化。
    
    参数:
    - input_windows: 窗口划分后的数据, 形状为 (batch_size, num_windows, N, window_size, out_dim)
    
    返回:
    - 平均池化后的窗口特征，形状为 (batch_size, num_windows, N, out_dim)
    """
    # 这里我们输入的维度是 [B, num_windows, N, window_size, out_dim]

    batch_size, num_windows, N, window_size, out_dim = input_windows.shape
    # 使用 view 来调整数据形状为 (batch_size * num_windows * N, window_size, out_dim)
    input_reshaped = input_windows.reshape(batch_size * num_windows * N, out_dim, window_size)
    # 对 window_size 维度进行池化，保持 out_dim 不变
    pool = nn.AvgPool1d(kernel_size, stride)  # 对每个窗口大小进行池化
    pooled_windows = pool(input_reshaped)  # 输出形状为 [B * num_windows * N, 1, out_dim]
    # 恢复数据形状为 (batch_size, num_windows, N, out_dim)
    pooled_windows = pooled_windows.reshape(batch_size, num_windows, N, out_dim)
    
    return pooled_windows
    
class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim):
        """
        基于时间窗口特征学习图结构的构造函数。
        
        参数:
        - input_dim: 每个时间窗口的特征维度 (out_dim)
        """
        super(Dot_Graph_Construction_weights, self).__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, window_features):
        """
        基于窗口特征学习图结构。
        
        参数:
        - window_features: 形状为 (batch_size, num_windows, out_dim)
        
        返回:
        - 邻接矩阵, 形状为 (batch_size, num_windows, num_windows)
        """
        # 通过线性映射学习特征的权重
        window_features = self.mapping(window_features)  # (batch_size, num_windows, out_dim-->mapping_dim)
        
        # 转置以便进行批量矩阵乘法
        window_features_transposed = window_features.transpose(1, 2)  # (batch_size, mapping_dim, num_windows)
        
        # 计算窗口之间的相似性矩阵
        Adj = torch.bmm(window_features, window_features_transposed)  # (batch_size, num_windows, num_windows)
        
        # 创建掩码矩阵，去除对角线元素
        num_windows = window_features.size(1)
        eyes_like = torch.eye(num_windows).repeat(window_features.size(0), 1, 1).to(window_features.device)  # (batch_size, num_windows, num_windows)
        eyes_like_inf = eyes_like * 1e8  # 生成大数掩码
        Adj = F.leaky_relu(Adj - eyes_like_inf)  # 激活并去除自连接权重
        
        # 使用 softmax 进行行归一化
        Adj = F.softmax(Adj, dim=-1)
        
        # 恢复自连接
        Adj = Adj + eyes_like  # 加上单位矩阵，保留自连接
        
        return Adj  # 返回形状为 (batch_size, num_windows, num_windows)


def Mask_Matrix(num_node, time_length, decay_rate):
    Adj = tr.ones(num_node * time_length, num_node * time_length).cuda()
    for i in range(time_length):
        v = 0
        for r_i in range(i,time_length):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (r_i) * num_node
            idx_e_col = (r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1
        v=0
        for r_i in range(i+1):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (i-r_i) * num_node
            idx_e_col = (i-r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1

    return Adj 


class GraphNormalization(nn.Module):
    def __init__(self, num_features):
        super(GraphNormalization, self).__init__()
        self.epsilon = 1e-5
        self.alpha = nn.Parameter(tr.ones(num_features))
        self.beta = nn.Parameter(tr.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        normalized_x = (x - mean) / tr.sqrt(var + self.epsilon)
        return self.alpha.view(1, -1, 1) * normalized_x + self.beta.view(1, -1, 1)

class MPNN_mk_v2(nn.Module):
    def __init__(self, input_dim, output_dim, k, way_multi_field='sum'):
        """
        自定义图卷积扩散网络，支持多重感受野。

        参数:
        - input_dim: 输入特征维度
        - output_dim: 输出特征维度
        - k: 感受野的大小 (传播步数)
        - way_multi_field: 'cat' 或 'sum'，决定多重感受野的拼接或求和
        """
        super(MPNN_mk_v2, self).__init__()
        self.k = k  # 图卷积的层数
        self.way_multi_field = way_multi_field  # 'cat' 表示拼接，'sum' 表示求和

        # 创建 k 层线性变换
        self.theta = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(self.k)])

        # 图归一化
        self.graph_norm = GraphNormalization(output_dim)

    def forward(self, X, A):
        """
        前向传播。
        
        参数:
        - X: 输入特征矩阵，形状为 (batch_size, num_windows, input_dim)
        - A: 邻接矩阵，形状为 (batch_size, num_windows, num_windows)
        
        返回:
        - GCN_output: 输出特征矩阵，形状为 (batch_size, num_windows, output_dim)
        """
        batch_size, num_windows, input_dim = X.size()
        
        GCN_output_ = []

        # 图卷积扩散过程
        for kk in range(self.k):
            if kk == 0:
                A_ = A  # 初始邻接矩阵
            else:
                A_ = tr.bmm(A_, A)  # 邻接矩阵自乘，扩大感受野
            
            # 传播图卷积信息 (A * X)，并进行线性映射
            out_k = self.theta[kk](tr.bmm(A_, X))  # 线性变换，(batch_size, num_windows, output_dim)
            GCN_output_.append(out_k)

        # 根据 'cat' 或 'sum' 模式，处理多层图卷积的输出
        if self.way_multi_field == 'cat':
            GCN_output_ = tr.cat(GCN_output_, -1)  # 拼接操作
        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)  # 求和操作

        # 图归一化
        GCN_output_ = self.graph_norm(tr.transpose(GCN_output_, 1, 2))  # 转置为 (batch_size, output_dim, num_windows)后进行归一化
        GCN_output_ = tr.transpose(GCN_output_, 1, 2)  # 转置回 (batch_size, num_windows, output_dim)

        return F.leaky_relu(GCN_output_)


import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    ★ 修订点 ★
    - 允许 tensor_cur=None → 从 tensor_hist 派生 Query
    - 其它逻辑与两路输入时完全一致，保证可比性
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        # layer norm
        self.layer_norm_hist = nn.LayerNorm(input_dim)
        self.layer_norm_cur  = nn.LayerNorm(input_dim)

        # QKV 线性映射
        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear   = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)

        # 输出映射
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, tensor_hist, tensor_cur=None):
        """
        tensor_hist : (B, num_windows, N, input_dim)   # 历史（来自 3.4）
        tensor_cur  : (B, target_size, N, input_dim)   # 当前窗口（来自 3.3）
                      若为 None，则自动从 tensor_hist 构造
        """
        # ---------- 1. 归一化 ----------
        tensor_hist = self.layer_norm_hist(tensor_hist)

        if tensor_cur is None:
            # ★ 退化方案：取最近一个历史窗口作为“当前” ★
            #   也可改成 mean-pool: tensor_hist.mean(dim=1, keepdim=True)
            tensor_cur = tensor_hist[:, -1:, :, :]     # shape (B, 1, N, D)
        else:
            tensor_cur = self.layer_norm_cur(tensor_cur)

        # ---------- 2. 获取形状 ----------
        B, num_w,   N, D = tensor_hist.size()
        _, tgt_sz, _, _  = tensor_cur.size()

        # ---------- 3. Q K V ----------
        K = self.key_linear(tensor_hist)     # (B, num_w, N, H)
        V = self.value_linear(tensor_hist)   # (B, num_w, N, H)

        # 循环计算每个目标时间步的 Query
        out_steps = []
        for t in range(tgt_sz):
            Q = self.query_linear(tensor_cur[:, t, :, :])   # (B, N, H)
            Q = Q.unsqueeze(2).permute(0, 2, 1, 3)          # (B, 1, N, H)

            # 注意力 (B, 1, N, num_w)
            attn = torch.einsum('bqnh,bknh->bqnk', Q, K)
            attn = F.softmax(attn, dim=3)

            # 聚合 (B, 1, N, H)
            attn_out = torch.einsum('bqnk,bknh->bqnh', attn, V)
            out_steps.append(attn_out)

        # 合并并映射
        attn_out = torch.stack(out_steps, dim=1).squeeze(2)   # (B, tgt_sz, N, H)
        out      = self.fc(attn_out)                          # (B, tgt_sz, N, output_dim)
        return out


# 原始模型
class DualGraphSTNet(nn.Module):
    def __init__(self, matrix_decay, num_features, window_size, target_size, out_dim, gru_dim, Graph_learner_n_hid, Graph_learner_n_head_dim, num_heads, num_nodes, temperature, max_diffusion_step, num_rnn_layers, device):
        super(DualGraphSTNet, self).__init__()
        self.device = device
        self.matrix_decay = matrix_decay
        self.num_features = num_features
        self.window_size = window_size
        self.target_size = target_size
        self.out_dim = out_dim
        self.gru_dim = gru_dim
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.Graph_learner_n_hid = Graph_learner_n_hid
        self.Graph_learner_n_head_dim = Graph_learner_n_head_dim
        self.temperature = temperature
        self.max_diffusion_step = max_diffusion_step
        self.num_rnn_layers = num_rnn_layers

        self.linear1 = nn.Linear(1, gru_dim)
        # 第二层输入处理
        self.positional_encoding = PositionalEncoding(device, gru_dim)
        self.conv_graph_st = Conv_GraphST
        self.window_pooling = window_pooling
        self.dot_graph_weights = Dot_Graph_Construction_weights(out_dim)  
        self.mask_matrix = Mask_Matrix

        # 图卷积
        self.BN = nn.BatchNorm1d(out_dim)
        self.mpnn = MPNN_mk_v2(out_dim, out_dim, k=2)

        # 注意：此时没有 tensor_cur
        self.attentionFusion = AttentionFusion(out_dim, 2*out_dim, out_dim)

        # GRU
        self.gru = nn.GRU(input_size=out_dim, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(in_features=128, out_features=target_size * out_dim)

        '''
        # 预测模型
        self.prediction_module = PredictionModule(in_dim=out_dim, hid_dim=64, out_dim=num_features, n_layers=2, dropout=0.2)
        self.reconstruction_model = ReconstructionModel(in_dim=out_dim, latent_dim=128)
        '''
        
    def encoder(self, inputs, adj, head):
        """
        Encoder forward pass
        Inputs: [batch_size, num_nodes, feature_dim, len_sequence]
        adj: [num_nodes, num_nodes]
        """
        # print("shape of input", inputs.shape) # torch.Size([128, 51, 64, 30]) 
        # print("shape of adj", adj.shape) # [2550, 4]
        encoder_hidden_state = None
        encoder_hidden_state_tensor = torch.zeros(inputs.shape).to(self.device)
        for t in range(self.window_size):
            _, encoder_hidden_state = self.encoder_model[head](inputs[..., t], adj, encoder_hidden_state)
            encoder_hidden_state_tensor[..., t] = encoder_hidden_state[-1, ...].reshape(-1, self.num_nodes,
                                                                                        self.gru_dim)
        return encoder_hidden_state_tensor
        
    def forward(self, input_window, previous_steps):

        # 处理 previous_steps (第二层)
        previous_steps = self.linear1(previous_steps.unsqueeze(-1)) # [batch_size, T=previous_steps, N(features), gru_dim]        
        prev_features_encoded = self.positional_encoding(previous_steps) # [batch_size, T, N, gru_dim]
        windowed_data = self.conv_graph_st(prev_features_encoded, self.window_size, stride=self.window_size) # [B, num_windows, N, window_size, out_dim]
        #print("Shape of 划分时间窗口 [B, num_windows, N, window_size, out_dim]:",windowed_data.shape)
        pooled_data = self.window_pooling(windowed_data, self.window_size, self.window_size) # [B, num_windows, N, out_dim]
        #print("Shape of 池化 [B, num_windows, N, out_dim]:",pooled_data.shape)
        batch_size, num_windows, N, out_dim = pooled_data.shape        
        pooled_data_ =  pooled_data.view(batch_size, num_windows * N, out_dim) # [B, num_windows * N, out_dim]
        adj_matrix = self.dot_graph_weights(pooled_data_) # [B, num_windows * N, num_windows * N]
        #print("Shape of 邻接矩阵 [B, num_windows * N, num_windows * N]:",adj_matrix.shape)
        decay_matrix = self.mask_matrix(num_windows, N, decay_rate=self.matrix_decay) # [num_windows * N, num_windows * N]
        #print("Shape of 衰减矩阵 [num_windows * N, num_windows * N]:",decay_matrix.shape)
        final_adj_matrix = adj_matrix * decay_matrix # [batch_size, num_windows * N, num_windows * N]
        pooled_data_ = self.BN(pooled_data_.permute(0, 2, 1)).permute(0, 2, 1) # [B, num_windows * N, out_dim]
        #print("Shape of mpnn 输入数据(归一化) [B, num_windows * N, out_dim]:",pooled_data_.shape)
        output2 = self.mpnn(pooled_data_, final_adj_matrix)  # [B, num_windows * N, out_dim]
    
        # 检查第二层输出 torch.Size([256, 9, 64])
        # print("Shape of output2 [B, num_windows * N, out_dim]:", output2.shape)

        # 特征融合       
        # 通过 reshape 将数据重塑为 [B, num_windows, N, out_dim]
        tensor_hist = output2.view(batch_size, num_windows, N, -1) # [B, num_windows, N, input_dim]

        # print("Shape of tensor1 [B, num_windows, N, input_dim]:", tensor1.shape)
        # print("Shape of tensor2 [B, targetsize, N, input_dim] :", tensor2.shape)
        fused_features = self.attentionFusion(tensor_hist)           # 自动退化为自注意力
        # print("Shape of fused_features [B, targetsize, N, input_dim] :", fused_features.shape)
        output = fused_features.mean(dim=-1) # [B, target_size, N]
        # print("Shape of output [B, target_size, N]:", output.shape)
        
        # 进行最大最小归一化
        # 计算每个特征的最小值和最大值
        #min_val = output.min(dim=-1, keepdim=True)[0]  # 最小值 [bs, target_size, num_sensors, 1]
        #max_val = output.max(dim=-1, keepdim=True)[0]  # 最大值 [bs, target_size, num_sensors, 1]
        
        # 最大最小归一化
        #normalized_output = (output - min_val) / (max_val - min_val)
        return output


       