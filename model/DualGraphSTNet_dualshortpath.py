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

_DUALV4_SOFTGRAPH = os.environ.get('DUALV4_SOFTGRAPH', '') == '1'  # C4: Gumbel 软采样(hard=False)


class MLP(nn.Module):
    """Two-layer fully-connected ELU network with batch normalization."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        # fc1: maps each input vector of size n_in to a hidden vector of size n_hid
        # applied to the last dimension of the input tensor
        self.fc1 = nn.Linear(n_in, n_hid)    # transforms [..., n_in] -> [..., n_hid]
        # fc2: maps each hidden vector of size n_hid to an output vector of size n_out
        self.fc2 = nn.Linear(n_hid, n_out)   # transforms [..., n_hid] -> [..., n_out]
        # batch normalization over the output dimension n_out
        self.bn = nn.BatchNorm1d(n_out)
        # dropout probability for regularization
        self.dropout_prob = do_prob
        # initialize weights and biases
        self.init_weights()

    def init_weights(self):
        # Xavier normal initialization for Linear layers, and constant biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                # for batch norm: weight=1, bias=0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, input_data):
        # input_data shape: [batch_size, num_features, seq_len]
        # reshape to [batch_size*num_features, seq_len] for BatchNorm1d
        x = input_data.view(-1, input_data.size(-1))           # ([B*N, seq_len])
        x = self.bn(x)                                         # ([B*N, seq_len])
        # reshape back to [batch_size, num_features, seq_len]
        return x.view(input_data.size(0), input_data.size(1), -1)  # ([B, N, seq_len])

    def forward(self, input_data):
        # input_data shape: [batch_size, num_features, seq_len]
        # 1) First fully-connected layer + ELU activation
        x = self.fc1(input_data)                              # ([B, N, n_hid])
        x = F.elu(x)
        # 2) Dropout (training mode only)
        x = F.dropout(x, self.dropout_prob, training=self.training)  # ([B, N, n_hid])
        # 3) Second fully-connected layer + ELU activation
        x = self.fc2(x)                                       # ([B, N, n_out])
        x = F.elu(x)
        # 4) Batch normalization over the output channels
        x = self.batch_norm(x)                                # ([B, N, n_out])
        # output shape: [batch_size, num_features, n_out]
        return x


class Graph_learner(nn.Module):
    def __init__(self, n_in, n_hid, n_head_dim, head, num_nodes, temperature, device, do_prob=0.):
        super(Graph_learner, self).__init__()
        # store hyperparameters
        self.n_in = n_in               # input feature dimension for MLP
        self.n_hid = n_hid             # hidden dimension
        self.n_head_dim = n_head_dim   # dimension of each attention head
        self.head = head               # number of attention heads
        self.num_nodes = num_nodes     # number of graph nodes (features)
        self.temperature = temperature # temperature for Gumbel-softmax sampling
        self.device = device           # torch device (cpu or cuda)

        # MLP to process input sequences
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        # linear projections for queries and keys
        self.Wq = nn.Linear(n_hid, n_head_dim * head)
        self.Wk = nn.Linear(n_hid, n_head_dim * head)
        # initialize projection weights
        for m in [self.Wq, self.Wk]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs):
        # inputs shape: [B, seq_len, N]
        # permute to [B, N, seq_len] for MLP
        X = inputs.permute(0, 2, 1)                           # ([B, N, n_in])
        # pass through MLP: output shape [B, N, n_hid]
        X = self.mlp1(X)                                     # ([B, N, n_hid])

        # compute query and key projections
        Xq = self.Wq(X)                                      # ([B, N, head*n_head_dim])
        Xk = self.Wk(X)                                      # ([B, N, head*n_head_dim])
        B, N, _ = Xq.shape

        # reshape for multi-head: [B, N, head, n_head_dim]
        Xq = Xq.view(B, N, self.head, self.n_head_dim)       # ([B, N, head, d])
        Xk = Xk.view(B, N, self.head, self.n_head_dim)       # ([B, N, head, d])
        # permute to [B, head, N, n_head_dim]
        Xq = Xq.permute(0, 2, 1, 3)                          # ([B, head, N, d])
        Xk = Xk.permute(0, 2, 1, 3)                          # ([B, head, N, d])

        # 1) compute raw attention scores (dot product)
        # scores shape: [B, head, N, N]
        scores = torch.matmul(Xq, Xk.transpose(-1, -2))

        # mask to exclude self-connections (diagonal)
        mask_loc = torch.eye(self.num_nodes, dtype=bool, device=self.device)  # ([N, N])
        # select off-diagonal scores: shape [B, head, N*(N-1)]
        scores_offdiag = scores.masked_select(~mask_loc).view(B, self.head, -1)
        # rearrange for softmax: [B, N*(N-1), head]
        scores_offdiag = scores_offdiag.permute(0, 2, 1)    # ([B, N*(N-1), head])
        # softmax over heads
        prob = F.softmax(scores_offdiag, dim=-1)            # ([B, N*(N-1), head])

        # 2) sample edges via Gumbel-softmax (hard)
        edges = gumbel_softmax(torch.log(prob + 1e-5), tau=self.temperature, hard=not _DUALV4_SOFTGRAPH)  # C4: 软图时 hard=False  ([B, N*(N-1), head])

        # 3) build adjacency matrices: start with ones [head, B, N, N]
        adj_list = torch.ones(self.head, B, self.num_nodes, self.num_nodes, device=self.device)

        # mask for off-diagonal positions, expanded to [head, B, N, N]
        base_mask = (~torch.eye(self.num_nodes, dtype=bool, device=self.device)).unsqueeze(0).unsqueeze(0)  # ([1,1,N,N])
        mask = base_mask.expand(self.head, B, -1, -1)       # ([head, B, N, N])

        # assign sampled edges to adjacency
        # edges.permute(2,0,1) -> [head, B, N*(N-1)], then flatten to fill masked positions
        adj_list[mask] = edges.permute(2, 0, 1).flatten()   # ([head, B, N, N])

        # return:
        # prob shape: [B, N*(N-1), head]
        # adj_list shape: [head, B, N, N]
        return prob, adj_list


# The DCGRU unit can take as input the adjacency matrix produced by the preceding multi-head attention step and then perform spatio-temporal convolution over this graph structure to output feature representations for each sensor.
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

        x0_list, x1_list = [x0_0], [x1_0]   # C5修复: diffstep=0 时也有初始列表(否则 UnboundLocalError)
        if self._max_diffusion_step == 0:
            pass
        else:
            x0_1 = torch.matmul(adj_mx0, x0_0)
            x1_1 = torch.matmul(adj_mx1, x1_0)
            x0_list, x1_list = [x0_0, x0_1], [x1_0, x1_1]

            for k in range(2, self._max_diffusion_step + 1):
                x0_2 = torch.matmul(adj_mx0, x0_1)
                x1_2 = torch.matmul(adj_mx1, x1_1)
                # x0_0 = torch.cat([x0_0, x0_1], dim=0)
                # x1_0 = torch.cat([x1_0, x1_1], dim=0)
                x0_list.append(x0_1)
                x1_list.append(x1_1)
                x0_1 = x0_2
                x1_1 = x1_2
        num_matrices = self._max_diffusion_step + 1  # Adds for x itself.

        # x0_0 = x0_0.permute(1, 2, 3, 0)  # [3, 90, 128]
        # x1_0 = x1_0.permute(1, 2, 3, 0)  # [3, 90, 128]
        x0_0 = torch.cat(x0_list, dim=0).permute(1, 2, 3, 0)
        x1_0 = torch.cat(x1_list, dim=0).permute(1, 2, 3, 0)

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

# The encoder can be a single-layer or multi-layer DCGRU model, depending on the `num_rnn_layers` parameter specified during instantiation.
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
        
        # Depthwise convolution: applies a separate convolution kernel to each input channel
        self.depthwise = nn.Conv1d(input_channels, input_channels, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, groups=input_channels, bias=bias)
        
        # Pointwise convolution: applies a 1x1 convolution to the output of the depthwise convolution
        # to mix channel information
        self.pointwise = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
        # Batch normalization and activation function
        self.batchnorm = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, device, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        # Create a positional encoding matrix of shape (1, max_len, d_model)
        pe = torch.zeros(1, max_len, d_model).to(self.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(self.device)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(self.device)  # [d_model//2]

        # Apply sine to even indices
        pe[0, :, 0::2] = torch.sin(position * div_term)  # [1, max_len, d_model//2]
        # Apply cosine to odd indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # [1, max_len, d_model//2]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: Input tensor of shape [batch_size, T, N, out_dim]
        :return: Output tensor with positional encoding added, same shape [batch_size, T, N, out_dim]
        """
        # Get the time dimension length T from the input
        time_length = x.size(1)  # T
        # Extract positional encodings matching the current time dimension
        pe = self.pe[:, :time_length, :]  # Shape: [1, T, d_model]

        # Apply positional encoding along the time dimension only
        # x shape: [batch_size, T, N, out_dim]
        # Positional encoding is added only along the time axis, keeping N and out_dim unchanged
        x = x + pe.unsqueeze(2)  # Broadcast to [batch_size, T, N, out_dim]

        return self.dropout(x)



def Conv_GraphST(input, time_window_size, stride): 
    """
    Divide the input time series into multiple non-overlapping time windows.

    Args:
    - input: Input data of shape (batch_size, T, N, out_dim)
    - time_window_size: Size of each time window
    - stride: Sliding step size along the time axis

    Returns:
    - Windowed data of shape (batch_size, num_windows, N, time_window_size, out_dim)
    """
    # Get input dimensions
    batch_size, time_length, num_sensors, out_dim = input.shape
    
    # Calculate the number of valid sliding windows
    num_windows = (time_length - time_window_size) // stride + 1
    
    # Rearrange input for unfolding along the time axis
    input_unf = input.permute(0, 1, 3, 2)  # (batch_size, T, out_dim, N)
    
    # Apply unfold to create sliding windows along the time dimension
    input_windows = input_unf.unfold(dimension=1, size=time_window_size, step=stride)
    
    # Rearrange shape to (batch_size, num_windows, N, time_window_size, out_dim)
    input_windows = input_windows.permute(0, 1, 3, 4, 2)
    
    return input_windows


def window_pooling(input_windows, kernel_size, stride):
    """
    Perform average pooling on each time window.

    Args:
    - input_windows: Windowed data of shape (batch_size, num_windows, N, window_size, out_dim)

    Returns:
    - Pooled window features of shape (batch_size, num_windows, N, out_dim)
    """
    # Input shape: [B, num_windows, N, window_size, out_dim]
    batch_size, num_windows, N, window_size, out_dim = input_windows.shape

    # Reshape for 1D pooling: (B * num_windows * N, out_dim, window_size)
    input_reshaped = input_windows.reshape(batch_size * num_windows * N, out_dim, window_size)

    # Apply average pooling along the time window axis
    pool = nn.AvgPool1d(kernel_size, stride)
    pooled_windows = pool(input_reshaped)  # Output shape: [B * num_windows * N, out_dim, 1]

    # Reshape back to (batch_size, num_windows, N, out_dim)
    pooled_windows = pooled_windows.reshape(batch_size, num_windows, N, out_dim)
    
    return pooled_windows


class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim):
        """
        Module for learning graph structure based on window features.

        Args:
        - input_dim: Feature dimension of each time window (i.e., out_dim)
        """
        super(Dot_Graph_Construction_weights, self).__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, window_features):
        """
        Learn adjacency matrix based on window features.

        Args:
        - window_features: Tensor of shape (batch_size, num_windows, out_dim)

        Returns:
        - Adjacency matrix of shape (batch_size, num_windows, num_windows)
        """
        # Map features to a new space for similarity computation
        window_features = self.mapping(window_features)  # (batch_size, num_windows, out_dim)

        # Transpose for batch matrix multiplication
        window_features_transposed = window_features.transpose(1, 2)  # (batch_size, out_dim, num_windows)

        # Compute similarity matrix between windows
        Adj = torch.bmm(window_features, window_features_transposed)  # (batch_size, num_windows, num_windows)

        # Create mask to remove self-connections (diagonal entries)
        num_windows = window_features.size(1)
        eyes_like = torch.eye(num_windows).repeat(window_features.size(0), 1, 1).to(window_features.device)  # (batch_size, num_windows, num_windows)
        eyes_like_inf = eyes_like * 1e8  # Large mask value to suppress diagonal
        Adj = F.leaky_relu(Adj - eyes_like_inf)  # Apply activation and remove self-connections

        # Row-wise softmax normalization
        Adj = F.softmax(Adj, dim=-1)

        # Restore self-connections by adding identity matrix
        Adj = Adj + eyes_like
        
        return Adj  # Shape: (batch_size, num_windows, num_windows)



def Mask_Matrix(num_node, time_length, decay_rate, device="cuda"):
    # Adj = tr.ones(num_node * time_length, num_node * time_length).cuda()
    Adj = tr.full((num_node * time_length, num_node * time_length),
                   1., device=device, dtype=tr.float32)
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
        Customized graph convolution diffusion network supporting multi-scale receptive fields.

        Args:
        - input_dim: Dimension of input features
        - output_dim: Dimension of output features
        - k: Size of the receptive field (number of propagation steps)
        - way_multi_field: 'cat' or 'sum', determines whether to concatenate or sum multi-scale outputs
        """
        super(MPNN_mk_v2, self).__init__()
        self.k = k  # Number of graph convolution layers (propagation steps)
        self.way_multi_field = way_multi_field  # 'cat' for concatenation, 'sum' for summation

        # Create k linear transformation layers
        self.theta = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(self.k)])

        # Graph normalization module
        self.graph_norm = GraphNormalization(output_dim)

    def forward(self, X, A):
        """
        Forward pass.

        Args:
        - X: Input feature matrix, shape (batch_size, num_windows, input_dim)
        - A: Adjacency matrix, shape (batch_size, num_windows, num_windows)

        Returns:
        - GCN_output: Output feature matrix, shape (batch_size, num_windows, output_dim)
        """
        batch_size, num_windows, input_dim = X.size()
        
        GCN_output_ = []

        # Graph convolution diffusion process
        for kk in range(self.k):
            if kk == 0:
                A_ = A  # Use original adjacency matrix for the first step
            else:
                A_ = tr.bmm(A_, A)  # Multiply adjacency matrix with itself to expand the receptive field
            
            # Perform graph convolution (A * X) followed by a linear transformation
            out_k = self.theta[kk](tr.bmm(A_, X))  # Shape: (batch_size, num_windows, output_dim)
            GCN_output_.append(out_k)

        # Merge multi-scale outputs either by concatenation or summation
        if self.way_multi_field == 'cat':
            GCN_output_ = tr.cat(GCN_output_, dim=-1)  # Concatenate along feature dimension
        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)  # Sum across k steps

        # Apply graph normalization
        GCN_output_ = self.graph_norm(tr.transpose(GCN_output_, 1, 2))  # Transpose to (batch_size, output_dim, num_windows)
        GCN_output_ = tr.transpose(GCN_output_, 1, 2)  # Transpose back to (batch_size, num_windows, output_dim)

        return F.leaky_relu(GCN_output_)  # Apply non-linear activation



class AttentionFusion(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(AttentionFusion, self).__init__()
        
        # Layer normalization layers
        self.layer_norm1 = nn.LayerNorm(input_dim)  # Normalize historical information
        self.layer_norm2 = nn.LayerNorm(input_dim)  # Normalize target information
        
        # Learnable attention weight projections
        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)
        
        # Output projection layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Save hidden dimension
        self.hidden_dim = hidden_dim

    def forward(self, tensor1, tensor2):
        # tensor1: (B, num_windows, N, input_dim) --> historical features
        # tensor2: (B, target_size, N, input_dim) --> target features to be predicted
        
        # Apply Layer Normalization
        tensor1 = self.layer_norm1(tensor1)  # Normalize historical information
        tensor2 = self.layer_norm2(tensor2)  # Normalize target information

        # Get shape of input tensors
        bs, num_windows, num_sensors, input_dim = tensor1.size()
        _, target_size, _, _ = tensor2.size()
        
        # Initialize output list
        output_list = []

        # Compute attention step by step for each target timestep
        for t in range(target_size):
            query = self.query_linear(tensor2[:, t, :, :])  # Shape: (B, N, hidden_dim)
            query = query.unsqueeze(2)  # Add a temporal dimension: (B, N, 1, hidden_dim)
            query = query.permute(0, 2, 1, 3)  # Reshape to (B, 1, N, hidden_dim)

            # Keys and values span the full historical time window
            key = self.key_linear(tensor1)    # (B, num_windows, N, hidden_dim)
            value = self.value_linear(tensor1)  # (B, num_windows, N, hidden_dim)

            # Compute attention scores using dot product
            attn_scores = torch.einsum('bqnh,bknh->bqnk', query, key)  # (B, 1, N, num_windows)
            attn_scores = F.softmax(attn_scores, dim=3)  # Softmax over time dimension

            # Weighted sum of values
            attn_output = torch.einsum('bqnk,bknh->bqnh', attn_scores, value)  # (B, 1, N, hidden_dim)
            
            # Append current timestep's output
            output_list.append(attn_output)

        # Concatenate attention outputs for all target timesteps
        attn_output = torch.stack(output_list, dim=1).squeeze(2)  # (B, target_size, N, hidden_dim)

        # Final linear projection to output dimension
        output = self.fc(attn_output)  # (B, target_size, N, output_dim)

        return output


class VAEBottleneck(nn.Module):
    """B6 检测头: 对融合输出做变分瓶颈(编码到隐高斯→重参数化→解码重构)。
    在最后一维(=N 特征维)上操作。返回 (重构, 隐变量KL标量)。"""
    def __init__(self, dim):
        super(VAEBottleneck, self).__init__()
        self.enc = nn.Linear(dim, dim)
        self.fc_mu = nn.Linear(dim, dim)
        self.fc_logvar = nn.Linear(dim, dim)
        self.dec = nn.Linear(dim, dim)

    def forward(self, x):
        h = F.relu(self.enc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = tr.exp(0.5 * logvar)
        z = mu + std * tr.randn_like(std)
        recon = self.dec(z)
        kl = -0.5 * tr.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, kl


class ModelWithLinearMapping(nn.Module):
    def __init__(self, target_size, input_dim, output_dim):
        super(ModelWithLinearMapping, self).__init__()
        
        # 定义线性层，分别映射 instant 和 history 部分的时间维度
        self.linear_instant = nn.LazyLinear(target_size)  # 对时间维度5映射到 target_size
        self.linear_history = nn.LazyLinear(target_size)  # 对时间维度25映射到 target_size
        
        self.target_size = target_size
        
    def forward(self, output1, output2):
        # output1: [B, target_size, N]
        # output2: [B, windows_num, N]
        instant = output1  # [B, target_size, N]
        history = output2  # [B, windows_num, N]
        # 2. 交换 instant 和 history 的维度，使时间维度成为最后一维
        instant = instant.permute(0, 2, 1)  # instant: [B, N, 5]
        history = history.permute(0, 2, 1)  # history: [B, N, 4]
        #print("Instant shape:", instant.shape)  # [B, 5, N]
        #print("History shape:", history.shape)  # [B, 4, N]
        # 3. 使用线性层将时间维度映射到 target_size
        instant_mapped = self.linear_instant(instant)  # [B, N, 5] -> [B, N, target_size]
        history_mapped = self.linear_history(history)  # [B, N, 25] -> [B, N, target_size]
        
        # 4. 恢复原始维度顺序（将时间维度恢复到第二维）
        instant_mapped = instant_mapped.permute(0, 2, 1)  # [B, target_size, N]
        history_mapped = history_mapped.permute(0, 2, 1)  # [B, target_size, N]
        
        # 5. 对两个映射后的向量进行加和
        fused_output = instant_mapped + history_mapped
        
        return fused_output

# Original model
class DualGraphSTNet(nn.Module):
    def __init__(self, matrix_decay, num_features, window_size, target_size, out_dim, gru_dim,
                 Graph_learner_n_hid, Graph_learner_n_head_dim, num_heads, num_nodes, temperature,
                 max_diffusion_step, num_rnn_layers, device):
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

        # First-stage input projection
        self.linear1 = nn.Linear(1, gru_dim)

        # Temporal graph structure learner (learns graph based on time dimension)
        self.graph_learner = Graph_learner(
            window_size,
            Graph_learner_n_hid,
            Graph_learner_n_head_dim,
            num_heads,
            num_nodes,
            temperature,
            device
        )

        # Multi-head spatio-temporal encoder (each head processes the graph separately)
        self.encoder_model = nn.ModuleList([
            EncoderModel(
                self.device,
                gru_dim,
                gru_dim,
                max_diffusion_step,
                num_nodes,
                num_rnn_layers
            ) for _ in range(self.num_heads)
        ])

        # Positional encoding for the time dimension
        self.positional_encoding = PositionalEncoding(device, gru_dim)

        # Windowed temporal convolution module
        self.conv_graph_st = Conv_GraphST

        # Window-level average pooling module
        self.window_pooling = window_pooling

        # Constructs attention-based graph over time windows
        self.dot_graph_weights = Dot_Graph_Construction_weights(out_dim)
        # C8 消融(环境变量驱动): 去学习窗图, 用全连接(ones)替代学到的相似度图
        self.no_dotgraph = os.environ.get('DUALV4_NODOTGRAPH', '') == '1'
        # B6 检测头: 融合输出后加 VAE 变分瓶颈重构 target;  B7: 预测+重构联合输出
        self.vae = os.environ.get('DUALV4_VAE', '') == '1'
        self.joint = os.environ.get('DUALV4_JOINT', '') == '1'
        if self.vae or self.joint:
            self.vae_head = VAEBottleneck(num_features)
        self._latent_kl = None

        # Mask matrix for temporal decay
        self.mask_matrix = Mask_Matrix
        decay_mask = Mask_Matrix(num_nodes, window_size, matrix_decay).cpu()
        self.register_buffer("decay_mask", decay_mask, persistent=False)

        # Graph convolution with normalization
        self.BN = nn.BatchNorm1d(out_dim)
        self.mpnn = MPNN_mk_v2(out_dim, out_dim, k=2)

        # Feature fusion across sensors via attention
        self.attentionFusion = AttentionFusion(out_dim, 2 * out_dim, out_dim)

        # GRU for temporal decoding
        self.gru = nn.GRU(input_size=out_dim, hidden_size=128, batch_first=True)

        # Final output projection to predict target_size × out_dim
        self.fc = nn.Linear(in_features=128, out_features=target_size * out_dim)
        self.simplefusion = ModelWithLinearMapping(target_size, out_dim, num_features)
        
    def encoder(self, inputs, adj, head):
        """
        Encoder forward pass
        inputs: [batch_size, num_nodes, feature_dim, len_sequence]
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
        # Process input_window (first layer) [batch_size, window_size, features]
        input_window = input_window.permute(0, 2, 1)  # (num_samples, N=features, T=window_size)
        input_projected = self.linear1(input_window.unsqueeze(-1))  # [B, N, T, gru_dim]
        input_projected = input_projected.permute(0, 1, 3, 2)  # [B, N, gru_dim, T]

        # Check shape torch.Size([256, 51, 32, 30])
        # print("Shape of input_projected:", input_projected.shape)

        prob, adj_list = self.graph_learner(input_window.permute(0, 2, 1))  # [B, N * (N-1), head], [head, B, N, N]
        # Check shape torch.Size([4, 256, 51, 51])
        # print("Shape of adj_list:", adj_list.shape)

        # Multi-head attention + Encoder
        state_for_output = torch.zeros(input_projected.shape).to(self.device)  # [B, N, gru_dim, T]
        # print("Shape of state_for_output [B, N, 64, T]:", state_for_output.shape)
        # state_for_output = (state_for_output.unsqueeze(0)).repeat(self.num_heads - 1, 1, 1, 1, 1) # [1, B, N, gru_dim, T] --> [self.num_heads - 1, B, N, gru_dim, T]
        state_for_output = (
            state_for_output
            .unsqueeze(0)                       # [1, B, N, D, T]
            .expand(self.num_heads - 1, -1, -1, -1, -1)   # [H-1, B, N, D, T]
        )
        # print("Shape of state_for_output [3, B, N, 64, T]:", state_for_output.shape)
        for head in range(self.num_heads - 1):
            state_for_output[head, ...] = self.encoder(input_projected, adj_list[head + 1, ...], head)  # [H-1, B, N, gru_dim, T]
        state_for_output2 = torch.mean(state_for_output, 0).permute(0, 1, 3, 2)  # [B, N, T, gru_dim]
        # print("Shape of state_for_output2 [B, N, T, 64]:", state_for_output2.shape)
        output1 = state_for_output2[:, :, -self.target_size:, :]  # [B, N, target_size, gru_dim]

        # ************Check output shape torch.Size([256, 51, 30])*********************
        # print("Shape of output1 [B, N, target_size, gru_dim]:", output1.shape)

        # Process previous_steps (second layer)
        previous_steps = self.linear1(previous_steps.unsqueeze(-1))  # [batch_size, T, N, gru_dim]
        prev_features_encoded = self.positional_encoding(previous_steps)  # [batch_size, T, N, gru_dim]
        windowed_data = self.conv_graph_st(prev_features_encoded, self.window_size, stride=self.window_size)  # [B, num_windows, N, window_size, out_dim]
        # print("Shape of time-window split [B, num_windows, N, window_size, out_dim]:", windowed_data.shape)
        pooled_data = self.window_pooling(windowed_data, self.window_size, self.window_size)  # [B, num_windows, N, out_dim]
        # print("Shape of pooling [B, num_windows, N, out_dim]:", pooled_data.shape)
        batch_size, num_windows, N, out_dim = pooled_data.shape
        pooled_data_ = pooled_data.view(batch_size, num_windows * N, out_dim)  # [B, num_windows * N, out_dim]
        if self.no_dotgraph:
            # C8: 不学习窗图, 用全连接矩阵(结构仅由后面的 decay_matrix 提供)
            adj_matrix = torch.ones(batch_size, num_windows * N, num_windows * N, device=self.device)
        else:
            adj_matrix = self.dot_graph_weights(pooled_data_)  # [B, num_windows * N, num_windows * N]
        # print("Shape of adjacency matrix [B, num_windows * N, num_windows * N]:", adj_matrix.shape)
        decay_matrix = self.mask_matrix(num_windows, N, decay_rate=self.matrix_decay, device=self.device)  # [num_windows * N, num_windows * N]
        # print("Shape of decay matrix [num_windows * N, num_windows * N]:", decay_matrix.shape)
        final_adj_matrix = adj_matrix * decay_matrix  # [batch_size, num_windows * N, num_windows * N]
        pooled_data_ = self.BN(pooled_data_.permute(0, 2, 1)).permute(0, 2, 1)  # [B, num_windows * N, out_dim]
        # print("Shape of mpnn input (normalized) [B, num_windows * N, out_dim]:", pooled_data_.shape)
        output2 = self.mpnn(pooled_data_, final_adj_matrix)  # [B, num_windows * N, out_dim]

        # Check second-layer output torch.Size([256, 9, 64])
        # print("Shape of output2 [B, num_windows * N, out_dim]:", output2.shape)

        # Feature fusion
        # Reshape to [B, num_windows, N, out_dim]
        tensor1 = output2.view(batch_size, num_windows, N, -1).mean(dim=-1)  # [B, num_windows, N]
        tensor2 = output1.permute(0, 2, 1, 3).mean(dim=-1)  # [B, target_size, N]
        fused_output = self.simplefusion(tensor2, tensor1)  # [B, target_size, N]

        # B6: 用VAE重构替换输出;  B7: 输出=0.5*(预测+重构)联合;  隐变量KL存 self._latent_kl 供训练
        if self.joint:
            recon, self._latent_kl = self.vae_head(fused_output)
            fused_output = 0.5 * (fused_output + recon)
        elif self.vae:
            fused_output, self._latent_kl = self.vae_head(fused_output)
        else:
            self._latent_kl = fused_output.new_zeros(())

        return prob, fused_output



       