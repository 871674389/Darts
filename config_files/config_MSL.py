import numpy as np

class Config(object):
    def __init__(self):
        # 数据配置
        self.device = 'cuda:0'
        self.batch_size = 64
        self.T = 7499
        self.learning_rate = 0.001
        self.epochs = 200
        self.train_filename = 'data/MSL/MSL_train_noisy.npy' # (496800, 51) --> (8280, 51) 1min=时间戳
        self.test_filename_data = 'data/MSL/MSL_test.npy' # (449919, 51) --> (7499, 51)
        # self.test_filename_labels = 'data/SWAT/labels.npy'
        self.anomaly_file = 'data/MSL/MSL_test_anomaly_intervals.csv'
        # 'data/SWAT/labels.npy' # (449919, 51)

        # 数据加载器配置
        self.val_split = 0.2
        self.downsampling_fre = 1
        '''
        数据样本总数: 79486 个
        总批次数: 621 个
        每个完整批次的样本数: 128 个
        最后一个批次的样本数: 126 个
        ----------------------------
        总样本数: 397426 个
        总批次数: 3105 个
        全批次数: 3104 批次，每批 128 个样本
        最后一个批次: 114 个样本
        '''
        
        # 模型配置
        self.num_features = 55  # 输入特征的数量 (传感器数量)
        self.window_size = 30  # 时间窗口大小
        self.step_size = 5
        self.target_size = 5 # 单步预测vs多步预测
        '''
        为了满足预测窗口之间是连续时间戳
        需要保证“步长=预测窗口”
        window_size = 300秒（5分钟） 或 150秒（2.5分钟）
        step_size = 150秒（窗口大小的一半）
        '''
        self.previous_steps_length = 300  # 不包含input_window 之前的时间步长度
        self.out_dim = 64  # 输出维度
        self.gru_dim = 64  # GRU 隐藏层维度
        self.Graph_learner_n_hid = 64
        self.Graph_learner_n_head_dim = 32

        self.matrix_decay = 0.8
    
        self.num_nodes = 55  # 节点数量 (传感器数量)
        self.num_heads = 3  # 多头注意力的头数
        self.temperature = 0.5
        self.prior = np.array([0.9, 0.05, 0.05])

        # 其他模型相关配置
        self.max_diffusion_step = 1  # 图卷积扩散步数
        self.num_rnn_layers = 1  # RNN 层数
        self.EncoderModel_n_hid = 32

        # 预测模块重构模块
        # self.pre_n_layers = 2

        # 异常分数计算的参数：
        # self.kernel_sigma = 3   # 5秒到10秒 如果数据波动较大或噪声较多，可以选择较大的值。如果数据波动较小，可以选择较小的值。
        # self.long_window = 3600   # 600秒（10分钟） 到 3600秒（1小时）
        # self.short_window = 180   # 180秒（3分钟） 到 300秒（5分钟）
        # self.constant_std = 1e-6   # 这个值通常足够小，防止除零错误

        # 训练保存配置
        self.save_dir = 'experiments/MSL_noise'
        self.output_dir = 'pics/MSL'
        self.log_dir = 'logs/MSL'
        self.score_dir = 'output/MSL_noise'
        self.results_dir = 'results/MSL_noise'
        self.start_epoch = 0
        self.model_save_path = "experiments/MSL_noise/epoch_89_lr_0.000328.params"
        # epoch_15_lr_0.001000.params epoch_15_lr_0.001
        # epoch_31_lr_0.003000.params epoch_31_lr_0.003
        self.save_result = True
        self.save_interval = 10  # 模型保存间隔 (每 n 个 epoch 保存一次)
        self.scores_save_path = "output/MSL_noise/epoch_89_lr_0.0003276800000000001/anomaly_scores.pkl"




