import numpy as np
import os
_M = os.environ.get("DUALV4_SMD_MACHINE", "machine-1-1")
class Config(object):
    def __init__(self):
        self.device = "cuda:0"
        self.batch_size = 128
        self.T = 7499
        self.learning_rate = 0.001
        self.epochs = 200
        self.train_filename = f"data/SMD/{_M}/train.npy"
        self.test_filename_data = f"data/SMD/{_M}/test.npy"
        self.anomaly_file = f"data/SMD/{_M}/label.csv"
        self.val_split = 0.2
        self.downsampling_fre = 1
        self.num_features = 38
        self.window_size = 30
        self.step_size = 5
        self.target_size = 5
        self.previous_steps_length = 300
        self.out_dim = 64
        self.gru_dim = 64
        self.Graph_learner_n_hid = 64
        self.Graph_learner_n_head_dim = 32
        self.matrix_decay = 0.65
        self.num_nodes = 38
        self.num_heads = 3
        self.temperature = 0.5
        self.prior = np.array([0.9, 0.05, 0.05])
        self.max_diffusion_step = 1
        self.num_rnn_layers = 1
        self.EncoderModel_n_hid = 32
        self.save_dir = f"experiments/SMD/{_M}"
        self.output_dir = f"pics/SMD/{_M}"
        self.log_dir = f"logs/SMD/{_M}"
        self.score_dir = f"output/SMD/{_M}"
        self.results_dir = f"results/SMD/{_M}"
        self.start_epoch = 0
        self.model_save_path = f"experiments/SMD/{_M}/epoch_0_lr_0.001000.params"
        self.save_result = True
        self.save_interval = 10
        self.scores_save_path = f"output/SMD/{_M}/anomaly_scores.pkl"
