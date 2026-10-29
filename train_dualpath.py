import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.utils import *
from lib.evaluate import *
import numpy as np
import sys
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config_files.config_general import Config  # 假设配置文件中定义了Config类 
from model.DualGraphSTNet_dualshortpath import *  # 假设模型在此模块中定义
from modules import *
import logging
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import json
from profiling import profile_memory, profile_throughput
import time

# ==== 消融/变体开关(环境变量驱动, 不设则行为完全不变) ====
_DUALV4_NOKL = os.environ.get('DUALV4_NOKL', '') == '1'   # C10: 去KL图先验损失, loss=NLL only
_DUALV4_VAE   = os.environ.get('DUALV4_VAE', '') == '1'    # B6: VAE重构头, 损失加 β·隐变量KL
_DUALV4_JOINT = os.environ.get('DUALV4_JOINT', '') == '1'  # B7: 预测+重构联合, 同样加 β·隐变量KL
_VAE_BETA = 0.1

def set_random_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置CPU上的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置GPU上的随机种子
    torch.backends.cudnn.deterministic = True  # 保证每次返回的结果相同
    torch.backends.cudnn.benchmark = False  # 关闭CUDNN优化以保持结果可复现

# 设置随机种子
set_random_seed(int(os.environ.get("DUALV4_SEED", "42")))  # env-driven seed

# 训练过程
def train_epoch(model, train_loader, optimizer, device, config):

    model.train()

    # 计算批次平均误差
    total_loss = 0
    total_nll_loss = 0
    total_kl_loss = 0

    # 在训练开始前记录时间
    start_time = time.time()
    
    prior = config.prior
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    log_prior = log_prior.to(device)

    # 初始化用于存储批次 “预测、重构、KL散度” 误差的列表
    all_channel_error = []
    all_timestamp_error = []

    # 初始化梯度范数列表
    grad_norms = []
    entropy_log = []  # 用于存储每个批次的熵

    # 在训练开始前记录初始内存
    initial_memory = torch.cuda.memory_allocated()

    # print("length trainloader:", len(train_loader)) # length trainloader: 621
    for batch_idx, (input_window, previous_steps, target_window) in enumerate(train_loader):
        optimizer.zero_grad()
        input_window = input_window.to(device).requires_grad_(True)      # AMGSRML will see grads here # (B, T, N)
        previous_steps = previous_steps.to(device).requires_grad_(True)    # MSSGML will see grads here # (B, T, N)
        target_window = target_window.to(device) # (B, T, N)
        variance = 1

        # 前向传播
        
        # C = torch.cat((input_window, previous_steps), dim=1)  # 在 T 维度上拼接，得到 C 的形状为 (B, 2T, N)
        prob, out = model(input_window, previous_steps) # [B, N * (N-1), head]
        prob = prob.to(device)
        # print("prob device:", torch.mean(prob, 1))
        # print("log_prior device:", log_prior)
        # 计算预测损失
        loss_kl = kl_categorical(torch.mean(prob, 1).to(device), log_prior, 1).to(device)
        loss_nll, loss_nll_channel, loss_nll_timestamp, _ = nll_gaussian_with_intermediate(out, target_window, variance)
        loss = loss_nll if _DUALV4_NOKL else (loss_kl + loss_nll)
        if _DUALV4_VAE or _DUALV4_JOINT:
            loss = loss + _VAE_BETA * model._latent_kl
        # print("loss_nll_channel shape:", loss_nll_channel.shape) # torch.Size([128, 1, 51])
        
        # 计算训练时间
        end_time = time.time()
        epoch_time = end_time - start_time

        # 记录内存使用情况
        current_memory = torch.cuda.memory_allocated()
        max_memory = torch.cuda.max_memory_allocated()
        #print(f"Epoch Training Time: {epoch_time:.2f} seconds")
        #print(f"Max Memory Usage: {max_memory / 1024**2:.2f} MB")

        # === TOKEN-MIX ENTROPY (one-liner style) =========================
        def _entropy(grad_tensor, eps=1e-12):
            """
            grad_tensor: (B, T, N) or (B, T', N)   absolute grads summed over channels already
            returns     : (B,) Shannon entropy
            """
            B = grad_tensor.size(0)
            alpha = grad_tensor.abs().sum(dim=-1)      # → (B, T)   汇聚 N 维特征
            alpha = alpha.flatten(1)                   # → (B, T)   每条序列展开为 token 向量
            alpha = alpha / (alpha.sum(dim=1, keepdim=True) + eps)
            return -(alpha * (alpha + eps).log()).sum(dim=1)  # (B,)

        loss.backward()

        # 计算梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 计算批次平均误差
        total_loss += loss.item()
        total_nll_loss += loss_nll.item()
        total_kl_loss += loss_kl.item()

    
        # 将当前批次的误差追加到列表中
        all_channel_error.append(loss_nll_channel.detach().cpu().numpy())
        all_timestamp_error.append(loss_nll_timestamp.detach().cpu().numpy())

        '''
        if batch_idx % 50 == 0:
            logging.info(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}, Grad Norm: {total_norm:.4f}")
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}, Grad Norm: {total_norm:.4f}")        
        '''

    torch.cuda.empty_cache()

    # 计算批次平均误差
    avg_loss = total_loss / len(train_loader)
    avg_nll_loss = total_nll_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    #print("length all_channel_error:", len(all_channel_error))


    # 2) 抽取梯度
    '''
    H_win  = _entropy(input_window.grad)           # AMGSRML 熵
    entropy_log.append(H_win.mean().item())       # 用一个 list 把每批次的熵收起来
    '''
    H_win  = _entropy(input_window.grad)           # AMGSRML 熵
    H_hist = _entropy(previous_steps.grad)         # MSSGML 熵

    lam    = (config.num_features * config.window_size) / \
            (config.num_features * (config.window_size + previous_steps.size(1)))
    H_dual = lam * H_win + (1 - lam) * H_hist      # Dual-Path 总熵
    H_dual = H_win
    entropy_log.append(H_dual.mean().item())       # 用一个 list 把每批次的熵收起来

    return entropy_log, avg_loss, avg_nll_loss, avg_kl_loss, all_channel_error, all_timestamp_error, grad_norms, epoch_time, max_memory

# 验证过程
def validate_epoch(model, val_loader, config, device):
    model.eval()
    # 计算批次平均误差
    total_loss = 0
    total_nll_loss = 0
    total_kl_loss = 0
    
    prior = config.prior
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    log_prior = log_prior.to(device)
    
    # 初始化存储误差的列表
    all_channel_error = []
    all_timestamp_error = []
    
    with torch.no_grad():
        for batch_idx, (input_window, previous_steps, target_window) in enumerate (val_loader):
            input_window = input_window.to(device) # (B, T, N)
            previous_steps = previous_steps.to(device) # (B, T, N)
            target_window = target_window.to(device) # (B, T, N)
            variance = 1

            #C = torch.cat((input_window, previous_steps), dim=1) # 在 T 维度上拼接，得到 C 的形状为 (B, 2T, N)         
            # 前向传播，获取预测窗口和重构损失
            prob, out = model(input_window, previous_steps)
            prob = prob.to(device)
            #print("prob device:", prob.device)
            #print("log_prior device:", log_prior.device)
            #print("prob device:", prob.shape)
            #print("log_prior device:", log_prior.shape)

            # 计算预测损失
            loss_kl = kl_categorical(torch.mean(prob, 1), log_prior, 1).to(device)
            loss_nll, loss_nll_channel, loss_nll_timestamp, _ = nll_gaussian_with_intermediate(out, target_window, variance)
            loss = loss_nll if _DUALV4_NOKL else (loss_kl + loss_nll)
            if _DUALV4_VAE or _DUALV4_JOINT:
                loss = loss + _VAE_BETA * model._latent_kl

            # 计算批次平均误差
            total_loss += loss.item()
            total_nll_loss += loss_nll.item()
            total_kl_loss += loss_kl.item()
       
            # 将当前批次的误差追加到列表中
            all_channel_error.append(loss_nll_channel.detach().cpu().numpy())
            all_timestamp_error.append(loss_nll_timestamp.detach().cpu().numpy())

            '''
            if batch_idx % 50 == 0:
                logging.info(f"Validation Batch {batch_idx}/{len(val_loader)} - Loss: {loss.item():.4f}")
                print(f"Validation Batch {batch_idx}/{len(val_loader)} - Loss: {loss.item():.4f}")
            '''

    # 计算批次平均误差
    avg_loss = total_loss / len(val_loader)
    avg_nll_loss = total_nll_loss / len(val_loader)
    avg_kl_loss = total_kl_loss / len(val_loader)



    return avg_loss, avg_nll_loss, avg_kl_loss, all_channel_error, all_timestamp_error

# 主训练过程
config = Config()
os.makedirs(config.output_dir, exist_ok=True)
device = config.device
train_loader, val_loader = load_data_train(config.train_filename, config, config.device)
learning_rates = [config.learning_rate]

best_val_loss = float('inf')
best_lr = None
start_epoch = config.start_epoch
patience = 20  # 提前停止的耐心值


# 初始化记录每个 epoch 的训练和验证损失、学习率
train_losses = []
val_losses = []
lrs = []
grad_norms_epoch = []  # 记录每个 epoch 的平均梯度范数
entropy_log_epoch = []  # 记录每个 epoch 的熵

# 设置日志配置
os.makedirs(config.log_dir, exist_ok=True)
log_file_path = os.path.join(config.log_dir, "training.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir, exist_ok=True)


for lr in learning_rates:
    logging.info(f"Training with learning rate: {lr}")
    print(f"Training with learning rate: {lr}")
    
    # 初始化模型
    model = DualGraphSTNet(
        matrix_decay = config.matrix_decay,
        num_features=config.num_features,
        window_size=config.window_size,
        out_dim=config.out_dim,
        target_size=config.target_size,
        gru_dim=config.gru_dim,
        Graph_learner_n_hid = config.Graph_learner_n_hid,
        Graph_learner_n_head_dim=config.Graph_learner_n_head_dim,
        num_heads=config.num_heads,
        num_nodes=config.num_nodes,
        temperature=config.temperature,
        max_diffusion_step=config.max_diffusion_step, 
        num_rnn_layers=config.num_rnn_layers,
        device=config.device
    ).to(device)

    # 添加L2正则化（权重衰减）可以帮助缓解梯度爆炸问题，同时促进模型的泛化能力
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5  # 添加适当的权重衰减
    )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.8, 
        patience=5,
        min_lr=1e-6,
        verbose=True
    )

    if start_epoch > 0:
        params_filename = os.path.join(config.save_dir, f'epoch_{start_epoch}_lr_{lr:.6f}.params')
        model.load_state_dict(torch.load(params_filename))
        logging.info(f'Starting from epoch {start_epoch}, loaded weights from {params_filename}')
        print(f'Starting from epoch {start_epoch}, loaded weights from {params_filename}')
    
    # 提前停止计数器
    early_stop_counter = 0
    train_times = []
    memory_usages = []
    
    for epoch in range(start_epoch, config.epochs):
        # 使用当前的正样本权重训练
        entropy_log, train_avg_loss, train_avg_nll_loss, train_avg_kl_loss, all_channel_error, all_timestamp_error, grad_norms, epoch_time, max_memory = train_epoch(model, train_loader, optimizer, device, config)
        
        train_times.append(epoch_time)
        memory_usages.append(max_memory / 1024**2)  # 转换为MB

        # ----------- 保存为 JSON -----------
        json_path = os.path.join(config.output_dir, "train_times.json")
        with open(json_path, "w") as f:
            json.dump(train_times, f)
        print(f"Saved train_times to {json_path}")

        json_path = os.path.join(config.output_dir, "memory_usages.json")
        with open(json_path, "w") as f:
            json.dump(memory_usages, f)
        print(f"Saved memory_usages to {json_path}")

        # 计算每个 epoch 的平均梯度范数
        avg_grad_norm = np.mean(grad_norms)
        grad_norms_epoch.append(avg_grad_norm)

        # 记录每个 epoch 的熵
        entropy_log_epoch.append(np.mean(entropy_log))

        # 验证过程
        val_avg_loss, val_avg_nll_loss, val_avg_kl_loss, _, _ = validate_epoch(model, val_loader, config, device)
        
        params_filename = os.path.join(config.save_dir, f'epoch_{epoch}_lr_{optimizer.param_groups[0]["lr"]:.6f}.params')
        
        # 打印并记录当前 epoch 的损失信息
        log_message = (f"Epoch {epoch}/{config.epochs - 1} - "
                    f"Train Loss: {train_avg_loss:.4f}, Train Pred Loss: {train_avg_nll_loss:.4f}, "
                    f"Train KL Loss: {train_avg_kl_loss:.4f} - "
                    f"Val Loss: {val_avg_loss:.4f}, Val Pred Loss: {val_avg_nll_loss:.4f}, "
                    f"Val KL Loss: {val_avg_kl_loss:.4f}")

        logging.info(log_message)
        print(log_message)

        # 记录当前 epoch 的损失和学习率
        train_losses.append(train_avg_loss)
        val_losses.append(val_avg_loss)
        lrs.append(optimizer.param_groups[0]["lr"])  # 获取当前学习率并记录

        # 调整学习率
        scheduler.step(val_avg_loss)
        
        # 获取并打印当前的学习率
        current_lr = scheduler.get_last_lr()[0] # 获取当前学习率
        print(f"Epoch {epoch + 1}, Current Learning Rate: {current_lr:.6f}")



        # 保存最佳模型
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_lr = lr
            torch.save(model.state_dict(), params_filename)
            logging.info(f'Save parameters to file: {params_filename}')
            print(f'Save parameters to file: {params_filename}')
            
            channel_scores, global_scores = get_train_score(
                all_pred_per_channel=all_channel_error,
                epoch=epoch,
                lr=current_lr,
                config=config
            )
            
            early_stop_counter = 0  # 重置计数器
        else:
            early_stop_counter += 1  # 增加计数器
        
        # 提前停止条件
        if early_stop_counter >= patience:
            logging.info("Early stopping triggered.")
            print("Early stopping triggered.")
            break
          
        
# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
os.makedirs(config.output_dir, exist_ok=True)
pic_file_path = os.path.join(config.output_dir, "loss_curve.png")
plt.savefig(pic_file_path, dpi=300)
plt.close()

# 绘制学习率曲线
plt.figure(figsize=(10, 5))
plt.plot(range(len(lrs)), lrs, label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate over Epochs")
plt.legend()
pic_file_path = os.path.join(config.output_dir, "lr_curve.png")
plt.savefig(pic_file_path, dpi=300)
plt.close()

# 绘制梯度范数曲线
plt.figure(figsize=(10, 5))
plt.plot(range(len(grad_norms_epoch)), grad_norms_epoch, label="Average Gradient Norm")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Average Gradient Norm over Epochs")
plt.legend()
pic_file_path = os.path.join(config.output_dir, "grad_norm_curve.png")
plt.savefig(pic_file_path, dpi=300)
plt.close()

# ----------- 保存为 JSON -----------
json_path = os.path.join(config.output_dir, "entropy_log_epoch_dual.json")
with open(json_path, "w") as f:
    json.dump(entropy_log_epoch, f)
print(f"Saved entropy log to {json_path}")
# ----------- ② 绘图 -----------
'''
epochs = np.arange(1, len(entropy_log_epoch) + 1)

plt.figure(figsize=(6,4))
plt.plot(epochs, entropy_log_epoch, marker='o', label='Dual-Path')

# 如果有对照曲线，取消注释
# plt.plot(epochs, entropy_single, marker='s', linestyle='--', label='Single-Path')

plt.xlabel('Epoch')
plt.ylabel('Token-Mix Entropy')
plt.title('Training Dynamics of Token-Mix Entropy')
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
pic_file_path = os.path.join(config.output_dir, "entropy_curve.png")
plt.savefig(pic_file_path, dpi=300)
plt.close()
'''
# 取验证集的第 1 个 batch，当 “代表样本”
val_iter = iter(val_loader)
input_window, previous_steps, _ = next(val_iter)
batch = (input_window.to(device), previous_steps.to(device))
mem_mb = profile_memory(model, batch, device)
print(f"[PROFILE] peak mem = {mem_mb:.1f} MiB")
fps    = profile_throughput(model, val_loader, device)
print(f"[PROFILE] peak_mem={mem_mb:.1f} MiB | throughput={fps:.1f} samples/s")


logging.info(f"Best learning rate: {best_lr} with validation loss: {best_val_loss}")
print(f"Best learning rate: {best_lr} with validation loss: {best_val_loss}")
logging.info("Training finished!")
print("Training finished!")
