import torch, time

# dual-path model
def profile_memory(model, batch, device):
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(*batch)
    torch.cuda.synchronize(device)
    mem = torch.cuda.max_memory_allocated(device) / 1024**2
    return mem   # MiB
                               

# single-path model
def profile_memory_single(model, batch, device):
    # 重置 GPU 的内存统计
    torch.cuda.reset_peak_memory_stats(device)
    
    # 只传递 input_window 给模型，确保只传递一个参数
    batch = (batch[0].to(device),)  # batch[0] 应该是 input_window

    with torch.no_grad():
        _ = model(*batch)  # 传递单个参数给 forward()
    
    # 同步 GPU 操作，确保内存统计完整
    torch.cuda.synchronize(device)
    
    # 获取最大内存使用量（MiB）
    mem = torch.cuda.max_memory_allocated(device) / 1024**2
    return mem  # 返回内存使用量，以 MiB 为单位



# dual-path model
def profile_throughput(model, loader, device, warmup=10, measure=50):
    model.eval(); total=0
    with torch.no_grad():
        # warm-up
        for i,(inp, hist, _) in enumerate(loader):
            if i==warmup: break
            _ = model(inp.to(device), hist.to(device))
        torch.cuda.synchronize(device)
        tic = time.time()
        for i,(inp, hist, _) in enumerate(loader):
            if i==measure: break
            _ = model(inp.to(device), hist.to(device))
            total += inp.size(0)
        torch.cuda.synchronize(device)
    return total / (time.time() - tic)           # samples / second



# single-path model
def profile_throughput_single(model, loader, device, warmup=10, measure=50):
    model.eval()  # 设置模型为评估模式
    total = 0
    with torch.no_grad():
        # warm-up
        for i, (inp, hist, _) in enumerate(loader):
            if i == warmup: break
            _ = model(inp.to(device))  # 只传递 inp 参数
        torch.cuda.synchronize(device)
        
        # 开始计时并测量吞吐量
        tic = time.time()
        
        for i, (inp, hist, _) in enumerate(loader):
            if i == measure: break
            _ = model(inp.to(device))  # 只传递 inp 参数
            total += inp.size(0)
        
        torch.cuda.synchronize(device)
    
    # 返回吞吐量：每秒处理的样本数
    return total / (time.time() - tic)  # samples / second
