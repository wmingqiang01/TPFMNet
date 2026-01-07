import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import datetime
from config import configs
# from utils.utils import SIC_dataset
from utils.utils_2 import SIC_dataset
from utils.metrics import *
from utils.tools import setup_logging
from models.ConvNeXt_uncertain import ConvNext 
from models.TPFMNet import TPFMNet 
from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.metrics import r2_score

# --- 数据集和数据加载器设置 ---
dataset_train = SIC_dataset(
    configs.full_data_path,
    configs.train_period[0],
    configs.train_period[1],
    configs.input_gap,
    configs.input_length,
    configs.pred_shift,
    configs.pred_gap,
    configs.pred_length,
    samples_gap=1,
)

dataset_vali = SIC_dataset(
    configs.full_data_path,
    configs.eval_period[0],
    configs.eval_period[1],
    configs.input_gap,
    configs.input_length,
    configs.pred_shift,
    configs.pred_gap,
    configs.pred_length,
    samples_gap=1,
)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=configs.batch_size,
    shuffle=True,
    num_workers=configs.num_workers,
    persistent_workers=True,
    prefetch_factor=32,
)

dataloader_vali = DataLoader(
    dataset_vali,
    batch_size=configs.batch_size_vali,
    shuffle=False,
    num_workers=configs.num_workers,
)

# 添加测试数据集和数据加载器
dataset_test = SIC_dataset(
    configs.full_data_path,
    configs.eval_period[0],
    configs.eval_period[1],
    configs.input_gap,
    configs.input_length,
    configs.pred_shift,
    configs.pred_gap,
    configs.pred_length,
    samples_gap=1,
)

dataloader_test = DataLoader(
    dataset_test,
    batch_size=configs.batch_size_vali,
    shuffle=False,
    num_workers=configs.num_workers,
)

# --- 指标计算函数（已修改为仅计算海洋部分） ---
def calculate_metrics(mu, targets, dataset, land_mask, device, prefix=""):
    mu_np = mu[:, :, 0, :, :].detach().cpu().numpy()
    targets_np = targets[:, :, 0, :, :].detach().cpu().numpy()
    land_mask_np = land_mask.cpu().numpy()

    pred_flat = mu_np.flatten()
    target_flat = targets_np.flatten()

    nan_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))

    num_batches = mu_np.shape[0]
    num_times = mu_np.shape[1]
    # land_mask=True 假设为海洋区域
    land_mask_flat = np.tile(land_mask_np.flatten(), num_batches * num_times)
    ocean_mask = land_mask_flat == True 

    valid_mask = nan_mask & ocean_mask
    pred_valid = pred_flat[valid_mask]
    target_valid = target_flat[valid_mask]

    if len(pred_valid) == 0:
        metrics = {
            f"{prefix}loss": np.nan,
            f"{prefix}mae_sic": np.nan,
            f"{prefix}rmse_sic": np.nan,
            f"{prefix}R^2_sic": np.nan,
        }
        # 返回一个 nan tensor 避免计算错误
        return metrics, torch.tensor(np.nan) 

    # 计算 loss 时不进行反归一化，使用MSE损失
    mse_loss = np.mean((pred_valid - target_valid) ** 2)

    # SIC大于等于0.15的才算
    # 注意：这里直接在 pred_valid 和 target_valid 上应用了 > 0.15 过滤，这可能会改变它们的维度。
    # 为了保持一致性，我们应该对整个有效集应用过滤，或仅对计算 MAE/RMSE 时进行过滤。
    # 按照代码逻辑，我们对 pred/target 进行 SIC 阈值过滤后再计算 MAE/RMSE
    pred_sic = pred_valid * (pred_valid >= 0.15)
    targets_sic = target_valid * (target_valid >= 0.15)

    # 重新计算有效 SIC 上的 MAE/RMSE
    mae = np.mean(np.abs(pred_sic - targets_sic))
    rmse = np.sqrt(np.mean((pred_sic - targets_sic)**2))
    r_squared = r2_score(targets_sic, pred_sic) if len(targets_sic) > 0 else np.nan

    metrics = {
        f"{prefix}loss": mse_loss,
        f"{prefix}mae_sic": mae,
        f"{prefix}rmse_sic": rmse,
        f"{prefix}R^2_sic": r_squared,
    }
    return metrics, torch.tensor(mse_loss)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # 创建日志目录
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    log_dir = os.path.join(configs.train_log_path, f"ConvNeXt_uncertain_{configs.input_length}_{configs.pred_length}")
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join("pretrained_models")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(os.path.join(log_dir, "train.log"))
    
    # 初始化模型
    # model = TPFMNet(T=configs.input_length, C=configs.input_dim, uncertainty_type='laplacian').to(device)
    model = TPFMNet(T=configs.input_length, C=configs.input_dim, uncertainty_type='gaussian').to(device)
    land_mask_np = np.load(configs.mask_path)
    land_mask = torch.from_numpy(land_mask_np).to(device)
    
    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=configs.lr,
        epochs=configs.num_epochs,
        steps_per_epoch=len(dataloader_train),
    )
    
    # ********************* 修改点 1：使用复合分数初始化 *********************
    best_composite_score = float('inf')
    patience_counter = 0
    
    # 训练开始时间
    start_time = time.time()
    total_batches = len(dataloader_train)
    
    # 打印训练信息
    logger.info(f"开始训练 TPFMNet_uncertain 模型")
    logger.info(f"训练集样本数: {len(dataset_train)}, 验证集样本数: {len(dataset_vali)}, 测试集样本数: {len(dataset_test)}")
    logger.info(f"批次大小: {configs.batch_size}, 验证/测试批次大小: {configs.batch_size_vali}")
    logger.info(f"学习率: {configs.lr}, 最大训练轮数: {configs.num_epochs}, 早停耐心值: {configs.patience}")
    logger.info(f"输入序列长度: {configs.input_length}, 预测序列长度: {configs.pred_length}")
    logger.info(f"设备: {device}")
    
    # 获取模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    # 训练循环
    for epoch in range(configs.num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_losses = []
        train_pbar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{configs.num_epochs} [Train]", leave=False)
          
        batch_times = []
        for batch_idx, batch in enumerate(train_pbar):
            batch_start = time.time()
            inputs, targets = batch
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            
            # 对输入数据应用mask
            ocean_mask = land_mask.float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            inputs = inputs * ocean_mask
            
            optimizer.zero_grad()
            # 假设 model(inputs, targets, is_training=True) 返回 (mu, sigma), total_loss (NLL)
            (mu, sigma), total_loss = model(inputs, targets, is_training=True) 
            
            metrics, _ = calculate_metrics(mu, targets, dataset_train, land_mask, device)
            total_loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            if configs.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), configs.clipping_threshold)
            
            optimizer.step()
            scheduler.step()
            
            # 记录当前批次的损失和学习率
            current_lr = scheduler.get_last_lr()[0]
            train_losses.append(total_loss.item())
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}", 
                'mae_sic':f"{metrics['mae_sic']:.4f}",
                'rmse_sic':f"{metrics['rmse_sic']:.4f}",
                'R^2_sic':f"{metrics['R^2_sic']:.4f}"
            })
        
        # 验证阶段
        model.eval()
        val_metrics_sum = {"val_loss": 0, "val_mae_sic": 0, "val_rmse_sic": 0, "val_R^2_sic": 0}
        val_count = 0
        val_pbar = tqdm(dataloader_vali, desc=f"Epoch {epoch+1}/{configs.num_epochs} [Validation]", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                inputs, targets = batch
                inputs, targets = inputs.to(device).float(), targets.to(device).float()
                
                # 对输入数据应用mask
                ocean_mask = land_mask.float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
                inputs = inputs * ocean_mask
                
                (mu, sigma), total_loss = model(inputs, targets, is_training=False)
                metrics, _ = calculate_metrics(mu, targets, dataset_vali, land_mask, device, prefix="val_")
                
                # 累加指标，处理可能的 nan
                for k, v in metrics.items():
                    if not np.isnan(v):
                        val_metrics_sum[k] += v
                val_count += 1
                
                # 更新验证进度条
                val_pbar.set_postfix({'val_loss': f"{metrics['val_loss']:.4f}"})
        
        # 计算平均验证指标
        val_metrics = {k: v / val_count if val_count > 0 else np.nan for k, v in val_metrics_sum.items()}
        
        # ********************* 修改点 2：计算复合分数 *********************
        # 简单相加三个要最小化的指标作为复合分数
        current_composite_score = (
            val_metrics["val_loss"] + 
            val_metrics["val_mae_sic"] + 
            val_metrics["val_rmse_sic"]
        )
        
        # 计算本轮训练时间
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        
        # 记录日志
        train_loss = sum(train_losses) / len(train_losses)
        logger.info(f"Epoch {epoch+1}/{configs.num_epochs} - 用时: {epoch_time:.2f}s - 平均批次时间: {avg_batch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.6f} - "
                            f"Val Loss: {val_metrics['val_loss']:.6f} - "
                            f"Val MAE: {val_metrics['val_mae_sic']:.6f} - "
                            f"Val RMSE: {val_metrics['val_rmse_sic']:.6f} - "
                            f"Val R^2: {val_metrics['val_R^2_sic']:.6f} - "
                            f"Composite Score: {current_composite_score:.6f}") # 打印复合分数
        
        # ********************* 修改点 3：使用复合分数保存模型 *********************
        if current_composite_score < best_composite_score:
            best_composite_score = current_composite_score
            patience_counter = 0
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'composite_score': best_composite_score, # 记录复合分数
                },
                os.path.join(checkpoint_dir, f"TPFMNet_{configs.input_length}_{configs.pred_length}_gaussian.pth")
            )
            logger.info(f"Saved best model with Composite Score: {best_composite_score:.6f}")
        else:
            patience_counter += 1
            
        # 早停
        if configs.early_stopping and patience_counter >= configs.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 计算总训练时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"训练完成! 总用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    logger.info(f"最佳验证复合分数: {best_composite_score:.6f}")
    
    # 打印最终模型路径
    final_model_path = os.path.join(checkpoint_dir, f"TPFMNet_{configs.input_length}_{configs.pred_length}.pth")
    logger.info(f"最佳模型保存路径: {final_model_path}")
    
    # --- 测试阶段 ---
    logger.info("开始计算测试集指标")
    
    # 加载最佳模型
    checkpoint = torch.load(final_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_metrics_sum = {"test_loss": 0, "test_mae_sic": 0, "test_rmse_sic": 0, "test_R^2_sic": 0}
    test_count = 0
    test_pbar = tqdm(dataloader_test, desc="Test", leave=False)
    
    with torch.no_grad():
        for batch in test_pbar:
            inputs, targets = batch
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            
            # 对输入数据应用mask
            ocean_mask = land_mask.float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
            inputs = inputs * ocean_mask
            
            (mu, sigma), total_loss = model(inputs, targets, is_training=False)
            metrics, _ = calculate_metrics(mu, targets, dataset_test, land_mask, device, prefix="test_")
            
            for k, v in metrics.items():
                if not np.isnan(v):
                    test_metrics_sum[k] += v
            test_count += 1
            
            # 更新测试进度条
            test_pbar.set_postfix({'test_loss': f"{metrics['test_loss']:.4f}"})
    
    # 计算平均测试指标
    test_metrics = {k: v / test_count for k, v in test_metrics_sum.items()}
    
    # 记录测试日志
    logger.info(f"Test Loss: {test_metrics['test_loss']:.6f} - "
                f"Test MAE: {test_metrics['test_mae_sic']:.6f} - "
                f"Test RMSE: {test_metrics['test_rmse_sic']:.6f} - "
                f"Test R^2: {test_metrics['test_R^2_sic']:.6f}")