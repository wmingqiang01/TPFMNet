import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import laplace, norm  # 用于 PDF
import sys
import os
import warnings
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score

# 假设您的项目结构，如果路径不同请自行调整
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from config import configs
# from utils.utils_2 import SIC_dataset
# from models.TPFMNet import TPFMNet

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------------

def add_land_overlay(ax, land_mask, origin='upper'):
    """添加陆地覆盖层（海洋为True，陆地为False）。支持指定 origin 与底图一致"""
    if land_mask is not None:
        land_color = "#d2b48c"
        # 屏蔽海洋(True)，仅显示陆地(False)
        land = np.ma.masked_where(land_mask == True, land_mask)
        ax.imshow(land, cmap=ListedColormap([land_color]), origin=origin)

def process_model_output(output):
    """统一处理模型输出形状"""
    if len(output.shape) == 5:
        return output[0, :, 0, :, :].cpu().numpy()
    elif len(output.shape) == 4 and output.shape[1] == 1:
        return output.squeeze(1).cpu().numpy()
    else:
        return output.cpu().numpy()

# ----------------------------------------------------------------------------
# 核心指标计算函数 (修改版)
# ----------------------------------------------------------------------------

def calculate_metrics(predictions, targets, threshold=0.15, land_mask=None, lower=None, upper=None):
    """
    计算评估指标，去掉陆地部分，并添加置信区间指标、ACC 和 IIEE。
    
    Grid Size: 25km x 25km = 625 km^2
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # 移除NaN值
    nan_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    
    # 处理海洋掩模（海洋=True，陆地=False）
    mask_flat = None
    if land_mask is not None:
        if predictions.ndim == 3:  # (time, lat, lon)
            num_times = predictions.shape[0]
            mask_flat = np.tile(land_mask.flatten(), num_times)
        else:  # 2D (lat, lon)
            mask_flat = land_mask.flatten()
        ocean_mask = (mask_flat == True)
    else:
        ocean_mask = np.ones_like(pred_flat, dtype=bool)
    
    # 组合掩模
    valid_mask = nan_mask & ocean_mask
    pred_valid = pred_flat[valid_mask]
    target_valid = target_flat[valid_mask]
    
    if len(pred_valid) == 0:
        return {"error": "No valid data points"}
    
    # 1. 基础回归指标
    mae = np.mean(np.abs(pred_valid - target_valid))
    mse = np.mean((pred_valid - target_valid) ** 2)
    rmse = np.sqrt(mse)
    
    r2 = np.nan
    acc_binary = np.nan
    bacc = np.nan
    corr = np.nan
    iiee_area = np.nan
    
    try:
        # R2 Score
        if len(pred_valid) > 1:
            r2 = r2_score(target_valid, pred_valid)
        
        # 2. 二分类指标 (SIE: Sea Ice Extent)
        pred_binary = (pred_valid > threshold).astype(int)
        target_binary = (target_valid > threshold).astype(int)
        
        acc_binary = accuracy_score(target_binary, pred_binary)
        bacc = balanced_accuracy_score(target_binary, pred_binary)
        
        # 3. 空间相关系数 (Spatial ACC / Correlation)
        # 计算皮尔逊相关系数
        if len(pred_valid) > 1 and np.std(pred_valid) > 1e-9 and np.std(target_valid) > 1e-9:
            corr = np.corrcoef(pred_valid, target_valid)[0, 1]
            
        # 4. 综合冰缘误差 (IIEE)
        # IIEE = Area of (Overestimation + Underestimation)
        # 计算差异像素数
        mismatch_pixels = np.sum(np.abs(pred_binary - target_binary))
        
        # 转换为面积 (百万平方公里)
        # 1 pixel = 25km * 25km = 625 km^2
        # 10^6 km^2 = 1,000,000 km^2
        grid_area_km2 = 25 * 25
        iiee_area = (mismatch_pixels * grid_area_km2) / 1e6
        
    except Exception as e:
        # print(f"Metrics calc warning: {e}")
        pass
    
    # 5. 置信区间指标
    coverage = np.nan
    mean_width = np.nan
    sharpness = np.nan
    
    if lower is not None and upper is not None:
        lower_flat = lower.flatten()
        upper_flat = upper.flatten()
        lower_valid = lower_flat[valid_mask]
        upper_valid = upper_flat[valid_mask]
        target_valid_ci = target_flat[valid_mask]
        
        # 覆盖率：真实值在CI内的比例
        coverage = np.mean((target_valid_ci >= lower_valid) & (target_valid_ci <= upper_valid))
        # CI平均宽度
        mean_width = np.mean(upper_valid - lower_valid)
        # 尖锐性
        if mean_width > 0:
            sharpness = 1.0 / mean_width
        else:
            sharpness = np.inf
    
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
        'acc': acc_binary, 'bacc': bacc, 
        'corr': corr, 'iiee': iiee_area, # 单位：10^6 km^2
        'n_valid_points': len(pred_valid),
        'coverage': coverage, 'mean_width': mean_width, 'sharpness': sharpness
    }

def print_detailed_metrics(overall_metrics, timestep_metrics, uncertainty, predictions, targets, pred_times, land_mask):
    """打印详细评估指标，包含 ACC、IIEE 和 每月平均不确定性"""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # 处理海洋掩模用于不确定性统计
    if land_mask is not None:
        ocean_mask = land_mask == True
        unc_flat = uncertainty.flatten()
        mask_all_flat = np.tile(land_mask.flatten(), uncertainty.shape[0])
        uncertainty_masked = unc_flat[mask_all_flat == True]
    else:
        ocean_mask = None
        uncertainty_masked = uncertainty.flatten()
    
    print("\n" + "="*120)
    print("完整评估指标报告 (仅海洋区域)")
    print("="*120)
    
    # 1. 总体指标
    print("\n1. 总体指标 (IIEE单位: 10^6 km^2):")
    print("-" * 60)
    key_metrics = ['mae', 'rmse', 'acc', 'corr', 'iiee']
    
    # 先打印关键指标
    if 'mae' in overall_metrics: print(f"{'MAE':>12}: {overall_metrics['mae']:.6f}")
    if 'rmse' in overall_metrics: print(f"{'RMSE':>12}: {overall_metrics['rmse']:.6f}")
    if 'acc' in overall_metrics: print(f"{'Binary ACC':>12}: {overall_metrics['acc']:.6f}")
    if 'corr' in overall_metrics: print(f"{'Spatial ACC':>12}: {overall_metrics['corr']:.6f}")
    if 'iiee' in overall_metrics: print(f"{'IIEE (Area)':>12}: {overall_metrics['iiee']:.6f}")
            
    print("-" * 30)
    # 打印剩余指标
    for metric, value in overall_metrics.items():
        if metric not in key_metrics and metric != 'n_valid_points':
             print(f"{metric.upper():>12}: {value:.6f}")
    if 'n_valid_points' in overall_metrics:
        print(f"{'VALID_PTS':>12}: {overall_metrics['n_valid_points']:,}")
    
    # 2. 每个时间步详细指标
    print("\n2. 每个时间步详细指标:")
    print("-" * 150)
    header = (f"{'月份':>5} {'时间':>6} {'MAE':>9} {'RMSE':>9} {'ACC':>9} {'BACC':>9} "
              f"{'Corr':>9} {'IIEE':>9} {'COV':>8} {'WIDTH':>9} {'SHARP':>9} {'UNC':>9}")
    print(header)
    print("-" * 150)
    
    for t in range(min(12, len(timestep_metrics))):
        timestep_key = f'timestep_{t+1}'
        if timestep_key in timestep_metrics and 'error' not in timestep_metrics[timestep_key]:
            m = timestep_metrics[timestep_key]
            time_label = pred_times[t] if pred_times is not None else f"T{t+1}"
            
            # 计算当前月份的平均不确定性
            if ocean_mask is not None:
                unc_mean_t = np.mean(uncertainty[t][ocean_mask])
            else:
                unc_mean_t = np.mean(uncertainty[t])

            print(f"{months[t]:>6} {str(time_label)[-4:]:>6} "
                  f"{m['mae']:>9.5f} {m['rmse']:>9.5f} "
                  f"{m['acc']:>9.5f} {m['bacc']:>9.5f} "
                  f"{m['corr']:>9.5f} {m['iiee']:>9.5f} "  
                  f"{m['coverage']:>8.4f} {m['mean_width']:>9.4f} {m['sharpness']:>9.4f} "
                  f"{unc_mean_t:>9.5f}")
    
    # 3. 不确定性统计 (仅海洋)
    print("\n3. 不确定性统计 (仅海洋):")
    print("-" * 50)
    if len(uncertainty_masked) > 0:
        print(f"{'平均不确定性':>15}: {np.mean(uncertainty_masked):.6f}")
        print(f"{'不确定性标准差':>15}: {np.std(uncertainty_masked):.6f}")
        print(f"{'最大不确定性':>15}: {np.max(uncertainty_masked):.6f}")
    else:
        print("No valid ocean points")
    
    # 4. 季节性分析 (包含新增指标)
    print("\n4. 季节性分析 (仅海洋):")
    print("-" * 100)
    print(f"{'季节':>12} {'MAE':>10} {'Corr':>10} {'IIEE':>10} {'Uncertainty':>12}")
    print("-" * 100)
    
    seasons = {
        '春季 (3-5月)': [2, 3, 4],
        '夏季 (6-8月)': [5, 6, 7], 
        '秋季 (9-11月)': [8, 9, 10],
        '冬季 (12-2月)': [11, 0, 1]
    }
    
    for season_name, month_indices in seasons.items():
        valid_indices = [i for i in month_indices if i < len(timestep_metrics)]
        if valid_indices:
            maes, corrs, iiees, uncs = [], [], [], []
            
            for i in valid_indices:
                # Metrics
                key = f'timestep_{i+1}'
                if key in timestep_metrics:
                    maes.append(timestep_metrics[key]['mae'])
                    corrs.append(timestep_metrics[key]['corr'])
                    iiees.append(timestep_metrics[key]['iiee'])
                
                # Uncertainty
                if land_mask is not None:
                    uncs.append(np.mean(uncertainty[i][ocean_mask]))
                else:
                    uncs.append(np.mean(uncertainty[i]))
            
            if maes:
                print(f"{season_name:>12} {np.mean(maes):10.5f} {np.mean(corrs):10.5f} "
                      f"{np.mean(iiees):10.5f} {np.mean(uncs):12.5f}")
    print("\n" + "="*120)
# ----------------------------------------------------------------------------
# 加载与评估流程
# ----------------------------------------------------------------------------

def load_model_and_data(checkpoint_path=None, uncertainty_type="laplacian", start_time=202101, end_time=202212):
    """加载模型和测试数据"""
    # 请确保这些包在您的环境中可用

    from config import configs
    from utils.utils_2 import SIC_dataset
    from models.TPFMNet import TPFMNet

    # 初始化模型
    model = TPFMNet(T=configs.input_length, C=configs.input_dim, uncertainty_type='laplacian')
    
    if checkpoint_path is None:
        checkpoint_path = f'./pretrained_models/TPFMNet{configs.input_length}_{configs.pred_length}.pth'
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
            # 兼容处理 weights_only 参数
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"已加载预训练模型: {checkpoint_path}")
        except Exception as e:
            print(f"加载模型时出错: {e}")
    else:
        print(f"警告: 未找到模型 {checkpoint_path}，使用随机初始化")
    
    model.eval()
    
    # 加载测试数据
    dataset_test = SIC_dataset(
        configs.full_data_path,
        start_time, end_time,
        configs.input_gap, configs.input_length,
        configs.pred_shift, configs.pred_gap, configs.pred_length,
        samples_gap=1,
    )
    
    # 获取第一个样本用于演示
    input_data, target_data = dataset_test[0]
    input_data = torch.from_numpy(input_data).unsqueeze(0).float()
    
    # 获取目标和时间
    all_targets = dataset_test.GetTargets()
    targets = all_targets[0, :, 0, :, :] # (time, lat, lon)
    
    all_times = dataset_test.GetTimes()
    sample_times = all_times[0]
    pred_times = sample_times[configs.input_length:]
    
    print(f"输入形状: {input_data.shape}, 目标形状: {targets.shape}")
    
    # 加载掩模
    try:
        land_mask = np.load("ocean_mask.npy")
        print(f"已加载海洋掩模: {land_mask.shape}")
    except:
        print("Warning: ocean_mask.npy not found")
        land_mask = None
    
    return model, input_data, targets, pred_times, land_mask

def evaluate_model(model, input_data, targets, pred_times, land_mask, threshold=0.15):
    """评估模型性能"""
    print("开始评估模型性能...")
    
    with torch.no_grad():
        mu, uncertainty, lower, upper = model(input_data, is_training=False, return_ci=True)
    
    mu = process_model_output(mu)
    uncertainty = process_model_output(uncertainty)
    lower = process_model_output(lower)
    upper = process_model_output(upper)
    
    min_timesteps = min(mu.shape[0], targets.shape[0])
    mu = mu[:min_timesteps]
    targets = targets[:min_timesteps]
    uncertainty = uncertainty[:min_timesteps]
    lower = lower[:min_timesteps]
    upper = upper[:min_timesteps]
    
    # 计算总体指标
    overall_metrics = calculate_metrics(mu, targets, threshold, land_mask, lower, upper)
    
    # 计算每步指标
    timestep_metrics = {}
    for t in range(min_timesteps):
        metrics_t = calculate_metrics(mu[t], targets[t], threshold, land_mask, lower[t], upper[t])
        timestep_metrics[f'timestep_{t+1}'] = metrics_t
    
    print_detailed_metrics(overall_metrics, timestep_metrics, uncertainty, mu, targets, pred_times, land_mask)
    
    return mu, uncertainty

# ----------------------------------------------------------------------------
# 可视化函数 (保持原有功能，仅简化结构以适应单文件)
# ----------------------------------------------------------------------------

def visualize_true_distribution(mu_val, unc_val, uncertainty_type='laplacian'):
    """可视化分布 PDF"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if uncertainty_type == 'laplacian':
        dist = laplace
    else:
        dist = norm
    x = np.linspace(0, 1, 500)
    pdf = dist.pdf(x, loc=mu_val, scale=unc_val)
    ax.plot(x, pdf, 'r-', label='Predicted PDF')
    ax.axvline(mu_val, color='g', linestyle='--', label='Mean')
    ax.legend()
    return fig

def visualize_uncertainty(model, input_data, targets, pred_times, land_mask, save_all_months=True):
    """生成可视化图表 (新增：绘制95%置信区间上下界)"""
    print("开始生成可视化...")
    with torch.no_grad():
        # 获取模型输出，包含置信区间 lower 和 upper
        mu, uncertainty, lower, upper = model(input_data, is_training=False, return_ci=True)
        
    mu = process_model_output(mu)
    uncertainty = process_model_output(uncertainty)
    lower = process_model_output(lower)
    upper = process_model_output(upper)
    
    # 计算覆盖热图
    coverage_map = np.zeros_like(mu)
    for t in range(mu.shape[0]):
        coverage_map[t] = ((targets[t] >= lower[t]) & (targets[t] <= upper[t])).astype(float)
        
    ocean_cmap = LinearSegmentedColormap.from_list('ocean_white', ['#003366', '#ffffff'], N=256)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    if save_all_months:
        # 确保只处理存在的月份数
        for idx in range(min(12, mu.shape[0])):
            time_label = pred_times[idx] if pred_times is not None else idx+1
            t_dir = f'./uncertain/timestep_{time_label}_{months[idx]}'
            os.makedirs(t_dir, exist_ok=True)
            
            # 绘图配置列表
            # 格式: (数据, colormap, vmin, vmax, 文件名)
            plot_configs = [
                (mu[idx], ocean_cmap, 0, 1, 'predicted_mean'),
                (targets[idx], ocean_cmap, 0, 1, 'ground_truth'),
                # --- 新增部分: 绘制置信区间上下界 ---
                (lower[idx], ocean_cmap, 0, 1, 'ci_lower_bound'),
                (upper[idx], ocean_cmap, 0, 1, 'ci_upper_bound'),
                # ---------------------------------
                (uncertainty[idx], 'Reds', 0, np.max(uncertainty[idx]), 'uncertainty'),
                (coverage_map[idx], 'Blues', 0, 1, 'coverage_heatmap')
            ]
            
            for data, cmap, vmin, vmax, fname in plot_configs:
                fig, ax = plt.subplots(figsize=(8, 8))
                # 使用 origin='lower' 保持一致，根据您的数据方向可能需要调整
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                ax.axis('off')
                
                # 添加陆地掩模
                add_land_overlay(ax, land_mask, origin='lower')
                
                plt.savefig(os.path.join(t_dir, f'{fname}.svg'), format='svg', bbox_inches='tight', pad_inches=0)
                plt.close()
                
            # 简单绘制平均分布 (PDF)
            avg_mu = np.mean(mu[idx])
            avg_unc = np.mean(uncertainty[idx])
            fig_dist = visualize_true_distribution(avg_mu, avg_unc)
            fig_dist.savefig(os.path.join(t_dir, 'true_distribution.svg'))
            plt.close(fig_dist)
            
            print(f"已保存 {months[idx]} 图片到 {t_dir}")
# ----------------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------------

def main(checkpoint_path=None, uncertainty_type="laplacian", threshold=0.15, save_all_months=True, start_time=202101, end_time=202212):
    print(">>> 启动评估脚本 <<<")
    
    # 1. 加载
    model, input_data, targets, pred_times, land_mask = load_model_and_data(checkpoint_path, uncertainty_type, start_time, end_time)
    
    # 2. 评估 (含新指标计算)
    mu, uncertainty = evaluate_model(model, input_data, targets, pred_times, land_mask, threshold)
    
    # 3. 可视化
    visualize_uncertainty(model, input_data, targets, pred_times, land_mask, save_all_months)
    
    # 4. 最终摘要重算 (确保包含所有指标)
    print("\n=== 最终摘要 (包含 ACC 和 IIEE) ===")
    with torch.no_grad():
        _, _, lower, upper = model(input_data, is_training=False, return_ci=True)
    lower = process_model_output(lower)[:mu.shape[0]]
    upper = process_model_output(upper)[:mu.shape[0]]
    
    final_metrics = calculate_metrics(mu, targets, threshold, land_mask, lower, upper)
    
    print(f"总体 MAE: {final_metrics['mae']:.6f}")
    print(f"总体 RMSE: {final_metrics['rmse']:.6f}")
    print(f"总体 R²: {final_metrics['r2']:.6f}")
    print(f"总体 Binary ACC: {final_metrics['acc']:.6f}")
    print(f"总体 Spatial ACC (Corr): {final_metrics['corr']:.6f}")
    print(f"总体 IIEE (百万 km²): {final_metrics['iiee']:.6f}")
    print(f"覆盖率 (Coverage): {final_metrics['coverage']:.6f}")
    
    print("\n所有任务完成。")

if __name__ == "__main__":
    main(
        # checkpoint_path='/root/shared-nvme/wmq/my_model/best/TPFMNet_12_12.pth',
        checkpoint_path='/root/shared-nvme/wmq/my_model/best/TPFMNet_12_12.pth',
        uncertainty_type="laplacian",
        threshold=0.15,
        save_all_months=True,
        start_time=202101,
        end_time=202212
    )
