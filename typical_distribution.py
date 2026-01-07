import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import laplace
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# 1. 辅助函数：添加陆地掩码
# ----------------------------------------------------------------------------
def add_land_overlay(ax, land_mask, origin='lower'):
    """添加陆地覆盖层（海洋为True，陆地为False）"""
    if land_mask is not None:
        land_color = "#d2b48c" 
        # 原始mask: True=海洋, False=陆地
        # 我们需要显示陆地，所以Mask掉海洋(True)
        land_to_show = np.ma.masked_where(land_mask == True, land_mask)
        # 这里的 zorder=10 保证掩码在热图之上
        ax.imshow(land_to_show, cmap=ListedColormap([land_color]), origin=origin, zorder=10)

# ----------------------------------------------------------------------------
# 2. 绘图函数：极简风格 (无标题无标注) -> 现在增加了标注和真值线
# ----------------------------------------------------------------------------
def plot_clean_laplace_pdf(mu, b, true_val, filename):
    """绘制极简风格的拉普拉斯分布，增加真值线和坐标轴标签"""
    x = np.linspace(0, 1, 1000)
    y = laplace.pdf(x, loc=mu, scale=b)

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 绘图
    ax.plot(x, y, color='#003366', linewidth=2.5, label='Predicted PDF')
    # 均值虚线
    ax.axvline(mu, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8, label='Predicted Mean')
    # 真值实线
    ax.axvline(true_val, color='green', linestyle='-', linewidth=2.0, alpha=0.9, label='True Value')
    
    # 填充颜色
    ax.fill_between(x, y, color='#003366', alpha=0.1)
    
    # 设置范围
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    
    # 移除顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 保留坐标轴刻度，但字体设大
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 增加坐标轴标签
    ax.set_xlabel("Sea Ice Concentration", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    
    # 添加图例
    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> PDF图已保存: {os.path.basename(filename)}")

# ----------------------------------------------------------------------------
# 3. 选点逻辑 (强制空间分离)
# ----------------------------------------------------------------------------
def get_distinct_points_with_truth(mu_map, unc_map, target_map, land_mask, min_dist=30):
    points = []
    rows, cols = mu_map.shape
    
    if land_mask is not None:
        base_mask = (land_mask == True) # True是海洋
    else:
        base_mask = np.ones_like(mu_map, dtype=bool)
    
    base_mask = base_mask & (~np.isnan(mu_map)) & (~np.isnan(unc_map))
    exclusion_mask = np.zeros_like(mu_map, dtype=bool)

    def mask_around(y, x, dist):
        yy, xx = np.ogrid[:rows, :cols]
        return ((yy - y)**2 + (xx - x)**2) <= dist**2

    def find_best_index(metric_map, criteria='max', extra_mask=None):
        mask = base_mask & (~exclusion_mask)
        if extra_mask is not None:
            mask = mask & (~extra_mask)
            
        if np.sum(mask) == 0: return None
            
        valid_indices = np.where(mask)
        valid_values = metric_map[valid_indices]
        
        if len(valid_values) == 0: return None
            
        if criteria == 'max':
            local_idx = np.argmax(valid_values)
        elif criteria == 'min':
            local_idx = np.argmin(valid_values)
        elif criteria == 'abs_diff_mid': 
            mid_v = (np.max(valid_values) + np.min(valid_values)) / 2
            local_idx = np.argmin(np.abs(valid_values - mid_v))
        
        return valid_indices[0][local_idx], valid_indices[1][local_idx]

    definitions = [
        ('Max_Ice', mu_map, 'max'),
        ('High_Uncertainty', unc_map, 'max'),
        ('Transition_Zone', mu_map, 'abs_diff_mid'),
        ('Open_Water', mu_map, 'min')
    ]

    # --------------------------------------------------------
    # 边缘屏蔽逻辑 (防止选点太靠边)
    # --------------------------------------------------------
    border_margin = 50  # 边缘保留距离(像素)
    border_mask = np.zeros_like(mu_map, dtype=bool)
    if rows > 2 * border_margin and cols > 2 * border_margin:
        border_mask[:border_margin, :] = True
        border_mask[-border_margin:, :] = True
        border_mask[:, :border_margin] = True
        border_mask[:, -border_margin:] = True

    for p_type, map_src, crit in definitions:
        # 如果是Open_Water，施加边缘Mask
        extra = None
        if p_type == 'Open_Water':
            extra = border_mask
            
        res = find_best_index(map_src, crit, extra_mask=extra)
        if res:
            py, px = res
            points.append({
                'type': p_type,
                'y': py, 'x': px,
                'mu': mu_map[py, px],
                'b': unc_map[py, px],
                'true': target_map[py, px]
            })
            exclusion_mask |= mask_around(py, px, min_dist)

    return points

# ----------------------------------------------------------------------------
# 4. 主程序
# ----------------------------------------------------------------------------
def main_clean_plot(checkpoint_path, target_month_str='202209'):
    # 引入依赖
    try:
        from config import configs
        from utils.utils_2 import SIC_dataset
        from models.TPFMNet import TPFMNet
    except ImportError:
        print("Error: 请在项目根目录运行。")
        return

    print(f">>> 开始处理: {target_month_str} (Zorder修正版) <<<")

    # 1. 模型与数据
    model = TPFMNet(T=configs.input_length, C=configs.input_dim, uncertainty_type='laplacian')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    dataset = SIC_dataset(
        configs.full_data_path,
        start_time=202101, end_time=202212,
        input_gap=configs.input_gap, input_length=configs.input_length,
        pred_shift=configs.pred_shift, pred_gap=configs.pred_gap, pred_length=configs.pred_length,
        samples_gap=1,
    )

    all_times = dataset.GetTimes()
    target_idx = -1
    time_offset = -1
    target_int = int(target_month_str)
    
    for i in range(len(all_times)):
        pred_times = all_times[i][configs.input_length:]
        if target_int in pred_times:
            target_idx = i
            time_offset = np.where(pred_times == target_int)[0][0]
            break
            
    if target_idx == -1:
        print("未找到指定月份数据")
        return

    print(f"  -> 找到数据: Sample {target_idx}, Offset {time_offset}")

    # 获取数据与真值
    input_data, _ = dataset[target_idx]
    all_targets_raw = dataset.GetTargets() 
    target_map = all_targets_raw[target_idx, time_offset, 0, :, :]
    
    # 推理
    input_tensor = torch.from_numpy(input_data).unsqueeze(0).float()
    with torch.no_grad():
        mu, uncertainty, _, _ = model(input_tensor, is_training=False, return_ci=True)
    
    mu_map = mu[0, time_offset, 0, :, :].cpu().numpy()
    unc_map = uncertainty[0, time_offset, 0, :, :].cpu().numpy()
    
    try:
        land_mask = np.load("ocean_mask.npy")
        land_mask = land_mask.astype(bool)
    except:
        print("Warning: ocean_mask.npy 未找到，将不显示陆地")
        land_mask = None

    # 选点
    selected_points = get_distinct_points_with_truth(mu_map, unc_map, target_map, land_mask, min_dist=40)
    
    # 输出表格
    print("\n" + "="*80)
    print(f"{'Type':<20} | {'Coord(Y,X)':<12} | {'Pred(Mean)':<12} | {'Uncertainty':<12} | {'TRUE VALUE':<12}")
    print("-" * 80)
    for p in selected_points:
        print(f"{p['type']:<20} | {p['y']:3d}, {p['x']:3d}    | {p['mu']:.6f}     | {p['b']:.6f}      | {p['true']:.6f}")
    print("="*80 + "\n")

    # 绘图
    output_dir = f'./final_clean_plots_{target_month_str}'
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Map 1: 选点位置图 ---
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    # 使用 Blues_r: 0(水)为深蓝, 1(冰)为白色
    im1 = ax1.imshow(mu_map, vmin=0, vmax=1, cmap='Blues_r', origin='lower')
    
    # 叠加陆地掩码 (Zorder=10)
    add_land_overlay(ax1, land_mask, origin='lower')
    
    # 统一使用红色圆点加黑边
    for i, p in enumerate(selected_points):
        # *** 关键修复 ***: 将 scatter 和 text 的 zorder 设为 20，确保在陆地(zorder=10)之上
        # c='red' (红点), edgecolors='black' (黑边), marker='o' (圆点)
        ax1.scatter(p['x'], p['y'], c='red', edgecolors='black', marker='o', s=150, linewidths=1.5, zorder=20)
        ax1.text(p['x']+5, p['y']+5, str(i+1), color='yellow', fontsize=14, fontweight='bold', zorder=20)
    
    ax1.axis('off')
    # 增加Colorbar
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(output_dir, 'map_locations_clean.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # --- Map 2: 不确定性图 ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    im2 = ax2.imshow(unc_map, cmap='Reds', origin='lower')
    # 叠加陆地掩码
    add_land_overlay(ax2, land_mask, origin='lower')
    ax2.axis('off')
    # 增加Colorbar
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(output_dir, 'map_uncertainty_clean.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # --- Plots: 4个分布图 ---
    for i, p in enumerate(selected_points):
        fname = os.path.join(output_dir, f"{i+1}_{p['type']}.png")
        plot_clean_laplace_pdf(p['mu'], p['b'], p['true'], fname)

    print(f"所有图片已保存至: {output_dir}")

if __name__ == "__main__":
    # 路径配置
    ckpt = '/root/shared-nvme/wmq/my_model/best/TPFMNet_12_12.pth'
    main_clean_plot(ckpt, '202209')
