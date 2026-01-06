import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================
# 全局设备设置
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==============================
# 辅助函数（GPU 版）
# ==============================
def process_model_output(output):
    if len(output.shape) == 5:
        return output[0, :, 0, :, :].cpu().numpy()
    elif len(output.shape) == 4 and output.shape[1] == 1:
        return output.squeeze(1).cpu().numpy()
    else:
        return output.cpu().numpy()

# ------------------------------
# GPU 版 SICNetseason BACC（单帧）
# ------------------------------
@torch.no_grad()
def calculate_iiee_gpu(pred_tensor, target_tensor, threshold=0.15, ocean_mask_tensor=None):
    """
    输入: (H, W) 或 (C, H, W) 的 torch.Tensor，已在 GPU 上
    返回: iiee (km², float scalar on cpu)
    """
    if pred_tensor.dim() == 3:
        pred_tensor = pred_tensor.squeeze(0)
    if target_tensor.dim() == 3:
        target_tensor = target_tensor.squeeze(0)

    # 二值化
    pred_bin = (pred_tensor > threshold).float()
    target_bin = (target_tensor > threshold).float()

    # 掩码无效值
    valid = ~(torch.isnan(pred_tensor) | torch.isnan(target_tensor))
    if ocean_mask_tensor is not None:
        valid = valid & ocean_mask_tensor

    # 错误格子数
    error = (pred_bin != target_bin) & valid
    num_error = error.sum().item()               # 在 GPU 上求和后转 cpu scalar
    iiee = num_error * 625.0                      # 25km × 25km = 625 km²
    return iiee

# ==============================
# 历史最大 SIE 计算（GPU 批量加速）
# ==============================
@torch.no_grad()
def compute_max_sie_historical_gpu(target_months=[6,7,8,9], threshold=0.15, grid_area=625.0,
                                 hist_start=197901, hist_end=201912):
    from config import configs
    from utils.utils_2 import SIC_dataset

    print(f"Loading historical data {hist_start}-{hist_end} on GPU to compute max SIE ...")
    hist_dataset = SIC_dataset(
        configs.full_data_path,
        hist_start, hist_end,
        configs.input_gap, configs.input_length,
        configs.pred_shift, configs.pred_gap, configs.pred_length,
        samples_gap=1,
    )

    # 加载 land_mask → GPU
    try:
        land_mask_np = np.load("ocean_mask.npy")  # True = ocean
        ocean_mask = torch.from_numpy(land_mask_np).bool().to(device)
    except:
        ocean_mask = None
        print("Warning: ocean_mask.npy not found, will use whole grid")

    max_sie = {m: 0.0 for m in target_months}
    all_times = hist_dataset.GetTimes()
    input_len = configs.input_length

    for idx in tqdm(range(len(hist_dataset)), desc="Historical Max SIE"):
        _, y = hist_dataset[idx]                                   # y: (T_pred, C, H, W)
        y_tensor = torch.from_numpy(y).to(device)                  # (T,1,H,W)

        times = all_times[idx][input_len:]                         # 对应 y 的时间

        for t in range(y_tensor.shape[0]):
            month = int(str(times[t])[4:6])
            if month not in target_months:
                continue

            target_frame = y_tensor[t, 0]                               # (H,W)
            valid = ~(torch.isnan(target_frame))
            if ocean_mask is not None:
                valid = valid & ocean_mask

            ice_cells = (target_frame > threshold) & valid
            sie_area = ice_cells.sum().item() * grid_area

            if sie_area > max_sie[month]:
                max_sie[month] = sie_area

    print("Historical max SIE (million km²):")
    for m, area in max_sie.items():
        print(f"  Month {m:02d}: {area/1e6:.3f} M km²")
    return max_sie

# ==============================
# 热图绘制（不变）
# ==============================
def plot_heatmap(data_matrix, months, lead_times, save_path="bacc_heatmap_sicnetseason_gpu.png"):
    data_percent = data_matrix.T * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.RdYlBu_r
    vmin, vmax = 70, 100
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(data_percent, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("BACC (%)", rotation=-90, va="bottom", fontsize=11)

    ax.set_xticks(np.arange(len(lead_times)))
    ax.set_xticklabels(lead_times, fontsize=11)
    ax.set_xlabel("Forecast Lead Time (months)", fontsize=12)

    month_map = {6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec',1:'Jan'}
    ax.set_yticks(np.arange(len(months)))
    ax.set_yticklabels([month_map[m] for m in months], fontsize=11)
    ax.set_ylabel("Target Month", fontsize=12)

    for i in range(len(months)):
        for j in range(len(lead_times)):
            val = data_percent[i, j]
            if not np.isnan(val):
                rgba = cmap(norm(val))
                lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                color = "black" if lum > 0.5 else "white"
                ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                        color=color, fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f">>> Heatmap saved: {save_path}")
    plt.close()

# ==============================
# 主评估函数（全 GPU）
# ==============================
@torch.no_grad()
def evaluate_and_plot_gpu(model, dataset, land_mask_np, max_sie_dict,
                          target_months=[6,7,8,9], max_lead=6, threshold=0.15, grid_area=625.0):

    model.to(device)
    model.eval()

    # land_mask → GPU
    ocean_mask = None
    if land_mask_np is not None:
        ocean_mask = torch.from_numpy(land_mask_np).bool().to(device)

    # 累计 IIEE
    results = {m: {l: {"iiee": 0.0, "count": 0} for l in range(1, max_lead+1)} 
               for m in target_months}

    all_times = dataset.GetTimes()
    from config import configs
    input_len = configs.input_length

    print("\nStarting GPU evaluation ...")
    for idx in tqdm(range(len(dataset)), desc="Eval"):
        x, y = dataset[idx]
        x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(device)   # (1,T_in,C,H,W)
        y_tensor = torch.from_numpy(y).to(device)                        # (T_pred,1,H,W)

        mu, _, _, _ = model(x_tensor, is_training=False, return_ci=True)
        mu = mu.squeeze(0)[:, 0]                                         # (T_pred, H, W)

        times = all_times[idx][input_len:]

        for t_idx in range(min(mu.shape[0], len(times), max_lead)):
            lead = t_idx + 1
            month_str = str(times[t_idx])
            if len(month_str) < 6: continue
            month = int(month_str[4:6])
            if month not in target_months: continue

            pred_frame = mu[t_idx]
            target_frame = y_tensor[t_idx, 0]

            iiee = calculate_iiee_gpu(pred_frame, target_frame, threshold, ocean_mask)

            results[month][lead]["iiee"] += iiee
            results[month][lead]["count"] += 1

    # ========== 计算最终 BACC ==========
    data_matrix = np.zeros((max_lead, len(target_months)))   # ← 正确定义！

    print("\n" + "="*80)
    print("SICNetseason-style BACC (Historical Max SIE normalization)")
    print("="*80)
    for i, lead in enumerate(range(1, max_lead+1)):
        for j, month in enumerate(target_months):
            cnt = results[month][lead]["count"]
            if cnt > 0:
                avg_iiee = results[month][lead]["iiee"] / cnt
                max_sie = max_sie_dict.get(month, 1e9)  # 防止除0
                bacc = max(0.0, 1 - avg_iiee / max_sie)
                data_matrix[i, j] = bacc
                print(f"Lead {lead} | Month {month:02d} → {bacc*100:6.2f}%  (n={cnt:3d}, maxSIE={max_sie/1e6:.3f}M km²)")
            else:
                data_matrix[i, j] = np.nan
                print(f"Lead {lead} | Month {month:02d} →   —   (no sample)")

    plot_heatmap(data_matrix, target_months, list(range(1, max_lead+1)),
                 save_path="bacc_heatmap_sicnetseason_gpu.png")
# ==============================
# 主程序
# ==============================
def main():
    ckpt_path = '/root/shared-nvme/wmq/my_model/pretrained_models/TPFUNet_12_12.pth'

    # 1. 加载模型
    from config import configs
    from models.TPFUNet import TPFUNet
    model = TPFUNet(T=configs.input_length, C=configs.input_dim, uncertainty_type='laplacian')
    ckpt = torch.load(ckpt_path, map_location=device,weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    print("Model loaded")

    # 2. 加载测试数据：SICNetseason 相同年份 2000-2019
    from utils.utils_2 import SIC_dataset
    dataset = SIC_dataset(configs.full_data_path, 200001, 201912,  # start_time=200001, end_time=201912
                          configs.input_gap, configs.input_length,
                          configs.pred_shift, configs.pred_gap, configs.pred_length,
                          samples_gap=1)
    print(f"Test dataset loaded: 2000-2019 ({len(dataset)} samples)")

    # 3. 加载 land_mask
    try:
        land_mask = np.load("ocean_mask.npy")
        print("ocean_mask.npy loaded")
    except:
        land_mask = None
        print("No ocean_mask.npy, will use full grid")

    # 4. 计算历史最大 SIE（1979-2019，不变）
    max_sie_dict = compute_max_sie_historical_gpu(
        target_months=[6,7,8,9],
        threshold=0.15,
        grid_area=625.0,
        hist_start=197901,
        hist_end=201912
    )

    # 5. 正式评估（GPU）
    evaluate_and_plot_gpu(
        model=model,
        dataset=dataset,
        land_mask_np=land_mask,
        max_sie_dict=max_sie_dict,
        target_months=[6,7,8,9],
        max_lead=6,
        threshold=0.15,
        grid_area=625.0
    )

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()