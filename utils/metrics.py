"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-04-13 11:15:02
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-04-14 15:33:20
FilePath: /root/arctic_sic_prediction/metrics.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import numpy as np
import torch

Max_SIE = 25889

def mse_func(pred, true):
    mse = torch.mean((pred - true) ** 2, dim=[2, 3, 4]).mean(dim=1)
    return mse.mean().item()


def rmse_func(pred, true):
    mse = torch.mean((pred - true) ** 2, dim=[2, 3, 4]).mean(dim=1)
    rmse = torch.sqrt(mse)
    return rmse.mean().item()


def mae_func(pred, true):
    mae = torch.mean(torch.abs(pred - true), dim=[2, 3, 4]).mean(dim=1)
    return mae.mean().item()

def nse_func(pred, true):
    squared_error = torch.mean((pred - true) ** 2, dim=[2, 3, 4]).mean(dim=1)
    mean_observation = torch.mean(true, dim=[2, 3, 4]).mean(dim=1)
    mean_observation = (
        mean_observation.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    )
    squared_deviation = torch.mean((true - mean_observation) ** 2, dim=[2, 3, 4]).mean(
        dim=1
    )
    nse = 1 - squared_error / squared_deviation
    return nse.mean().item()
    # return 1

def PSNR(pred, true):
    mse = torch.mean((pred - true) ** 2, dim=[2, 3, 4]).mean(dim=1)
    PSNR = 10 * np.log10(1.0 / mse.mean().item())
    return PSNR
    # return 1

def BACC_func(pred, true, mask):
    # 使用布尔索引将大于0.15的位置设置为1，其他地方设置为0
    # print(pred.shape)
    pred_binary = torch.where(
        pred > 0.15,
        torch.tensor(1.0, device=pred.device),
        torch.tensor(0.0, device=pred.device),
    )
    true_binary = torch.where(
        true > 0.15,
        torch.tensor(1.0, device=true.device),
        torch.tensor(0.0, device=true.device),
    )
    IIEE = torch.sum(torch.abs(pred_binary - true_binary) * mask, dim=[2, 3, 4]).mean(
        dim=1
    )
    IIEE = IIEE.mean().item()
    IIEE = min(IIEE, Max_SIE)
    BACC = 1 - IIEE / Max_SIE
    return BACC
