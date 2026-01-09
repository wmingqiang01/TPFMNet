# TPFUNet: 海冰密集度预测 (Sea Ice Concentration Prediction)

本项目 **TPFUNet** 是用于海冰密集度（SIC）预测的深度学习模型。该框架提供了一套完整的流程，包括数据处理、模型训练、评估验证以及结果可视化。

## 📂 项目结构

```
TPFUNet/
├── models/                     # 模型架构定义
│   ├── TPFUNet.py              # TPFUNet 核心代码
│   └── ...
├── utils/                      # 工具库
│   ├── utils_2.py              # 数据集加载器 (SIC_dataset)
│   ├── metrics.py              # 评估指标计算
│   └── ...
├── download_data/              # 数据下载与处理脚本
├── config.py                   # 全局配置文件 (参数设置中心)
├── train_uncertain.py          # 主训练与验证脚本
├── diff_lead_time.py           # 不同预见期 (Lead Time) 的性能分析
├── visualize_confidence_interval.py # 预测结果与置信区间可视化
└── README.md                   # 项目说明文档
```

## 🛠️ 环境依赖

请确保安装了以下 Python 库：

*   Python 3.x
*   PyTorch (建议 1.10+)
*   NumPy
*   Einops (用于张量操作)
*   Timm (用于视觉模型组件)
*   Scikit-learn
*   Tqdm (进度条)

安装核心依赖：

```bash
pip install torch numpy einops timm scikit-learn tqdm
```

## 💾 数据准备

本项目数据来源于OSI SAF 450-a的月度北极海冰密集度数据。

1.  **数据存放**：默认目录为 `download_data/`。
2.  **配置路径**：在 `config.py` 中修改 `configs.full_data_path`。
    *   默认指向：`./download_data/full_sic_update.nc`
3.  **数据处理**：如果需要下载或重新组织数据，请参考 `download_data/` 目录下的脚本。

## 🚀 使用指南

### 1. 配置参数

所有实验参数均在 `config.py` 中管理。你可以修改：
*   **训练参数**：`batch_size` (批大小), `lr` (学习率), `num_epochs` (训练轮数)。
*   **时序参数**：
    *   `input_length`: 输入的历史时间步长 (默认 12)。
    *   `pred_length`: 预测的未来时间步长 (默认 12)。
*   **训练/评估周期**：`train_period` 和 `eval_period` 定义了训练集和测试集的时间范围。

### 2. 开始训练

运行以下命令启动训练：

```bash
python train_uncertain.py
```

程序将自动：
*   加载配置的数据集。
*   初始化指定模型。
*   进行训练并定期在验证集上评估。
*   保存训练日志 (`train_logs/`) 和模型权重。

### 3. 评估与可视化

*   **指标计算**：模型在训练过程中会自动计算 RMSE, MAE, R2 等指标（定义在 `utils/metrics.py`）。
*   **可视化**：使用 `visualize_confidence_interval.py` 生成预测结果图，包括不确定性区间。
*   **预见期分析**：运行 `diff_lead_time.py` 分析模型在不同预测时间长度下的表现稳定性。
