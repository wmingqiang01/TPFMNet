# TPFUNet: Sea Ice Concentration Prediction

This project implements **TPFUNet** and various other deep learning models for Sea Ice Concentration (SIC) prediction. It provides a framework for training, evaluating, and visualizing forecasting models on meteorological data.

## Project Structure

```
TPFUNet/
├── models/                     # Model architectures (TPFUNet, ConvNeXt, etc.)
├── utils/                      # Utilities, metrics, and dataset loaders
│   ├── utils_2.py              # Dataset implementation (SIC_dataset)
│   ├── metrics.py              # Evaluation metrics
│   └── ...
├── download_data/              # Scripts for data acquisition and processing
├── config.py                   # Central configuration file
├── train_uncertain.py          # Main training and validation script
├── diff_lead_time.py           # Analysis for different lead times
├── visualize_confidence_interval.py # Visualization tools
└── ...
```

## Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- NumPy
- Einops
- Timm
- Scikit-learn
- Tqdm
- Natten (Optional, for specific attention mechanisms)

You can install the core dependencies using pip:

```bash
pip install torch numpy einops timm scikit-learn tqdm
```

## Data Preparation

The project uses NetCDF (`.nc`) files for training and evaluation.

1.  **Data Location**: By default, the system looks for data in the `download_data/` directory.
2.  **Configuration**: Check `config.py` for the `configs.full_data_path` setting.
    *   Default: `./download_data/full_sic_update.nc`
3.  **Download/Organize**: Use the scripts in `download_data/` to download or format your data if needed.

## Configuration

All training and model parameters are defined in `config.py`. Key configurations include:

*   **Model Selection**: Uncomment the desired model in `configs.model` (e.g., `"TPFUNet"`, `"ConvNeXt_uncertain"`).
*   **Training Params**: `batch_size`, `lr` (learning rate), `num_epochs`.
*   **Data Params**:
    *   `input_length`: Number of input time steps.
    *   `pred_length`: Number of prediction time steps.
    *   `train_period` / `eval_period`: Date ranges for training and evaluation.
*   **Hardware**: `configs.device` (default is `cuda:0`).

## Usage

### Training

To start training the selected model:

```bash
python train_uncertain.py
```

The script will:
*   Load the dataset configured in `config.py`.
*   Initialize the model.
*   Run the training loop with validation.
*   Save logs and checkpoints.

### Evaluation & Visualization

*   **Metrics**: Custom metrics are calculated in `utils/metrics.py`.
*   **Visualization**: Use `visualize_confidence_interval.py` to generate visualizations of the model's predictions and uncertainty intervals.
*   **Lead Time Analysis**: Use `diff_lead_time.py` to analyze performance across different prediction horizons.

## Supported Models

The framework is designed to support various spatiotemporal forecasting models:

*   **TPFUNet**
*   **ConvNeXt / ConvNeXt_uncertain**
*   **SimVP**
*   **Swin Transformer**
*   **TAU**
*   And others (refer to `config.py` for the full list of integrated models).

## License

[License Information]
