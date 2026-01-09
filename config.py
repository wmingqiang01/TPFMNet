import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.device = torch.device("cuda:0")
configs.batch_size_vali = 1
configs.batch_size = 1
configs.lr = 1e-3  
configs.weight_decay = 0.01
configs.num_epochs = 200
configs.early_stopping = False
configs.patience = configs.num_epochs // 10
configs.gradient_clipping = True
configs.clipping_threshold = 1.0
configs.layer_norm = True
configs.display_interval = 50
configs.num_workers = 16

# data related
configs.img_size = (432,432)

configs.input_dim = 1  #  input_dim: 输入张量对应的通道数
configs.output_dim = 1  #  output_dim: 输出张量对应的通道数

configs.mask_path = "ocean_mask.npy"
# configs.mask_path = "arctic_mask.npy"

configs.input_length = 12  # 每轮训练输入多少张数据
configs.pred_length = 12 # 每轮训练输出多少张数据

configs.input_gap = 1  # 每张输入数据之间的间隔
configs.pred_gap = 1  # 每张输出数据之间的间隔

configs.pred_shift = configs.pred_gap * configs.pred_length

configs.train_period = (198801, 201912)
configs.eval_period = (202001, 202408)

# configs.train_period = (197901, 201512)
# configs.eval_period = (201601, 202012)


# paths
configs.full_data_path = ["./download_data/full_sic_update.nc",     ]
configs.train_log_path = "train_logs"
configs.test_results_path = "test_results"

configs.feat_mixing_hidden_channels = 8
configs.embed_dim = 128