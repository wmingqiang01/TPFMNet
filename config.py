import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

configs.model = "ConvLSTM"   # 爆显存
# configs.model = "PredRNN"    #爆显存
# configs.model = "PredRNNv2"   # 爆显存
# configs.model = "E3DLSTM"    # 爆显存
# configs.model = "SimVP"        # 已完成
# configs.model = "TAU"     # 已完成
# configs.model = "ConvNeXt"      # 已完成
# configs.model = "ConvNeXt_V2"
# configs.model = "InceptionNeXt"
# configs.model = "Swin_Transformer"
# configs.model = "SICFN"
# configs.model = "SICTeDev"
# configs.model = "Swin_Transformer"   # 已完成
# configs.model = "SwinFreq"    # 爆显存
# configs.model = "SICWare"       # 已完成
# configs.model = "Swin_Wavelet"   # 已完成
# configs.model = "Swin_Wavelet_Learnable"   
# configs.model = "TSMixer"
# configs.model = "ConvNeXt_uncertain"
# configs.model = "Mamba"

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
# configs.img_size = (448, 304)
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

# model related
configs.kernel_size = (3, 3)
configs.patch_size = (4, 4)
configs.hidden_dim = (
    96,
    96,
    96,
    96,
)  # hidden_dim: 隐藏状态的神经单元个数，也就是隐藏层的节点数，应该可以按计算需要设置。

configs.decouple_beta = 0.1  # PredRNNv2

configs.kernel_size_3D = (2, 2, 2)  # E3DLSTM

# SimVP
configs.hid_S = 64
configs.hid_T = 512
configs.N_T = 8
configs.N_S = 4
configs.spatio_kernel_enc = 3
configs.spatio_kernel_dec = 3
configs.act_inplace = True

# Mamba specific configs
configs.temporal_fusion_layers = 4
configs.d_state = 32
configs.use_attention = True

# TAU
configs.mlp_ratio = 4
configs.drop = 0.0
configs.drop_path = 0.1

# ConvNeXt
configs.use_grn = True

# ConvNeXt_V2
configs.depths = [3, 3, 9, 3]
configs.dims = [96, 192, 384, 768]
configs.uncertainty_type = 'gaussian'

# SICFN
configs.patch_embed_size = (4, 4)
configs.dropcls = 0.0
configs.fno_blocks = 8
configs.fno_bias = True
configs.fno_softshrink = 0.0

# paths
configs.full_data_path = [
    # "./download_data/full_sic.nc",      # SIC
    # "./download_data/full_t2m.nc",      # 2m temperature
    # "./download_data/full_u10.nc",      # 10m U wind
    # "./download_data/full_v10.nc",      # 10m V wind
    # "./download_data/full_siv_u.nc",    # Sea ice velocity U
    # "./download_data/full_siv_v.nc",    # Sea ice velocity V
    "./download_data/full_sic_update.nc",      
]
configs.train_log_path = "train_logs"
configs.test_results_path = "test_results"

configs.feat_mixing_hidden_channels = 8
configs.embed_dim = 128