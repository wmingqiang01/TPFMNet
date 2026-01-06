"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2023-11-23 12:45:21
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2023-11-25 00:23:25
FilePath: /arctic_sic_prediction/model_factory.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
"""

import torch.nn as nn
import numpy as np
from utils.utils import unfold_StackOverChannel, fold_tensor
from models.ConvLSTM import ConvLSTM
from models.PredRNN import PredRNN
from models.PredRNNv2 import PredRNNv2
from models.MotionRNN import MotionRNN
from models.E3DLSTM import E3DLSTM
from models.SimVP import SimVP
from models.TAU import TAU
from models.ConvNeXt import ConvNext
from models.ConvNeXt_uncertain import ConvNext as ConvNextUncertain
from models.convnextv2 import ConvNeXtV2
from models.InceptionNeXt import InceptionNeXt
from models.Swin_Transformer import Swin_Transformer
from models.SICFN import SICFN
from models.SwinFreq import SwinFreq
from models.SICWave import SICWare
from models.mamba import Mamba
from models.TSMixer import TSMixer


# from models.VMRNN import VMRNN_B_Model
from models.wast import WaST
from models.SICTeDev import SICTeDev
from models.Swin_Wavelet import Swin_Wavelet


class IceNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        if configs.model == "ConvLSTM":
            self.base_net = ConvLSTM(
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hidden_dim,
                configs.output_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.kernel_size,
            )
        elif configs.model == "Mamba":
            self.base_net = Mamba(
                in_chans=configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                out_chans=configs.output_dim * configs.patch_size[0] * configs.patch_size[1],
                dims=configs.hidden_dim[0] if isinstance(configs.hidden_dim, (list, tuple)) else configs.hidden_dim,
                spatio_kernel=getattr(configs, 'kernel_size', 3),
                N_S=getattr(configs, 'N_S', 4),
                temporal_fusion_layers=getattr(configs, 'temporal_fusion_layers', 4),
                d_state=getattr(configs, 'd_state', 32),
                use_attention=getattr(configs, 'use_attention', True),
            )
        elif configs.model == "PredRNN":
            self.base_net = PredRNN(
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hidden_dim,
                configs.output_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.kernel_size,
                configs.layer_norm,
            )
        elif configs.model == "PredRNNv2":
            self.base_net = PredRNNv2(
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hidden_dim,
                configs.output_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.kernel_size,
                configs.decouple_beta,
                configs.layer_norm,
            )
        elif configs.model == "MotionRNN":
            self.base_net = MotionRNN(
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hidden_dim,
                configs.output_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.kernel_size,
                configs.decouple_beta,
                configs.layer_norm,
            )
        elif configs.model == "E3DLSTM":
            self.base_net = E3DLSTM(
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hidden_dim,
                configs.output_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.kernel_size_3D,
                configs.layer_norm,
            )
        elif configs.model == "SimVP":
            self.base_net = SimVP(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.act_inplace,
            )
        elif configs.model == "TAU":
            self.base_net = TAU(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.act_inplace,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
            )
        elif configs.model == "Swin_Transformer":
            self.base_net = Swin_Transformer(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.act_inplace,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
            )
        elif configs.model == "Swin_Wavelet" or configs.model == "Swin_Wavelet_Origin":
            self.base_net = Swin_Wavelet(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.act_inplace,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
            )
        elif configs.model == "SwinFreq":
            self.base_net = SwinFreq(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.act_inplace,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
            )
        elif configs.model == "ConvNeXt":
            self.base_net = ConvNext(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.act_inplace,
                configs.mlp_ratio,
                configs.use_grn,
                configs.drop_path,
            )
        elif configs.model == "ConvNeXt_uncertain":
            self.base_net = ConvNextUncertain(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.act_inplace,
                configs.mlp_ratio,
                configs.use_grn,
                configs.drop_path,
                uncertainty_type="laplacian"
            )
        elif configs.model == "ConvNeXt_V2":
            self.base_net = ConvNeXtV2(
                in_chans=configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                out_chans=configs.output_dim * configs.patch_size[0] * configs.patch_size[1],
                depths=getattr(configs, 'depths', [3, 3, 9, 3]),
                dims=getattr(configs, 'dims', [96, 192, 384, 768]),
                drop_path_rate=getattr(configs, 'drop_path', 0.),
                spatio_kernel=getattr(configs, 'spatio_kernel_enc', 3),
                N_S=getattr(configs, 'N_S', 4),
                temporal_fusion_layers=getattr(configs, 'temporal_fusion_layers', 2),
                uncertainty_type=getattr(configs, 'uncertainty_type', 'gaussian')
            )
        elif configs.model == "SICFN":
            self.base_net = SICFN(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.patch_embed_size,
                configs.fno_blocks,
                configs.fno_bias,
                configs.fno_softshrink,
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.act_inplace,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
                configs.dropcls,
            )
        elif configs.model == "SICWare":
            self.base_net = SICWare(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.patch_embed_size,
                configs.fno_blocks,
                configs.fno_bias,
                configs.fno_softshrink,
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.act_inplace,
                configs.mlp_ratio,
                configs.drop,
                configs.drop_path,
                configs.dropcls,
            )
        # elif configs.model == "VMRNN_B_Model":
        #     self.base_net = VMRNN_B_Model(
        #         configs.input_length,
        #         configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
        #         configs.img_size[0] / configs.patch_size[0],
        #         configs.img_size[1] / configs.patch_size[1],
        #         configs.patch_size,
        #         configs.embed_dim,
        #         configs.depths,
        #         configs.num_heads,
        #         configs.window_size,
        #         configs.drop_rate,
        #         configs.attn_drop_rate,
        #         configs.drop_path_rate,
        #     )
        elif configs.model == "WaST":
            self.base_net = WaST(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.encoder_dim,
                configs.block_list,
                configs.drop_path_rate,
                configs.mlp_ratio,
            )
        elif configs.model == "SICTeDev":
            self.base_net = SICTeDev(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.hid_S,
                configs.hid_T,
                configs.N_S,
                configs.N_T,
                configs.spatio_kernel_enc,
                configs.spatio_kernel_dec,
                configs.act_inplace,
            )
        elif configs.model == "TSMixer":
            self.base_net = TSMixer(
                input_length=configs.input_length,
                forecast_length=configs.pred_length,
                channels=configs.input_dim,
                spatial_size=configs.img_size,
                feat_mixing_hidden_channels=configs.feat_mixing_hidden_channels,
                no_mixer_layers=3,
                dropout=configs.drop,
                patch_size=configs.patch_size,
                embed_dim=configs.embed_dim,
            )

        else:
            raise ValueError("错误的网络名称，不存在%s这个网络" % configs.model)
        self.patch_size = configs.patch_size
        self.img_size = configs.img_size

    def forward(self, inputs, targets):
        # print(np.unique(inputs.cpu()))
        result = self.base_net(
            unfold_StackOverChannel(inputs, self.patch_size),
            unfold_StackOverChannel(targets, self.patch_size),
        )
        
        # 处理不同的返回值格式
        if len(result) == 2:
            # 检查第一个元素是否为元组（不确定性模型）
            if isinstance(result[0], tuple):
                # ConvNeXtV2等不确定性模型: ((mu, uncertainty), loss)
                (mu, uncertainty), loss = result
                mu = fold_tensor(mu, self.img_size, self.patch_size)
                uncertainty = fold_tensor(uncertainty, self.img_size, self.patch_size)
                return (mu, uncertainty), loss
            else:
                # 普通模型: (outputs, loss)
                outputs, loss = result
                outputs = fold_tensor(outputs, self.img_size, self.patch_size)
                return outputs, loss
        elif len(result) == 3:
            outputs, loss, distribution_params = result
            outputs = fold_tensor(outputs, self.img_size, self.patch_size)
            return outputs, loss, distribution_params
        else:
            raise ValueError(f"Unexpected number of return values: {len(result)}")
