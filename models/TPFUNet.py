import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from timm.models.layers import DropPath, to_2tuple
from torch.utils.checkpoint import checkpoint
import sys
import os
import math
import torch.fft # 引入傅里叶变换库

# --- 1. 环境检查与工具函数 (保持不变) ---
try:
    from natten.functional import na2d
    has_natten = False
except ImportError:
    has_natten = False

def convert_dilated_to_nondilated(kernel, d):
    k = kernel.shape[-1]
    large_k = (k - 1) * d + 1
    large_kernel = torch.zeros((kernel.shape[0], kernel.shape[1], large_k, large_k), dtype=kernel.dtype, device=kernel.device)
    for i in range(k):
        for j in range(k):
            large_kernel[:, :, i * d, j * d] = kernel[:, :, i, j]
    return large_kernel

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)

    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)
    if attempt_use_lk_impl and need_large_impl:
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            if DepthWiseConv2dImplicitGEMM is not None and in_channels == out_channels and out_channels == groups and stride == 1 and dilation == 1:
                return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
        except ImportError:
            pass
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     dilation=dilation, groups=groups, bias=bias)

def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)

def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    fused_weight = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    fused_bias = bn.bias + (conv_bias - bn.running_mean) * bn.weight / std
    return fused_weight, fused_bias

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_kernel.size(2) // 2 - equivalent_kernel.size(2) // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel

# --- 2. 基础组件层 (LayerNorm, SE, GRN, LayerScale) (保持不变) ---

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()

class GRN(nn.Module):
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

# --- 3. 结构化重参数模块 (DilatedReparamBlock) (保持不变) ---

class DilatedReparamBlock(nn.Module):
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2, dilation=1,
                                    groups=channels, bias=deploy, attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(f'dil_conv_k{k}_{r}',
                                 nn.Conv2d(channels, channels, kernel_size=k, stride=1, padding=(r * (k - 1) + 1) // 2,
                                           dilation=r, groups=channels, bias=False))
                self.__setattr__(f'dil_bn_k{k}_{r}', get_bn(channels, use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f'dil_conv_k{k}_{r}')
            bn = self.__getattr__(f'dil_bn_k{k}_{r}')
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__(f'dil_conv_k{k}_{r}')
                bn = self.__getattr__(f'dil_bn_k{k}_{r}')
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                     padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__(f'dil_conv_k{k}_{r}')
                self.__delattr__(f'dil_bn_k{k}_{r}')

# --- [NEW] 4. 新增：频域混合模块 (FrequencyMixer) ---

class FrequencyMixer(nn.Module):
    """
    频域混合器：将输入变换到频域，进行全局特征交互后再变换回空域。
    这提供了全局感受野，与 Attn 和 Conv 形成互补。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 使用 1x1 卷积在频域进行通道融合
        # 输入维度为 2*dim 是因为我们将实部和虚部拼接在一起处理
        self.freq_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim * 2, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 2D 快速傅里叶变换 (RFFT)
        # 结果形状: [B, C, H, W//2 + 1] (复数)
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        # 2. 拼接实部和虚部 -> [B, 2C, H, W//2 + 1]
        x_cat = torch.cat([x_fft.real, x_fft.imag], dim=1)
        
        # 3. 频域特征处理 (全局交互)
        # 1x1 Conv 可以在不同频率分量间共享权重，具有平移不变性
        freq_feat = self.freq_conv(x_cat)
        
        # 4. 拆分回实部和虚部
        real, imag = torch.chunk(freq_feat, 2, dim=1)
        x_fft_new = torch.complex(real, imag)
        
        # 5. 2D 逆快速傅里叶变换 (IRFFT)
        # 必须指定输出尺寸 s=(H, W)，因为 rfft2 对于奇偶宽度的输入可能产生相同的输出形状
        x_out = torch.fft.irfft2(x_fft_new, s=(H, W), dim=(-2, -1), norm='ortho')
        
        return x_out

# --- 5. 辅助模块 (h_sigmoid, h_swish, CoordAtt) (保持不变) ---

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class TPFU(nn.Module):
    def __init__(self, dim, kernel_size=7, smk_size=5, num_heads=2, mlp_ratio=4.0, 
                 res_scale=False, ls_init_value=1e-6, drop_path=0., 
                 norm_layer=LayerNorm2d, use_gemm=False, deploy=False, 
                 use_checkpoint=False, use_grn=True):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        self.smk_size = smk_size
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.res_scale = res_scale
        
        self.norm1 = norm_layer(dim)

        # --- Path A: Robust Attention Mechanism ---
        self.weight_query = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1, bias=False),
            nn.BatchNorm2d(dim // 2)
        )
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(dim, dim // 2, 1, bias=False),
            nn.BatchNorm2d(dim // 2)
        )
        self.weight_proj = nn.Conv2d(49, kernel_size ** 2 + smk_size ** 2, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 1.0)
        self.scale = (dim // num_heads) ** -0.5

        self.get_rpb()

        # --- Path B: Structural (Reparam) Branch ---
        # [修改]：合并了原有的 Local Mixer (Conv7x7) 和 Reparam 模块。
        # 使用 DilatedReparamBlock 作为统一的结构化局部特征提取器。
        # 它包含了大核卷积和多尺度空洞卷积，能力覆盖了简单的 Conv7x7。
        self.structural_mixer = DilatedReparamBlock(dim, kernel_size=19, deploy=deploy, 
                                      use_sync_bn=False, attempt_use_lk_impl=use_gemm)
        
        # --- [NEW] Path C: Frequency Domain Branch ---
        # [新增]：频域分支，用于提取全局频域特征
        self.freq_mixer = FrequencyMixer(dim)
        
        # Fusion Projection
        self.fusion_proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1)
        )
        
        self.spatial_robust_att = CoordAtt(dim, reduction=16)
        
        self.ls1 = LayerScale(dim, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # MLP with GRN
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(mlp_ratio * dim), 1),
            nn.GELU(),
            GRN(int(mlp_ratio * dim)) if use_grn else nn.Identity(),
            nn.Conv2d(int(mlp_ratio * dim), dim, 1)
        )
        self.ls2 = LayerScale(dim, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.trunc_normal_(self.rpb1, std=0.02)
        nn.init.trunc_normal_(self.rpb2, std=0.02)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size // 2] = height - (kernel_size - 1)
        num_repeat_w[kernel_size // 2] = width - (kernel_size - 1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (
                    2 * kernel_size - 1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size ** 2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size ** 2))
        return attn + rpb

    def _forward_mixer(self, x):
        identity = x
        x = self.norm1(x)
        
        B, C, H, W = x.shape
        
        # --- Path A: Robust Attention ---
        query = self.weight_query(x) * self.scale
        key = self.weight_key(x)

        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        
        weight_feat = einsum(query, key, 'b g c n, b g c l -> b g n l')
        weight_feat = weight_feat / torch.clamp(self.temperature, min=0.01, max=5.0)

        weight_feat = rearrange(weight_feat, 'b g n l -> b l g n').contiguous()
        weight_feat = self.weight_proj(weight_feat) 
        weight_feat = rearrange(weight_feat, 'b l g (h w) -> b g h w l', h=H, w=W)

        attn1_w, attn2_w = torch.split(weight_feat, [self.smk_size ** 2, self.kernel_size ** 2], dim=-1)
        
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        
        attn1 = self.apply_rpb(attn1_w, self.rpb1.to(x.device), H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2_w, self.rpb2.to(x.device), H, W, self.kernel_size, *rpb2_idx)
        
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)

        v = self.v_proj(x)
        v_heads = rearrange(v, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)
        
        if has_natten:
            out_attn1 = na2d(attn1, v_heads[0], kernel_size=self.smk_size)
            out_attn2 = na2d(attn2, v_heads[1], kernel_size=self.kernel_size)
        else:
            pad1, pad2 = self.smk_size // 2, self.kernel_size // 2
            v1 = F.pad(v_heads[0].flatten(0, 1).permute(0, 3, 1, 2), (pad1,)*4, mode='replicate')
            v2 = F.pad(v_heads[1].flatten(0, 1).permute(0, 3, 1, 2), (pad2,)*4, mode='replicate')
            
            v1_unfold = F.unfold(v1, self.smk_size).view(B, self.num_heads, C//2//self.num_heads, self.smk_size**2, H, W)
            v2_unfold = F.unfold(v2, self.kernel_size).view(B, self.num_heads, C//2//self.num_heads, self.kernel_size**2, H, W)
            
            out_attn1 = torch.einsum('bghwk, bghwck -> bghwc', attn1, v1_unfold.permute(0,1,4,5,2,3))
            out_attn2 = torch.einsum('bghwk, bghwck -> bghwc', attn2, v2_unfold.permute(0,1,4,5,2,3))

        path_a = torch.cat([out_attn1, out_attn2], dim=-1) 
        path_a = rearrange(path_a, 'b g h w c -> b (g c) h w')

        # --- Path B: Merged Structural Branch ---
        path_b = self.structural_mixer(v)

        # --- Path C: Frequency Branch ---
        path_c = self.freq_mixer(v)

        # --- Fusion ---
        mixed = path_a + path_b + path_c
        mixed = self.fusion_proj(mixed)
        mixed = self.spatial_robust_att(mixed)
        
        return identity + self.drop_path(self.ls1(mixed))

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self._forward_mixer, x, use_reentrant=False)
        else:
            x = self._forward_mixer(x)
        
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(self.ls2(x))
        return x

    def reparm(self):
        # 只需要对合并后的 structural_mixer 进行重参数化
        if hasattr(self.structural_mixer, 'merge_dilated_branches'):
            self.structural_mixer.merge_dilated_branches()

# --- 7. 后续 Encoder / Decoder / Main Model 保持不变 ---

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, upsampling=False, act_norm=False, act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(
                *[nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
                  nn.PixelShuffle(2)]
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, downsampling=False, upsampling=False, act_norm=True, act_inplace=True):
        super(ConvSC, self).__init__()
        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, upsampling=upsampling, padding=padding, act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        return self.conv(x)

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0], act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s, act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):
        enc_skips = []
        latent = x
        for layer in self.enc:
            latent = layer(latent)
            enc_skips.append(latent)
        return enc_skips[-1], enc_skips[:-1]

class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True, uncertainty_type='gaussian'):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.uncertainty_type = uncertainty_type
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s, act_inplace=act_inplace) for s in samplings[:-1]],
            ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1], act_inplace=act_inplace)
        )
        self.readout_mu = nn.Conv2d(C_hid, C_out, 1)
        self.readout_uncertainty = nn.Conv2d(C_hid, C_out, 1)
        self.bias_correction = nn.Conv2d(C_out, C_out, kernel_size=1)

    def forward(self, hid, enc_skips=None):
        if enc_skips is None: enc_skips = []
        enc_skips = enc_skips[::-1]
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
            if i < len(enc_skips):
                hid = hid + enc_skips[i]
        hid = self.dec[-1](hid)
        mu = torch.sigmoid(self.readout_mu(hid))
        mu = mu + torch.tanh(self.bias_correction(mu)) * 0.1
        mu = torch.clamp(mu, 0, 1)
        uncertainty = self.readout_uncertainty(hid)
        if self.uncertainty_type == 'gaussian':
            uncertainty = torch.log(1 + torch.exp(uncertainty)) + 1e-3
        elif self.uncertainty_type == 'laplacian':
            uncertainty = torch.exp(uncertainty) + 1e-3
        return mu, uncertainty

class MetaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=4.0, use_grn=True, drop_path=0.0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block = TPFU(
            dim=in_channels,
            kernel_size=7,
            smk_size=5,
            num_heads=4,
            mlp_ratio=mlp_ratio,
            res_scale=False,
            ls_init_value=1e-6,
            drop_path=drop_path,
            use_grn=use_grn
        )

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)

class MidMetaNet(nn.Module):
    def __init__(self, channel_in, channel_hid, N2, mlp_ratio=4.0, use_grn=True, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        enc_layers = [MetaBlock(channel_in, channel_hid, mlp_ratio, use_grn, drop_path=dpr[0])]
        for i in range(1, N2 - 1):
            enc_layers.append(MetaBlock(channel_hid, channel_hid, mlp_ratio, use_grn, drop_path=dpr[i]))
        enc_layers.append(MetaBlock(channel_hid, channel_in, mlp_ratio, use_grn, drop_path=drop_path))
        
        self.enc = nn.Sequential(*enc_layers)
        
        self.bottleneck = TPFU(
            dim=channel_in, 
            kernel_size=7, 
            smk_size=5, 
            num_heads=4, 
            mlp_ratio=mlp_ratio, 
            res_scale=False, 
            ls_init_value=1e-6, 
            drop_path=drop_path, 
            norm_layer=LayerNorm2d, 
            use_gemm=False, 
            deploy=False, 
            use_checkpoint=False,
            use_grn=use_grn
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        z = self.bottleneck(z)
        y = z.reshape(B, T, C, H, W)
        return y
    
    def reparm(self):
        for m in self.modules():
            if hasattr(m, 'reparm'):
                m.reparm()
            if hasattr(m, 'merge_dilated_branches'):
                m.merge_dilated_branches()

class TPFUNet(nn.Module):
    def __init__(self, T=12, C=1, hid_S=32, hid_T=512, N_S=6, N_T=6, spatio_kernel_enc=3, spatio_kernel_dec=3,
                 act_inplace=False, mlp_ratio=4.0, use_grn=True, drop_path=0.1, uncertainty_type='gaussian'):
        super(TPFUNet, self).__init__()
        self.uncertainty_type = uncertainty_type
        self.T = T
        self.C = C
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace, uncertainty_type=uncertainty_type)
        self.hid = MidMetaNet(
            hid_S * T,
            hid_T,
            N_T,
            mlp_ratio=mlp_ratio,
            use_grn=use_grn,
            drop_path=drop_path,
        )

    def gaussian_log_likelihood(self, z, mu, sigma):
        sigma_stable = torch.clamp(sigma, min=1e-4, max=10.0)
        log_sigma = torch.log(sigma_stable)
        log_2pi = torch.log(torch.tensor(2 * math.pi, device=sigma.device, dtype=sigma.dtype))
        return -0.5 * (log_2pi + 2 * log_sigma + (z - mu)**2 / (sigma_stable**2))

    def laplacian_log_likelihood(self, z, mu, b):
        b_stable = torch.clamp(b, min=1e-4, max=10.0)
        log_b = torch.log(b_stable)
        return -torch.log(torch.tensor(2.0)) - log_b - torch.abs(z - mu) / b_stable

    def forward(self, input_x, targets=None, is_training=True, return_ci=False):
        B, T, C, H, W = input_x.shape
        
        x_batch = input_x.view(B * T, C, H, W)
        embed_batch, skip_batch = self.enc(x_batch)
        
        _, C_, H_, W_ = embed_batch.shape
        embed = embed_batch.view(B, T, C_, H_, W_)
        
        hid = self.hid(embed)
        
        hid_batch = hid.view(B * T, C_, H_, W_)
        param_mu_batch, param_uncertainty_batch = self.dec(hid_batch, skip_batch)
        
        param_mu = param_mu_batch.view(B, T, C, H, W)
        param_uncertainty = param_uncertainty_batch.view(B, T, C, H, W)
        
        mu, uncertainty = param_mu, param_uncertainty
        
        if targets is not None:
            if self.uncertainty_type == 'gaussian':
                log_likelihood = self.gaussian_log_likelihood(targets, mu, uncertainty)
            elif self.uncertainty_type == 'laplacian':
                log_likelihood = self.laplacian_log_likelihood(targets, mu, uncertainty)
            neg_log_likelihood_loss = -torch.mean(log_likelihood)
            return (mu, uncertainty), neg_log_likelihood_loss
        else:
            lower, upper = None, None
            if return_ci:
                lower, upper = self.compute_confidence_interval(mu, uncertainty)
                return mu, uncertainty, lower, upper
            else:
                return mu, uncertainty

    def compute_confidence_interval(self, mu, uncertainty, confidence_level=0.95):
        if self.uncertainty_type == 'gaussian':
            if confidence_level == 0.95: z = 1.96
            elif confidence_level == 0.99: z = 2.576
            elif confidence_level == 0.90: z = 1.645
            else:
                alpha = 1 - confidence_level
                z = (-2 * torch.log(torch.tensor(alpha/2))).sqrt().item()
            lower = torch.clamp(mu - z * uncertainty, 0, 1)
            upper = torch.clamp(mu + z * uncertainty, 0, 1)
        elif self.uncertainty_type == 'laplacian':
            alpha = 1 - confidence_level
            p_lower = alpha / 2
            p_upper = 1 - alpha / 2
            lower_quantile = mu + uncertainty * torch.log(torch.tensor(2 * p_lower))
            upper_quantile = mu - uncertainty * torch.log(torch.tensor(2 * (1 - p_upper)))
            lower = torch.clamp(lower_quantile, 0, 1)
            upper = torch.clamp(upper_quantile, 0, 1)
        return lower, upper
    
    def switch_to_deploy(self):
        """推理加速：合并重参数化分支"""
        if hasattr(self.hid, 'reparm'):
            self.hid.reparm()