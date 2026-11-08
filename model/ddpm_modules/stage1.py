import numpy as np
from pydantic import PositiveInt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange
from torch.autograd import Function
import pywt
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = int(in_batch/(r**2)),in_channel, r * in_height, r * in_width
    x1 = x[0:out_batch, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class OverlapPatchEmbed(nn.Module):#重叠嵌入
    def __init__(self, in_c=3, embed_dim=48, bias=False):#embed_dim：输入数据转换为固定大小的向量表示时，这个向量的长度或维度
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)#输入图像转换为特征图

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class MySequential(nn.Sequential):#处理单个输入和元组形式的多个输入
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)#输入是元组（tuple），则使用*inputs将元组解包后传递给模块
            else:
                inputs = module(inputs)
        return inputs 


class Depth_conv(nn.Module):#深度卷积
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
    
class FeedForwardd(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForwardd, self).__init__()

        self.x1_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x1_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x2_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x2_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x3_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x3_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x4_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x4_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)
        self.x5_Conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.x5_Conv5 = nn.Conv2d(dim, dim, kernel_size=5,padding=2, bias=bias)

        self.outCon = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        x1 = self.x1_Conv5(self.x1_Conv1(x))
        x2 = self.x2_Conv5(self.x2_Conv1(x))
        x3 = self.x3_Conv5(self.x3_Conv1(x))

        x1 = x2 + self.x4_Conv5(self.x4_Conv1(F.gelu(x1) * x2))
        x3 = x2 + self.x5_Conv5(self.x5_Conv1(F.gelu(x3) * x2))

        return self.outCon(torch.concat([x1,x3],dim=1))
    
class FeedForwards(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForwards, self).__init__()

        hidden_features = int(dim * ffn_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x
    
class FeedForwardm(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForwardm, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))


    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        # print(x.dtype)
        #
        w_ll = w_ll.to(dtype=x.dtype)
        w_lh = w_lh.to(dtype=x.dtype)
        w_hl = w_hl.to(dtype=x.dtype)
        w_hh = w_hh.to(dtype=x.dtype)
        #
        
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)
class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        #
        filters = filters.to(dtype=x.dtype)
        # print(torch.eq(filters_, filters))
        #
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            #
            filters = filters.to(dtype=dx.dtype)
            # print(torch.eq(filters_, filters))
            #            
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.dim = dim

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SpectralProcessor(nn.Module):
    """频域专用处理器"""
    def __init__(self, channels):
        super().__init__()
        # self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv = nn.Conv2d(channels, channels, 1, padding=0)  
        
    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x)
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        
        mag = self.conv(mag)
        pha = self.conv(pha)
        
        return torch.fft.irfft2(torch.polar(mag, pha), s=(H, W))

class CrossDomainGate(nn.Module):
    """跨域门控注意力"""
    def __init__(self, channels):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels//8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.freq_att = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels//8, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, spatial_feat, freq_feat):
        # 空间域指导频域
        spatial_guide = self.spatial_att(spatial_feat)
        # 频域指导空间域
        freq_guide = self.freq_att(freq_feat)
        
        return freq_guide, spatial_guide  # 交换指导信号
    
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class SpatialDomainfeatures1(nn.Module):
    def __init__(self, nChannels):
        super().__init__()
        # 多尺度卷积（简化padding计算）
        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(nChannels, nChannels, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(nChannels, nChannels, kernel_size=7, padding=3, bias=False)
        self.detail_conv = nn.Conv2d(nChannels * 3, nChannels, kernel_size=1, padding=0, bias=False)  
        self.norm = nn.InstanceNorm2d(nChannels * 3) 
        self.ca = ChannelAttention(nChannels * 3)
        self.conv6 = nn.Conv2d(nChannels * 3, nChannels, kernel_size=1, padding=0, bias=False)
        self.gelu = nn.GELU() 
        
    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_cat = self.norm(x_cat)
        x_attn = self.ca(x_cat)  
        x_fused = self.gelu(self.conv6(x_attn))
        x_detail = self.gelu(self.detail_conv(x_attn))  
        p2p = x_fused + x_detail + x                 
        return p2p

class SpatialDomainfeatures(nn.Module):
    def __init__(self, nChannels):
        super().__init__()
        # 多尺度卷积（简化padding计算）
        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(nChannels, nChannels, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(nChannels, nChannels, kernel_size=7, padding=3, bias=False)
        self.detail_conv = nn.Conv2d(nChannels * 3, nChannels, kernel_size=1, padding=0, bias=False)  
        self.norm = nn.InstanceNorm2d(nChannels * 3) 
        self.ca = ChannelAttention(nChannels * 3)
        self.conv6 = nn.Conv2d(nChannels * 3, nChannels, kernel_size=1, padding=0, bias=False)
        self.gelu = nn.GELU() 


    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_cat = self.norm(x_cat)
        x_attn = self.ca(x_cat)  
        x_fused = self.gelu(self.conv6(x_attn))
        return x_fused + x 
    

from einops import rearrange  # 用于维度重排（需安装einops库：pip install einops）

class LayerNorm2d(nn.Module):
    """
    适配2D特征图的LayerNorm，输入形状：(B, C, H, W)
    对通道维度（C）进行归一化，保持空间维度（H, W）不变
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)  # 核心归一化层（对最后一维）
        self.dim = dim  # 通道数

    def forward(self, x):
        # 输入x形状：(B, C, H, W)
        # 步骤1：维度重排为(B, H, W, C)，让通道维在最后（适配LayerNorm）
        x = rearrange(x, "b c h w -> b h w c")
        # 步骤2：对通道维（最后一维）归一化
        x = self.norm(x)
        # 步骤3：转回原形状(B, C, H, W)
        return rearrange(x, "b h w c -> b c h w")

class Swish(nn.Module):
    """
    Swish激活函数：x * sigmoid(x)
    自带门控机制，输出范围(-∞, +∞)，比ReLU更平滑
    """
    def forward(self, x):
        return x * torch.sigmoid(x)  # 核心计算：x乘以sigmoid(x)

    
class DirectionAwareHighFreqAttention(nn.Module):
    def __init__(self, dim, num_directions=4, bias=True):  # num_directions改为4（3方向+1普通）
        super().__init__()
        # 1. 方向卷积：垂直、水平、对角线 + 新增普通卷积分支
        self.direction_convs = nn.ModuleList([
            # 垂直方向卷积（检测垂直边缘）
            nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=bias),
            # 水平方向卷积（检测水平边缘）
            nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=bias),
            # 对角线方向卷积（检测45°/135°边缘）
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias, groups=dim),
            # 新增：普通3x3卷积分支（捕捉局部综合语义，补充无方向细节）
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        ])
        
        # 2. 初始化对角线卷积核
        self._init_diagonal_kernel()
        
        # 3. 注意力层：适配num_directions=4（总通道数=dim*4）
        total_channels = dim * num_directions
        # 分组数设为dim（确保total_channels能被dim整除，避免通道不匹配错误）
        self.attn = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, 1, 1, 0, groups=4),
            nn.Sigmoid()
        )
        
        # 4. 融合层：将4个分支的特征融合回原dim通道
        self.fusion = nn.Conv2d(total_channels, dim, kernel_size=1, bias=bias)
        # 高频增强系数（可学习）
        self.high_boost = nn.Parameter(torch.tensor(0.1))

    def _init_diagonal_kernel(self):
        """初始化对角线卷积核（增强45°/135°方向响应）"""
        diag_conv = self.direction_convs[2]  # 第3个是对角线卷积
        with torch.no_grad():
            diag_kernel = torch.tensor([
                [0., 1., 0.],
                [1., 0., 1.],
                [0., 1., 0.]
            ]).float()
            # 扩展为深度卷积需要的形状：[dim, 1, 3, 3]
            diag_kernel = diag_kernel.repeat(diag_conv.out_channels, 1, 1, 1)
            diag_conv.weight.copy_(diag_kernel)

    def forward(self, x):
        # 1. 提取4个分支的特征：3个方向特征 + 1个普通卷积特征
        features = [conv(x) for conv in self.direction_convs]  # 列表长度=4，每个元素形状[B,dim,H,W]
        
        # 2. 拼接特征：通道维度拼接，形状变为[B, dim*4, H, W]
        concat_feats = torch.cat(features, dim=1)
        
        # 3. 注意力加权：学习每个通道的重要性
        attn_weights = self.attn(concat_feats)  # 形状和concat_feats一致
        weighted_feats = concat_feats * attn_weights  # 逐通道加权
        
        # 4. 特征融合：从dim*4通道融合回dim通道
        fused_feats = self.fusion(weighted_feats)
        
        # 5. 残差连接 + 高频增强（可学习系数控制增强强度）
        return x + self.high_boost * fused_feats

class DTB1(nn.Module):
    def __init__(self, dim, ffn_factor, bias, LayerNorm_type):
        super(DTB1, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.MSA = DirectionAwareHighFreqAttention(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForwards(dim, ffn_factor, bias)

    def forward(self, x):
        residual = x        
        x_global = self.MSA(self.norm1(x))          
        x = self.ffn(self.norm2(x_global))      
        return x + residual

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def reflect(x, minx, maxx): #用于将数值矩阵中的值反射到指定的最小值和最大值范围内
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.
    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.
    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def mypad(x, pad, mode='constant', value=0): #用于对 PyTorch 张量进行类似 NumPy 的 2D 填充操作
    """ Function to do numpy like padding on tensors. Only works for 2-D
    padding.
    Inputs:
        x (tensor): tensor to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
        mode (str): 'symmetric', 'wrap', 'constant, 'reflect', 'replicate', or
            'zero'. The padding technique.
    """
    if mode == 'symmetric':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,xe]
        # horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l+m2, dtype='int32'), -0.5, l-0.5)
            return x[:,:,:,xe]
        # Both
        else:
            m1, m2 = pad[0], pad[1]
            l1 = x.shape[-1]
            xe_row = reflect(np.arange(-m1, l1+m2, dtype='int32'), -0.5, l1-0.5)
            m1, m2 = pad[2], pad[3]
            l2 = x.shape[-2]
            xe_col = reflect(np.arange(-m1, l2+m2, dtype='int32'), -0.5, l2-0.5)
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]
    elif mode == 'periodic':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            xe = np.arange(x.shape[-2])
            xe = np.pad(xe, (pad[2], pad[3]), mode='wrap')
            return x[:,:,xe]
        # Horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            xe = np.arange(x.shape[-1])
            xe = np.pad(xe, (pad[0], pad[1]), mode='wrap')
            return x[:,:,:,xe]
        # Both
        else:
            xe_col = np.arange(x.shape[-2])
            xe_col = np.pad(xe_col, (pad[2], pad[3]), mode='wrap')
            xe_row = np.arange(x.shape[-1])
            xe_row = np.pad(xe_row, (pad[0], pad[1]), mode='wrap')
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:,:,i,j]

    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        return F.pad(x, pad)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))
    
def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None): #准备二维分解滤波器组
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.
    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to
    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0_col = np.array(h0_col[::-1]).ravel()
    h1_col = np.array(h1_col[::-1]).ravel()
    t = torch.get_default_dtype()
    if h0_row is None:
        h0_row = h0_col
    else:
        h0_row = np.array(h0_row[::-1]).ravel()
    if h1_row is None:
        h1_row = h1_col
    else:
        h1_row = np.array(h1_row[::-1]).ravel()
    h0_col = torch.tensor(h0_col, device=device, dtype=t).reshape((1,1,-1,1))
    h1_col = torch.tensor(h1_col, device=device, dtype=t).reshape((1,1,-1,1))
    h0_row = torch.tensor(h0_row, device=device, dtype=t).reshape((1,1,1,-1))
    h1_row = torch.tensor(h1_row, device=device, dtype=t).reshape((1,1,1,-1))

    return h0_col, h1_col, h0_row, h1_row 

def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None): #准备二维合成滤波器组
    """
    Prepares the filters to be of the right form for the sfb2d function.  In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.
    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to
    Returns:
        (g0_col, g1_col, g0_row, g1_row)
    """
    g0_col = np.array(g0_col).ravel()
    g1_col = np.array(g1_col).ravel()
    t = torch.get_default_dtype()
    if g0_row is None:
        g0_row = g0_col
    if g1_row is None:
        g1_row = g1_col
    g0_col = torch.tensor(g0_col, device=device, dtype=t).reshape((1,1,-1,1))
    g1_col = torch.tensor(g1_col, device=device, dtype=t).reshape((1,1,-1,1))
    g0_row = torch.tensor(g0_row, device=device, dtype=t).reshape((1,1,1,-1))
    g1_row = torch.tensor(g1_row, device=device, dtype=t).reshape((1,1,1,-1))

    return g0_col, g1_col, g0_row, g1_row

def afb1d_atrous(x, h0, h1, mode='symmetric', dim=-1, dilation=1): #对图像进行非下采样的小波变换
    """ 1D analysis filter bank (along one dimension only) of an image without
    downsampling. Does the a trous algorithm.
    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).
        dilation (int): dilation factor. Should be a power of 2.
    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    # Calculate the pad size
    L2 = (L * dilation)//2
    pad = (0, 0, L2-dilation, L2) if d == 2 else (L2-dilation, L2, 0, 0)
    #ipdb.set_trace()
    x = mypad(x, pad=pad, mode=mode)
    lohi = F.conv2d(x, h, groups=C, dilation=dilation)
    return lohi #(N, 2*C, H, W)

def afb2d_atrous(x, filts, mode='symmetric', dilation=1): #二维非下采样小波变换
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to `afb1d_atrous`
    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by `prep_filt_afb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling `prep_filt_afb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
        dilation (int): dilation factor for the filters. Should be 2**level
    Returns:
        y: Tensor of shape (N, C, 4, H, W)
    """
    tensorize = [not isinstance(f, torch.Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                h0, h1, device=x.device)
        else:
            h0_col = h0
            h0_row = h0.transpose(2,3)
            h1_col = h1
            h1_row = h1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                *filts, device=x.device)
        else:
            h0_col, h1_col, h0_row, h1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lohi = afb1d_atrous(x, h0_row, h1_row, mode=mode, dim=3, dilation=dilation)
    y = afb1d_atrous(lohi, h0_col, h1_col, mode=mode, dim=2, dilation=dilation)

    return y 

def sfb1d_atrous(lo, hi, g0, g1, mode='symmetric', dim=-1, dilation=1, #一维非下采样合成滤波器
                 pad1=None, pad=None):
    """ 1D synthesis filter bank of an image tensor with no upsampling. Used for
    the stationary wavelet transform.
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)
    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)

    # Calculate the padding size.
    # With dilation, zeros are inserted between the filter taps but not after.
    # that means a filter that is [a b c d] becomes [a 0 b 0 c 0 d].
    centre = L / 2
    fsz = (L-1)*dilation + 1
    newcentre = fsz / 2
    before = newcentre - dilation*centre

    # When conv_transpose2d is done, a filter with k taps expands an input with
    # N samples to be N + k - 1 samples. The 'padding' is really the opposite of
    # that, and is how many samples on the edges you want to cut out.
    # In addition to this, we want the input to be extended before convolving.
    # This means the final output size without the padding option will be
    #   N + k - 1 + k - 1
    # The final thing to worry about is making sure that the output is centred.
    short_offset = dilation - 1
    centre_offset = fsz % 2
    a = fsz//2
    b = fsz//2 + (fsz + 1) % 2
    
    #pad = (0, 0, a, b) if d == 2 else (a, b, 0, 0)
    pad = (0, 0, b, a) if d == 2 else (b, a, 0, 0)
    lo = mypad(lo, pad=pad, mode=mode)
    hi = mypad(hi, pad=pad, mode=mode)

    #unpad = (fsz - 1, 0) if d == 2 else (0, fsz - 1)
    unpad = (fsz, 0) if d == 2 else (0, fsz)
    
    y = F.conv_transpose2d(lo, g0, padding=unpad, groups=C, dilation=dilation) + \
        F.conv_transpose2d(hi, g1, padding=unpad, groups=C, dilation=dilation)

    return y/(2*dilation)  #(N, C, H, W)

def sfb2d_atrous(ll, lh, hl, hh, filts, mode='symmetric'):
    """ Does a single level 2d wavelet reconstruction of wavelet coefficients.
    Does separate row and column filtering by two calls to `sfb1d_atrous`
    Inputs:
        ll (torch.Tensor): lowpass coefficients
        lh (torch.Tensor): horizontal coefficients
        hl (torch.Tensor): vertical coefficients
        hh (torch.Tensor): diagonal coefficients
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by `prep_filt_sfb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling `prep_filt_sfb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    tensorize = [not isinstance(x, torch.Tensor) for x in filts]
    if len(filts) == 2:
        g0, g1 = filts
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(g0, g1)
        else:
            g0_col = g0
            g0_row = g0.transpose(2,3)
            g1_col = g1
            g1_row = g1.transpose(2,3)
    elif len(filts) == 4:
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(*filts)
        else:
            g0_col, g1_col, g0_row, g1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lo = sfb1d_atrous(ll, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d_atrous(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d_atrous(lo, hi, g0_row, g1_row, mode=mode, dim=3)

    return y  #(N, C, H, W)
class SWTForward(nn.Module):
    """ Performs a 2d Stationary wavelet transform (or undecimated wavelet
    transform) of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme. PyWavelets uses only periodization so we use this
            as our default scheme.
        """
    def __init__(self, J=1, wave='db1', mode='symmetric'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.h0_col = nn.Parameter(filts[0], requires_grad=False)
        self.h1_col = nn.Parameter(filts[1], requires_grad=False)
        self.h0_row = nn.Parameter(filts[2], requires_grad=False)
        self.h1_row = nn.Parameter(filts[3], requires_grad=False)

        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the SWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            List of coefficients for each scale. Each coefficient has
            shape :math:`(N, C_{in}, 4, H_{in}, W_{in})` where the extra
            dimension stores the 4 subbands for each scale. The ordering in
            these 4 coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
        """

        device = x.device
        h0_col = self.h0_col.to(device)
        h1_col = self.h1_col.to(device)
        h0_row = self.h0_row.to(device)
        h1_row = self.h1_row.to(device)
        ll = x
        filts = (h0_col, h1_col, h0_row, h1_row)
        for j in range(self.J):
            y = afb2d_atrous(ll, filts, self.mode)
        return y
    
class SWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image
    Args:
        wave (str or pywt.Wavelet): Which wavelet to use
        C: deprecated, will be removed in future
    """
    def __init__(self, wave='db1', mode='symmetric'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
        # Prepare the filters
        
        filts = prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.g0_col = nn.Parameter(filts[0], requires_grad=False)
        self.g1_col = nn.Parameter(filts[1], requires_grad=False)
        self.g0_row = nn.Parameter(filts[2], requires_grad=False)
        self.g1_row = nn.Parameter(filts[3], requires_grad=False)

        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        ll = coeffs[:, 0:3, :, :]  # [B,3,H,W]
        lh = coeffs[:, 3:6, :, :]  # [B,3,H,W]
        hl = coeffs[:, 6:9, :, :]  # [B,3,H,W]
        hh = coeffs[:, 9:12, :, :] # [B,3,H,W]

        filts = (self.g0_col, self.g1_col, self.g0_row, self.g1_row)
        recon_img = sfb2d_atrous(ll, lh, hl, hh, filts, mode=self.mode) 
        return recon_img

from .color_space import *
class Pre_enhance2(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[1, 2, 4, 6],
                 ffn_factor = 2.0,   
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Pre_enhance2, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.patch_embed1 = OverlapPatchEmbed(9, dim)

        self.encoder_level1 =  SpatialDomainfeatures(nChannels=dim) 
        self.encoder_level11 = MySequential(*[
            DTB1(dim=dim, ffn_factor=ffn_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.down1_21 = Downsample(dim)
        
        self.encoder_level2 = SpatialDomainfeatures(nChannels=dim * 2 ** 1)
        self.encoder_level21 = MySequential(*[
            DTB1(dim=int(dim * 2 ** 1), ffn_factor=ffn_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.down2_31 = Downsample(int(dim * 2 ** 1))

        self.latent = SpatialDomainfeatures(nChannels=dim * 2 ** 2)
        self.latent1 = MySequential(*[
             DTB1(dim=int(dim * 2 ** 2), ffn_factor=ffn_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.up3_21 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level21 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2), int(dim * 1), kernel_size=1, bias=bias)
        self.reduce_chan_level11 = nn.Conv2d(int(dim * 2), int(dim * 1), kernel_size=1, bias=bias)


        self.decoder_level2 = SpatialDomainfeatures(nChannels=dim * 2 ** 1)
        self.decoder_level21 = MySequential(*[
            DTB1(dim=int(dim * 2 ** 1), ffn_factor=ffn_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_11 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = SpatialDomainfeatures(nChannels=dim)
        self.decoder_level11 = MySequential(*[
            DTB1(dim=int(dim), ffn_factor=ffn_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])


        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output1 = nn.Conv2d(int(dim), 9, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.upH1 = Upsample(int(dim * 2 ** 3))
        self.upL1 = Upsample(int(dim * 2 ** 3))
        self.finalproj = nn.Conv2d(2 * out_channels, out_channels, 1, 1, 0)

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
    
    def forward(self, inp_img):

        swt = SWTForward(J=1).to(inp_img.device)
        iswt = SWTInverse().to(inp_img.device)
        wavelet_inp = swt(inp_img)
        B = inp_img.shape[0]
        # coeffs = wavelet_inp.view(B, 3, 4, 256, 256) 
        if len(wavelet_inp.shape) == 5:
            # 如果是(N, C, 4, H, W)格式
            _, C, _, H, W = wavelet_inp.shape
            coeffs = wavelet_inp  # 直接使用返回的形状 (B, C, 4, H, W)
        elif len(wavelet_inp.shape) == 4:
            # 如果是(N, 4*C, H, W)格式，需要重新reshape
            _, CHW, H, W = wavelet_inp.shape
            C = CHW // 4  # 每个子带的通道数
            coeffs = wavelet_inp.view(B, C, 4, H, W)
        else:
            raise ValueError(f"Unexpected wavelet input shape: {wavelet_inp.shape}")
        LL_band = coeffs[:,:,0,:,:] 
        LH_band = coeffs[:,:,1,:,:]
        HL_band = coeffs[:,:,2,:,:]
        HH_band = coeffs[:,:,3,:,:]

        input_highs = torch.cat([LH_band, HL_band, HH_band], dim=1) 

        inp_enc_level1 = self.patch_embed(LL_band)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        inp_enc_level11 = self.patch_embed1(input_highs)
        out_enc_level11 = self.encoder_level11(inp_enc_level11)
        inp_enc_level21 = self.down1_21(out_enc_level11)
        out_enc_level21 = self.encoder_level21(inp_enc_level21)
        inp_enc_level31 = self.down2_31(out_enc_level21)

        latent = self.latent(inp_enc_level3)
        latent1 = self.latent1(inp_enc_level31)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)#192
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)#96
        out_dec_level2= self.decoder_level2(inp_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)#96
        out_dec_level1 = self.output(out_dec_level1)

        inp_dec_level21 = self.up3_21(latent1)
        inp_dec_level21 = torch.cat([inp_dec_level21, out_enc_level21], 1)
        inp_dec_level21 = self.reduce_chan_level21(inp_dec_level21)
        out_dec_level21 = self.decoder_level21(inp_dec_level21)
        inp_dec_level11 = self.up2_11(out_dec_level21)
        inp_dec_level11 = torch.cat([inp_dec_level11, out_enc_level11], 1)
        inp_dec_level11 = self.reduce_chan_level11(inp_dec_level11)
        out_dec_level11 = self.decoder_level11(inp_dec_level11)
        out_dec_level11 = self.output1(out_dec_level11)
        coeffs = torch.cat([out_dec_level1, out_dec_level11], dim=1)
        recon_img = iswt(coeffs) 
        return recon_img, out_dec_level11


        # lab_img = rgb2lab(final)
        # L1 = lab_img[:, 0:1, :, :]  # 亮度通道 [B, 1, H, W]
        # AB = lab_img[:, 1:3, :, :]  # 色彩通道 [B, 2, H, W]

        # import os
        # # 将 NumPy 数组转换为 PyTorch 张量
        # # print("Tensor min/max:",stage1.min(),stage1.max())
        # image_reconstructed_tensor = final.float()
        # x_img = tensor2img(image_reconstructed_tensor)  # 将张量转换为图像
        # save_path = os.path.join('results', 'x_image.png')  # 保存路径
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建目录
        # imwrite(x_img, save_path)  # 保存图像
        # print(f"x_ 图像已保存到: {save_path}")
