import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    # input = F.normalize(input, p=2, dim=1, eps=1e-12)
    # input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b)

# model



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules

import numbers

from einops import rearrange


##########################################################################
## Layer Norm

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

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
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


class PA_MSA1(nn.Module):#多头注意力
    def __init__(self, dim, num_heads, bias,t_dim = 3):
        super(PA_MSA1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        self.q_T = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv_T = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.q_concat = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)
        
        self.k_T = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv_T = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.k_concat = nn.Conv2d(2*dim, dim, kernel_size=1, bias=bias)

        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        
        k = self.k_dwconv(self.k(x))
       
        v = self.v_dwconv(self.v(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
       
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
       
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, time_emb_dim=None):
        super(TransformerBlock, self).__init__()
        self.time_mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim)
        ) if time_emb_dim is not None else None
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.atten = PA_MSA1(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForwardm(dim, ffn_expansion_factor, bias)

    def forward(self, x, time, HH):
        # print(x.shape)
        # print(HH.shape)
        x = x + HH
        # print(x.shape)
        x1 = x + self.atten(self.norm1(x)+time)
        x2 = x1 + self.ffn(self.norm2(x1)+time)
        return x2
    
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

class EnhancedCrossDomainGate(nn.Module):
    """增强版跨域门控注意力"""
    def __init__(self, channels):
        super().__init__()
        # 空间域注意力（更深的网络结构）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels//4, channels//8, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels//8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        # 频域注意力（添加通道注意力机制）
        self.freq_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        # 跨域融合门控
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels*2, channels//8, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels//8, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, spatial_feat, freq_feat):
        spatial_guide = self.spatial_att(spatial_feat)
        freq_guide = self.freq_att(freq_feat)
        
        # 动态融合门控
        fusion = self.fusion_gate(torch.cat([spatial_feat, freq_feat], dim=1))
        return freq_guide * fusion, spatial_guide * (1 - fusion)



class EnhancedFreBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhancedFreBlock, self).__init__()
        # 多尺度频域处理
        self.mag_processor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels*2, out_channels, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.pha_processor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.LeakyReLU(0.1)
        )
        # 残差连接
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x_ori = x
        _, _, H, W = x.shape
        
        # 傅里叶变换
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        
        # 多尺度处理
        mag = self.mag_processor(mag)
        pha = self.pha_processor(pha)
        
        # 逆变换
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        
        return x_out + self.residual(x_ori)

class ImprovedMixer4(nn.Module):
    def __init__(self, dim, time_emb_dim=None, **kwargs):
        super().__init__()
        self.dim = dim
        
        # 增强的频域处理模块
        self.freblock = EnhancedFreBlock(dim, dim)
        
        # 时间嵌入处理
        self.time_mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim*2),
            Swish(),
            nn.Linear(dim*2, dim)
        ) if time_emb_dim is not None else None
        
        self.cross_gate = EnhancedCrossDomainGate(dim)
        
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(dim*2, dim, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim, dim, 1)
        )
        self.final_proj = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim*2, dim, 1)
        )

    def forward(self, x, time, HH):
        x_ori = x + HH
        if time is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(time)
            x_ori = x_ori + time_emb.unsqueeze(2).unsqueeze(3)
        x_freq = self.freblock(x_ori)
        spatial_gate, freq_gate = self.cross_gate(x_ori, x_freq)
        x_spatial = x_ori * spatial_gate
        x_freq = x_freq * freq_gate
        fused = self.multi_scale_fusion(torch.cat([x_spatial, x_freq], dim=1))
        return self.final_proj(torch.cat([fused, x_ori], dim=1)) + x_ori

class Mixer4(nn.Module):
    def __init__(self, dim, time_emb_dim=None, **kwargs):
        super().__init__()
        
        self.dim = dim
        # self.freblock = EnhancedFreBlock(dim, dim)
        self.freblock = FreBlock(dim, dim)

        self.time_mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim)
        ) if time_emb_dim is not None else None

        self.finalproj = nn.Conv2d(2 * dim, dim, 1, 1, 0)
        self.cross_gate = CrossDomainGate(dim)

    def forward(self, x, time):

        x_ori = x
        if time is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(time)
            x_ori = x_ori + time_emb.unsqueeze(2).unsqueeze(3)
        x_freq = self.freblock(x_ori)

        spatial_gate, freq_gate = self.cross_gate(x_ori, x_freq)
        x_ori = x_ori * spatial_gate
        x_freq = x_freq * freq_gate
        fused = self.finalproj(torch.cat([x_ori, x_freq], dim=1))

        return fused + x_ori
      

    def _gaussian_filter(self, x):
        """生成高斯低通滤波器（用于提取低频分量）"""
        B, C, H, W = x.shape
        # 生成高斯核（中心低频，边缘高频衰减）
        xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W//2+1), indexing='ij')
        center = (H // 2, (W//2+1) // 2)
        sigma = min(H, W) * 0.1  # 高斯核标准差（控制低频范围）
        gaussian = torch.exp(-((xx - center[0])**2 + (yy - center[1])** 2) / (2 * sigma**2))
        gaussian = gaussian.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1).to(x.device)
        return gaussian

class FreBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(in_channels,out_channels,1,1,0),
            nn.LeakyReLU(0.1, inplace=True))
        self.processpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self,x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out1 = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out1

class Mixer3(nn.Module):
    def __init__(self, dim, time_emb_dim=None, **kwargs):
        super().__init__()
        
        self.dim = dim
        self.freblock = FreBlock(dim, dim)

        self.time_mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim)
        ) if time_emb_dim is not None else None

        self.finalproj = nn.Conv2d(2 * dim, dim, 1, 1, 0)

    def forward(self, x, time, HH):
        B, C, H, W = x.shape
        if time is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(time)
            x = x + HH + time_emb.unsqueeze(2).unsqueeze(3)

        x_freq = self.freblock(x)
        x_out = torch.cat((x, x_freq), 1)
        x_out = self.finalproj(x_out)
        return x_out + x
    

    
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
       
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)
        out = self.project_out(out)
        return out

class ResnetBloc_eca1(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, mid_layer=False, with_mixer=False, with_lowreso=False,with_ca=False):
        super().__init__()
        self.mid_layer = mid_layer
        self.with_mixer = with_mixer
        self.with_lowreso = with_lowreso
        self.with_ca = with_ca

        if mid_layer:
            self.mid = TransformerBlock(dim=int(dim), num_heads=dim, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias')
        if with_mixer:
            self.mixer = Mixer3(dim=int(dim))
        if with_lowreso:
            self.attn = ImprovedMixer4(dim=int(dim))
        if with_ca:
            self.ca = Mixer4(dim=int(dim))
        
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        )
    def forward(self, x, time_emb, HH):
        time = self.mlp(time_emb)[:, :, None, None]
        if self.mid_layer :
            x = self.mid(x, time, HH)
        if self.with_mixer :
            x = self.mixer(x, time, HH)
        if self.with_lowreso :
            x = self.attn(x, time, HH)
        if self.with_ca :
            x = self.ca(x,time)

        return x
    
class Encoder_stage6(nn.Module):#小波调制参数
    def __init__(
            self,
            in_channel=9,
            inner_channel=48,
            norm_groups=24,
            heads = [1,2,4,8]
    ):
        super().__init__()

        dim = inner_channel
        time_dim = inner_channel
        # x
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, dim, kernel_size=3, stride=1, padding=1, bias=False))

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
    
        self.block1 = ResnetBloc_eca1(dim=dim, dim_out=dim, time_emb_dim=time_dim, norm_groups=norm_groups,
                                     mid_layer=False, with_mixer=True, with_lowreso=False,with_ca=False)
        self.block2 = ResnetBloc_eca1(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, mid_layer=True, with_mixer=False, with_lowreso=False,with_ca=False)
        self.block3 = ResnetBloc_eca1(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, mid_layer=False, with_mixer=False, with_lowreso=True,with_ca=False)
 
        self.conv_up2 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 2), (dim * 2 ** 2) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_up1 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 1), (dim * 2 ** 1) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))
        
        self.conv_up0 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 1), (dim * 2 ** 1) * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))



        self.conv_cat2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=False)
       
        self.decoder_block2 = ResnetBloc_eca1(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, mid_layer=True, with_mixer=False, with_lowreso=False,with_ca=False)
        self.decoder_blockl = ResnetBloc_eca1(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, mid_layer=False, with_mixer=True, with_lowreso=False,with_ca=False)
        
        self.proj_kv1 = nn.Conv2d(9, dim, kernel_size=1)

        self.proj_kv2 = nn.Sequential(  
            nn.Conv2d(9, dim * 2 ** 1, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        self.proj_kv3 = nn.Sequential( 
            nn.Conv2d(9, dim * 2 ** 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 2, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )

        self.cross = Chanel_Cross_Attention(dim=dim, num_head=heads[0], bias=True)
        self.cross1 = Chanel_Cross_Attention(dim=dim * 2 ** 1, num_head=heads[1], bias=True)
        self.cross2 = Chanel_Cross_Attention(dim=dim * 2 ** 2, num_head=heads[2], bias=True)

    def forward(self, x, t, laplacian):

        x1 = self.conv1(x)#3
        la1 = self.proj_kv1(laplacian)#1
        x1_la1 = self.cross(x1, la1)
        # print(x1_la1.shape)
        # print(x1.shape)
        x1 = self.block1(x1, t, x1_la1)


        x2 = self.conv2(x1)#3
        la2 = self.proj_kv2(laplacian)#3
        x2_la2 = self.cross1(x2, la2)
        # print(x2_la2.shape)
        # print(x2.shape)
        x2 = self.block2(x2, t, x2_la2)


        x3 = self.conv3(x2)#3
        la3 = self.proj_kv3(laplacian)#3
        x3_la3 = self.cross2(x3, la3)
        # print(x3.shape)
        # print(x3_la3.shape)
        x3 = self.block3(x3, t, x3_la3)
        x3 = self.block3(x3, t, x3_la3)

        de_level2 = self.conv_up2(x3)#3
        # print(de_level2.shape)
        de_level2 = torch.cat([de_level2, x2], 1)
        de_level2 = self.conv_cat2(de_level2)#1
        # print(de_level2.shape)
        # print(x2_la2.shape)
        de_level2 = self.decoder_block2(de_level2, t, x2_la2)

        de_level1 = self.conv_up1(de_level2)#3
        x2_la2 = self.conv_up0(x2_la2)#3
        de_level1 = torch.cat([de_level1, x1], 1)
        # print(de_level1.shape)
        # print(x2_la2.shape)
        mid_feat = self.decoder_blockl(de_level1, t, x2_la2)

        return mid_feat, x2_la2

class UNet_low(nn.Module):
    def __init__(
        self,
        in_channel=9,
        out_channel=3,
        inner_channel=48,
        norm_groups=24,
        with_time_emb=True,
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None


        dim = inner_channel
        self.encoder_water = Encoder_stage6(in_channel=in_channel, inner_channel=inner_channel, norm_groups=norm_groups)

        self.refine = ResnetBloc_eca1(dim=dim*2**1, dim_out=dim*2**1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, mid_layer=True, with_mixer=False, with_lowreso=False,with_ca=False)
        self.de_predict = nn.Sequential(nn.Conv2d(dim * 2 ** 1, out_channel, kernel_size=1, stride=1))


    def forward(self, x, time, laplacian):        
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        mid_feat, x2_la2 = self.encoder_water(x, t, laplacian)
        mid_feat2 = self.refine(mid_feat, t, x2_la2)

        return self.de_predict(mid_feat2)