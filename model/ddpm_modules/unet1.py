import math
from typing import Tuple, Union, List
import numpy as np
import torch
from torch import nn
from einops import rearrange

class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

    def forward(self, x):
        x = self.PEG(x) + x
        return x


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, head_num):
        super(Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query_layer, key_layer, value_layer):
        B, N, C = query_layer.size()
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        _, _, _, d = query_layer.size()
        # 此处的d就是函数transpose_for_scores中的attention_head_size
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        attention_out = context_layer.view(*new_context_layer_shape)

        return attention_out


# Intra-Sequence Self-Attention，即序列内自注意力
class Intra_SA(nn.Module):
    def __init__(self, dim, head_num):
        super(Intra_SA, self).__init__()
        self.hidden_size = dim // 2
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_h
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_v
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous()
        feature_h = feature_h.view(B * H, W, C//2)
        feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous()
        feature_v = feature_v.view(B * W, H, C//2)
        qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)
        qkv_v = torch.chunk(self.qkv_local_v(feature_v), 3, dim=2)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]

        if H == W:
            query = torch.cat((q_h, q_v), dim=0)
            key = torch.cat((k_h, k_v), dim=0)
            value = torch.cat((v_h, v_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(q_h, k_h, v_h)
            attention_output_v = self.attn(q_v, k_v, v_v)
            attention_output_h = attention_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x


# Inter-Sequence Self-Attention，即序列间自注意力
class Inter_SA(nn.Module):
    def __init__(self,dim, head_num):
        super(Inter_SA, self).__init__()
        self.hidden_size = dim
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.conv_h = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_h
        self.conv_v = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_v
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        self.ffn = Mlp(self.hidden_size)
        self.fuse_out = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = torch.chunk(self.conv_h(x_input[0]), 3, dim=1)
        feature_v = torch.chunk(self.conv_v(x_input[1]), 3, dim=1)
        query_h, key_h, value_h = feature_h[0], feature_h[1], feature_h[2]
        query_v, key_v, value_v = feature_v[0], feature_v[1], feature_v[2]

        horizontal_groups = torch.cat((query_h, key_h, value_h), dim=0)
        horizontal_groups = horizontal_groups.permute(0, 2, 1, 3).contiguous()
        horizontal_groups = horizontal_groups.view(3*B, H, -1)
        horizontal_groups = torch.chunk(horizontal_groups, 3, dim=0)
        query_h, key_h, value_h = horizontal_groups[0], horizontal_groups[1], horizontal_groups[2]

        vertical_groups = torch.cat((query_v, key_v, value_v), dim=0)
        vertical_groups = vertical_groups.permute(0, 3, 1, 2).contiguous()
        vertical_groups = vertical_groups.view(3*B, W, -1)
        vertical_groups = torch.chunk(vertical_groups, 3, dim=0)
        query_v, key_v, value_v = vertical_groups[0], vertical_groups[1], vertical_groups[2]

        if H == W:
            query = torch.cat((query_h, query_v), dim=0)
            key = torch.cat((key_h, key_v), dim=0)
            value = torch.cat((value_h, value_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(query_h, key_h, value_h)
            attention_output_v = self.attn(query_v, key_v, value_v)
            attention_output_h = attention_output_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x


class Stripformer(nn.Module):
    def __init__(self, input_channels, flag=1):
        super(Stripformer, self).__init__()
        head_num = 5
        dim = 320
        self.Trans_block_1 = Intra_SA(dim, head_num)
        self.Trans_block_2 = Inter_SA(dim, head_num)
        self.conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=320, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels=320, out_channels=input_channels, kernel_size=1)
        self.flag = flag

    def forward(self, x):
        x = self.conv_1(x)
        if self.flag:
            hx = x
            for _ in range(6):
                hx = self.Trans_block_1(hx)
                hx = self.Trans_block_2(hx)

            hx = self.conv_2(hx)
            # print('在运行stripformer12')
            return hx
        elif self.flag == 0:
            hx = self.Trans_block_1(x)
            hx = self.Trans_block_2(hx)

            hx = self.conv_2(hx)
            # print('在运行stripformer2')
            return hx
        
class MLP(nn.Module):
    def __init__(self, embedding_size, forward_expansion=2):
        super(MLP, self).__init__()
        self.conv = nn.Conv2d(in_channels=embedding_size, out_channels=embedding_size, kernel_size=3, padding=1)
        self.fc = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size),
            nn.LayerNorm(embedding_size)
        )

    def forward(self, x):
        self.fc.to(x.device)
        self.conv.to(x.device)
        B, C, H, W = x.shape

        hx = self.conv(x)
        hx = rearrange(hx, 'b c h w -> b (h w) c')
        hx = self.fc(hx)
        hx = rearrange(hx, 'b (h w) c -> b c h w', h=H, w=W)
        hx = self.conv(hx)
        # print('在运行MLP')

        return hx + x


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 dropout: float = 0.1, is_noise: bool = True):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.is_noise = is_noise
        self.act1 = Swish()
        self.norm1 = Normalize(in_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer

        self.norm2 = Normalize(in_channels=out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        if self.is_noise:
            self.time_emb = nn.Linear(time_channels, out_channels)
            self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.norm1(x)
        h = self.conv1(self.act1(h))
        # Add time embeddings
        if self.is_noise:
            h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.norm2(h)
        h = self.conv2(self.dropout(self.act2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, flag, is_noise: bool = True):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, is_noise=is_noise)
        self.att = Stripformer(input_channels=out_channels, flag=0)
        self.mlp = MLP(embedding_size=out_channels)
        self.flag = flag

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.mlp(x)

        if self.flag >= 2:
            x = self.att(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, flag2, is_noise: bool = True, flag1=0):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels, is_noise=is_noise)
        self.att = Stripformer(input_channels=out_channels, flag=0)
        self.mlp = MLP(embedding_size=out_channels)
        self.flag1 = flag1
        self.flag2 = flag2

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        if self.flag1:
            x = self.mlp(x)
        if self.flag1 and self.flag2 >= 2:
            x = self.att(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int, is_noise: bool = True):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, is_noise=is_noise)
        self.dia1 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=2, padding=get_pad(16, 3, 1, 2))
        self.dia2 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=4, padding=get_pad(16, 3, 1, 4))

        self.sf = Stripformer(input_channels=n_channels)

        self.dia3 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=8, padding=get_pad(16, 3, 1, 8))
        self.dia4 = nn.Conv2d(n_channels, n_channels, 3, 1, dilation=16, padding=get_pad(16, 3, 1, 16))
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, is_noise=is_noise)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.dia1(x)
        x = self.dia2(x)

        x = self.sf(x)

        x = self.dia3(x)
        x = self.dia4(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class DenoiserUNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, input_channels: int = 6, output_channels: int = 3, n_channels: int = 32,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4), n_blocks: int = 2, is_noise: bool = True):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        ' n_channels '是我们将图像转换成的初始特征映射中的通道数
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        ' ch_mults '是每个resolution下的通道号列表。通道数为' ch_mults[i] * n_channels '
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        ' is_attn '是一个布尔值列表，表示是否在每个resolution下使用注意力模块
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        ' n_blocks '是每个resolution下' UpDownBlocks '的数量
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map 将图像投影到特征映射中
        self.image_proj = nn.Conv2d(input_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.is_noise = is_noise
        if is_noise:
            self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # Add `n_blocks`
            for j in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise, flag=i))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, is_noise=False)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = n_channels * ch_mults[i]
            for j in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise, flag2=i))
            # Final block to reduce the number of channels
            in_channels = n_channels * (ch_mults[i-1] if i >= 1 else 1)
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_noise=is_noise, flag1=1, flag2=i))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        self.act = Swish()
        self.final = nn.Conv2d(in_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor = torch.tensor([0]).cuda()):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        # Get time-step embeddings
        if self.is_noise:
            t = self.time_emb(t)
        else:
            t = None
        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            print(x.shape)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)


        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)

                # x = attention(x)

            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()

                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.final(self.act(x))


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)