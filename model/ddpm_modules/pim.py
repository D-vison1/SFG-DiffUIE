import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight
        # return x / torch.sqrt(sigma+1e-5) * self.weight

import numbers
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

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)




class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# Phase integration module
class PIM(nn.Module):
    def __init__(self, channel):
        super(PIM, self).__init__()

        self.processmag = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0))

        self.processpha = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0))

    def forward(self, rgb_x, ycbcr_x):
        rgb_fft = torch.fft.rfft2(rgb_x, norm='backward')
        ycbcr_fft = torch.fft.rfft2(ycbcr_x, norm='backward')
        rgb_amp = torch.abs(rgb_fft)
        rgb_phase = torch.angle(rgb_fft)

        ycbcr_amp = torch.abs(ycbcr_fft)
        ycbcr_phase = torch.angle(ycbcr_fft)

        rgb_amp = self.processmag(rgb_amp)
        rgb_phase = self.processmag(rgb_phase)

        ycbcr_amp = self.processmag(ycbcr_amp)
        ycbcr_phase = self.processmag(ycbcr_phase)

        mix_phase = rgb_phase + ycbcr_phase

        out_rgb = torch.fft.irfft2(rgb_amp * torch.exp(1j * mix_phase), norm='backward')
        out_ycbcr = torch.fft.irfft2(ycbcr_amp * torch.exp(1j * mix_phase), norm='backward')
        return out_rgb, out_ycbcr



class CAB(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# Lightweight Cross Attention
class IAM(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(IAM, self).__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim, LayerNorm_type='withBisa')
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


# Interaction Attention Module
class IAMB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.proj = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        self.rgb_cab = IAM(self.dim)
        self.ycbcr_cab = IAM(self.dim)

    def forward(self, in_feats):
        out_rgb = self.rgb_cab(in_feats[0], in_feats[1])
        out_ycbcr = self.ycbcr_cab(in_feats[1], in_feats[0])

        x = self.proj(out_ycbcr + out_rgb)
        return x
    
class OverlapPatchEmbed(nn.Module):#重叠嵌入
    def __init__(self, in_c=3, embed_dim=48, bias=False):#embed_dim：输入数据转换为固定大小的向量表示时，这个向量的长度或维度
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)#输入图像转换为特征图

    def forward(self, x):
        x = self.proj(x)
        return x
    
# COLOR ENHANCEMENT MODULE
class CEM(nn.Module):
    def __init__(self, channel):
        super(CEM, self).__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x1_amp1 = torch.mean(x, dim=1, keepdim=True)
        x1_amp1 = x - x1_amp1
        att = self.adaptive_avg_pool(x1_amp1)
        att = F.softmax(att, 1)
        return att
    

class BI_color_Guidance_Bridge(nn.Module):
    def __init__(self, dim):
        super(BI_color_Guidance_Bridge, self).__init__()

        self.iam = IAMB(dim)
    def forward(self, rgb, lab):
        x = self.iam([rgb, lab])
        return x

















































































































import os
import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]
        

def rgb_to_hsv_tensor(tensor):
    # 将张量转换为 NumPy 格式，并调整为 (H, W, C)
    tensor_np = tensor.permute(0, 2, 3, 1).cpu().detach().numpy() # (B, H, W, C)
    tensor_np = (tensor_np * 255).astype(np.uint8)

    # 使用 OpenCV 进行 RGB 到 HSV 的转换
    hsv_np = np.zeros_like(tensor_np)
    for i in range(tensor_np.shape[0]):
        hsv_np[i] = cv2.cvtColor(tensor_np[i], cv2.COLOR_RGB2HSV)

    # 将结果转换回张量，并调整为 (B, C, H, W)
    hsv_tensor = torch.from_numpy(hsv_np).float() / 255.0
    return hsv_tensor.permute(0, 3, 1, 2)
