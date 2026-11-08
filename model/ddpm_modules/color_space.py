import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(0)
# device='cpu'
def rgb2hsi(img):
    img = torch.clamp(img, 0, 1)
    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]
    i = (r + g + b) / 3
    s = 1 - 3 * img.min(1)[0] / (r + g + b + 1e-5)
    x1 = (2 * r - b - g) / 2
    x2 = ((r - g) ** 2 + (r - b) * (g - b) + 1e-5) ** 0.5
    angle = torch.arccos(x1 / x2) / 2 / torch.pi
    # h = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)
    h = (b <= r) * angle + (b > r) * (1 - angle)
    h = h.unsqueeze(1)
    s = s.unsqueeze(1)
    i = i.unsqueeze(1)
    out = torch.cat((h, s, i), dim=1)
    return out


def hsi2rgb(img):
    img = torch.clamp(img, 0, 1)
    h = img[:, 0, :, :]
    s = img[:, 1, :, :]
    i = img[:, 2, :, :]
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
    h1 = torch.zeros_like(h)
    hi0 = (h < 1 / 3)
    hi2 = (h >= 2 / 3)
    hi1 = 1 - hi0.int() - hi2.int()
    hi1 = (hi1 == 1)
    h1[hi0] = 2 * torch.pi * h[hi0]
    h1[hi1] = 2 * torch.pi * (h[hi1] - 1 / 3)
    h1[hi2] = 2 * torch.pi * (h[hi2] - 2 / 3)
    p = i * (1 - s)
    q = i * (1 + s * torch.cos(h1) / (torch.cos(torch.pi / 3 - h1) + 1e-5))

    r[hi0] = q[hi0]
    b[hi0] = p[hi0]
    g[hi0] = 3 * i[hi0] - r[hi0] - b[hi0]

    g[hi1] = q[hi1]
    r[hi1] = p[hi1]
    b[hi1] = 3 * i[hi1] - r[hi1] - g[hi1]

    b[hi2] = q[hi2]
    g[hi2] = p[hi2]
    r[hi2] = 3 * i[hi2] - g[hi2] - b[hi2]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    out = torch.cat((r, g, b), dim=1)
    return out

def rgb2hsv(img):
        img = torch.clamp(img, 0, 1)
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + 1e-5))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + 1e-5))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + 1e-5))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + 1e-5)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

def hsv2rgb(hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb


MAT_RGB2XYZ  = torch.Tensor([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]]).to(device)
MAT_XYZ2RGB = torch.Tensor([[ 3.2405, -1.5372, -0.4985],
                            [-0.9693,  1.8760,  0.0416],
                            [ 0.0556, -0.2040,  1.0573]]).to(device)

XYZ_REF_WHITE = torch.Tensor([0.95047, 1.0, 1.08883]).to(device)


def rgb2lab(rgb):
    rgb=torch.clamp(rgb,0,1)
    return xyz_to_lab(rgb_to_xyz(rgb))

def lab2rgb(lab):
    lab=torch.clamp(lab,0,1)
    return xyz_to_rgb(lab_to_xyz(lab))

def rgb_to_xyz(rgb):

    # convert dtype from uint8 to float
    # xyz = rgb.astype(np.float64) / 255.0
    # xyz = rgb.astype(np.float64)
    xyz = rgb
    # gamma correction
    mask = xyz > 0.04045
    abc=torch.zeros_like(xyz)
    abc[mask] = ((xyz[mask]  + 0.055) / 1.055)**2.4
    abc[~mask] = xyz[~mask]/12.92
    xyz = abc.permute(0, 2, 3, 1)
    # linear transform
    xyz = torch.matmul(xyz , MAT_RGB2XYZ.T)
    xyz = xyz.permute(0, 3, 1, 2)
    return xyz
def xyz_to_lab(xyz):


    xyz=xyz.permute(0, 2, 3, 1)
    xyz = xyz/XYZ_REF_WHITE

    # nonlinear transform
    mask = xyz > 0.008856
    xyz[mask] = torch.pow(xyz[mask], 1.0 / 3.0)
    xyz[~mask] = 7.787 * xyz[~mask] + 16.0 / 116.0
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    # linear transform
    lab = torch.zeros_like(xyz)
    # lab = torch.zeros(xyz.shape, requires_grad=True)
    lab[..., 0] = (116.0 * y) - 16.0  # L channel
    lab[..., 1] = 500.0 * (x - y)  # a channel
    lab[..., 2] = 200.0 * (y - z)  # b channel
    lab[..., 0] = lab[..., 0]/100  # L channel
    lab[..., 1] = (lab[..., 1]+86.183030)/184.416084  # a channel
    lab[..., 2] = (lab[..., 2]+107.857300)/202.335422 # b channel
    lab=lab.permute(0, 3, 1, 2)
    return lab

def lab_to_xyz(lab):


    lab=lab.permute(0, 2, 3, 1)
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l=l*100
    a=a*184.416084-86.183030
    b=b*202.335422-107.857300
    xyz = torch.zeros_like(lab)
    # xyz = torch.zeros(lab.shape,requires_grad=True)
    xyz[..., 1] = (l + 16.0) / 116.0
    xyz[..., 0] = a / 500.0 + xyz[..., 1]
    xyz[..., 2] = xyz[..., 1] - b / 200.0
    # index = xyz[..., 2] < 0
    # xyz[index, 2] = 0
    xyz = torch.clamp(xyz, min=0.0) 

    # nonlinear transform
    mask = xyz > 0.2068966
    xyz[mask] = torch.pow(xyz[mask], 3.0)
    xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787

    # de-normalization
    xyz = xyz*XYZ_REF_WHITE
    xyz=xyz.permute(0, 3, 1, 2)
    return xyz

def xyz_to_rgb(xyz):


    rgb = xyz.permute(0, 2, 3, 1)
    rgb = torch.matmul(rgb, MAT_XYZ2RGB.T)

    # gamma correction
    mask = rgb > 0.0031308
    rgb[mask] = 1.055 * torch.pow(rgb[mask], 1.0 / 2.4) - 0.055
    rgb[~mask] = rgb[~mask] * 12.92

    # clip and convert dtype from float to uint8
    # rgb = np.round(255.0 * np.clip(rgb, 0, 1)).astype(np.uint8)
    rgb = torch.clip(rgb, 0, 1)
    rgb = rgb.permute(0, 3, 1, 2)
    return rgb

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

    
def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

    
class OverlapPatchEmbed(nn.Module):#重叠嵌入
    def __init__(self, in_c=3, embed_dim=48, bias=False):#embed_dim：输入数据转换为固定大小的向量表示时，这个向量的长度或维度
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)#输入图像转换为特征图

    def forward(self, x):
        x = self.proj(x)
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
