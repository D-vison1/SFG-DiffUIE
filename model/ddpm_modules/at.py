import torch
from torch import nn
import numpy as np
from torchvision.transforms import ToTensor,ToPILImage
from PIL import ImageFilter,Image
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ConditionNet(nn.Module):
    def __init__(self, n_block=10, n_conv_each_block=2, conv_index='128', sr_in_channel=128, support_size=128):
        super(ConditionNet, self).__init__()
        self.support_size = support_size
        self.n_block = n_block
        self.n_conv_each_block = n_conv_each_block
        self.n_modulation = n_block * n_conv_each_block
        self.conv_index = conv_index #决定网络的输出通道
        if conv_index == '128':
            self.condition_channel = 128
        elif self.conv_index == '64':
            self.condition_channel = 64
        else:
            raise ValueError('Illegal VGG conv_index!!!')
        self.sr_in_channel = sr_in_channel

        self.condition = self.get_VGG_condition()  #构建的 VGG 网络
        initialize_weights(self.condition, 0.1)


    def get_VGG_condition(self):
        cfg = [64, 64, 'P', 128, 128, 'P']
        if self.conv_index == '128':
            cfg_idx = cfg[:5]
        elif self.conv_index == '64':
            cfg_idx = cfg[:2]
        else:
            raise ValueError('Illegal VGG conv_index!!!')
        return self._make_layers(cfg_idx)

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        # in_channels = 3 * self.support_size
        in_channels = self.support_size
        for v in cfg:
            if v == 'P':
                layers += [nn.AvgPool2d(kernel_size=3, stride=1, padding=1)]
            else:
                conv2d = DynamicConv(in_planes=in_channels, out_planes=v, kernel_size=3, stride=1, padding=1,
                                    bias=False)
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def reset_support_size(self, support_size):
        self.support_size = support_size

    def forward(self, support_x):
        """return batch_size condition_features
        Input:
        For training
        support_x [batch_size, support_size*3, h, w]
        For testing batch_size = 1
        support_x [1, support_size*3, h, w]

        """

        support_conditional_feature = self.condition(support_x)  # [batch_size, 128, h/2, w/2]
        return support_conditional_feature


class Attention(nn.Module):
    def __init__(self, in_planes, ratio, K, temprature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temprature = temprature
        assert in_planes > ratio
        hidden_planes = in_planes // ratio
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        )

        if init_weight:
            self._initialize_weights()

    def update_temprature(self):
        if self.temprature > 1:
            self.temprature -= 1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att).view(x.shape[0], -1)  # bs,K
        return F.softmax(att / self.temprature, -1)


class DynamicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True, K=4,
                 temprature=30, ratio=2, init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = Attention(in_planes=in_planes, ratio=ratio, K=K, temprature=temprature,
                                   init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if self.init_weight:
            self._initialize_weights()

        # TODO 初始化

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K
        x = x.contiguous().view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k

        if self.bias is not None:
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h, w)
        return output

def load_part_of_model(new_model, src_model_path, s):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        if k in m_dict.keys():
            param = src_model.get(k)
            if param.shape == m_dict[k].data.shape:
                m_dict[k].data = param
                print('loading:', k)
            else:
                print('shape is different, not loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict, strict=s)
    return new_model

def load_part_of_model2(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        k2 = k.replace('denoise_fn.', '')
        if k2 in m_dict.keys():
            # print(k)
            param = src_model.get(k)
            if param.shape == m_dict[k2].data.shape:
                m_dict[k2].data = param
                print('loading:', k)
            # else:
            #     print('shape is different, not loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict)
    return new_model

def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    # ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # if img_np.shape[0] == 1:
    #     ar = ar[0]
    # else:
    #     assert img_np.shape[0] == 3, img_np.shape
    #     ar = ar.transpose(1, 2, 0)

    # return Image.fromarray(ar)
    return ToPILImage()(img_np)
    

def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    # return img_var.detach().cpu().numpy()[0]
    return img_var.detach().cpu().numpy().transpose(1, 2, 0) 

def get_A(x):
    # x_np = np.clip(torch_to_np(x), 0, 1)
    # x_pil = np_to_pil(x_np)
    # h, w = x_pil.size
    # windows = (h + w) / 2
    # A = x_pil.filter(ImageFilter.GaussianBlur(windows))
    # A = ToTensor()(A).to(x.device)
    # return A.unsqueeze(0)

    processed_images = []

    for i in range(x.size(0)):
        img_np = torch_to_np(x[i])
        img_pil = np_to_pil(img_np)
        
        h, w = img_pil.size
        windows = (h + w) / 2
        
        A_pil = img_pil.filter(ImageFilter.GaussianBlur(windows))
        
        A_tensor = ToTensor()(A_pil).to(x.device)
        processed_images.append(A_tensor)
    
    return torch.stack(processed_images, dim=0)

class TNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.aNet = ConditionNet(support_size=input_channels)
        self.final = nn.Conv2d(128, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        a = self.final(self.aNet(x))
        return a
 
class PM(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.tNet = TNet(input_channels,1)
        self.aNet = TNet(input_channels,3)

    def forward(self, x): 
        a = self.aNet(get_A(x)+x)  #background
        t = self.tNet(x)       #transmit
        return a, t
    