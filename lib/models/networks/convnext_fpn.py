import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import math

# https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2
from .dcn import DeformableConv2d

num_out_channel = {"convnext_tiny"  : 768,
                   "convnext_small" : 768,
                   "convnext_base" : 1024,
                   "convnext_large" : 1536,
                   "convnext_xlarge" : 1536}

BN_MOMENTUM = 0.1

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        #self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

        self.conv = DeformableConv2d(chi, cho, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        #self.conv = nn.Conv2d(chi, cho, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        for name, m in self.actf.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, num_features_out, heads, head_conv, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        self.heads = heads
        self.inplanes = num_features_out
        self.deconv_with_bias = False
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        #self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        #self.head.weight.data.mul_(head_init_scale)
        #self.head.bias.data.mul_(head_init_scale)

        ## intermediate stage output
        self.stages_output = []
        ##

        ###
        # used for deconv layers
        self.deconv_layer1 = self._make_deconv_layer(384, 4)#(256, 4)
        self.deconv_layer2 = self._make_deconv_layer(192, 4)#(128, 4)
        self.deconv_layer3 = self._make_deconv_layer(96, 4)#(64, 4)

        self.smooth_layer1 = DeformConv(384, 384)#(256, 256)
        self.smooth_layer2 = DeformConv(192, 192)#(128, 128)
        self.smooth_layer3 = DeformConv(96, 96)#(64, 64)

        self.project_layer1 = DeformConv(384, 384)
        self.project_layer2 = DeformConv(192, 192)
        self.project_layer3 = DeformConv(96, 96)

        ###
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(96, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes,
                    kernel_size=1, stride=1,
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes,
                  kernel_size=1, stride=1,
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters, num_kernels):

        layers = []

        kernel, padding, output_padding = \
            self._get_deconv_cfg(num_kernels)

        planes = num_filters
        fc = DeformableConv2d(self.inplanes, planes, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        '''
        fc = nn.Conv2d(self.inplanes, planes,
                 kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False)
        fill_fc_weights(fc)
        '''
        up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
        fill_up_weights(up)

        layers.append(fc)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        layers.append(up)
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        self.inplanes = planes

        return nn.Sequential(*layers)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            ###
            self.stages_output.append(x)
            ###
        return x#self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        ###x = self.head(x)
        p4 = self.stages_output[3]
        c3 = self.stages_output[2]
        c2 = self.stages_output[1]
        c1 = self.stages_output[0]

        '''
        Il nostro obbiettivo è cercare di ricostuire l'img originale -> 1/4 dell'img originale.
        smooth_layer1 -> è necessario per ridurre l'aliasing che si viene a creare...
        dovuto al fatto che si è fatto il merge con l'upsampled layer.

        '''

        p3 = self.smooth_layer1(self.deconv_layer1(p4) + self.project_layer1(c3))
        p2 = self.smooth_layer2(self.deconv_layer2(p3) + self.project_layer2(c2))
        p1 = self.smooth_layer3(self.deconv_layer3(p2) + self.project_layer3(c1))
        #print("*** p4.shape : ",p4.shape)
        #print("*** p3.shape : ",p3.shape)
        #print("*** p2.shape : ",p2.shape)
        #print("*** p1.shape : ",p1.shape)
        self.stages_output = []
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(p1)
            #print("*** : ",ret[head].shape)
        #xxx = input()
        return [ret]

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(heads, head_conv, pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(num_out_channel["convnext_tiny"], heads, head_conv=head_conv, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def convnext_small(heads, head_conv, pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(num_out_channel["convnext_small"], heads, head_conv=head_conv, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        print("*********************** pretrained model *******************")
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def convnext_base(heads, head_conv, pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(num_out_channel["convnext_base"], heads, head_conv=head_conv, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def convnext_large(heads, head_conv, pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(num_out_channel["convnext_large"], heads, head_conv=head_conv, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def convnext_xlarge(heads, head_conv, pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(num_out_channel["convnext_xlarge"], heads, head_conv=head_conv, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


_model_factory = {
    'convnextfpn_tiny' : convnext_tiny,
    'convnextfpn_small' : convnext_small,
    'convnextfpn_base' : convnext_base,
    'convnextfpn_large' : convnext_large,
    'convnextfpn_xlarge' : convnext_xlarge,
}

def get_conv_next_fpn(type_model, heads, head_conv=256):
    get_model = _model_factory[type_model]
    model = get_model(heads, head_conv=head_conv, pretrained=True, in_22k=True)

    return model
