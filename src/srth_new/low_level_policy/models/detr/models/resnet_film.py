import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet

from .efficientnet import FiLMBlock

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, use_film=False, language_embed_size=768):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_film = use_film
        if self.use_film:
            self.film = FiLMBlock(language_embed_size=language_embed_size, num_channels=planes)

    def forward(self, x, language_embed=None):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.use_film and language_embed is not None:
            out = self.film(out, language_embed)
        out += identity
        out = self.relu(out)
        return out

class CustomSequential(nn.Module):
    def __init__(self, *modules):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x, language_embed=None):
        for module in self.modules_list:
            if hasattr(module, 'use_film') and module.use_film:
                x = module(x, language_embed)
            else:
                x = module(x)
        return x
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class CustomResNet(ResNet):
    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_film=True):
        if dilate:
            self.dilation *= stride
            stride = 1
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, use_film))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, use_film=use_film))
        return CustomSequential(*layers)

    def forward(self, x, language_embed=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x, language_embed)
        x = self.layer2(x, language_embed)
        x = self.layer3(x, language_embed)
        x = self.layer4(x, language_embed)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def filmed_basic_block():
    return BasicBlock

def filmed_bottleneck():
    return Bottleneck

def _resnet(arch, block, layers, weights, use_film, **kwargs):
    model = CustomResNet(block, layers, **kwargs)
    return model

def film_resnet18(weights=None, use_film=True, **kwargs):
    return _resnet("resnet18", filmed_basic_block(), [2, 2, 2, 2], weights, use_film, **kwargs)

def film_resnet34(weights=None, use_film=True, **kwargs):
    return _resnet("resnet34", filmed_basic_block(), [3, 4, 6, 3], weights, use_film, **kwargs)

def film_resnet50(weights=None, use_film=True, **kwargs):
    return _resnet("resnet50", filmed_bottleneck(), [3, 4, 6, 3], weights, use_film, **kwargs)
