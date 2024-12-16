import torch
import torch.nn as nn
import functools


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_act_layer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU
    elif act_type == 'lrelu':
        act_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2)
    elif act_type == 'none':
        def act_layer(): return Identity()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return act_layer


# Adaptive instance normalization
class AdaptiveInstanceNorm3d(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveInstanceNorm3d, self).__init__()
        self.norm = nn.InstanceNorm3d(in_channels, affine=False, track_running_stats=False)

    def forward(self, x, mean, std, alpha=1.0):
        size = x.size()
        N, C = size[:2]
        mean = mean.view(N, C, 1, 1, 1).expand(size)
        std = std.view(N, C, 1, 1, 1).expand(size)
        norm = self.norm(x)
        out = std * norm + mean
        out = out * alpha + x * (1 - alpha)
        return out


# Basic convolution block with AdaIN layer
class ConvAdaInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(ConvAdaInBlock, self).__init__()
        act_layer = get_act_layer(act_type)
        self.conv_layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm_layer = AdaptiveInstanceNorm3d(out_channels)
        self.act_layer = act_layer()

    def forward(self, x, mean, std, alpha=1.0):
        conv = self.conv_layer(x)
        norm = self.norm_layer(conv, mean, std, alpha)
        out = self.act_layer(norm)
        return out


# Basic downsampling block with AdaIN layer
class DownAdaInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(DownAdaInBlock, self).__init__()
        self.down_layer = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.layers = ConvAdaInBlock(in_channels, out_channels, act_type)

    def forward(self, x, mean, std, alpha=1.0):
        down = self.down_layer(x)
        out = self.layers(down, mean, std, alpha)
        return out


# Basic upsampling block with AdaIN layer
class UpAdaInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(UpAdaInBlock, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.layers2 = ConvAdaInBlock(in_channels, out_channels, act_type)

    def forward(self, x1, x2, mean, std, alpha=1.0):
        x1 = self.layers1(x1)
        x = torch.cat([x2, x1], dim=1)
        out = self.layers2(x, mean, std, alpha)
        return out


# Common code generator
class CommonCode(nn.Module):
    def __init__(self, in_features, out_features):
        super(CommonCode, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Linear(out_features, out_features),
            nn.Linear(out_features, out_features),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x):
        out = self.layers(x)
        return out


# Individual code generator
class IndividualCode(nn.Module):
    def __init__(self, in_features, out_features):
        super(IndividualCode, self).__init__()
        self.mean_layer = nn.Linear(in_features, out_features)
        self.std_layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        mean = self.mean_layer(x)
        std = self.std_layer(x)
        return mean, std
