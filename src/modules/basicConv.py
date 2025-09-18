from collections import OrderedDict
import torch.nn as nn
import torch


def createCNNs(in_channel, channels, strides):
    """create 3*3 kernel CNNs"""
    layers = nn.Sequential()
    last_channel = in_channel
    for i, (out_channel,stride) in enumerate(zip(channels,strides)):
        layers.add_module(str(i * 4), nn.Conv2d(last_channel, out_channel,
                                                kernel_size=3, stride=1, padding=1, bias=True))
        layers.add_module(str(i * 4 + 1), nn.BatchNorm2d(out_channel))

        layers.add_module(str(i * 4 + 2), nn.LeakyReLU(negative_slope=0.1))

        layers.add_module(str(i * 4 + 3), nn.MaxPool2d(3, stride=stride, padding=1))

        last_channel = out_channel
    return layers


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, bn=False, activation_fn=True,
                 leaky_relu=True):
        """
        FC implement
        """
        super(Conv2d, self).__init__()
        if stride is None:
            stride = [1, 1]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = bn
        self.activation_fn = activation_fn

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if bn:
            self.bn_linear = nn.BatchNorm2d(out_channels)

        # TODO: origin is relu
        if activation_fn:
            self.relu = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        # x (b,n,s,c)

        x = x.permute(0, 3, 2, 1)  # (b,c,s,n)
        outputs = self.conv(x)
        if self.bn:
            outputs = self.bn_linear(outputs)

        if self.activation_fn:
            outputs = self.relu(outputs)

        outputs = outputs.permute(0, 3, 2, 1)  # (b,n,s,c)
        return outputs


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_activation=True,
                 use_leaky=True, bn=False):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if use_activation:
           relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)
        else:
           relu = nn.Identity()

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.composed_module(x)
        x = x.permute(0, 2, 1)
        return x
