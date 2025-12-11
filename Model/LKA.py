import torch.nn as nn
import torch

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LargeKernelAttention(nn.Module):

    def __init__(self, dim, kernel_size=21):
        super(LargeKernelAttention, self).__init__()

        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.conv_h = DepthwiseSeparableConv(dim, dim,
                                             kernel_size=(kernel_size, 1),
                                             padding=(padding, 0))
        self.conv_w = DepthwiseSeparableConv(dim, dim,
                                             kernel_size=(1, kernel_size),
                                             padding=(0, padding))

        self.conv3 = nn.Conv2d(dim, dim, 1)

        self.conv4 = nn.Conv2d(dim, dim, 1)

        self.act = nn.GELU()

        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x_h = self.conv_h(x)
        x_w = self.conv_w(x)
        x = x_h + x_w

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.act(x)

        x = torch.sigmoid(x) * identity

        x = self.conv4(x)
        return x