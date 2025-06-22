import torch
import torch.nn as nn


class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super(AdaptiveConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding= k // 2) for k in kernel_sizes
        ])
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, len(kernel_sizes), kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attn = self.attention(x)
        attn = attn.unsqueeze(2)
        conv_results = [conv(x) for conv in self.convs]
        conv_results = torch.stack(conv_results, dim=1)
        output = (attn * conv_results).sum(dim=1)
        return output



input_tensor = torch.randn(32, 7, 6, 16)  # batch_size=32, channels=7, height=6, width=16

kernel_sizes = [3, 5, 7]
adaptive_conv = AdaptiveConv(in_channels=7, out_channels=7, kernel_sizes=kernel_sizes)
output = adaptive_conv(input_tensor)

class SameConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=3):
        super(SameConv2D, self).__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x):
        return self.conv(x)


input_tensor = torch.randn(32, 7, 64, 64)  # batch_size=32, channels=7, height=64, width=64

conv2d = SameConv2D(in_channels=7, out_channels=7, kernel_size=3)

output = conv2d(input_tensor)
