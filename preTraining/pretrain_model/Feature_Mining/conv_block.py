import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGroupConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveGroupConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1) for _ in range(6)
        ])

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1)
        self.share_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1)
        self.share_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1)
        # self.shared_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):

        attn = self.attention(x)  # [batch_size, 3, height, width]
        conv1 = self.conv1(x)

        conv2_1 = self.share_conv1(x)
        conv2_2 = self.share_conv1(conv2_1)
        conv2 = conv2_2

        conv3_1 = self.share_conv2(x)
        conv3_2 = self.share_conv2(conv3_1)
        conv3_3 = self.share_conv2(conv3_2)
        conv3 = conv3_3


        conv_results = torch.stack([conv1, conv2, conv3], dim=1)  
        attn = attn.unsqueeze(2)
        output = (attn * conv_results).sum(dim=1)

        return output

if __name__ == '__main__':
    input_tensor = torch.randn(32, 7, 6, 16)  # batch_size=32, channels=7, height=6, width=16

    adaptive_group_conv = AdaptiveGroupConv(in_channels=7, out_channels=7)

    output = adaptive_group_conv(input_tensor)
    print(output.shape)
