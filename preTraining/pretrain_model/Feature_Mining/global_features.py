import torch
import torch.nn as nn

class global_F(nn.Module):
    def __init__(self, seq_len):
        super(global_F, self).__init__()
        len2 = seq_len // 2
        len4 = seq_len // 2 ** 2
        len8 = seq_len // 2 ** 3
        self.linear8 = nn.Linear(len8, len4)
        self.linear4 = nn.Linear(len4, len2)
        self.linear2 = nn.Linear(len2, seq_len)
        self.linear = nn.Linear(seq_len, seq_len)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(len2)
        self.global_avg_pool4 = nn.AdaptiveAvgPool1d(len4)
        self.global_avg_pool8 = nn.AdaptiveAvgPool1d(len8)
        self.activation = nn.ReLU()  

    def forward(self, x):
        # Down
        global_d2 = self.global_avg_pool(x)
        global_d4 = self.global_avg_pool4(x)
        global_d8 = self.global_avg_pool8(x)
        # Up
        global_u4 = self.activation(self.linear8(global_d8))
        global_u2 = self.activation(self.linear4(global_u4+global_d4))
        global_u = self.activation(self.linear2(global_u2 + global_d2))
        # predict
        outputs = self.activation(self.linear(global_u + x))

        return outputs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    seq_len = 96
    model = global_F(seq_len)
    x = torch.randn(32,7,96)
    o = model(x)

    input_np = x.squeeze().detach().numpy()
    output_np = o.squeeze().detach().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i in range(3):
        axes[0, i].imshow(input_np[i], cmap='viridis')
        axes[0, i].set_title(f'Input Channel {i + 1}')
        axes[1, i].imshow(output_np[i], cmap='viridis')
        axes[1, i].set_title(f'Output Channel {i + 1}')

    plt.suptitle('Global Average Pooling')
    plt.show()
    print(x.shape)
