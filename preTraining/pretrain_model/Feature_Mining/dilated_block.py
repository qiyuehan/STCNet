import torch
import torch.nn as nn
import torch.nn.functional as F

from DanceEnd.my_models.tools.deformable import Def_Conv
from Figs.plot_fig.adaptiveWeights.plot_adaWeights import plt_adaptive_Weights


class Adaptive_dilatedConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation=[1,2,5]):
        super(Adaptive_dilatedConv, self).__init__()

        self.convs = nn.ModuleList([
            Def_Conv(in_channels, in_channels, kernel_size=3, dilation_rate=d, groups=in_channels)
            for d in dilation
        ])


        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

     
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, len(dilation), kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):  # [32, 321, 6, 8]---# [32, 6, 8, 321]
        attention = self.attention(x) 
      
        conv_outputs = [conv(x) for conv in self.convs]
        # concatenated = torch.cat(conv_outputs, dim=1)
        concatenated = torch.stack(conv_outputs, dim=1) # [32, 3, 321, 6, 8] --([32, 3, 6, 8, 321]
        attn = attention.unsqueeze(2)  
        # get the attn score
        attn_weights = attn * concatenated
        output = (attn_weights + concatenated).sum(dim=1).squeeze() 
        output = self.out_conv(output) #[32, 321, 6, 8]
        # plot_ada_def
        b,l,c,_,_ = attn_weights.shape
        return output
