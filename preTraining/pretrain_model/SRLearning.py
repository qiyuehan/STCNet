import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from torch import nn

from Figs.plot_fig.flot_frequency.plot_diffFreq import plot_frequency
from my_models.pretrain_model.Feature_Mining.dilated_block import Adaptive_dilatedConv
from my_models.pretrain_model.Feature_Mining.global_features import global_F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.dwt = DWT1DForward(wave=configs.wave, J=1).to(configs.device)
        self.idwt = DWT1DInverse(wave=configs.wave).to(configs.device)
        self.batch_size = configs.batch_size
        self.global_F = global_F(configs.seq_len//2)
        self.Mixer = nn.Linear(configs.seq_len//2, configs.seq_len//2)

        self.seq2seq = nn.Linear(configs.seq_len, configs.seq_len)

        self.block_size = configs.block_size
        self.block_num = configs.seq_len//configs.block_size

        self.LTC = Adaptive_dilatedConv(in_channels=configs.enc_in, kernel_size=3)
        self.IVC = Adaptive_dilatedConv(in_channels=self.block_num, kernel_size=3)

        self.forecast = nn.Sequential(
            nn.Linear(configs.seq_len, configs.pred_len),
            nn.Linear(configs.pred_len, configs.pred_len)
        )
        self.masking = configs.masking


    def forward(self, batch_denoise):

        x_deno_mean = torch.mean(batch_denoise, dim=1, keepdim=True)
        x_deno_var = torch.var(batch_denoise, dim=1, keepdim=True) + 1e-5
        input_rev = (batch_denoise - x_deno_mean) / torch.sqrt(x_deno_var)

        b, s, c = input_rev.shape  # (32,96,321)
        block_input = torch.stack(torch.split(input_rev, split_size_or_sections=self.block_size, dim=1), dim=1)  # 32,6,16,321
        mask_block = block_input * self.masking  # ([32, 6, 16, 321]

        mask_input = mask_block.reshape(b, -1, c)  # ([32, 96, 321]

        input_den = mask_input.transpose(-1, -2)  # ([32, 321, 96]
        input_TF = torch.tensor(input_den, dtype=torch.float32)
        yl, yh = self.dwt(input_TF)  # ([32, 321, 96]

        # low global features yl=#([32, 321, 48]
        LF_features = self.global_F(yl)  # [32, 321, 48]

        # HF
        b, s, _ = yh[0].shape
        HF_blocks = torch.stack(torch.split(yh[0], split_size_or_sections=self.block_num, dim=-1), dim=-1) # [32, 321, 6, 8]
        #  HF_1: long-term correlations
        LT_cor = self.LTC(HF_blocks)  #[32, 321, 6, 8]
        #ablation 2:
        # LT_cor = HF_blocks

        # HF_2: inter_variables correlations
        IV_blocks = HF_blocks.permute(0, 2, 3, 1)  # [32, 6, 8, 321]
        IV_cor = self.IVC(IV_blocks)
        # ablation 3:
        # IV_cor = IV_blocks

        # Mixer
        HF_features = self.Mixer((LT_cor + IV_cor.permute(0, 3, 1, 2)).reshape(b, s, -1))

        REC_features = self.idwt((LF_features, [HF_features])) # [32, 8, 48]

        out_features = self.seq2seq(REC_features)
        reco_input = out_features.transpose(-1, -2) * torch.sqrt(x_deno_var) + x_deno_mean
        rec_fea = REC_features.transpose(-1, -2) * torch.sqrt(x_deno_var) + x_deno_mean
        # rec_fea = self.ELC(rec_fea)
        return reco_input, rec_fea

