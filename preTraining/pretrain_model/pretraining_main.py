import argparse
import torch
import random
import numpy as np

from my_models.pretrain_model.exp.exp_main_F import Exp_Main
from utils.standardmask import random_masking, loss2
from utils.tools import point_masking

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='ETTh1', help='Pretrain_model Name')
parser.add_argument('--model', type=str, required=False, default='FITS',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data_class', type=str, required=False, default='ETTh', help='dataset type')
parser.add_argument('--root_path', type=str, default=r.\dataset\', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')


# Masking
parser.add_argument('--block_size', type=int, default=16, help='subsequence length')
parser.add_argument('--mask_ratio', type=float, default=0.7, help='masking rate')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--features', type=str, default='M', help=' variables')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

parser.add_argument('--seed', type=int, default=2021, help='size of augmented data, i.e, 1 means double the size of dataset')
parser.add_argument('--wave', type=str, default='haar', help='haar, dbn')

args = parser.parse_args()

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_blocks = args.seq_len // args.block_size

args.masking, mask_index = point_masking(args.batch_size, num_blocks, args.block_size, args.enc_in, args.mask_ratio, args.device)
args.mask_index = mask_index.reshape(args.batch_size, -1, args.enc_in)



fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


print('Args in experiment:')
# print(args)

torch.cuda.empty_cache()

Exp = Exp_Main

if __name__ == '__main__':
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}__sl{}_ll{}_pl{}_{}'.format(
                args.model_id,
                args.model,
                args.data_class,
                args.seq_len,
                args.label_len,
                args.pred_len,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start Pre-training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.pre_train(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_ll{}_pl{}_{}'.format(
                args.model_id,
                args.model,
                args.data_class,
                args.seq_len,
                args.label_len,
                args.pred_len,
                 ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
