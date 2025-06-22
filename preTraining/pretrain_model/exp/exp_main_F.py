from data_provider.data_factory import data_provider
from my_models.pretrain_model import SRLearning
from my_models.pretrain_model.exp.exp_basic import Exp_Basic
from utils.metrics import calculate_metrics
from utils.standardmask import loss2

from utils.tools import EarlyStopping, adjust_learning_rate, point_masking

import torch
import torch.nn as nn
from torch import optim
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from thop import profile

warnings.filterwarnings('ignore')


# SRLearning_pru = prune_fc_layers(SRLearning, amount=0.1)
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')

        num_blocks = self.args.seq_len // self.args.block_size
        # self.masking, mask_index = point_masking(self.args.batch_size, num_blocks, self.args.block_size, self.args.enc_in, self.args.mask_ratio)
        # self.mask_index = mask_index.reshape(self.args.batch_size,  -1, self.args.enc_in)

        # self.masking = self.args.masking
        self.mask_index = self.args.mask_index

    def _build_model(self):
        model_dict = {
            'pre_training': SRLearning
        }
        pretrain_model = model_dict['pre_training'].Model(self.args).float()
 
        # pretrain_model = prune_fc_layers(pretrain_model,amount=0.5)
        return pretrain_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _pretrain_optimizer(self):
        model_optim = optim.Adam(self.pretrain_model.parameters(), lr=self.args.learning_rate)
        print('!!!!!!!!!!!!!!learning rate______F____!!!!!!!!!!!!!!!')
        print(self.args.learning_rate)
        return model_optim

    def _predict_optimizer(self):
        model_optim = optim.Adam(self.predict_model.parameters(), lr=self.args.learning_rate)
        print('!!!!!!!!!!!!!!learning rate______F____!!!!!!!!!!!!!!!')
        print(self.args.learning_rate)
        return model_optim
    def _noise_optimizer(self):
        noise_optim = optim.Adam(self.noisemodel.parameters(), lr=self.args.learning_rate)
        print('!!!!!!!!!!!!!!learning rate______F____!!!!!!!!!!!!!!!')
        print(self.args.learning_rate)
        return noise_optim

    def block_loss(self, preds, target):
        loss = (preds.cpu() - target.cpu()) ** 2
        mask = self.mask_index.cpu()
        loss = (loss * mask).sum() / mask.sum()
        return loss


    def block_loss_ablation(self, preds, target):
        loss = (preds.cpu() - target.cpu()) ** 2
        mask = self.mask_index.cpu()
        loss = (loss * mask).sum() / mask.sum()
        mse = loss
        mae = np.abs(preds.cpu() - target.cpu())
        mae = (mae * mask).sum() / mask.sum()
        return loss, mse, mae

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_profile(self, model):
        _input = torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in).to(self.device)
        macs, params = profile(model, inputs=(_input,))
        print('FLOPs: ', macs / 1e9, 'GFLOPs')
        print('params: ', params / 1e6, 'M')
        return macs, params


    def pre_train(self, name):
        # self._get_profile(self.model)
        # print('Trainable parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        path = os.path.join(self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, name=self.args.model_id)
        pretrain_optim = self._pretrain_optimizer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.pretrain_model.train()
            epoch_time = time.time()

            for i, (batch_denoise_x, batch_denoise_y) in enumerate(self.train_loader):
                iter_count += 1
                pretrain_optim.zero_grad()
                batch_denoise_x = batch_denoise_x.float().to(self.device)
                rec_input, in_Fea = self.pretrain_model(batch_denoise_x)


                loss = self.block_loss(rec_input, batch_denoise_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\t Pre-Training ===> iters: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\t Pre-Training ===> speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                pretrain_optim.step()

            print("Pre-Trainin ===> Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss,_,_ = self.pre_vali(self.vali_loader)
            test_loss, mse, mae = self.pre_vali(self.test_loader)

            print("Pre-Training ===> Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}  Test MSE: {4:.7f}  Test MAE: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, mse, mae))
            early_stopping(vali_loss, self.pretrain_model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(pretrain_optim, epoch + 1, self.args)
        return self.pretrain_model

    def pre_vali(self, vali_loader):
        total_loss = []
        mse_list =[]
        mae_list=[]
        self.pretrain_model.eval()
        with torch.no_grad():
            for i, (batch_denoise_x, batch_denoise_y) in enumerate(vali_loader):
                batch_denoise_x = batch_denoise_x.float().to(self.device)

                rec_out, pre_weights = self.pretrain_model(batch_denoise_x)
                pred = rec_out.detach().cpu()
                true = batch_denoise_x.detach().cpu()
                loss, mse, mae = self.block_loss_ablation(pred, true)
                total_loss.append(loss)
                mse_list.append(mse)
                mae_list.append(mae)

        total_loss = np.average(total_loss)
        total_mse = np.average(mse_list)
        total_mae = np.average(mae_list)
        self.pretrain_model.train()
        return total_loss, total_mse, total_mae


