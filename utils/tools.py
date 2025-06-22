import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

np.random.seed(0)
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    # if args.lradj == 'type1':
    #     lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False,name='ETTh1', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.name = name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, path + '/' + self.name+'.pth')
        self.val_loss_min = val_loss




def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def generate_mask(channels, rows, cols, ratio):
    matrix_3D = np.zeros((channels, rows, cols), dtype=int)

    for d in range(channels):
        for i in range(rows):
            rand_col = np.random.choice(cols, size=int(cols * (1 - ratio)), replace=False)
            matrix_3D[d, i, rand_col] = 1

    for d in range(channels):
        for j in range(cols):
            if np.sum(matrix_3D[d, :, j]) == 0:
                rand_row = np.random.randint(rows)
                matrix_3D[d, rand_row, j] = 1

    for j in range(cols):
        for k in range(channels):
            if np.sum(matrix_3D[k, :, j]) == 0:
                rand_row = np.random.randint(rows)
                matrix_3D[k, rand_row, j] = 1
        for r in range(rows):
            if np.sum(matrix_3D[:, r, j]) == 0:
                rand_row = np.random.randint(channels)
                matrix_3D[rand_row, r, j] = 1
    return torch.from_numpy(matrix_3D)

def point_masking( b, channels, rows, cols, mask_ratio, device):
    # 2 Random masking (32,60,16,8)
    # b, channels, rows, cols = x_block.shape
    mask_idx = []
    for i in range(b):
        mask_matrix = generate_mask(channels, rows, cols, mask_ratio)
        mask_idx.append(mask_matrix)

    masking = torch.stack(mask_idx, dim=0)
    # x_enc_masked = x_block * mask_index
    mask_index = (~masking.bool()).int()
    return masking.to(device), mask_index.to(device)  # The input data has been masked; The masked data index




def read_data(file_path, data_class, seq_len, period, norm=True):
    scaler = StandardScaler()
    df_raw = pd.read_csv(file_path, header=0, encoding='latin-1')
    if data_class == 'ETTh':
        border1s = [0, 12 * 30 * 24 + 4 * 30 * 24 - seq_len - period,  12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24 + 4 * 30 * 24 - period,   12 * 30 * 24 + 4 * 30 * 24,    12 * 30 * 24 + 8 * 30 * 24]
    elif data_class == 'ETTm':
        border1s = [0, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len - period,  12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - period, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        # num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = period
        num_train = len(df_raw) - num_test - num_vali
        border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]

    if norm:
        # train_data = df_data
        scaler.fit(df_data.values)
        data = scaler.transform(df_data.values)
        # data_wave = wavelet_noising_column(data)
    else:
        data = df_data.values

    pre_train = data[border1s[0]:border2s[0]]
    valid = data[border1s[1]:border2s[1]]
    test = data[border1s[2]:border2s[2]]
    # down_forecast_dataset = data[border1s[1]:border2s[2]]

    # pre_train_denoise = data_wave[border1s[0]:border2s[0]]
    # valid_denoise = data_wave[border1s[1]:border2s[1]]
    # test_denoise = data_wave[border1s[2]:border2s[2]]
    return pre_train, valid, test

def getDataMatrix(sequence, multi_uni="M", seq_len=96,  pre_len=48, stride = 1):
    if multi_uni=="MS":
        data_len = len(sequence) - seq_len - pre_len
        X = np.zeros(shape=(data_len, seq_len, sequence.shape[1]))
        T = np.zeros(shape=(data_len, pre_len))
        for i in range(data_len):
            X[i, :] = np.array(sequence[i:(i + seq_len)])
            T[i, :] = np.array(sequence.iloc[:, -1][(i + seq_len):(i + seq_len + pre_len)])
    else:

        data_len = int((len(sequence)-seq_len-pre_len)/stride)-1
        X = np.zeros(shape=(data_len, seq_len, sequence.shape[1]))
        T = np.zeros(shape=(data_len, pre_len, sequence.shape[1]))
        j = 0
        for i in range(data_len):
            X[i, :] = np.array(sequence[j:(j + seq_len)])
            T[i, :] = np.array(sequence[(j + seq_len):(j + seq_len + pre_len)])
            j = j + stride
    return X, T

def norm_data(sequence):
    scaler = StandardScaler()
    scaler.fit(sequence)
    seq = scaler.transform(sequence)
    return seq

