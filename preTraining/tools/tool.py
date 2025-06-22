import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
np.random.seed(0)


def read_data(file_path, data_class, seq_len, period, norm=True):
    scaler = StandardScaler()
    df_raw = pd.read_csv(file_path, encoding='latin-1')
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
        train_data = df_data[border1s[0]:border2s[0]]
        scaler.fit(train_data.values)
        data = scaler.transform(df_data.values)
        # data_wave = wavelet_noising_column(data)
    else:
        data = df_data.values

    pre_train = data[border1s[0]:border2s[0]]
    valid = data[border1s[1]:border2s[1]]
    test = data[border1s[2]:border2s[2]]

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

import matplotlib.pyplot as plt

def visual(true, preds=None, name='./pic/test.pdf'):
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')