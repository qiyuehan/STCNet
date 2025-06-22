
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_true, y_pred)

    # mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    nzero_indices = y_true !=0
    ape = np.abs((y_true[nzero_indices]-y_pred[nzero_indices])/y_true[nzero_indices])
    mape = np.mean(ape)*100

    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


    r2 = r2_score(y_true, y_pred)

    mae_naive, smape_naive = calculate_naive_metrics(y_true)

    owa = 0.5 * (mae / mae_naive + smape / smape_naive)
    return mse, mae, rmse, mape, smape, r2, owa



def calculate_naive_metrics(y_true):

    y_naive = np.roll(y_true, 1)
    y_naive[0] = y_true[0]  
    return calculate_m(y_true[1:], y_naive[1:])


def calculate_m(y_true, y_pred):

    mae = np.mean(np.abs(y_true - y_pred))
    ape = 2*np.abs(y_pred - y_true)
    mape = np.abs(y_pred)+np.abs(y_true)
    nz_idx = mape !=0
    smape = 100 * np.mean(ape[nz_idx]/mape[nz_idx])
    # smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    return mae, smape
