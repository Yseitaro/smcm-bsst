import numpy as np
from scipy import signal
import random
import os
import configparser

def rectification(data):
    """full-wave rectification
    信号の全波整流を計算する
    """
    return np.abs(data)

def remove_noise(data, order, sampleing_freq):
    # order = 4 #次数
    cutoff = np.array([60,63]) # カットオフ周波数

    # Butterworthフィルタの設計（`btype=bandstop`でバンドストップフィルタになる）
    b, a = signal.butter(order, cutoff, btype='bandstop', analog=False, fs=sampleing_freq)

    return signal.filtfilt(b, a, data, axis = 0)


def smoothing(data, order, sampling_freq, cutoff_freq):
    """Smoothing using butterworth lowpass filter
    ローパスフィルタを用いて信号を平滑化する
    """
    Wn = cutoff_freq / (sampling_freq/2)
    b, a = signal.butter(order, Wn, 'low')

    return signal.filtfilt(b, a, data, axis=0)

def remove_transient_state(filt_data, reject_rate):
    """Remove transient state from filtered signals
    データ冒頭の動作遷移の区間を除去する
    """
    n_samples, _ = filt_data.shape
    start_point = n_samples*reject_rate

    return filt_data[int(start_point):]

def get_shuffled_data(data, random_state):
    """Shuffle data
    データをシャッフルする
    """
    n_samples = data.shape[0]

    ind = random_state.choice(np.arange(0, n_samples, 1), 
                            size=n_samples, replace=False)

    return data[ind]