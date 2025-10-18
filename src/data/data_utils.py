from unicodedata import category
import numpy as np
from scipy import signal
import numbers
import errno
import os
import configparser
import sys
import pandas as pd
from .process_emg import rectification, remove_noise
from .process_emg import smoothing
from .process_emg import remove_transient_state
from .process_emg import get_shuffled_data
import errno
import os
import configparser
import torch
import torch.nn.functional as nnf
from torch.utils.data import TensorDataset
import pickle


def config_parser(path):
    """Read `config.ini` to get setting
    """
    config_ini = configparser.ConfigParser()

    if not os.path.exists(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    config_ini.read(path + '/config.ini', encoding='utf-8')

    read_setting = config_ini['SETTING']

    n_sub = read_setting.getint('n_sub')
    n_trial = read_setting.getint('n_trial')
    n_channel = read_setting.getint('n_channel')
    n_class = read_setting.getint('n_class')
    sampling_freq = read_setting.getint('sampling_freq')

    return n_sub, n_trial, n_channel, n_class, sampling_freq

def _make_write_directory(data_path, category, sub):
    """Make directory for reshaped files
    """
    path='{0}/{1}/sub{2}'.format(data_path, category, sub)
    os.makedirs(path, exist_ok=True)


def make_write_file_path(data_path, category, sub, clas, trial):
    """Generate a file path for reshaped files
    """
    _make_write_directory(data_path, category, sub)

    path = '{0}/{1}/sub{2}/class{3}trial{4}.csv'.\
            format(data_path,category, sub, clas, trial)
    return path


def get_delimiter(extension):
    """Get deliminater depending on extension of file.
    """
    if extension == 'txt':
        return None
    elif extension == 'csv':
        return ','
    elif extension == 'dat':
        return '\t'
    else:
        return None


def make_read_path(data_path,category, sub, class_, trial):
    """Generate a file path for raw data 
    """
    path = '{0}/{1}/sub{2}/class{3}trial{4}.csv'.\
            format(data_path,category, sub, class_, trial)

    return path

def process_data(data_path,order=2, sampling_freq=2000, 
                 cutoff_freq=1.0, reject_rate = 0.1): 
    """Rectification, smoothing, and extraction for raw data
    """

    n_sub, n_trial, _, n_class, _=config_parser(data_path)
    # print(f'n_class : {n_class}')

    print('Processing EMG signals...')
    for s in range(n_sub):
        for t in range(n_trial):
            print('\r                                       ', end='', flush=True)
            for c in range(n_class):
                print('\r > subject: {0},  trial: {1}, class: {2}'.format(s+1, t+1, c+1), end='', flush=True)
                # print(f'c+1:{c+1}')
                fname = make_read_path(data_path, 'raw', s+1, c+1, t+1)
                read_data = np.loadtxt(fname,delimiter=',')

                # Rectified and smoothed data
                filtered_data = remove_noise(read_data,order,sampling_freq)
                filtered_data = rectification(filtered_data)
                filtered_data = smoothing(filtered_data, order,         
                                                  sampling_freq, cutoff_freq)
                category = 'filtered/{:.1f}Hz'.format(cutoff_freq)
                fname = make_write_file_path(data_path, category, s+1, c+1, t+1)
                np.savetxt(fname=fname, X=filtered_data, fmt='%.7e', delimiter=',')

                # extracted data
                extracted_data = remove_transient_state(filtered_data, reject_rate)
                category = 'extracted/{:.1f}Hz'.format(cutoff_freq)
                fname = make_write_file_path(data_path, category, s+1, c+1, t+1)
                np.savetxt(fname=fname, X=extracted_data, fmt='%.7e', delimiter=',')
    print('\nDone!\n')



def concatenate_data_with_class(cfg,sub,trial_set,cutoff=None,data_type='raw'):

    n_channel = cfg.data.n_channel
    n_class = cfg.data.n_class
    data_path = cfg.path.data_root
    cutoff = cfg.preprocess.cutoff_freq
    dataset_name = cfg.cfg_name
    # if dataset_name == 'ninaprodb3_short':
    #     print(f'ninapro short ver')
    #     n_class = [4,5,10,11,12,13]
    # else:
    #     n_class = [i for i in range(n_class)]
    n_class = [i for i in range(n_class)]
    if cfg.cfg_name == 'ninaprodb3_subset2':
        n_class = [4,5,8,9,12,13]
    if cfg.cfg_name == 'ninaprodb3_subset':
        n_class = [4,5,10,11,12,13]

    data = np.empty((0,n_channel))
    label = np.empty(0,dtype=int)

    for t in trial_set:
        for i, c in enumerate(n_class):
            category = 'extracted/{:.1f}Hz'.format(cutoff)
            fname = make_read_path(data_path,category,sub+1,c+1,t+1)
            if dataset_name == 'khushaba1':
                # print(f'yes')
                read_data = np.loadtxt(fname,delimiter=',')[:,range(n_channel)]
            else:
                read_data = np.loadtxt(fname,delimiter=',')
            n_sample = read_data.shape[0]
            
            data = np.vstack((data, read_data))
            label = np.hstack((label, np.full(n_sample,i,dtype=int)))
    return data,label

def concatenate_data_with_class_by_motion(cfg, sub, t, c, cutoff=None, data_type='raw'):

    n_channel = cfg.data.n_channel
    n_class = cfg.data.n_class
    data_path = cfg.path.data_root
    cutoff = cfg.preprocess.cutoff_freq
    dataset_name = cfg.cfg_name
    

    data = np.empty((0,n_channel))
    label = np.empty(0,dtype=int)

    # t = trial_set[0]
    # c = class_set[0]
   
    category = 'extracted/{:.1f}Hz'.format(cutoff)
    fname = make_read_path(data_path,category,sub+1,c+1,t+1)

    if dataset_name == 'khushaba1':
        # print(f'yes')
        read_data = np.loadtxt(fname,delimiter=',')[:,range(n_channel)]
    else:
        read_data = np.loadtxt(fname,delimiter=',')
    n_sample = read_data.shape[0]
    
    data = np.vstack((data, read_data))
    label = np.hstack((label, np.full(n_sample,c,dtype=int)))
    
    return data,label


def reshape_tensor(x, y, n_class, window_size, sliding, fs):

    # convert ms → samples
    wlen = int(fs * window_size / 1000)
    hop  = int(fs * sliding    / 1000)

    n_channel = x.shape[1]
    x_list, y_list = [], []

    for c in range(n_class):
        mask = (y == c)
        if not mask.any():
            raise ValueError(f'No data for class {c}')

        x_c = x[mask]                     # all samples of class c
        n_samps = x_c.shape[0]
        if n_samps < wlen:
            raise ValueError(f'Not enough samples ({n_samps}) for window_size {wlen}')

        # number of windows for this class
        n_win = 1 + (n_samps - wlen) // hop

        for i in range(n_win):
            start = i * hop
            end   = start + wlen
            segment = x_c[start:end]      # shape (wlen, n_channel)

            # to tensor: (1, 1, n_channel, wlen)
            seg_t = (
                torch.from_numpy(segment)
                     .float()
                     .transpose(0, 1)      # → (n_channel, wlen)
                     .unsqueeze(0)         # → (1, n_channel, wlen)
                     .unsqueeze(1)         # → (1, 1, n_channel, wlen)
            )

            x_list.append(seg_t)
            y_list.append(c)

    # concatenate all windows
    x_tensor = torch.cat(x_list, dim=0)      # (total_windows, 1, n_channel, wlen)
    y_tensor = torch.tensor(y_list, dtype=torch.long)

    return x_tensor, y_tensor


def check_random_state(seed):
    """
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
        


def reshape_dataset_to_tensor(X, y=None, structure='static'):
    """Reshape dataset to tensor
    """
    X_tensor = torch.from_numpy(X).float()
    if structure == 'dynamic':
        pass
        # X_tensor = torch.unsqueeze(X_tensor, dim=1)
    if y is not None:
        y_tensor = torch.from_numpy(y).long()
        return TensorDataset(X_tensor, y_tensor)
    else:
        return X_tensor

