import numpy as np
from scipy import signal
import numbers
import errno
import os
import configparser
import torch
import sys
sys.path.append('..')
import pandas as pd
from src import utils
from src.data import data_utils
import yaml
from dotmap import DotMap
from src.data.data_utils import config_parser
from src.models.bayesian_fmsmm import BayesianFiniteMixtureScaleMixtureModel
from src.models.bayes_gmm import BayesianGaussianMixtureClassifier
from src.models.adaptive_pc import Adaptive_PC
from src.models.alda import AdaptiveLDA
from src.models.sklearn_qda import sklearnQDA
from src.models.asvm import Adaptive_SVM
from src.models.cnn import EMGClassificationCNN
from src.models.mlp import MLP
from src.models.optimization_nu import train_nu
from sklearn.metrics import accuracy_score

def experiment(model_name, cfg):

    n_sub = [i for i in range(cfg.data.n_sub)]
    n_trial = cfg.data.n_trial

    train_trial = cfg.validation.train_trial
    test_trial = [i for i in range(n_trial) if i not in train_trial]
    

    for s in n_sub:
        acc = []
        # モデルの定義
        model = select_model(model_name, cfg, s)

        # モデルの学習(初期学習)
        data_train, label_train = data_utils.concatenate_data_with_class(cfg,s,train_trial)
        
        data_val, label_val = None, None
        
        if 'CNN' in model_name or 'MLP' in model_name:
            data_train, label_train = data_utils.concatenate_data_with_class(cfg,s,[train_trial[0]])
            data_val, label_val = data_utils.concatenate_data_with_class(cfg,s,[train_trial[1]])
            
            # data_train, label_train = data_utils.reshape_tensor(data_train, label_train, cfg.data.n_class, cfg.preprocess.window_size, cfg.preprocess.sliding, cfg.preprocess.fs)
            data_val, label_val = data_utils.reshape_tensor(data_val, label_val, cfg.data.n_class, cfg.preprocess.window_size, cfg.preprocess.sliding, cfg.preprocess.fs)

        print(f'subject{s+1} initial training')
        model_initial_train(model_name, model, data_train, label_train, s, cfg, data_val, label_val)
        # model.fit(data_train, label_train, True)
        accuracy = accuracy_score(label_train, np.argmax(model.predict(data_train), axis = 1))
        print(f'initial accuracy : {accuracy}')
        acc.append(accuracy)

        # νの値を保存
        if model_name == 'BCMT_opt_nu':
            path_save_nu = f'{cfg.path.save}/parameters/{model_name}/{cfg.preprocess.cutoff_freq}Hz/'
            os.makedirs(path_save_nu, exist_ok=True)
            np.savetxt(path_save_nu + f'nu_sub{s+1}_optimization-fix-extracted.csv', model.nu)

        # テスト
        for t in test_trial:
            x_test, y_test = data_utils.concatenate_data_with_class(cfg,s,[t])
            
            if 'CNN' in model_name or 'MLP' in model_name:
                x_test, y_test = data_utils.reshape_tensor(x_test, y_test, cfg.data.n_class, cfg.preprocess.window_size, cfg.preprocess.sliding, cfg.preprocess.fs)

            y_prb = model.predict(x_test)
            y_prd = np.argmax(y_prb, axis = 1)
            accuracy = accuracy_score(y_prd, y_test)
            if 'CNN' in model_name or 'MLP' in model_name:
                y_prd = torch.from_numpy(y_prd).long()
            print(f'trial{t+1} accuracy : {accuracy}')
            acc.append(accuracy)
            # 逐次学習  
            model_sequential_train(model_name, model, x_test, y_test, y_prd, s, cfg)
            
        # if 'opt_nu' not in model_name:
            save_accuracy(acc, s, model_name, cfg)


def model_sequential_train(model_name, model, x_test, y_test, y_prd, s, cfg):
    if 'BCMT' in model_name:
        prediction_prb = model.predict(x_test)
        idx = np.max(prediction_prb, axis =1) > 0.5
        model.fit(x_test[idx], y_prd[idx], False)
    elif 'BCMG' in model_name:
        prediction_prb = model.predict(x_test)
        idx = np.max(prediction_prb, axis =1) > 0.5
        model.fit(x_test[idx], y_prd[idx])
    elif 'LabeledBSL' in model_name:
        model.fit(x_test, y_test, False)
    elif 'WithoutBSL' in model_name:
        return
    elif 'PC' in model_name:
        print(f'model_name : {model_name}')
        model.fit(x_test, y_prd, True)
    elif 'SVM' in model_name:
        print(f'model_name : {model_name}')
        model.fit(x_test, y_prd, True)
    elif 'CNN' in model_name:
        prediction_prb = model.predict(x_test)
        idx = np.max(prediction_prb, axis =1) > 0.5
        if len(x_test[idx]) == 0:
            print('No data for sequential training')
            return
        model.fit(x_test[idx], y_prd[idx], s, cfg, True)
    elif 'MLP' in model_name:
        prediction_prb = model.predict(x_test)
        idx = np.max(prediction_prb, axis =1) > 0.5
        if len(x_test[idx]) == 0:
            print('No data for sequential training')
            return
        model.fit(x_test[idx], y_prd[idx], s, cfg, True)
    else:
        print(f'{model_name} sequential training')
        prediction_prb = model.predict(x_test)
        idx = np.max(prediction_prb, axis =1) > 0.5
        model.fit(x_test[idx], y_prd[idx], True)


def model_initial_train(model_name, model, data_train, label_train, s, cfg, data_val=None, label_val=None):
    if 'BCMT' in model_name:
        model.fit(data_train,label_train, True)
    elif 'BCMG' in model_name:
        model.fit(data_train, label_train)
    elif 'WithoutBSL' in model_name:
        model.fit(data_train, label_train, True)
    elif 'LabeledBSL' in model_name:
        model.fit(data_train, label_train, True)
    elif 'CNN' in model_name:
        model.fit(data_train, label_train, s, cfg, False, data_val, label_val)
    elif 'MLP' in model_name:
        model.fit(data_train, label_train, s, cfg, False, data_val, label_val)
    else:
        print(f'model_name : {model_name}')
        model.fit(data_train, label_train, False)
     


def save_accuracy(acc, s, model_name, cfg):
    path = f'{cfg.path.save}/acc/{model_name}_fix-window-ft/'
    os.makedirs(path, exist_ok=True)
    np.savetxt(path + f'sub{s+1}_accuracy.csv', acc, delimiter=',')



def select_model(model_name, cfg, sub=None):
     
    if 'opt_nu' in model_name:
        print(f'optimized')
        return BayesianFiniteMixtureScaleMixtureModel(n_components=cfg.data.n_class, nu_optimize=True)
    elif 'BCMT' in model_name:
        nu = np.loadtxt(f'{cfg.path.save}/parameters/BCMT_opt_nu/{cfg.preprocess.cutoff_freq}Hz/nu_sub{sub+1}.csv',delimiter=',') 
        # nu = np.loadtxt(f'{cfg.path.save}/parameters/BCMT_opt_nu/1Hz/nu_sub{sub+1}.csv',delimiter=',')
        return BayesianFiniteMixtureScaleMixtureModel(n_components=cfg.data.n_class, nu_fix=nu[0], nu_optimize=False)
    elif 'LabeledBSL' in model_name:
        nu = np.loadtxt(f'{cfg.path.save}/parameters/BCMT_opt_nu/{cfg.preprocess.cutoff_freq}Hz/nu_sub{sub+1}.csv',delimiter=',')
        return BayesianFiniteMixtureScaleMixtureModel(n_components=cfg.data.n_class, nu_fix=nu[0], nu_optimize=False)
    elif 'WithoutBSL' in model_name:
        nu = np.loadtxt(f'{cfg.path.save}/parameters/BCMT_opt_nu/{cfg.preprocess.cutoff_freq}Hz/nu_sub{sub+1}.csv',delimiter=',')
        return BayesianFiniteMixtureScaleMixtureModel(n_components=cfg.data.n_class, nu_fix=nu[0], nu_optimize=False)
    elif 'BCMG' in model_name:
        return BayesianGaussianMixtureClassifier(n_class=cfg.data.n_class)
    elif 'LDA' in model_name:
        return AdaptiveLDA(n_class=cfg.data.n_class)
    elif 'QDA' in model_name:
        return sklearnQDA(n_class=cfg.data.n_class)
    elif 'PC' in model_name:

        return Adaptive_PC(n_class=cfg.data.n_class)
    elif 'SVM' in model_name:
        parameter = np.loadtxt(f'{cfg.path.save}/parameters/{model_name}/par_sub{sub+1}.csv', delimiter=',')
        return Adaptive_SVM(n_class=cfg.data.n_class, c=parameter[0], gamma=parameter[1])
    elif 'CNN' in model_name:
        return EMGClassificationCNN(input_dim=cfg.data.n_channel,n_class=cfg.data.n_class)
    elif 'MLP' in model_name:
        n_channel = int(cfg.data.n_channel * (cfg.preprocess.window_size * cfg.preprocess.fs/1000))
        return MLP(n_channel=n_channel, n_class=cfg.data.n_class)
    else:
        raise Exception('Error: undefined model.')
     


    
      

if __name__ == '__main__':

    print('----------------------------------------------------')
    print('\n')
    print('   furui_lab: 10  trial 6 class')
    print(' kanoga_long: 120 trial 8 class')
    print('kanoga_short: 5   trial 8 class')
    print('     ninapro: 10  trial 6 class')
    print('\n > ', end="")
    print('----------------------------------------------------')


    print('----------------------------------------------------')
    print('\n')
    print('             BCMT: Bayesian Classification Model based T-distribution')
    print('      BCMT_opt_nu: Optimized-nu Bayesian Classification Model based T-distribution')
    print('             BCMG: Bayesian Classification Model based Gaussian distribution')
    print('       WithoutBSL: Without Sequential learning')
    print('       LabeledBSL: Sequential labeld self training')
    print('              LDA: Adaptive LDA')
    print('              QDA: Adaptive QDA')
    print('       sklearnQDA: sklearn QDA')
    print('              SVM: Adaptive SVM')
    print('               PC: Adaptive PC')
    print('\n > ', end="")
    print('----------------------------------------------------')

    name_dataset_set = ['kanoga_long', 'khushaba1']
    model_names =   ['BCMT']

    for name_dataset in name_dataset_set:
        for m in range(len(model_names)):
            with open(f'../config/settings_{name_dataset}.yaml', 'r') as f:
                cfg = yaml.safe_load(f)
            cfg = DotMap(cfg, _dynamic=False)
            experiment(model_names[m], cfg)

    



