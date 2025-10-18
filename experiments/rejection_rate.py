import numpy as np
from scipy import signal
import numbers
import errno
import os
import configparser
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
from src.models.optimization_nu import train_nu
from sklearn.metrics import accuracy_score

def experiment(model_name, cfg, thre_sets):
    

    n_sub = cfg.data.n_sub
    n_trial = cfg.data.n_trial

    train_trial = cfg.validation.train_trial
    test_trial = [i for i in range(n_trial) if i not in train_trial]
    

    for s in range(n_sub):
        # s = 2
        acc = np.empty(cfg.data.n_trial - 1)
        # モデルの定義
        model = select_model(model_name, cfg, s)

        # モデルの学習(初期学習)
        data_train, label_train = data_utils.concatenate_data_with_class(cfg,s,train_trial)

        # data_train = data_train*1e3
        
        print(f'subject{s+1} initial training')
        model_initial_train(model_name, model, data_train, label_train)
        # model.fit(data_train, label_train, True)
        accuracy = accuracy_score(label_train, np.argmax(model.predict(data_train), axis = 1))
        print(f'initial accuracy : {accuracy}')
        acc[0] = accuracy

        # νの値を保存
        if model_name == 'BCMT_opt_nu':
            path_save_nu = f'{cfg.path.save}/parameters/{model_name}/{cfg.preprocess.cutoff_freq}Hz/'
            os.makedirs(path_save_nu, exist_ok=True)
            np.savetxt(path_save_nu + f'nu_sub{s+1}.csv', model.nu)

        # テスト
        for th in thre_sets:
            x_test_n_total = 0
            rejected_n_total = 0
            for t in test_trial:
                x_test, y_test = data_utils.concatenate_data_with_class(cfg,s,[t])
                x_test_n_total += len(x_test)
                # x_test = x_test*1e3
                
                y_prb = model.predict(x_test)
                y_prd = np.argmax(y_prb, axis = 1)
                accuracy = accuracy_score(y_prd, y_test)
                print(f'trial{t+1} accuracy : {accuracy}')
                acc[t-1] = accuracy
                # 逐次学習  
                rejected_n = model_sequential_train(model_name, model, x_test, y_test, y_prd, th)
                rejected_n_total += rejected_n
            
            rejection_rate = rejected_n_total / x_test_n_total
            print(f'rejection_rate : {rejection_rate*100}%')
            save_accuracy(acc, s, model_name, cfg, rejection_rate, th)
            acc[1:] = 0


                



def model_sequential_train(model_name, model, x_test, y_test, y_prd, th):
    if 'BCMT' in model_name:
        prediction_prb = model.predict(x_test)
        idx = np.max(prediction_prb, axis =1) > th
        model.fit(x_test[idx], y_prd[idx], False)
    elif 'BCMG' in model_name:
        prediction_prb = model.predict(x_test)
        idx = np.max(prediction_prb, axis =1) > th
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
    else:
        print(f'{model_name} sequential training')
        prediction_prb = model.predict(x_test)
        idx = np.max(prediction_prb, axis =1) > th
        model.fit(x_test[idx], y_prd[idx], True)
    
    

    return len(x_test) - len(x_test[idx])


def model_initial_train(model_name, model, data_train, label_train):
    if 'BCMT' in model_name:
        model.fit(data_train,label_train, True)
    elif 'BCMG' in model_name:
        model.fit(data_train, label_train)
    elif 'WithoutBSL' in model_name:
        model.fit(data_train, label_train, True)
    elif 'LabeledBSL' in model_name:
        model.fit(data_train, label_train, True)
    else:
        print(f'model_name : {model_name}')
        model.fit(data_train, label_train, False)
     


def save_accuracy(acc, s, model_name, cfg, rejection_rate, th):
    path = f'{cfg.path.save}/acc_by_rejection/2Hz/{model_name}/threhold{th}/'
    os.makedirs(path, exist_ok=True)
    np.savetxt(path + f'sub{s+1}_accuracy.csv', acc, delimiter=',')
    np.savetxt(path + f'sub{s+1}_rejection_rate.csv', np.array([rejection_rate]), delimiter=',')



def select_model(model_name, cfg, sub=None):
     
    if 'opt_nu' in model_name:
        print(f'optimized')
        return BayesianFiniteMixtureScaleMixtureModel(n_components=cfg.data.n_class, nu_optimize=True)
    elif 'BCMT' in model_name:
        nu = np.loadtxt(f'{cfg.path.save}/parameters/BCMT_opt_nu/{cfg.preprocess.cutoff_freq}Hz/nu_sub{sub+1}.csv',delimiter=',') 
        return BayesianFiniteMixtureScaleMixtureModel(n_components=cfg.data.n_class, nu_fix=nu[0], nu_optimize=False)
    elif 'LabeledBSL' in model_name:
        nu = np.loadtxt(f'{cfg.path.save}/parameters/BCMT_opt_nu/{cfg.preprocess.cutoff_freq}Hz/nu_sub{sub+1}.csv',delimiter=',')
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
        parameter = np.loadtxt(f'{cfg.path.save}/acc/{model_name}/par_sub{sub+1}.csv', delimiter=',')
        return Adaptive_SVM(n_class=cfg.data.n_class, c=parameter[0], gamma=parameter[1])
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


    # name_dataset = input("input the name of dataset :")


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


    # model_name = input("input he name of model :")



    name_dataset = 'kanoga_long'
    model_names =   ['BCMT','BCMG']

    # model_names =   ['BCMT_opt_nu','BCMT','WithoutBSL','LabeledBSL','BCMG','sklearnQDA','LDA','PC']
    # model_names =   ['BCMG']



    with open(f'../config/settings_{name_dataset}.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
    cfg = DotMap(cfg, _dynamic=False)

    thre_sets = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    for m in range(len(model_names)):
        experiment(model_names[m], cfg, thre_sets)


    # model_name = ['BCMT','LabeledBSL','WithoutBSL','BCMG','LDA','sklearnQDA','PC','SVM']
    # dataset_name = ['furui_lab','kanoga_long']

    # for d in range(len(dataset_name)):

    # for m in range(len(model_names)):
    #     with open(f'../config/settings_{name_dataset}.yaml', 'r') as f:
    #         cfg = yaml.safe_load(f)
    #     cfg = DotMap(cfg, _dynamic=False)
    #     experiment(model_names[m], cfg, thre_sets)


    



