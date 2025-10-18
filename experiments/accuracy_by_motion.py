import numpy as np
from scipy import signal
import numbers
import errno
import os
import configparser
import random
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


def experiment(model_name, cfg, seed_id):
    
    n_sub = [i for i in range(cfg.data.n_sub)]
    if cfg.cfg_name == 'ninaprodb3_subset':
        n_sub = [1, 4, 8, 10]
    if cfg.cfg_name == 'ninaprodb3_subset2':
        n_sub = [1, 4, 8, 10]
    n_trial = cfg.data.n_trial
    n_class = cfg.data.n_class
    
    class_trial = [i for i in range(n_class)]
    train_trial = cfg.validation.train_trial
    test_trial = [i for i in range(n_trial) if i not in train_trial]

    
    models = []

    print('\n Model name : {0}\n'.format(model_name))

    #　初期学習
    for s in n_sub:
        acc_initial = np.empty(0)

        model = select_model(model_name, cfg, s)
        data_train, label_train = data_utils.concatenate_data_with_class(cfg,s,train_trial)
        model_initial_train(model_name, model, data_train, label_train)
        # model.fit(data_train, label_train, True)
        accuracy = accuracy_score(label_train, np.argmax(model.predict(data_train), axis = 1))
        print(f'initial accuracy : {accuracy}')
        acc_initial = np.append(acc_initial, accuracy)
        save_accuracy(acc_initial, s, model_name, cfg, seed_id)

        models.append(model)


    # 逐次学習
    for i, t in enumerate(test_trial):

        random.shuffle(class_trial)

        for s in n_sub:
            model = models[s]
            acc_motion = np.empty(0)
            for c in class_trial:
                x_test, y_test = data_utils.concatenate_data_with_class_by_motion(cfg, s, t, c)
                y_prb = model.predict(x_test)
                y_prd = np.argmax(y_prb, axis = 1)
                accuracy = accuracy_score(y_prd, y_test)
                # print(f'trial{t+1} class{c+1} accuracy : {accuracy}')
                acc_motion = np.append(acc_motion, accuracy)
                model_sequential_train(model_name, model, x_test, y_test, y_prd)
                print('\r > Seed {0}, Trial: {1},  Subject: {2}, Class: {3} Accuracy: {4}'.format(seed_id+1, t+1, s+1, c+1, accuracy), end='', flush=True)
                # model.fit(data_test[idx], label_prediction[idx], False)
            # print(f'trial{t+1} accuracy : {acc_motion}')
            save_accuracy_by_motion(acc_motion, s, t, model_name, cfg, seed_id)

    print('\nDone!\n')


def model_sequential_train(model_name, model, x_test, y_test, y_prd):
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
        # print(f'model_name : {model_name}')
        model.fit(x_test, y_prd, True)
    else:
        # print(f'{model_name} sequential training')
        prediction_prb = model.predict(x_test)
        idx = np.max(prediction_prb, axis =1) > 0.5
        model.fit(x_test[idx], y_prd[idx], True)



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
     

def save_accuracy(acc, s, model_name, cfg, seed_id):
    path = f'{cfg.path.save}/acc_by_motion/{cfg.preprocess.cutoff_freq}Hz/{model_name}/seed{seed_id}/sub{s+1}/'
    os.makedirs(path, exist_ok=True)
    np.savetxt(path + f'initial.csv', acc, delimiter=',')

def save_accuracy_by_motion(acc_motion, s, t, model_name, cfg, seed_id):
    path = f'{cfg.path.save}/acc_by_motion/{cfg.preprocess.cutoff_freq}Hz/{model_name}/seed{seed_id}/sub{s+1}/'
    os.makedirs(path, exist_ok=True)
    np.savetxt(path + f'trial{t+1}.csv', acc_motion, delimiter=',')



def select_model(model_name, cfg, sub=None):
     
    if 'opt_nu' in model_name:
        print(f'optimized')
        return BayesianFiniteMixtureScaleMixtureModel(n_components=cfg.data.n_class, nu_optimize=True)
    elif 'BCMT' in model_name:
        nu = np.loadtxt(f'{cfg.path.save}/parameters/BCMT_opt_nu/{cfg.preprocess.cutoff_freq}Hz/nu_sub{sub+1}.csv',delimiter=',') 
        # nu = np.loadtxt(f'{cfg.path.save}/parameters/BCMT_opt_nu/1Hz/nu_sub{sub+1}.csv',delimiter=',')
        return BayesianFiniteMixtureScaleMixtureModel(n_components=cfg.data.n_class, nu_fix=1, nu_optimize=False)
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



    name_datasets = ['khushaba1']

    model_names =   ['BCMT']


    for name_dataset in name_datasets:
        with open(f'../config/settings_{name_dataset}.yaml', 'r') as f:
                cfg = yaml.safe_load(f)
        cfg = DotMap(cfg, _dynamic=False)
        seed_list = [7,8,9]

        for model_name in model_names:
            for i in seed_list:
                random.seed(i)
                experiment(model_name,cfg,i)



