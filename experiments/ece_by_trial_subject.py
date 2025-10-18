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
from src.models.asvm import Adaptive_SVM
from src.models.sklearn_qda import sklearnQDA
from src.models.optimization_nu import train_nu
from sklearn.metrics import accuracy_score

def experiment(model_name, cfg):
    
    n_sub = cfg.data.n_sub

    n_trial = cfg.data.n_trial

    train_trial = cfg.validation.train_trial
    test_trial = [i for i in range(n_trial) if i not in train_trial]


    for s in n_sub:

        acc = []
        # モデルの定義
        model = select_model(model_name, cfg, s)

        # モデルの学習(初期学習)
        data_train, label_train = data_utils.concatenate_data_with_class(cfg,s,train_trial)
        
        print(f'subject{s+1} initial training')
        model_initial_train(model_name, model, data_train, label_train)
        # model.fit(data_train, label_train, True)
        accuracy = accuracy_score(label_train, np.argmax(model.predict(data_train), axis = 1))
        print(f'initial accuracy : {accuracy}')
        acc.append(accuracy)

        # テスト
        for t in test_trial:
            probability = np.empty((0, cfg.data.n_class))
            ece_label = np.empty(0)

            x_test, y_test = data_utils.concatenate_data_with_class(cfg,s,[t])
            
            y_prb = model.predict(x_test)
            probability = np.vstack((probability, y_prb))
            y_prd = np.argmax(y_prb, axis = 1)
            ece_label = np.hstack((ece_label, y_test))
            accuracy = accuracy_score(y_prd, y_test)
            print(f'trial{t+1} accuracy : {accuracy}')
            acc.append(accuracy)
            # 逐次学習  
            # idx = np.max(prediction_prb, axis =1) > 0.9
            model_sequential_train(model_name, model, x_test, y_test, y_prd)
            # model.fit(data_test[idx], label_prediction[idx], False)

            ece_, acc_, conf_ = expected_calibration_error(probability, ece_label)
            save_accuracy(s, t, ece_, acc_, conf_, cfg, model_name)

    # return probability, ece_label


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
        print(f'model_name : {model_name}')
        model.fit(x_test, y_prd, True)
    else:
        print(f'{model_name} sequential training')
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
     


def save_accuracy(s, t, ece, acc, conf, cfg, model_name):
    # path = f'{cfg.path.save}/ece_by_trial/{model_name}/'
    path = f'{cfg.path.save}/ece_by_trial_st/{model_name}/sub{s+1}/'
    os.makedirs(path, exist_ok=True)
    np.savetxt(path + f'trial{t+1}_ece.csv', ece, delimiter=',')
    np.savetxt(path + f'trial{t+1}_ece_accuracy.csv', acc, delimiter=',')
    np.savetxt(path + f'trial{t+1}_ece_confidence.csv', conf, delimiter=',')


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
        parameter = np.loadtxt(f'{cfg.path.save}/acc/{model_name}/par_sub{sub+1}.csv', delimiter=',')
        return Adaptive_SVM(n_class=cfg.data.n_class, c=parameter[0], gamma=parameter[1])
    else:
        raise Exception('Error: undefined model.')
     


def expected_calibration_error(samples, true_labels, M=10):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    accuracy_arr = np.zeros(M)
    confidences_arr = np.zeros(M)

    ece = np.zeros(1)

    idx = 0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            accuracy_arr[idx] = accuracy_in_bin
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            confidences_arr[idx] = avg_confidence_in_bin
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
        idx += 1
    return ece, accuracy_arr, confidences_arr
    
      

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

 
    

    name_datasets = ['ninaprodb3_subset']
    model_names = ['BCMT']

    for name_dataset in name_datasets:
        for model_name in model_names:
            print(f'model_name : {model_name}')
            with open(f'../config/settings_{name_dataset}.yaml', 'r') as f:
                    cfg = yaml.safe_load(f)
            cfg = DotMap(cfg, _dynamic=False)
            
            experiment(model_name, cfg)


