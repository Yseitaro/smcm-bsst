from importlib.resources import path
import sys
sys.path.append('..')
from sklearn.metrics import accuracy_score
import numpy as np
from src.data import data_utils
import os
from src.models.asvm import Adaptive_SVM
import random
import yaml
from dotmap import DotMap
import codecs


def parameter_tuning(sub, cfg):

    train_trail = cfg.validation.train_trial
    
    for t in train_trail:
        if t == 0:
            data_train,label_train = data_utils.concatenate_data_with_class(cfg, sub, [t])
            data_train = data_train[::50]
            label_train = label_train[::50]
            print('get data')
        else:
            data_val,label_val = data_utils.concatenate_data_with_class(cfg, sub, [t])
            data_val = data_val[::50]
            label_val = label_val[::50]


    c = [2**-20,2**-15,2**-10,2**-5,2**0,2**5,2**10,2**15,2**20]
    gamma= [2**-20,2**-15,2**-10,2**-5,2**0,2**5,2**10,2**15,2**20]


    
    best_accuracy = -1
    for c_ in c:
        for gamma_ in gamma:
            print('before SVM')
            s3vm1 = Adaptive_SVM(n_class=cfg.data.n_class, c=c_, gamma=gamma_, cycle_subst_prop=0.5)
            s3vm1.fit(data_train, label_train, adapt=False)
            print(f'check')
            preds1 = np.argmax(s3vm1.predict(data_val),axis=1)
            accuracy_1=accuracy_score(preds1,label_val)                      
      
            s3vm2 = Adaptive_SVM(n_class=cfg.data.n_class, c=c_, gamma=gamma_, cycle_subst_prop=0.5)
            s3vm2.fit(data_val, label_val, adapt=False)
            preds2 = np.argmax(s3vm1.predict(data_train),axis=1)
            accuracy_2=accuracy_score(preds2,label_train)
   
            accuracy = (accuracy_1+accuracy_2)/2

            print(f'Accuracy:{accuracy} c:{c_} gamma:{gamma_}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_c = c_
                best_gamma = gamma_

            print(f"Best parameter: best_accuracy:{best_accuracy} c:{best_c} gamma:{best_gamma}")
    
    return {f'sub{sub+1}' : {'c' : best_c, 'gamma' : best_gamma}}


# if __name__ == "__main__":
def svm_tuning():

    print('----------------------------------------------------')
    print('\n')
    print('   furui_lab: 10  trial 6 class')
    print(' kanoga_long: 120 trial 8 class')
    print('kanoga_short: 5   trial 8 class')
    print('     ninapro: 10  trial 6 class')
    print('\n > ', end="")
    print('----------------------------------------------------')

    name_datasets = ['ninaprodb3_subset']

    for name_dataset in name_datasets:

        with open(f'../config/settings_{name_dataset}.yaml', 'r') as f:
                cfg = yaml.safe_load(f)
        cfg = DotMap(cfg, _dynamic=False)

        n_sub = cfg.data.n_sub

        for s in range(n_sub):
            print(f'subject {s+1} start')

            param_args = parameter_tuning(s, cfg)
            print(f"Best parameter: c:{param_args[f'sub{s+1}']['c']} gamma:{param_args[f'sub{s+1}']['gamma']}")

            if s == 0:
                path = f'../results/{name_dataset}/parameters/SVM'
                os.makedirs(path, exist_ok=True)
                np.savetxt(f'{path}/par_sub{s+1}.csv',[param_args[f'sub{s+1}']['c'],param_args[f'sub{s+1}']['gamma']],delimiter=',')
            else:
                np.savetxt(f'{path}/par_sub{s+1}.csv',[param_args[f'sub{s+1}']['c'],param_args[f'sub{s+1}']['gamma']],delimiter=',')

            print(f'subject {s+1} end')
            print('\n')
            print('----------------------------------------------------')


