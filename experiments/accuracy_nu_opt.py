import numpy as np
import sys
sys.path.append('..')
from src.models import bayesian_fmsmm
from src.data.data_utils import concatenate_data_with_class
from src.data.data_utils import config_parser
import os
import importlib
importlib.reload(bayesian_fmsmm)
from sklearn.metrics import accuracy_score
from src.models.parameter_tuning import nu_tune
from torch.nn import functional as F
from torch import nn
import torch
import numpy.linalg as LA
# 学習率スケジューラー
from torch.optim import lr_scheduler


class nu_optim(nn.Module):
    def __init__(self, nu, mu, Sigma, alpha):
        super(nu_optim, self).__init__()
        self.n_components = 6
        self.nu = nn.Parameter(torch.tensor(nu).float())
        self.mu = mu
        self.Sigma = Sigma
        self.alpha = alpha

    def log_likelihood(self, X, nu, mu, psi):
        """Calculate log-probability density function of scale mixture model
        Parameters
        ----------
            x:      input data
            nu:     degrees of freedom
            mu:     mean vector
            psi:    scale matrix
        Returns
        -------
            log_prob : ndarray, shape=(n_samples)
                Log-probability from a scale mixutre model
                > log p(x_n|z_n)
        """
        _, n_dim = X.shape
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        diffs_ = X.T - mu.reshape(-1, 1)
        delta_ = np.sum(diffs_ * (np.dot(LA.inv(psi), diffs_)), axis=0)
        delta_ = torch.tensor(delta_).float()


        log_prb1 = torch.special.gammaln((nu + n_dim) / 2.) - torch.special.gammaln(nu / 2.)
        log_prob2 = -0.5 * torch.log(torch.tensor(LA.det(psi))) - 0.5 * n_dim * torch.log(nu * torch.pi)
        log_prob3 = - (nu + n_dim) / 2. * torch.log(1 + delta_ / nu)
        log_prob = log_prb1 + log_prob2 + log_prob3

        return torch.sum(log_prob,dim=0)
    
    def _calculate_likelihood(self, X, S):
        """Calculate class-posterior probability
        """
        n_samples, _ = X.shape
  
        likelihood_ = torch.zeros(self.n_components)

        for c in range(self.n_components):
            likelihood_[c] = self.log_likelihood(X[S==c], self.nu, self.mu[c], self.Sigma[c])
        
        return torch.sum(likelihood_) / self.n_components
    
    def forward(self, x, s):
        
        # loss_function = 0
       
        prb = self._calculate_likelihood(x, s)
        
        return prb
   



if __name__ == '__main__':

    dataset_name = input('Enter the dataset name: ')
    root = f'../data/{dataset_name}'
    n_sub,n_trial,n_channel,n_class,sampleing_freq = config_parser(root)
    n_trial_training = 2

    
    for s in range(n_sub):

        
        acc_socre = np.empty(n_trial-n_trial_training+1)

        data_train = np.empty((1,n_channel))
        label_train = np.empty(1,dtype=int)
        nu_value = 1.0
        # nu_tune_ini = 1.0

        # 訓練データの読み込み
        loss_place = []
        for i in range(n_trial_training):
            data, label = concatenate_data_with_class(root,s,i)
            data_train = np.vstack((data_train,data))
            label_train = np.hstack((label_train,label))

            
        data_train = data_train[1:]
        label_train = label_train[1:]
        label_train = torch.from_numpy(label_train).to(torch.int64)
        iteration = 1000

        clasifier = bayesian_fmsmm.BayesianFiniteMixtureScaleMixtureModel(n_components=n_class,nu_fix=nu_value)
        clasifier.fit(data_train,label_train,True)

        nu_optimizer = nu_optim(clasifier.nu[0], clasifier.mu, clasifier.Sigma, clasifier.alpha)
        optimizedr = torch.optim.SGD(nu_optimizer.parameters(), lr=0.01)
        nu_optimizer.train()


        for i in range(iteration):
            y_pred = nu_optimizer(data_train)
            # print(y_pred[:10])
            true_label = F.one_hot(label_train, n_class)
            true_label = true_label.float()
            # print(true_label[:10])
            loss = F.cross_entropy(y_pred, true_label)

            loss.backward()
            optimizedr.step()
            optimizedr.zero_grad()

            # if i % 10 == 0:
            print(f'iteration : {i} loss : {loss.item()}, nu : {nu_optimizer.nu.data}')
            loss_place.append(loss.item())

        np.savetxt(f'sub_{s+1}_loss.csv',np.array(loss_place))
        np.savetxt(f'sub_{s+1}_nu_epoch_{iteration}.csv',nu_optimizer.nu.data.view(-1,1))