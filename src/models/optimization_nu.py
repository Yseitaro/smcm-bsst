import numpy as np
import sys
from src.data.data_utils import concatenate_data_with_class
from src.data.data_utils import config_parser
import os
import importlib
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch import nn
import torch
import numpy.linalg as LA
# 学習率スケジューラー
from torch.optim import lr_scheduler
import torch
from torch.utils.data import DataLoader, TensorDataset



class opt_nu(nn.Module):
    def __init__(self, nu, mu, Sigma, n_class):
        super(opt_nu, self).__init__()
        self.n_components = n_class
        self.nu = nn.Parameter(torch.tensor(nu[0]).float())
        self.mu = torch.from_numpy(mu).float()
        self.Sigma = torch.from_numpy(Sigma).float()

    def log_likelihood(self, X,  mu, psi):
        """Calculate log-probability density function of scale mixture model"""
    
        _, n_dim = X.shape
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        diffs_ = X.T - mu.reshape(-1, 1)

        inv_psi = torch.linalg.inv(psi)  # 逆行列にepsを加えて安定化
        delta_ = torch.sum(diffs_ * torch.matmul(inv_psi, diffs_), dim=0)

        log_prb1 = torch.special.gammaln((self.nu + n_dim) / 2.) - torch.special.gammaln(self.nu / 2.)
        
        log_prob2 = -0.5 * torch.logdet(psi) - 0.5 * n_dim * torch.log(self.nu * torch.pi)
        
        log_prob3 = - (self.nu + n_dim) / 2. * torch.log(1 + delta_ / (self.nu))  # delta_ / nu にepsを加えて安定化

        log_prob = -(log_prb1 + log_prob2 + log_prob3)

        return torch.mean(log_prob)
        

    
    def _calculate_likelihood(self, X, S):
        """Calculate class-posterior probability"""
        likelihood_ = torch.zeros(self.n_components)
        for c in range(self.n_components):
            likelihood_[c] = self.log_likelihood(X[S==c], self.mu[c], self.Sigma[c])
        return torch.sum(likelihood_) / self.n_components

    def forward(self, x, s):
        # self.nu.data = torch.clamp(self.nu.data, min=0.0001)
        prb = self._calculate_likelihood(x, s)
        return prb
    


def train_nu(X, S, nu, mu, Sigma, n_class, iteration_f):

    print(f'X shape : {X.shape}')
    print(f'S shape : {S.shape}')


    model = opt_nu(nu, mu, Sigma, n_class)


    optimizedr = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizedr = torch.optim.SGD(model.parameters(), lr=0.1)


    iteration = 1000

    loss_old = 0
    early_stopping_count = 0

    old_total_loss = 0
    old_nu = 100

    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(S).long()

    total_loss = 0
    model.train()

    for i in range(iteration):

        optimizedr.zero_grad()
        loss = model(X_tensor, Y_tensor)
        loss.backward()
        optimizedr.step()
        total_loss = loss.item()
        new_nu = model.nu.data


        if i % 10 == 0:
            print(f'nu optimization train iteration : {i} loss : {total_loss}, nu : {model.nu.data}')

        if np.abs(new_nu-old_nu) < 0.0000001:
            early_stopping_count += 1
            # print(f'count {early_stopping_count}')
        else:
            # print(f'reset')
            early_stopping_count = 0

        if early_stopping_count == 500:
            print('Early Stopping')
            break

        old_nu = np.copy(model.nu.data)

    return model.nu.detach().numpy()

# import numpy as np
# import sys
# from src.data.data_utils import concatenate_data_with_class
# from src.data.data_utils import config_parser
# import os
# import importlib
# from sklearn.metrics import accuracy_score
# from torch.nn import functional as F
# from torch import nn
# import torch
# import numpy.linalg as LA
# # 学習率スケジューラー
# from torch.optim import lr_scheduler

# class opt_nu(nn.Module):
#     def __init__(self, nu, mu, Sigma, n_class):
#         super(opt_nu, self).__init__()
#         self.n_components = n_class
#         self.nu = nn.Parameter(torch.tensor(nu[0]).float())
#         self.mu = mu
#         self.Sigma = Sigma
    

#     def log_likelihood(self, X, nu, mu, psi):
#         """Calculate log-probability density function of scale mixture model
#         Parameters
#         ----------
#             x:      input data
#             nu:     degrees of freedom
#             mu:     mean vector
#             psi:    scale matrix
#         Returns
#         -------
#             log_prob : ndarray, shape=(n_samples)
#                 Log-probability from a scale mixutre model
#                 > log p(x_n|z_n)
#         """
#         _, n_dim = X.shape
#         torch.pi = torch.acos(torch.zeros(1)).item() * 2
#         diffs_ = X.T - mu.reshape(-1, 1)
#         delta_ = np.sum(diffs_ * (np.dot(LA.inv(psi), diffs_)), axis=0)
#         delta_ = torch.tensor(delta_).float()


#         log_prb1 = torch.special.gammaln((nu + n_dim) / 2.) - torch.special.gammaln(nu / 2.)
#         log_prob2 = -0.5 * torch.log(torch.tensor(LA.det(psi))) - 0.5 * n_dim * torch.log(nu * torch.pi)
#         log_prob3 = - (nu + n_dim) / 2. * torch.log(1 + delta_ / nu)
#         log_prob = log_prb1 + log_prob2 + log_prob3

#         return -torch.mean(log_prob,dim=0)
    
#     def _calculate_likelihood(self, X, S):
#         """Calculate class-posterior probability
#         """
  
#         likelihood_ = torch.zeros(self.n_components)

#         for c in range(self.n_components):
#             likelihood_[c] = self.log_likelihood(X[S==c], self.nu, self.mu[c], self.Sigma[c])
        
#         return torch.sum(likelihood_) / self.n_components
    
#     def forward(self, x, s):
        
#         # loss_function = 0
       
#         prb = self._calculate_likelihood(x, s)
        
#         return prb


# def train_nu(X, S, nu, mu, Sigma, n_class, iteration_f):

#     model = opt_nu(nu, mu, Sigma, n_class)
#     optimizedr = torch.optim.SGD(model.parameters(), lr=0.01)
    

#     loss_array = []
#     iteration = 1000
    
#     loss_old = 0
#     early_stopping_count = 0

#     for i in range(iteration):
#         loss = model(X, S)
#         loss.backward()
#         optimizedr.step()
#         optimizedr.zero_grad()

#         loss_array.append(loss.item())
    
#         if i % 10 == 0:
#             print(f'nu optimization train iteration : {i} loss : {loss.item()}, nu : {model.nu.data}')
#             # loss_array.append(loss.item())
        
#         loss_difference = np.abs(loss_array[-1]- loss_old)


#         if loss_difference < 0.0000001:
#             early_stopping_count += 1
#         else:
#             early_stopping_count = 0
        
#         if early_stopping_count == 100:
#             print('Early Stopping')
#             break
#         # path = f'../results/furui_lab/BCMT_optimized_nu/'
#         # os.makedirs(path, exist_ok=True)
#         # np.savetxt(f'{path}loss_array_iteration{iteration_f+1}.csv',loss_array)
#         # print(f'saved')

#         loss_old = loss_array[-1]

#     return model.nu.detach().numpy()



