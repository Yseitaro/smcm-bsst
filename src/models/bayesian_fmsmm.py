# Author: Akira Furui <petit.stella.billy@gmail.com>
# Copyright(c) 2020
# License: MIT
from math import gamma
import numpy as np
from numpy.core.numeric import newaxis
import numpy.linalg as LA
from scipy.special import digamma, logsumexp
from scipy.special import gammaln
from scipy.optimize import brentq, newton, bisect
from scipy import linalg
from src.models.optimization_nu import train_nu
# from sklearn.cluster import KMeans
import math
from ..data.data_utils import check_random_state

class BayesianFiniteMixtureScaleMixtureModel:
    """
    Bayesian Finite Mixture of Scale Mixture Models

    Parameters
    ----------
        n_components : int, default=1
            Depending on the data and the value of the `weight_concentration_prior` 
            the model can decide to not use all the components by setting some 
            component to values very close to zero. The number of effective 
            components is therefore smaller than n_components.

        method : {'bisect', 'newton', 'brentq'}, default='bisect'
            Method for optimization of nu.

        nu_fix : float, default=0
            Set fixed nu. 'nu_fix=0' means nu is optimized during EM.
            If nu_fix set to > 0, EM iterates with fixed nu.
        
        max_iter : int, default=5000
            Maximum number of iterations of EM algorithm.
        
        tol : float, default=1e-4
            Relative tolerance with regards to log-marginal likelihood to declare convergence

        alpha_0 : float, default=None.
            The prior concentration parameter of Dirichlet distribution.

        m_0 : ndarray, shape=(n_dim,), default=None.
            The prior mean of the Gaussian-inverse-Wishart distribution.

        rho_0 : float, default=None.
            The prior weighting parameter of the Gaussian-inverse-Wishart distribution.

        W_0 : ndarray, shape=(n_dim, n_dim), default=None.
            The prior scale matrix of the Gaussian-inverse-Wishart distribution.

        eta_0 : float, default=None.
            The prior of the number of degrees of freedom on the Gaussian-inverse-Wishart distribution.

    """

    def __init__(self, n_components, method='bisect', init_method='random', nu_fix=1, max_iter=500, tol=1e-5, 
                 nu_optimize=False,sparse=False, pruning_rate=0.01, alpha_0=None, m_0=None, beta_0=None, W_0=None, eta_0=None, 
                 verbose=False, random_state=None):

        self.n_components = n_components  # Number of components
        self.method = method
        self.init_method = init_method
        self.nu_fix = nu_fix
        self.max_iter = max_iter
        self.tol = tol
        self.sparse = sparse
        self.pruning_rate = pruning_rate

        self.alpha_0 = alpha_0
        self.m_0 = m_0
        self.beta_0 = beta_0
        self.W_0 = W_0
        self.eta_0 = eta_0

        self.nu_optimize = nu_optimize
        self.verbose = verbose
        self.random_state = random_state


    def _check_prior_parameters(self, X):
        """
        Check that the parameters are well defined.

        Parameters
        ----------
        X : array-like of shape=(n_samples, n_dim)
        """

        if self.alpha_0 is None:
            self.alpha_0 = np.abs(np.random.randn(self.n_components))
        elif np.any(self.alpha_0 <= 0):
            raise ValueError("The variable `alpha_0` should be greater than 0.")

        if self.m_0 is None:
            # self.m_0 = X.mean(axis=0)
            self.m_0 = np.zeros([self.n_components,self.n_dim])

        if self.beta_0 is None:
            self.beta_0 = np.full(self.n_components,1)
            # self.beta_0 = 1.
        elif np.any(self.beta_0 <= 0.):
            raise ValueError("The variable `beta_0` should be greater than 0.")

        if self.eta_0 is None:
            self.eta_0 = np.full(self.n_components, self.n_dim + 1)
        elif np.any(self.eta_0 <= self.n_dim - 1.):
            raise ValueError("The variable `eta_0` should be greater than {}.".format(self.n_dim - 1))

        if self.W_0 is None:
            # self.W_0 = np.tile(np.atleast_2d(np.cov(X.T)), (self.n_components, 1, 1))
            scales = np.var(X, axis=0)
            # print(f'scales : {scales}')
            self.W_0 = np.tile(np.diag(scales)/(self.eta_0[0]+self.n_dim+1), (self.n_components, 1, 1))
            
            

    def fit(self, X, S, Flag):
        """
        Estimate model parameters with the variational inference

        Parameters
        ----------
        X : array-like of shape=(n_samples, n_features)
        S : array-like of shape=(n_samples)
        Flag : Wheter it's initial learning or not. (Boolean)
            True : initial learning
            False : sequential learning
        """
        self.n_samples, self.n_dim  = X.shape
        self.nk = np.sum(np.eye(self.n_components)[S],axis=0)
        digamma_b_old = np.empty(shape=(self.n_samples,self.n_components))

        self.Flag = Flag
        ''''If it is initial learning, we have to initialize parameters'''
        if Flag == True:
            self._check_prior_parameters(X)

            random_state = check_random_state(self.random_state)
            self.initialize_param(X, S,random_state)

        if Flag == False:
            self.alpha_0 = np.copy(self.alpha)
            self.beta_0 = np.copy(self.beta)
            self.m_0 = np.copy(self.m)
            self.eta_0 = np.copy(self.eta)
            self.W_0 = np.copy(self.W)

        for i in range(self.max_iter):
            if i!=0:
                digamma_b_old = np.copy(digamma_b)
           
            print('\rIteration: %d' % (i), end='')


            digamma_a, digamma_b = self._vb_e_step(X)

            alpha = np.copy(self.alpha)
            beta = np.copy(self.beta)
            m = np.copy(self.m)
            W = np.copy(self.W)
            eta = np.copy(self.eta)
            
            self._vb_m_step(X, S, digamma_a, digamma_b, Flag)
            '''初期学習の場合，νの更新を行う'''
            if self.nu_optimize == True:
                if Flag == True:
                    self._update_nu(X, S, i)

            if self._is_converged(beta, m, W, digamma_b, digamma_b_old,tol=1e-4):
                break    

        self._compute_model_params(self.nk)
    
    def get_params(self):
        """Get parameters
        """
        return {'alpha': self.alpha, 'mu': self.mu, 'Sigma': self.Sigma, 'nu': self.nu}

     
    def predict(self, X):
        """Predict using the scale mixture-based
        """
        X = _trans_1d_to_column(X)

        # if self.Flag == False:
        #     print(f'alpha : {self.alpha.__sizeof__()}')
        #     print(f'beta : {self.beta.__sizeof__()}')
        #     print(f'm : {self.m.__sizeof__()}')
        #     print(f'W : {self.W.__sizeof__()}')
        #     print(f'eta : {self.eta.__sizeof__()}')

        return self._calculate_posterior_probabolity(X)
    

    def _calculate_posterior_probabolity(self, X):
        """Calculate class-posterior probability
        """
        n_samples, _ = X.shape
        
        prior_prob_ = self.alpha / np.sum(self.alpha)
        posterior_prob_ = np.zeros(shape=(n_samples, self.n_components))

        for c in range(self.n_components):
            posterior_prob_[:, c] = prior_prob_[c] * pdf(X, self.nu[c], self.mu[c], self.Sigma[c])
        
        posterior_prob_ /= np.sum(posterior_prob_, axis=1, keepdims=True)
        
        return posterior_prob_


    def initialize_param(self, X, S,random_state):
        """Initialize parameters using k-means

        Parameters
        ----------
            X : Data for estimation (n_samples, n_dim) 
        """
        mean = np.zeros([self.n_components,self.n_dim])

        for k in range(self.n_components):
            idx_ = np.where(S == k)
            mean[k] = np.mean(X[idx_],axis=0)

        # Hyperparameters
        self.alpha = np.copy(self.alpha_0 + self.nk)
        # print(f'self.alpha:{self.alpha}')
        self.W = np.copy(self.W_0)
        self.eta = np.copy(self.eta_0 + self.nk)
        self.beta = np.copy(self.beta_0 + self.nk)
        self.m = np.copy(self.m_0 + mean)
        if self.nu_fix == 0:
            self.nu = (20.0 - 2.0) * random_state.rand(self.n_components) + 2.0
        else:
            self.nu = np.copy(self.nu_fix + np.zeros(self.n_components))


    def _vb_e_step(self, X):
        """VB-E step

        Parameters
        ----------
            X : array-like of shape=(n_samples, n_dim) 
        """
        self._compute_posterior_moments(X)

        
        # VBE-step for the scale variables
        digamma_a, digamma_b = self._compute_invgamma_params()

        # return ln_resp, digamma_a, digamma_b
        return digamma_a, digamma_b

    
    def _compute_invgamma_params(self):
        """Compute the parameters of inverse gamma distribution (27),(28)

        Returns
        -------
            digamma_a : array-like of shape=(n_components,)

            digamma_b : array-like of shape=(n_samples, n_components)
        """
        digamma_a = (self.nu + self.n_dim) / 2.
        digamma_b = 0.5 * (self.expect_Delta + self.nu)

        return digamma_a, digamma_b


    def _compute_posterior_moments(self, X):
        """Compute the moments of posterior distribution

        Parameters
        ----------
            X : array-like of shape=(n_samples, n_dim) 
            (44),(45),(46)
        """
        n_samples, _ = X.shape

        self.expect_Delta = np.empty(shape=(n_samples, self.n_components))
        for k in range(self.n_components):
            self.expect_Delta[:, k] = self.n_dim / self.beta[k] + self.eta[k] * self.quad(X,self.m[k], self.W[k])
        
    def _vb_m_step(self, X, S, digamma_a, digamma_b, Flag):
        """VB-E step

        Parameters
        ----------
            X : array-like of shape=(n_samples, n_dim)

            ln_resp : array-like of shape=(n_samples, n_components)

            digamma_a : array-like of shape=(n_components,)

            digamma_b : array-like of shape=(n_samples, n_components)

        Returns
        -------
            expect_inv_u : array-like of shape=(n_samples, n_components)
            
            expect_ln_u : array-like of shape=(n_samples, n_components)
        """
        resp = np.eye(self.n_components)[S]

        expect_inv_u, expect_ln_u = self._compute_invgamma_moments(digamma_a, digamma_b)

        omega_k, xk, sk = self._compute_stats(X, resp, expect_inv_u)

        # Update posterior distribution
        self._update_hyperparams(self.nk, omega_k, xk, sk, Flag)
     


    def _update_hyperparams(self, nk, omega_k, xk, sk, Flag):
        """Update posterior distribution for parameters (35),(37)~(40)

        Parameters
        ----------
            nk : array-like of shape=(n_components,)

            omega_k : array-like of shape=(n_components,)

            xk : array-like of shape=(n_components, n_dim)

            sk : array-like of shape=(n_components, n_dim, n_dim)

            Flag : wheter it is initial training or not. 
                   True  -> initial training
                   Flase -> sequential training
        """

                    
        for k in range(len(nk)):

            diff = xk[k] - self.m_0[k]

            self.alpha[k] = self.alpha_0[k] + nk[k]
            self.beta[k] = self.beta_0[k] + omega_k[k]

            self.m[k] = ((self.beta_0[k] * self.m_0[k] + omega_k[k] * xk[k]) /
                    self.beta[k])
            

            self.W[k] = (self.W_0[k] + omega_k[k] * sk[k] + 
                         self.beta_0[k] * omega_k[k] / self.beta[k] *
                         np.outer(diff, diff))
            
            
            self.eta[k] = self.eta_0[k] + nk[k]
  

    def _update_nu(self, X, S, iteration):
        _, mu, Sigma = self._compute_eap_params()
        print(f'\n')
        print(f'Start optimize ν')
        nu_opt = train_nu(X, S, self.nu, mu, Sigma, self.n_components, iteration)

        self.nu = nu_opt + np.zeros(self.n_components)
        print(f'Finish optimize ν')
        print(f'\n')


    

    def _compute_invgamma_moments(self, digamma_a, digamma_b):
        """Compute the moments of inverse gamma distribution (42),(43)

        Parameters
        ----------
            digamma_a : array-like of shape=(n_components,)

            digamma_b : array-like of shape=(n_samples, n_components)

        Returns
        -------
            expect_inv_u : array-like of shape=(n_samples, n_components)
            
            expect_ln_u : array-like of shape=(n_samples, n_components)
        """
        expect_inv_u = digamma_a / digamma_b
        expect_ln_u = np.log(digamma_b) - digamma(digamma_a)

        return expect_inv_u, expect_ln_u



    def _compute_stats(self, X, resp, expect_inv_u):
        """Compute statistics using posterior distribution of latent variables. (29),(30),(31),(32)

        Parameters
        ----------
            X : array-like of shape=(n_samples, n_dim)

            ln_resp : array-like of shape=(n_samples, n_components)

            expect_inv_u : array-like of shape=(n_samples, n_components)

        Returns
        -------
            nk : array-like of shape=(n_components,)

            omega_k : array-like of shape=(n_components,)

            xk : array-like of shape=(n_components, n_dim)

            sk : array-like of shape=(n_components, n_dim, n_dim)
        """

        omega_k = np.sum(expect_inv_u * resp, axis=0)+0.00001

        sk = np.empty(shape=(self.n_components, self.n_dim, self.n_dim))
        xk = np.empty(shape=(self.n_components, self.n_dim))

        for k in range(self.n_components):

            xk[k] = (np.dot((resp[:,k] * expect_inv_u[:,k]).T, X)) / omega_k[k]
            diff = X - xk[k].reshape(1, -1)
            sk[k] = np.dot((resp[:,k] * expect_inv_u[:,k]) * diff.T, diff) / omega_k[k]

        # print(f'sk : {sk}')
        return omega_k, xk, sk
    
    def _is_converged(self,beta, m, W, digamma_b, digamma_b_old,tol):

        Return = False

        diff_beta = np.abs(np.sum(self.beta - beta))

        diff_m = np.abs(np.sum(self.m -m))
       
        diff_w = np.abs(np.sum(self.W - W))

        diff_digamma_b = np.abs(np.sum(digamma_b-digamma_b_old))

        if diff_beta < tol and diff_m<tol and diff_w<tol and diff_digamma_b<tol:
            print(f'Converged')
            Return = True
        

        return Return
                


    
    def _compute_model_params(self, nk):
        """Compute the model parameters
        """

        self.pi, self.mu, self.Sigma = self._compute_eap_params()




    def _compute_eap_params(self):
        """Compute EAP for model parameters
        """
        pi = self.alpha / np.sum(self.alpha)
        mu = self.m
        Sigma = self.W / (self.eta - self.n_dim - 1)[:,np.newaxis,np.newaxis]

        return pi, mu, Sigma



    def quad(self, X, mu, W):
        """Compute quadratic term

        Parameters
        ----------
            X : array-like of shape=(n_samples, n_dim)
            
            mu : array_like of shape=(n_dim,)

            W : array_like of shape=(n_dim, n_dim)

        Return
        ------
            quad : float
                The quadratic term value
        """
        
        diffs_ = X.T - mu.reshape(-1, 1)
 
        return np.sum(diffs_ * (np.dot(np.linalg.inv(W), diffs_)), axis = 0)



def _is_converged(pre_J, J, tol):
    """Check convergence

    Parameters
    ----------
        pre_J : float

        J : float

        tol : float

    Returns
    -------
        bool : True or False
    """
    if np.abs(pre_J - J) < tol:
        return True
    else:
        return False



def _trans_1d_to_column(X):
    if X.ndim == 1:
        return X.reshape(1, -1)
    else:
        return X


def pdf_smm(X, nu, mu, psi):
    """Calculate probability density function of scale mixture model

    Parameters
    ----------
        x:      input data
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix
    """
    _, n_dim = X.shape

    diffs_ = X.T - mu.reshape(-1, 1)
    delta_ = np.sum(diffs_ * (np.dot(LA.inv(psi), diffs_)), axis = 0)

    p_x_ = math.gamma((nu + n_dim)/2)/math.gamma(nu/2) * \
            LA.det(psi)**(-1/2)/((math.pi*nu)**(n_dim/2)) * \
            (1 + delta_/nu)**(-(nu + n_dim)/2)

    return p_x_.T


def log_pdf_smm(X, nu, mu, psi):
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

    diffs_ = X.T - mu.reshape(-1, 1)
    delta_ = np.sum(diffs_ * (np.dot(LA.inv(psi), diffs_)), axis=0)
    
    log_prob = gammaln((nu + n_dim) / 2.) - gammaln(nu / 2.) - \
                0.5 * np.log(LA.det(psi)) - 0.5 * n_dim * np.log(nu * np.pi) - \
                (nu + n_dim) / 2. * np.log(1 + delta_ / nu)

    return log_prob.T

def pdf(X, nu, mu, Sigma):
    """Calculate probability density function of finite mixture of scale mixture model

    Parameters
    ----------
        x:      input data
        pi:     mixising coeffients
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix
    """

    probability = np.exp(log_pdf_smm(X, nu, mu, Sigma))

    return probability.T

def log_pdf(X, pi, nu, mu, Sigma):
    """Calculate log-probability density function of finite mixture of scale mixture model

    Parameters
    ----------
        x:      input data
        pi:     mixising coeffients
        nu:     degrees of freedom
        mu:     mean vector
        psi:    scale matrix
    """

    ln_p_X_ = np.log(pdf(X, pi, nu, mu, Sigma))

    return ln_p_X_.T



