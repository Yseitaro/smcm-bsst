import numpy as np
import numpy.linalg as LA
from scipy.special import gammaln
import math


class BayesianGaussianMixtureClassifier:
    """
    GMM-based Bayesian Classifier

    Parameters
    ----------
        n_dim : int, default=None.
            The number of dimensions of input data (optional)
            If you want to perform sequential learning without initial learning, 
            you need to specify this parameter.

        n_class : int, default=None.
            The number of classes to be discriminated (optional)
            If you want to perform sequential learning without initial learning, 
            you need to specify this parameter.

        alpha_0 : float, default=None.
            The prior concentration parameter of Dirichlet distribution.

        m_0 : ndarray, shape=(n_dim,), default=None.
            The prior mean of the Gaussian-inverse-Wishart distribution.

        invW_0 : ndarray, shape=(n_dim, n_dim), default=None.
            The prior inverse of the scale matrix of the Gaussian-inverse-Wishart distribution.

        beta_0 : float, default=None.
            The prior weighting parameter of the Gaussian-inverse-Wishart distribution.

        nu_0 : float, default=None.
            The prior of the number of degrees of freedom on the Gaussian-inverse-Wishart distribution.
    """
    def __init__(self, n_class=None, alpha0=None, m0=None, invW0=None, beta0=None, nu0=None):

        self.n_class = n_class

        self.alpha0 = alpha0
        self.m0 = m0
        self.invW0 = invW0
        self.beta0 = beta0
        self.nu0 = nu0

        self.prior_init_flag = False

    def _set_prior(self, X):
        """Set parameters for prior distribution
        """    

        self.alpha = np.abs(np.random.randn(self.n_class)) if self.alpha0 is None else self.alpha0  #平均0分散1の正規分布を作成 ディクリレ分布のパラメータ
        self.m = np.zeros([self.n_class, self.n_dim]) if self.m0 is None else self.m0 #ガウスウィシャート分布のパラメータm 
        # self.invW = np.tile(np.atleast_2d(np.cov(X.T)), (self.n_class, 1, 1))
        # self.invW = np.tile(np.identity(self.n_dim)[np.newaxis], (self.n_class, 1, 1)) if self.invW0 is None else self.invW0
        self.beta = np.ones(self.n_class) if self.beta0 is None else self.beta0
        self.nu = np.full(self.n_class, self.n_dim + 1) if self.nu0 is None else self.nu0
        scales = np.var(X, axis=0)
        self.invW = np.tile(np.diag(scales)/(self.nu[0]+self.n_dim+1), (self.n_class, 1, 1))

        self.prior_init_flag = True

    def _predictive_dist_x(self, m, W, beta, nu):
        """Calculate p(x*|...) in the predictive distribution
        """
        return multivariate_student_t(m, 
                    (1 - self.n_dim + nu) * beta * W / (1 + beta),
                    (1 - self.n_dim + nu))


    def fit(self, X, S):
        """Estimate the distributions of model parameters with Bayesian learning

        Parameters
        ----------
            X : ndarray, shape=(n_samples, n_dim)
                Training data for model.

            S : ndarray, shape=(n_samples, n_class)
                Hot-encoded class labels corresponding to `X`.
        """

        X = _trans_1d_to_column(X)
        S = np.copy(np.eye(self.n_class)[S])
        S = _trans_1d_to_column(S)

        _, self.n_dim = X.shape

        # Initialize prior distribution only during initial learning
        if self.prior_init_flag == False:
            self._set_prior(X)

        # Update posterior distributions
        for c in range(S.shape[1]):   # クラス数
            beta0_ = np.copy(self.beta[c])
            m0_ = np.copy(self.m[c])

            self.alpha[c] += np.sum(S[:,c])   
            self.beta[c] += np.sum(S[:,c])    
            self.m[c] = (np.sum(S[:,c].reshape(-1,1) * X, axis=0).T + beta0_ * self.m[c]) / self.beta[c]
            self.invW[c] = (np.dot((S[:,c].reshape(-1,1) * X).T, X) + 
                            beta0_ * np.dot(m0_.reshape(-1,1), m0_.reshape(-1,1).T) -
                            self.beta[c] * np.dot(self.m[c].reshape(-1,1), self.m[c].reshape(-1,1).T) +
                            self.invW[c])
            self.nu[c] += np.sum(S[:,c])


    def predict(self, X):
        """Calculate predictive distribution of class label

        Parameters
        ----------
            X : ndarray, shape=(n_samples, n_dim)
                Test data for prediction
        """
        X = _trans_1d_to_column(X)

        n_samples_, _ = X.shape
        s_ = np.zeros(shape=(n_samples_, self.n_class))

        self.eta = self.alpha / np.sum(self.alpha)

        for c in range(self.n_class):
            s_[:, c] = self.eta[c] * self._predictive_dist_x(self.m[c], 
                                                       np.linalg.inv(self.invW[c]), #np.linalg.pinv(self.invW[c]) 
                                                       self.beta[c], self.nu[c]).pdf(X)
        #s_ = np.exp(s_)

        s_ /= np.sum(s_, axis=1, keepdims=True) # Normalization
        
        return s_


def _trans_1d_to_column(X):
    if X.ndim == 1:
        return X.reshape(1, -1)
    else:
        return X


class multivariate_student_t:
    """Calculate log-probability density function of Student's t-distribution
    """
    def __init__(self, mu, lam, nu):
        self.n_dim = mu.shape[0]
        self.mu  = mu
        self.lam = lam
        self.nu  = nu

    def log_pdf(self, X):
        diffs_ = X.T - self.mu.reshape(-1,1)
        delta_ = np.sum(diffs_ * (np.dot(self.lam, diffs_)), axis=0)

        log_prob = gammaln((self.nu + self.n_dim) / 2.) - gammaln(self.nu / 2.) +\
                0.5 * np.log(LA.det(self.lam)) - 0.5 * self.n_dim * np.log(self.nu * np.pi) - \
                (self.nu + self.n_dim) / 2. * np.log(1 + delta_ / self.nu)

        return log_prob.T

    def pdf(self, X):
        return np.exp(self.log_pdf(X))

