"""
Adaptive LDA for EMG Classification

This implementation is based on the following papers:
1. Blumberg et al., "Adaptive Classification for Brain Computer Interfaces," in Proc. EMBC 2007, pp. 2536-2539, 2007.
2. Zhang et al., "An adaptation strategy of using LDA classifier for EMG pattern recognition," in Proc. EMBC 2013, pp. 4267-4270, 2013.
"""
from cmath import pi
import random
import numpy as np

class AdaptiveLDA:

    def __init__(self, n_class, cycle_subst_prop = 0.5):

        # fixed partとcycle partの割合を決めるハイパラ（論文2中では0.5を推奨）
        self.n_classes = n_class
        self.cycle_subst_prop = cycle_subst_prop
        

    def fit(self, X, S, adapt = False):
        """
        訓練データから射影ベクトルと識別関数のパラメータを計算

        - adapt == True の時は追加学習モード
        - adapt == False の時は初期学習モード
        """
        
        X = _trans_1d_to_column(X)
        # S = _trans_1d_to_column(S)


        S = np.eye(self.n_classes)[S]

        if adapt:
            X_train, S_train = self._cycle_substitution(X, S)

        else:
            X_train, S_train = X, S
            (self.X_fixed, self.S_fixed), (self.X_cycle, self.S_cycle) = \
                                                self._split_training_set(X_train, S_train)

        _, self.n_dims = X_train.shape
        
        self.mu, self.sigma_pool = self._fit_class_gaussian(X_train, S_train)

        print(f'x_train.nbytes: {X_train.nbytes}, s_train.nbytes: {S_train.nbytes}')
        print(f'X_fixed.nbytes: {self.X_fixed.nbytes}, X_cycle.nbytes: {self.X_cycle.nbytes}')
        print(f'S_fixed.nbytes: {self.S_fixed.nbytes}, S_cycle.nbytes: {self.S_cycle.nbytes}')
        # exit()


    def predict(self, X):
        """ガウス分布識別関数に基づき，クラスを予測
        """

        X = _trans_1d_to_column(X)

        n_data = len(X)
        g = np.empty((n_data, self.n_classes))

        # テストデータに対する識別関数値の計算 & クラス予測
        prior = 1. / self.n_classes
        for c in range(self.n_classes):
            g[:, c] = self._gaussian_discriminant_function(X, self.mu[c], self.sigma_pool, prior)+10000

        g /= np.sum(g, axis=1, keepdims=True)

        return g


    def _fit_class_gaussian(self, X, S):
        """クラスごとにガウス分布のパラメータを推定
        """
        # パラメータを入れるための空の配列を準備
        mu = np.empty((self.n_classes, self.n_dims))  # 平均ベクトル
        sigma = np.empty((self.n_classes, self.n_dims, self.n_dims))  # 共分散行列

        # 各クラスのパラメータを推定
        for c in range(S.shape[1]):
            mu[c] = np.sum(S[:,c].reshape(-1,1) * X, axis=0).T / np.sum(S[:,c])
            centered_X = X - mu[c]
            sigma[c] = np.dot((S[:,c].reshape(-1,1) * centered_X).T, centered_X)

        sigma_pool = np.sum(sigma,axis=0) / (len(X) - 1)

        return mu, sigma_pool

    def _gaussian_discriminant_function(self, X, mu_each, sigma_each, prior):
        """クラスごとにガウス分布識別関数を計算
        """
        centered_X = X.T - mu_each.reshape(-1,1)
        # quad = np.sum(centered_X * (np.dot(np.linalg.pinv(sigma_each), centered_X)), axis = 0)
        quad = np.sum(centered_X * (np.dot(np.linalg.inv(sigma_each), centered_X)), axis = 0)
        log_pro = (-1/2)*(quad + np.log(np.linalg.det(sigma_each)) - 2 * np.log(prior))

        return np.exp(log_pro)


    def _split_training_set(self, X, S):
        """
        初期訓練用データをfixed partとcycle partに分割する

        fixed partはずっと更新せず固定する
        cycle partは古いものから順に置換していく
        """
        _, n_channel = X.shape
        _, n_class = S.shape

        # X_fixed = [None] * n_channel
        # S_fixed = [None] * n_class

        # X_cycle = [None] * n_channel
        # S_cycle = [None] * n_class
        X_fixed = np.empty((0, n_channel))
        S_fixed = np.empty((0, n_class))
        X_cycle = np.empty((0, n_channel))
        S_cycle = np.empty((0, n_class))

        y = np.argmax(S, axis=1)

        for c in np.unique(y):

            n_samples = X[y==c].shape[0]
            ind = random.sample(range(n_samples), n_samples)
            
            X_tmp = X[y==c][ind]
            S_tmp = S[y==c][ind]

            X_fixed = np.vstack([X_fixed, 
                                 X_tmp[0:int(n_samples * self.cycle_subst_prop)]])
            S_fixed = np.vstack([S_fixed, 
                                 S_tmp[0:int(n_samples * self.cycle_subst_prop)]])

            X_cycle = np.vstack([X_cycle, 
                                    X_tmp[int(n_samples * self.cycle_subst_prop):]])
            S_cycle = np.vstack([S_cycle, 
                                    S_tmp[int(n_samples * self.cycle_subst_prop):]])

        return (X_fixed[1:], S_fixed[1:]), (X_cycle[1:], S_cycle[1:])



    def _cycle_substitution(self, X, S):
        """cycle partのデータを置換し，fixed partと結合して返す
        """

        y = np.argmax(S, axis=1)
        y_cycle = np.argmax(self.S_cycle, axis=1)

        for c in np.unique(y):
            rep_len = len(X[y == c])

            # cycle partの末尾に新規データを追加
            X_tmp = np.vstack([self.X_cycle[y_cycle == c], X[y == c]])
            S_tmp = np.vstack([self.S_cycle[y_cycle == c], S[y == c]])

            # 追加した分，cycle part冒頭のデータを取り除く
            self.X_cycle[y_cycle == c] = X_tmp[rep_len:]
            self.S_cycle[y_cycle == c] = S_tmp[rep_len:]

        return np.vstack([self.X_fixed, self.X_cycle]), np.vstack([self.S_fixed, self.S_cycle])


def _trans_1d_to_column(X):
    if X.ndim == 1:
        return X.reshape(1, -1)
    else:
        return X