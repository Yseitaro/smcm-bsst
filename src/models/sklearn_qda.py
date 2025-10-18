import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import os
import math
import random


class sklearnQDA:

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


        if adapt:
            X_train, S_train = self._cycle_substitution(X, S)

        else:
            X_train, S_train = X, S
            (self.X_fixed, self.S_fixed), (self.X_cycle, self.S_cycle) = \
                                                self._split_training_set(X_train, S_train)

        _, self.n_dims = X_train.shape

        self.qda = QuadraticDiscriminantAnalysis(store_covariance=True)
        self.qda.fit(X_train, S_train)

    def predict(self, X):

        X = _trans_1d_to_column(X)

        y_prob = self.qda.predict_proba(X)

        return y_prob
    

    def _split_training_set(self, X, S):
        """
        初期訓練用データをfixed partとcycle partに分割する

        fixed partはずっと更新せず固定する
        cycle partは古いものから順に置換していく
        """
        _, n_channel = X.shape
        # _, n_class = S.shape

        X_fixed = np.empty((0, n_channel))
        S_fixed = np.empty(0)

        X_cycle = np.empty((0, n_channel))
        S_cycle = np.empty(0)


        for c in np.unique(S):
            # print(y.shape)
            n_samples = X[S==c].shape[0]
            ind = random.sample(range(n_samples), n_samples)
            
            X_tmp = X[S==c][ind]
            S_tmp = S[S==c][ind]

            X_fixed = np.vstack([X_fixed, 
                                 X_tmp[0:int(n_samples * self.cycle_subst_prop)]])
            S_fixed = np.hstack([S_fixed, 
                                 S_tmp[0:int(n_samples * self.cycle_subst_prop)]])

            X_cycle = np.vstack([X_cycle, 
                                    X_tmp[int(n_samples * self.cycle_subst_prop):]])
            S_cycle = np.hstack([S_cycle, 
                                    S_tmp[int(n_samples * self.cycle_subst_prop):]])

        return (X_fixed[1:], S_fixed[1:]), (X_cycle[1:], S_cycle[1:])



    def _cycle_substitution(self, X, S):
        """cycle partのデータを置換し，fixed partと結合して返す
        """

        # y = np.copy(S)
        # y_cycle = np.copy(self.S_cycle)

        for c in np.unique(S):
            rep_len = len(X[S == c])

            # cycle partの末尾に新規データを追加
            X_tmp = np.vstack([self.X_cycle[self.S_cycle == c], X[S == c]])
            S_tmp = np.hstack([self.S_cycle[self.S_cycle == c], S[S == c]])

            # 追加した分，cycle part冒頭のデータを取り除く
            self.X_cycle[self.S_cycle == c] = X_tmp[rep_len:]
            self.S_cycle[self.S_cycle == c] = S_tmp[rep_len:]

        return np.vstack([self.X_fixed, self.X_cycle]), np.hstack([self.S_fixed, self.S_cycle])


 


def _trans_1d_to_column(X):
    if X.ndim == 1:
        return X.reshape(1, -1)
    else:
        return X