import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import accuracy_score
import os
import math
import random

class Adaptive_PC:

    def __init__(self, n_class,cycle_subst_prop = 0.5):

        self.n_classes = n_class
        self.cycle_subst_prop = cycle_subst_prop
    
    def fit(self, X, S, adapt = False):
        """
        訓練データから射影ベクトルと識別関数のパラメータを計算

        - adapt == True の時は追加学習モード
        - adapt == False の時は初期学習モード
        """
        X = _trans_1d_to_column(X)

        S = np.eye(self.n_classes)[S]


        # change S of the cast from int to float
        # S = S.astype(np.float64)

        if adapt:
            X_train, S_train = self._cycle_substitution(X, S)

        else:
            X_train, S_train = X, S
            (self.X_fixed, self.S_fixed), (self.X_cycle, self.S_cycle) = \
                                                self._split_training_set(X_train, S_train)

        _, self.n_dims = X_train.shape
        
        poly_X = self.polynominal_vector(X_train)
        
        # print(f'S_train : {S_train[0]}')
        self.W  = self._fit_w(poly_X, S_train)


    def polynominal_vector(self, X):
        '''
        turn data into polynominal vector
        data : n*4 dimention
        k = 2
        p(s) = [1,x1,x2,x3,x4,x1x2,x1x3,x1x4,x2x3,x2x4,x3x4,x1^2,x2^2,x3^2,x4^2]
        '''
        # p_vector = np.empty((len(data),15))

        one = np.copy(np.full(len(X),1))
        square = X * X
        comb = math.comb(self.n_dims, 2)

        polynominal = np.empty((len(X),comb))

        for i in range(len(X)):
            n = 0
            for j in range(0,len(X[i])-1):
                for k in range(j+1,len(X[i])):
                    polynominal[i][n] = X[i][j]*X[i][k]
                    n += 1
        
        p_vector = np.hstack((one.reshape(-1,1),X,polynominal,square))

        return p_vector


    def _fit_w(self, X, S):
        '''
        X : polynominal vector
        label : class label (one-hot)
        return optimized w
        '''
        # print(S.dtype)
        # print(f'S_train : {S}')
        W = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),S)

        return W

    def predict(self, X):
        """
        wに基づいてクラスラベルを予測
        """
        X = _trans_1d_to_column(X)
        poly_X = self.polynominal_vector(X)

        Y = np.dot(poly_X,self.W)

        return Y

    def _split_training_set(self, X, S):
        """
        初期訓練用データをfixed partとcycle partに分割する

        fixed partはずっと更新せず固定する
        cycle partは古いものから順に置換していく
        """
        _, n_channel = X.shape
        _, n_class = S.shape

        X_fixed = np.empty((0, n_channel))
        S_fixed = np.empty((0, n_class))

        X_cycle = np.empty((0, n_channel))
        S_cycle = np.empty((0, n_class))

        # X_fixed = [None] * n_channel
        # S_fixed = [None] * n_class

        # X_cycle = [None] * n_channel
        # S_cycle = [None] * n_class

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


