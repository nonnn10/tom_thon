#https://qiita.com/0NE_shoT_/items/b702ab482466df6e5569
#教科書

from sklearn import datasets
import pandas as pd
df = pd.read_csv('../data/isigaki_miyako_iriomote_wavedata/isigaki_2006_2018.csv', header=None)

import numpy as np
from sklearn.preprocessing import LabelEncoder
#データのロード
X = df.loc[:, [1,4]].values
#print(X)
y = df.loc[:, 0].values


#データの分割（テスト用とトレーニング用）
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)
print('Labels counts in y[0 1]:', np.bincount(y))
print('Labels counts in y_train[0 1]:', np.bincount(y_train))
print('Labels counts in y_test[0 1]:', np.bincount(y_test))

#標準化
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print(X_train_std)

#pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
print(X)

#Scikit-learnによるロジスティック回帰
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=11) # ロジスティック回帰モデルのインスタンスを作成
lr.fit(X_train_std, y_train) # ロジスティック回帰モデルの重みを学習

#モデルの保存
import pickle
import os
dest = os.path.join('classifier','pkl-objects')#パスの結合
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(lr,
            open(os.path.join(dest,'Logistic.pkl'),
            'wb'))#'classifier/pkl-objects/ensemble.pklで保存)
