#https://drumato.hatenablog.com/entry/2018/11/17/234633
#教書のsvmの箇所（ch3）
#https://qiita.com/kazuki_hayakawa/items/18b7017da9a6f73eba77（直線での分類）

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

#SVM実行（波の高さと風速の二つの特徴量のみ）
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=1, gamma=0.7, C=1.0, probability=True)#kernel='linear'kernel='rbf'
svm.fit(X_train_std, y_train)

#predict = svm.predict(X_test_std)
#print(predict)
from sklearn.metrics import accuracy_score

# トレーニングデータに対する精度
pred_train = svm.predict(X_train_std)
accuracy_train = accuracy_score(y_train, pred_train)
print('トレーニングデータに対する正解率： %.4f' % accuracy_train)
#print(y_train)
pred_test = svm.predict(X_test_std)
accuracy_test = accuracy_score(y_test, pred_test)
print('テストデータに対する正解率： %.4f' % accuracy_test)
print(y_train)
print(pred_train)

#パイプラインの作成
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pipe_svm = make_pipeline(StandardScaler(),
                        #PCA(n_components=2),
                        SVC(kernel='rbf', 
                        random_state=1, 
                        gamma=0.7, 
                        C=1.0, 
                        probability=True))


pipe_svm.fit(X_train, y_train)
y_pred = pipe_svm.predict(X_test)
print('Test Accuracy: %.3f' % pipe_svm.score(X_test, y_test))

#K分割交差検証
import numpy as np
from sklearn.model_selection import StratifiedKFold
    

kfold = StratifiedKFold(n_splits=10,
                        random_state=1).split(X_train_pca, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_svm.fit(X_train_pca[train], y_train[train])
    score = pipe_svm.score(X_train_pca[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.5f' % (k+1,
          np.bincount(y_train[train]), score))
    
print('\nCV accuracy: %.5f +/- %.5f' % (np.mean(scores), np.std(scores)))


#交差検証でのモデルの正解率
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_svm,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=-1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.5f +/- %.5f' % (np.mean(scores), np.std(scores)))

#適合率、再現率、F1スコア
from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

#モデルの保存
import pickle
import os
dest = os.path.join('classifier','pkl-objects')#パスの結合
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(svm,
            open(os.path.join(dest,'SVM.pkl'),
            'wb'))#'classifier/pkl-objects/ensemble.pklで保存)
