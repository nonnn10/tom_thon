#データの前処理を忘れてはいけない！！！！！！
from flask import Flask, render_template, request
#from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(Path().resolve())  #現在のこのファイルの絶対パス？
#print(cur_dir)
lr = pickle.load(open(os.path.join('classifier','pkl-objects','Logistic.pkl'), mode='rb'))  #lrモデルのロード
knc = pickle.load(open(os.path.join('classifier','pkl-objects','Near_K.pkl'), mode='rb'))  #kncモデルのロード
svm = pickle.load(open(os.path.join('classifier','pkl-objects','SVM.pkl'), mode='rb'))  #svmモデルのロード
forest = pickle.load(open(os.path.join('classifier','pkl-objects','random_forest.pkl'), mode='rb'))  #random_forestモデルのロード
#clf = pickle.load(open(os.path.join(cur_dir,'Cancellation_forecast','classifier','pkl_objects','Logistic.pkl'),'rb'))
#db = os.path.join(cur_dir,'databases','reviews.sqlite')
#print(lr)

dest = 'databases'

#dbname1の作成理由として新しく取得するデータは標準化という前処理が必要なのでそのために全データを使用し標準化する
dbname = os.path.join(dest,'isigaki_test.db')#6つのデータ
dbname1 = os.path.join(dest,'TEST.db')       #全てのデータ
conn = sqlite3.connect(dbname)               #データベースを表すコネクションオブジェクトの作成
conn1 = sqlite3.connect(dbname1)
cur = conn.cursor()                          #コネクションオブジェクトに対して全行の取得
cur1 = conn1.cursor()

# dbをpandasで読み出す。
df = pd.read_sql('SELECT * FROM sample', conn)
df1= pd.read_sql('SELECT * FROM sample', conn1)

#print(df)

cur.close()
#conn = close()
df=df[df.columns[df.columns != 'index']]      #index列の削除
df1=df1[df1.columns[df1.columns != 'index']]
#print(df)
#print(df1)
#df.head()
#df1.head()
print("index:",df.index.dtype)                #indexのタイプ確認（行）
print("column:",df.columns.dtype)             #columnsのタイプ確認（列）

#df = df.rename(columns={c:int(c) for c in df.columns})

X1 = df1.loc[:,['wind_speed','wave_height']].values   #X1に説明変数のwind_speedとwave_heightを代入
#Y1 = df1.loc[:, 'label'].values
#print(X1)
sc = StandardScaler()                         #標準化のオブジェクト
sc.fit(X1)                                    #全データを使用した標準化


def classify(model):#分類結果を返す関数
    X = df.loc[:,['wind_speed','wave_height']].values  #Xに説明変数のwind_speedとwave_heightを代入
    Y = df.loc[:, 'label'].values  #Yに真のクラスラベルを代入
    X_std = sc.transform(X)        #データの標準化（前処理）
    #print(X_std)
    print("使用したオブジェクト",model)
    print(Y)                       #真のクラスラベルの表示
    y = model.predict(X_std)         #クラスラベルの予測
    print(y)                       #予測したクラスラベルの表示
    proba = model.predict_proba(X_std)  #クラスの予測確率
    print(proba)                   #予測確率の表示
    return y, proba                #戻り値として予測したラベル、確率を返す

print(classify(forest))



text = classify(lr)
#print(text)

@app.route('/')
def home():
    return render_template('index.html',text = text)
          

if __name__ == "__main__":
    app.run(debug=True)

