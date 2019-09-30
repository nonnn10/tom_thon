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


def classify(model,df):#分類結果を返す関数、引数1は使用したいモデル、引数2は予測したいデータ
    X = df.loc[:,['wind_speed','wave_height']].values  #Xに説明変数のwind_speedとwave_heightを代入
    Y = df.loc[:, 'label'].values  #Yに真のクラスラベルを代入
    X_std = sc.transform(X)        #データの標準化（前処理）
    #print(X_std)
    #print("使用したオブジェクト",model)
    #print(Y)                       #真のクラスラベルの表示
    y = model.predict(X_std)         #クラスラベルの予測
    #print(y)                       #予測したクラスラベルの表示
    proba = model.predict_proba(X_std)  #クラスの予測確率
    #print(proba)#予測確率の表示

    return proba                #戻り値として予測したラベル、確率を返す

#print(classify(forest,df))


def Change_to_percentage(proba):   #欠航する確率を返す関数、引数1にはclassify(model,df)関数を入れる
    ans = np.array([])             #numpy配列の作成
    p = proba                      #引数を変数に代入
    for i in range(len(p)):        #配列の数だけループ
        proba1=p[i]                #ループ数の配列番号の値を変数に代入
        for l in range(len(proba1)):    #配列の数だけループ
            a = np.array(['{:.0%}'.format(proba1[l])])  #配列番号がループ数の箇所の数値をパーセント表記に変更しa変数に格納
            if l != 0 :                                 #ループ数が0以外なら実行
                ans = np.append(ans,a)                  #ans変数にa変数を格納
    return ans

print(Change_to_percentage(classify(forest,df)))

"""
arr = np.array([])
arr = np.append(arr, np.array([1, 2, 3]))
arr = np.append(arr, np.array([4, 5]))
arr

print(ans)
arr = np.empty((0,3), int)
arr = np.append(arr, np.array([[1, 2, 3]]), axis=0)
arr = np.append(arr, np.array([[4, 5, 0]]), axis=0)
arr
"""

text = Change_to_percentage(classify(forest,df))
#print(text)

day1 = text[0]
day2 = text[1]
day3 = text[2]
day4 = text[3]
day5 = text[4]
day6 = text[5]
#print(day1)
#print(day2)


@app.route('/')
def home():
    return render_template('index.html',day1 = day1, day2 = day2, day3 = day3, day4 = day4, day5 = day5,day6 = day6)
          

if __name__ == "__main__":
    app.run(debug=True)
