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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import re

#app = Flask(__name__)

def columndelet(df,cdl): #列名を削除をする関数
    for i in cdl:
        df=df[df.columns[df.columns != i]]
        
    return df 

dest = 'databases2'

def df_table(dest,dbname):#db内のtable名取得 引数(ファイルへのパス,ファイル名<-str型)
    dbname = os.path.join(dest,dbname)#データ
    conn = sqlite3.connect(dbname)               #データベースを表すコネクションオブジェクトの作成
    cur = conn.cursor()                          #コネクションオブジェクトに対して全行の取得
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;") #データベース内のtable名を取得してくる

    table_list = []
    for x in cur.fetchall(): #データベース内のtable名を取得してきたものからtable_listに格納
        table_list.append(x)

    cur.close()
    return table_list

table_list=df_table(dest,'learningData.db')
        
cdl=['Date','Time','Date0','Date1','Date2','Date3','Date4','Date5','Date6','Date7','Date8','Date9']#消したいカラム名
print(table_list[1])
#-----------------140個のモデル学習-------------------------------------
dest = 'databases2'


dbname = os.path.join(dest,'learningData.db')#データ
conn = sqlite3.connect(dbname)               #データベースを表すコネクションオブジェクトの作成
cur = conn.cursor()                          #コネクションオブジェクトに対して全行の取得

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;") #データベース内のtable名を取得してくる
list = []
count = 0
for x in cur.fetchall(): #データベース内のtable名を取得してきたものからlistに格納
    count+=1
    list.append(x)
    if count == 140:
        break

cur.close()
cdl=['Date','Time','Date0','Date1','Date2','Date3','Date4','Date5','Date6','Date7','Date8','Date9']#消したいカラム名

for i in range(len(list)):
    df = pd.read_sql("SELECT * FROM '%s'" % list[i], conn)
    df=columndelet(df,cdl)
    
    X = df.loc[:, 'Wave_swell0':'Wind_Max_Speed9'].values
    y = df.loc[:, 'Alert'].values
    
    #処理なし
    pipe_forest = make_pipeline(RandomForestClassifier(criterion='gini',n_estimators=25, 
                            random_state=1,
                            n_jobs=2))

    pipe_forest.fit(X, y)

    #標準化
    pipe_forest_std = make_pipeline(StandardScaler(),
                            #PCA(n_components=2),
                            RandomForestClassifier(criterion='gini',
                            n_estimators=25, 
                            random_state=1,
                            n_jobs=2))

    pipe_forest_std.fit(X, y)

    #pca
    pipe_forest_pca = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            RandomForestClassifier(criterion='gini',
                            n_estimators=25, 
                            random_state=1,
                            n_jobs=2))

    pipe_forest_pca.fit(X, y)
    
#----------------------------------モデルの保存--------------------------------
    #モデルの保存
    dest = os.path.join('classifier','learn-objct')#パスの結合
    if not os.path.exists(dest):
        os.makedirs(dest)
    ln_name = str(list[i])
    ln_name = ln_name.strip(")").strip("(").strip(",").strip("'")
    pickle.dump(pipe_forest_std, #保存したいモデル
                open(os.path.join(dest,'random_forest{0}.pkl'.format(ln_name)),
                'wb'))#'classifier/pkl-objects/ensemble.pklで保存)
#----------------------------------------------------------------------------

def classify(model,X):#分類結果を返す関数、引数1は使用したいモデル、引数2は予測したいデータ
    y = model.predict(X)         #クラスラベルの予測
    proba = model.predict_proba(X)  #クラスの予測確率

    return proba  

def Change_to_percentage(proba):   #欠航する確率を返す関数、引数1にはclassify(model,df)関数を入れる
    ans = np.array([])             #numpy配列の作成
    p = proba                      #引数を変数に代入
    for i in range(len(p)):        #配列の数だけループ
        proba1=p[i]                #ループ数の配列番号の値を変数に代入
        #print(proba1)
        for l in range(len(proba1)):    #配列の数だけループ
            a = np.array(['{:.0%}'.format(proba1[l])])#配列番号がループ数の箇所の数値をパーセント表記に変更しa変数に格納
            if l != 0 :#ループ数が0以外なら実行
                ans = np.append(ans,a)                  #ans変数にa変数を格納
            elif 1 == len(proba1):
                ans = np.append(ans,np.array(['{:.0%}'.format(0.0)]))  
    return ans

def ln_table_change(table_list,i):#table名を予測データとの対応した文字列に変換
    ln_name = str(table_list[i])
    ln_name = ln_name.strip(")").strip("(").strip(",").strip("'")
    ln_name=ln_name.rsplit('_', 1)[-1]
    ln_name=re.sub(r'\d+','',ln_name)
    return ln_name

def table_judgment(table_list,i):
    a=0
    predict_table_list = ["hateruma","hatoma","ohara","uehara","kohama","kurosima","taketomi","ishigaki"]
    if ln_table_change(table_list,i) == predict_table_list[0]:
        a = 0#hateruma
    elif ln_table_change(table_list,i) == predict_table_list[1]:
        a = 1#hatoma
    elif ln_table_change(table_list,i) == predict_table_list[2]:
        a = 2#ohara
    elif ln_table_change(table_list,i) == predict_table_list[3]:
        a = 3#uehara
    elif ln_table_change(table_list,i) == predict_table_list[4]:
        a = 4#kohama
    elif ln_table_change(table_list,i) == predict_table_list[5]:
        a = 5#kurosima
    elif ln_table_change(table_list,i) == predict_table_list[6]:
        a = 6#taketomi
    elif ln_table_change(table_list,i) == predict_table_list[7]:
        a = 7#uehara
    
    return a

def LDB_list_name(table_list):#table名がLDBのものだけ抽出
    LDB_list=[]
    for i in range(len(table_list)):
        ln_name = str(table_list[i])
        ln_name = ln_name.strip(")").strip("(").strip(",").strip("'")
        ldb_judg = ln_name.split('_', 1)[0]
        if ldb_judg == 'LDB':
            LDB_list.append(ln_name)

    return LDB_list
predict_table_list = ["hateruma","hatoma","ohara","uehara","kohama","kurosima","taketomi","ishigaki"]
proba_list=[] #webサイトに表示する確率
ans = np.array([]) 
count=0
for i in range(len(table_list)):
    count+=1
    pre_tab_num = table_judgment(table_list,i)      #予測データtableの指定
    learn_df = pd.read_sql("SELECT * FROM '%s'" % table_list[i], conn)
    learn_df = columndelet(learn_df,cdl)
    learn_X = learn_df.loc[:, 'Wave_swell0':'Wind_Max_Speed9'].values
    sc = StandardScaler()                         #標準化のオブジェクト
    sc.fit(learn_X)  
    
    predict_df = pd.read_sql("SELECT * FROM '%s'" % predict_table_list[pre_tab_num], conn)#予測したいデータ
    predict_df = columndelet(predict_df,cdl)
    predict_X = predict_df[-1:]#.loc[-1:, 'Wave_swell0':'Wind_Max_Speed9'].values
    X_std = sc.transform(predict_X)

    ln_name = str(table_list[i])
    ln_name = ln_name.strip(")").strip("(").strip(",").strip("'")
    model = pickle.load(open(os.path.join('classifier','learn-objct','random_forest{0}.pkl'.format(ln_name)), mode='rb'))  #モデルのロード

    predict_list=classify(model,X_std)
     
    a=Change_to_percentage(predict_list)
    ans = np.append(ans,a) 
    if count == 140:
        break


#dbnameと確率の辞書型
dic = dict(zip(LDB_list_name(table_list), ans)) 
val = dic[LDB_list_name(table_list)[0]]#使い方

#---------------------------------紳之亮-----------------------------------
"""
text = Change_to_percentage(classify(model,df))
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
    return render_template('sakisima.html',day1 = day1, day2 = day2, day3 = day3, day4 = day4, day5 = day5,day6 = day6)
        
@app.route('/kaisetu')
def kaisetu():
    return render_template('index_2.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=80,threaded=True)
"""