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
from flask_pure import Pure

app = Flask(__name__)
app.config['PURECSS_RESPONSIVE_GRIDS'] = True
app.config['PURECSS_USE_CDN'] = True
app.config['PURECSS_USE_MINIFIED'] = True
Pure(app)

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
val = dic[LDB_list_name(table_list)[30]]#使い方
#print(dic[LDB_haterumajima_hateruma1[0]])
#print(val)

#hateruma_hateruma0 = dic[LDB_list_name(table_list)[0]]

@app.route('/all')
def home():
    return render_template('index.html')
        
@app.route('/kaisetu')
def kaisetu():
    return render_template('index_2.html')

@app.route('/')
def main():
    return render_template('index_pure.html')

@app.route('/hatoma')
def hatoma():
    return render_template('hatoma.html', hatoma_day1=dic[LDB_list_name(table_list)[20]], hatoma_day2=dic[LDB_list_name(table_list)[21]], hatoma_day3=dic[LDB_list_name(table_list)[22]], hatoma_day4=dic[LDB_list_name(table_list)[23]], hatoma_day5=dic[LDB_list_name(table_list)[24]], hatoma_day6=dic[LDB_list_name(table_list)[25]], hatoma_day7=dic[LDB_list_name(table_list)[26]], hatoma_day8=dic[LDB_list_name(table_list)[27]], hatoma_day9=dic[LDB_list_name(table_list)[28]], hatoma_day10=dic[LDB_list_name(table_list)[29]], hatoma_isigaki1=dic[LDB_list_name(table_list)[30]], hatoma_isigaki2=dic[LDB_list_name(table_list)[31]], hatoma_isigaki3=dic[LDB_list_name(table_list)[32]], hatoma_isigaki4=dic[LDB_list_name(table_list)[33]], hatoma_isigaki5=dic[LDB_list_name(table_list)[34]], hatoma_isigaki6=dic[LDB_list_name(table_list)[35]], hatoma_isigaki7=dic[LDB_list_name(table_list)[36]], hatoma_isigaki8=dic[LDB_list_name(table_list)[37]], hatoma_isigaki9=dic[LDB_list_name(table_list)[38]], hatoma_isigaki10=dic[LDB_list_name(table_list)[39]])


@app.route('/hateruma')
def hateruma():
    return render_template('hateruma.html', hateruma_day1=dic[LDB_list_name(table_list)[0]], hateruma_day2=dic[LDB_list_name(table_list)[1]], hateruma_day3=dic[LDB_list_name(table_list)[2]], hateruma_day4=dic[LDB_list_name(table_list)[3]], hateruma_day5=dic[LDB_list_name(table_list)[4]], hateruma_day6=dic[LDB_list_name(table_list)[5]], hateruma_day7=dic[LDB_list_name(table_list)[6]], hateruma_day8=dic[LDB_list_name(table_list)[7]], hateruma_day9=dic[LDB_list_name(table_list)[8]], hateruma_day10=dic[LDB_list_name(table_list)[9]], isigaki_day1=dic[LDB_list_name(table_list)[10]], isigaki_day2=dic[LDB_list_name(table_list)[11]], isigaki_day3=dic[LDB_list_name(table_list)[12]], isigaki_day4=dic[LDB_list_name(table_list)[13]], isigaki_day5=dic[LDB_list_name(table_list)[14]], isigaki_day6=dic[LDB_list_name(table_list)[15]], isigaki_day7=dic[LDB_list_name(table_list)[16]], isigaki_day8=dic[LDB_list_name(table_list)[17]], isigaki_day9=dic[LDB_list_name(table_list)[18]], isigaki_day10=dic[LDB_list_name(table_list)[19]])


@app.route('/ohara')
def ohara():
    return render_template('ohara.html', ohara_isigaki1=dic[LDB_list_name(table_list)[40]], ohara_isigaki2=dic[LDB_list_name(table_list)[41]], ohara_isigaki3=dic[LDB_list_name(table_list)[42]], ohara_isigaki4=dic[LDB_list_name(table_list)[43]], ohara_isigaki5=dic[LDB_list_name(table_list)[24]], ohara_isigaki6=dic[LDB_list_name(table_list)[45]], ohara_isigaki7=dic[LDB_list_name(table_list)[46]], ohara_isigaki8=dic[LDB_list_name(table_list)[47]], ohara_isigaki9=dic[LDB_list_name(table_list)[48]], ohara_isigaki10=dic[LDB_list_name(table_list)[49]], ohara_day1=dic[LDB_list_name(table_list)[50]], ohara_day2=dic[LDB_list_name(table_list)[51]], ohara_day3=dic[LDB_list_name(table_list)[52]], ohara_day4=dic[LDB_list_name(table_list)[53]], ohara_day5=dic[LDB_list_name(table_list)[54]], ohara_day6=dic[LDB_list_name(table_list)[55]], ohara_day7=dic[LDB_list_name(table_list)[56]], ohara_day8=dic[LDB_list_name(table_list)[57]], ohara_day9=dic[LDB_list_name(table_list)[58]], ohara_day10=dic[LDB_list_name(table_list)[59]])


@app.route('/uehara')
def uehara():
    return render_template('uehara.html', uehara_isigaki1=dic[LDB_list_name(table_list)[60]], uehara_isigaki2=dic[LDB_list_name(table_list)[61]], uehara_isigaki3=dic[LDB_list_name(table_list)[62]], uehara_isigaki4=dic[LDB_list_name(table_list)[63]], uehara_isigaki5=dic[LDB_list_name(table_list)[64]], uehara_isigaki6=dic[LDB_list_name(table_list)[65]], uehara_isigaki7=dic[LDB_list_name(table_list)[66]], uehara_isigaki8=dic[LDB_list_name(table_list)[67]], uehara_isigaki9=dic[LDB_list_name(table_list)[68]], uehara_isigaki10=dic[LDB_list_name(table_list)[69]], uehara_day1=dic[LDB_list_name(table_list)[70]], uehara_day2=dic[LDB_list_name(table_list)[71]], uehara_day3=dic[LDB_list_name(table_list)[72]], uehara_day4=dic[LDB_list_name(table_list)[73]], uehara_day5=dic[LDB_list_name(table_list)[74]], uehara_day6=dic[LDB_list_name(table_list)[75]], uehara_day7=dic[LDB_list_name(table_list)[76]], uehara_day8=dic[LDB_list_name(table_list)[77]], uehara_day9=dic[LDB_list_name(table_list)[78]], uehara_day10=dic[LDB_list_name(table_list)[79]])


@app.route('/kohama')
def kohama():
    return render_template('kohama.html', kohama_isigaki1=dic[LDB_list_name(table_list)[80]], kohama_isigaki2=dic[LDB_list_name(table_list)[81]], kohama_isigaki3=dic[LDB_list_name(table_list)[82]], kohama_isigaki4=dic[LDB_list_name(table_list)[83]], kohama_isigaki5=dic[LDB_list_name(table_list)[84]], kohama_isigaki6=dic[LDB_list_name(table_list)[85]], kohama_isigaki7=dic[LDB_list_name(table_list)[86]], kohama_isigaki8=dic[LDB_list_name(table_list)[87]], kohama_isigaki9=dic[LDB_list_name(table_list)[88]], kohama_isigaki10=dic[LDB_list_name(table_list)[89]], kohama_day1=dic[LDB_list_name(table_list)[90]], kohama_day2=dic[LDB_list_name(table_list)[91]], kohama_day3=dic[LDB_list_name(table_list)[92]], kohama_day4=dic[LDB_list_name(table_list)[93]], kohama_day5=dic[LDB_list_name(table_list)[94]], kohama_day6=dic[LDB_list_name(table_list)[95]], kohama_day7=dic[LDB_list_name(table_list)[96]], kohama_day8=dic[LDB_list_name(table_list)[97]], kohama_day9=dic[LDB_list_name(table_list)[98]], kohama_day10=dic[LDB_list_name(table_list)[99]])


@app.route('/kurosima')
def kurosima():
    return render_template('kurosima.html', kurosima_isigaki1=dic[LDB_list_name(table_list)[100]], kurosima_isigaki2=dic[LDB_list_name(table_list)[101]], kurosima_isigaki3=dic[LDB_list_name(table_list)[102]], kurosima_isigaki4=dic[LDB_list_name(table_list)[103]], kurosima_isigaki5=dic[LDB_list_name(table_list)[104]], kurosima_isigaki6=dic[LDB_list_name(table_list)[105]], kurosima_isigaki7=dic[LDB_list_name(table_list)[106]], kurosima_isigaki8=dic[LDB_list_name(table_list)[107]], kurosima_isigaki9=dic[LDB_list_name(table_list)[108]], kurosima_isigaki10=dic[LDB_list_name(table_list)[109]], kurosima_day1=dic[LDB_list_name(table_list)[110]], kurosima_day2=dic[LDB_list_name(table_list)[111]], kurosima_day3=dic[LDB_list_name(table_list)[112]], kurosima_day4=dic[LDB_list_name(table_list)[113]], kurosima_day5=dic[LDB_list_name(table_list)[114]], kurosima_day6=dic[LDB_list_name(table_list)[115]], kurosima_day7=dic[LDB_list_name(table_list)[116]], kurosima_day8=dic[LDB_list_name(table_list)[117]], kurosima_day9=dic[LDB_list_name(table_list)[118]], kurosima_day10=dic[LDB_list_name(table_list)[119]])


@app.route('/taketomi')
def taketomi():
    return render_template('taketomi.html', taketomi_isigaki1=dic[LDB_list_name(table_list)[120]], taketomi_isigaki2=dic[LDB_list_name(table_list)[121]], taketomi_isigaki3=dic[LDB_list_name(table_list)[122]], taketomi_isigaki4=dic[LDB_list_name(table_list)[123]], taketomi_isigaki5=dic[LDB_list_name(table_list)[124]], taketomi_isigaki6=dic[LDB_list_name(table_list)[125]], taketomi_isigaki7=dic[LDB_list_name(table_list)[126]], taketomi_isigaki8=dic[LDB_list_name(table_list)[127]], taketomi_isigaki9=dic[LDB_list_name(table_list)[128]], taketomi_isigaki10=dic[LDB_list_name(table_list)[129]], taketomi_day1=dic[LDB_list_name(table_list)[130]], taketomi_day2=dic[LDB_list_name(table_list)[131]], taketomi_day3=dic[LDB_list_name(table_list)[132]], taketomi_day4=dic[LDB_list_name(table_list)[133]], taketomi_day5=dic[LDB_list_name(table_list)[134]], taketomi_day6=dic[LDB_list_name(table_list)[135]], taketomi_day7=dic[LDB_list_name(table_list)[136]], taketomi_day8=dic[LDB_list_name(table_list)[137]], taketomi_day9=dic[LDB_list_name(table_list)[138]], taketomi_day10=dic[LDB_list_name(table_list)[139]])

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=8080,threaded=True)

