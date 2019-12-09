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
from sklearn.pipeline import make_pipeline
import re
from flask_pure import Pure
import datetime
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from functools import partial
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score
import decimal
import sys

app = Flask(__name__)
app.config['PURECSS_RESPONSIVE_GRIDS'] = True
app.config['PURECSS_USE_CDN'] = True
app.config['PURECSS_USE_MINIFIED'] = True
Pure(app)

#-----------------------------関数定義--------------------------------------

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

def classify(model, X):  # 分類結果を返す関数、引数1は使用したいモデル、引数2は予測したいデータ
    #y = model.predict(X)  # クラスラベルの予測
    proba = model.predict_proba(X)  # クラスの予測確率
    return proba

def columndelet(df, cdl):  # 列名を削除をする関数
    for i in cdl:
        df = df[df.columns[df.columns != i]]
    return df


def Change_to_percentage(proba):  # 欠航する確率を返す関数、引数1にはclassify(model,df)関数を入れる
    ans = np.array([])  # numpy配列の作成
    p = proba  # 引数を変数に代入
    for i in range(len(p)):  # 配列の数だけループ
        proba1 = p[i]  # ループ数の配列番号の値を変数に代入
        # print(proba1)
        for l in range(len(proba1)):  # 配列の数だけループ
            a = np.array(['{:.0%}'.format(proba1[l])])  # 配列番号がループ数の箇所の数値をパーセント表記に変更しa変数に格納
            if l != 0:  # ループ数が0以外なら実行
                ans = np.append(ans, a)  # ans変数にa変数を格納
            elif 1 == len(proba1):
                ans = np.append(ans, np.array(['{:.0%}'.format(0.0)]))
    return ans


def ln_table_change(table_list,i):#table名を予測データとの対応した文字列に変換
    ln_name = str(table_list[i])
    ln_name = ln_name.strip(")").strip("(").strip(",").strip("'")
    ln_name=ln_name.rsplit('_', 1)[-1]
    ln_name=re.sub(r'\d+','',ln_name)
    return ln_name

def table_judgment(table_list, i):
    a = 0
    predict_table_list = ["hateruma", "hatoma", "ohara", "uehara", "kohama", "kurosima", "taketomi", "ishigaki"]
    if ln_table_change(table_list, i) == predict_table_list[0]:
        a = 0  # hateruma
    elif ln_table_change(table_list, i) == predict_table_list[1]:
        a = 1  # hatoma
    elif ln_table_change(table_list, i) == predict_table_list[2]:
        a = 2  # ohara
    elif ln_table_change(table_list, i) == predict_table_list[3]:
        a = 3  # uehara
    elif ln_table_change(table_list, i) == predict_table_list[4]:
        a = 4  # kohama
    elif ln_table_change(table_list, i) == predict_table_list[5]:
        a = 5  # kurosima
    elif ln_table_change(table_list, i) == predict_table_list[6]:
        a = 6  # taketomi
    elif ln_table_change(table_list, i) == predict_table_list[7]:
        a = 7  # uehara
    return a

def LDB_list_name(table_list):  # table名がLDBのものだけ抽出
    LDB_list = []
    for i in range(len(table_list)):
        ln_name = str(table_list[i])
        ln_name = ln_name.strip(")").strip("(").strip(",").strip("'")
        ldb_judg = ln_name.split('_', 1)[0]
        if ldb_judg == 'LDB':
            LDB_list.append(ln_name)
    return LDB_list

def model_save(model,prepro_num):#ハイパーパラメータ調整modelの前処理ごとのパイプライン
    prepro_list = ['non', 'std', 'ptf']
    if prepro_num == 0:
        pipe_model = make_pipeline(model)
    elif prepro_num == 1:
        pipe_model = make_pipeline(StandardScaler(),
                                   model)
    elif prepro_num == 2:
        pipe_model = make_pipeline(PowerTransformer(copy=True, method='yeo-johnson', standardize=True),
                                   model)
    return model

# ハイパーパラメータ関数
def objective(X_train, y_train, X_test, y_tesy, prepro, trial):
    """最小化する目的関数"""
    #print(prepro)
    # 使う分類器は XGB or RF
    classifier = trial.suggest_categorical('classifier', ['xgb.XGBClassifier'])#,'RandomForestClassifier'])

    # 選ばれた分類器で分岐する
    if classifier == 'RandomForestClassifier':
        # RF のとき
        # 調整するハイパーパラメータ(パラメータの追加を行った場合mainの方でも追加する)
        params = {
            'n_estimators': int(trial.suggest_loguniform('n_estimators', 1e+0, 1e+3)),
            'max_depth': int(trial.suggest_int('max_depth', 2, 32)),
            'criterion': trial.suggest_categorical("criterion", ["gini", "entropy"]),
            'min_samples_split': int(trial.suggest_int('min_samples_split',2, 32)),
            'max_features':int(trial.suggest_int('max_features',2,128))
        }
        model = RandomForestClassifier(**params)#,random_state=3)

    if classifier == 'xgb.XGBClassifier':
        # XGB のとき
        # 調整するハイパーパラメータ
        params = {
            'learning_rate': trial.suggest_uniform('learning_rate', 1e-1, 1e0),
            'max_depth': int(trial.suggest_int('max_depth', 2, 32)),
            'subsample': trial.suggest_uniform('subsample', 1e-1, 1e0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 1e-2, 1e0),
            'gamma':trial.suggest_uniform('gamma', 1e-2, 1),
            'olsample_bytree':trial.suggest_uniform('olsample_bytree', 1e-2, 1),
            'alpha':trial.suggest_uniform('alpha', 1e-2, 1)
        }
        model = xgb.XGBClassifier(**params, objective='binary:logistic')

    model.fit(X_train, y_train)
    return f1_score(y_true=y_test, y_pred=model.predict(X_test))#accuracy_score(y_test, model.predict(X_test))#最小化なので1から正解率を引く

#-----------------------------関数定義終了--------------------------------------

dest = 'databases2'
table_list = []
table_list=df_table(dest,'learningData.db') #learnData.dbから全tabel取得
table_list = LDB_list_name(table_list)      # 学習用のtableのみ抽出  # データベース内のtable名を取得してくる

count = 0

dbname = os.path.join(dest, "learningData.db")  # データ
conn = sqlite3.connect(dbname)  # データベースを表すコネクションオブジェクトの作成
cur = conn.cursor()
cdl = ['Date', 'Time', 'Date0', 'Date1', 'Date2', 'Date3', 'Date4', 'Date5', 'Date6', 'Date7', 'Date8','Date9']
print("モデル学習を実行しますか？実行:0,実行しない:1,を入力してください")#モデル学習実行or未実行
exe = int(input())
print(exe)
if exe == 0:#0なら学習しない
    for i in range(len(table_list)):
        df = pd.read_sql("SELECT * FROM '%s'" % table_list[i], conn)
        df = columndelet(df, cdl)
        print(df.columns)
        print(table_list[i])
        # 処理なし
        X = df.loc[:, 'Wave_swell0':'Wind_Max_Speed9'].values#Wave_swell0-->'Wave_height0'
        y = df.loc[:, 'Alert'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y,random_state=0)
        print('Labels counts in y[0 1]:', np.bincount(y))
        print('Labels counts in y_train[0 1]:', np.bincount(y_train))
        print('Labels counts in y_test[0 1]:', np.bincount(y_test))

        # 標準化
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        # ptf
        #pt = PowerTransformer()
        pt=PowerTransformer(copy=True, method='yeo-johnson', standardize=True)
        pt.fit(X_train)
        X_train_ptf = pt.transform(X_train)
        X_test_ptf = pt.transform(X_test)


        # --------------------ハイパーパラメータ探索--------------------------------
        #optenaのdocument-->https://optuna.readthedocs.io/en/stable/reference/index.html
        train_list = [X_train, X_train_std, X_train_ptf]  # 前処理で分けたトレーニングデータ(生、標準化、正規分布)
        test_list = [X_test, X_test_std, X_test_ptf]  # 前処理で分けたテストデータ(生、標準化、正規分布)
        prepro_list = ['non', 'std', 'ptf']  # 前処理のリスト(生、標準化、正規分布)
        prepro_num = 0  # 前処理の番号(0,1,2)
        bv = 0  # 一番いいスコア
        hyp_clf = ''  # 変数の初期化
        #https://optuna.readthedocs.io/en/stable/faq.html
        optuna.logging.set_verbosity(optuna.logging.WARNING)#oputenaのログ出力停止
        for l in range(len(train_list)):  # train_list分ループ(3回る)
            f = partial(objective, train_list[l],y_train, test_list[l],y_test,prepro_list[l])     # 目的関数にデータを適用する
            study = optuna.create_study(direction='maximize')       # 最適化のセッションを作る
            study.optimize(f, n_trials=30)      # 30 回試行する
            if bv <= study.best_value:#study.best_valueはF値
                bv = study.best_value
                hyp_clf = study.best_params['classifier']

                if hyp_clf == 'xgb.XGBClassifier':#xgbのパラメータ保存
                    #print('xgb')
                    md = int(study.best_params['max_depth'])
                    #print("ベストmd",md)
                    lr = study.best_params['learning_rate']
                    ss = study.best_params['subsample']
                    cb = study.best_params['colsample_bytree']
                    gm = study.best_params['gamma']
                    ob = study.best_params['olsample_bytree']
                    ap = study.best_params['alpha']
                    model = xgb.XGBClassifier(max_depth=md, learning_rate=lr, subsample=ss, colsample_bytree=cb,
                                              gamma=gm,olsample_bytree=ob,alpha=ap)
                    model.fit(train_list[l], y_train)
                    prepro_num = l
                    #print("更新時の前処理",l)
                    y_pred = model.predict(test_list[prepro_num])

                elif hyp_clf == 'RandomForestClassifier':#RFのパラメータ保存
                    #print('RF')
                    ne = int(study.best_params['n_estimators'])
                    md = int(study.best_params['max_depth'])
                    cr = study.best_params['criterion']
                    mss= study.best_params['min_samples_split']
                    mf = study.best_params['max_features']
                    model = RandomForestClassifier(n_estimators=ne, max_depth=md, criterion=cr,
                                                   min_samples_split=mss,max_features=mf)
                    model.fit(train_list[l], y_train)
                    prepro_num = l
                    #print("更新時の前処理", l)
                    y_pred = model.predict(test_list[prepro_num])
                del study#study変数の初期化
                #prepro_num = l
        """
        print("トータルベスト",bv)
        y_pred_proba=model.predict_proba(test_list[prepro_num])
        y_pred=model.predict(test_list[prepro_num])
        print(y_pred_proba)
        print("予想",y_pred)
        print("正解",y_test)
        print('{0}_{1}.pkl'.format(ln_name,prepro_list[prepro_num]))
        print('Precision: %.5f' % precision_score(y_true=y_test, y_pred=y_pred))
        print('Recall: %.5f' % recall_score(y_true=y_test, y_pred=y_pred))
        print('Last F1: %.5f' % f1_score(y_true=y_test, y_pred=y_pred))
        print("最終前処理",prepro_list[prepro_num])
        """
        # ----------------------------------モデルの保存--------------------------------
        dest = os.path.join('classifier', 'learn-objct')  # パスの結合
        if not os.path.exists(dest):
            os.makedirs(dest)
        ln_name = str(table_list[i])
        ln_name = ln_name.strip(")").strip("(").strip(",").strip("'")#学習に使用したtable名
        model = model_save(model, prepro_num)#パイプライン作成
        pickle.dump(model,  # 保存したいモデル
                    open(os.path.join(dest,'{0}.pkl'.format(ln_name)),'wb'))  # 'classifier/pkl-objects/ensemble.pklで保存)
        #----------------------------------------------------------------------------
#ここから学習未実行スタート
#----------------------------------------------------------------------------------------------
predict_table_list = ["hateruma","hatoma","ohara","uehara","kohama","kurosima","taketomi","ishigaki"]
proba_list=[] #webサイトに表示する確率
ans = np.array([]) 
count=0
dif_list=[]
for i in range(len(table_list)):
    df = pd.read_sql("SELECT * FROM '%s'" % table_list[i], conn)
    df = columndelet(df, cdl)
    df_proba = pd.read_sql("SELECT * FROM '%s'" % table_list[i], conn)#サンプルデータの確率出すためのdf
    X = df.loc[:, 'Wave_swell0':'Wind_Max_Speed9'].values#Wave_swell0-->'Wave_height0'
    y = df.loc[:, 'Alert'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y,random_state=0)
    pre_tab_num = table_judgment(table_list,i)      #予測データtableの指定

    predict_df = pd.read_sql("SELECT * FROM '%s'" % predict_table_list[pre_tab_num], conn)#予測したいデータ
    predict_df = columndelet(predict_df,cdl)
    predict_X = predict_df[-1:]#一番最後に最新データが入っているので[-1:]で取得

    ln_name = str(table_list[i])
    ln_name = ln_name.strip(")").strip("(").strip(",").strip("'")
    model = pickle.load(open(os.path.join('classifier','learn-objct','{0}.pkl'.format(ln_name)), mode='rb'))  #モデルのロード
    #model.fit(learn_X,learn_y)#使い方modelにpickleを入れた後にfitさせる
    predict_X = predict_X.loc[:, 'Wave_swell0':'Wind_Max_Speed9'].values
    predict_list=classify(model, predict_X)#予測確率を返す
     
    a=Change_to_percentage(predict_list)#%表記に直す
    ans = np.append(ans, a)

    #----------------------------サンプルデータそれぞれに対して学習し誤差計算------------------------
    #正解確率の辞書型作成
    date_x = df_proba.iloc[0, 0]  # 日付
    canseled_rate = {}  # 欠航割合
    alert_num = 0  # 同じ日付数
    alert_ca = 0  # 同じ日付で欠航ラベルの「1」の数(欠航している)
    #res=[]
    for i in range(len(df_proba)):
        now_date = df_proba.iloc[i, 0]
        if date_x == now_date:                # 日付
            alert_num = alert_num + 1
            if 1 == int(df_proba.iloc[i, 2]):  # 欠航ラベル判定
                alert_ca = alert_ca + 1
                # print(date_x,i)
        elif date_x != now_date:
            cr = alert_ca / alert_num
            canseled_rate.setdefault(date_x, cr)#辞書型に格納
            alert_num = 1
            alert_ca = 0
            if 1 == int(df_proba.iloc[i, 2]):  # 欠航ラベル判定
                alert_ca = alert_ca + 1

        if len(df_proba) - 1 == i:
            cr = alert_ca / alert_num
            canseled_rate.setdefault(date_x, cr)#辞書型に格納
        date_x = df_proba.iloc[i, 0]           # 日付更新

    #誤差確率算出
   
    loo = LeaveOneOut()
    result_sum=0
    count=0
    twofivepar=0

    print("index番号、予想ラベル、正解ラベル、予確率、正確率、予-正")
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]   
        model.fit(X_train,y_train) # 学習させる
        pred = model.predict(X_test)  # テストデータからラベルを予測する
        pred_proba = model.predict_proba(X_test) 
        result = pred_proba[0,1] - canseled_rate[df_proba.iloc[int(test_index),0]]#誤差計算
        if abs(result) >= 0.2:#20%以上の誤差のもの
            twofivepar=round(twofivepar+abs(result),2)
            count=count+1
        result_sum = float(round(result_sum,2))+float(round(result,2))
        
        #print(test_index,pred,y[test_index],round(pred_proba[0,1],2),round(canseled_rate[df_proba.iloc[int(test_index),0]],2),round(result,2))
    if count == 0:
        dif_list.append(0)
    else:
        dif_list.append(twofivepar / count)

#dbnameと確率の辞書型
dic = dict(zip(LDB_list_name(table_list), ans))          #確率の辞書型
dic_dif = dict(zip(LDB_list_name(table_list), dif_list)) #誤差の辞書型
#print(dic_dif)


li_day = []
now = datetime.date.today()
for i in range(10):
    days = now + datetime.timedelta(days=i)
    month = days.month
    day = days.day
    monthday = str(month)+'/'+str(day)
    pd = li_day.append(monthday)

        
@app.route('/kaisetu')
def kaisetu():
    return render_template('kaisetu.html')

@app.route('/')
def main():
    return render_template('index_pure.html')

@app.route('/hatoma')
def hatoma():
    return render_template('hatoma.html', hatoma_day1=dic["LDB_hatomajima_hatoma0"], hatoma_day2=dic["LDB_hatomajima_hatoma1"], hatoma_day3=dic["LDB_hatomajima_hatoma2"], hatoma_day4=dic["LDB_hatomajima_hatoma3"], hatoma_day5=dic["LDB_hatomajima_hatoma4"], hatoma_day6=dic["LDB_hatomajima_hatoma5"], hatoma_day7=dic["LDB_hatomajima_hatoma6"], hatoma_day8=dic["LDB_hatomajima_hatoma7"], hatoma_day9=dic["LDB_hatomajima_hatoma8"], hatoma_day10=dic["LDB_hatomajima_hatoma9"], hatoma_isigaki1=dic["LDB_hatomajima_ishigaki0"], hatoma_isigaki2=dic["LDB_hatomajima_ishigaki1"], hatoma_isigaki3=dic["LDB_hatomajima_ishigaki2"], hatoma_isigaki4=dic["LDB_hatomajima_ishigaki3"], hatoma_isigaki5=dic["LDB_hatomajima_ishigaki4"], hatoma_isigaki6=dic["LDB_hatomajima_ishigaki5"], hatoma_isigaki7=dic["LDB_hatomajima_ishigaki6"], hatoma_isigaki8=dic["LDB_hatomajima_ishigaki7"], hatoma_isigaki9=dic["LDB_hatomajima_ishigaki8"], hatoma_isigaki10=dic["LDB_hatomajima_ishigaki9"], hatomagosaday1=dic_dif["LDB_hatomajima_hatoma0"], hatomagosa_day2=dic_dif["LDB_hatomajima_hatoma1"], hatomagosa_day3=dic_dif["LDB_hatomajima_hatoma2"], hatomagosa_day4=dic_dif["LDB_hatomajima_hatoma3"], hatomagosa_day5=dic_dif["LDB_hatomajima_hatoma4"], hatomagosa_day6=dic_dif["LDB_hatomajima_hatoma5"], hatomagosa_day7=dic_dif["LDB_hatomajima_hatoma6"], hatomagosa_day8=dic_dif["LDB_hatomajima_hatoma7"], hatomagosa_day9=dic_dif["LDB_hatomajima_hatoma8"], hatomagosa_day10=dic_dif["LDB_hatomajima_hatoma9"], hatomagosa_isigaki1=dic_dif["LDB_hatomajima_ishigaki0"], hatomagosa_isigaki2=dic_dif["LDB_hatomajima_ishigaki1"], hatomagosa_isigaki3=dic_dif["LDB_hatomajima_ishigaki2"], hatomagosa_isigaki4=dic_dif["LDB_hatomajima_ishigaki3"], hatomagosa_isigaki5=dic_dif["LDB_hatomajima_ishigaki4"], hatomagosa_isigaki6=dic_dif["LDB_hatomajima_ishigaki5"], hatomagosa_isigaki7=dic_dif["LDB_hatomajima_ishigaki6"], hatomagosa_isigaki8=dic_dif["LDB_hatomajima_ishigaki7"], hatomagosa_isigaki9=dic_dif["LDB_hatomajima_ishigaki8"], hatomagosa_isigaki10=dic_dif["LDB_hatomajima_ishigaki9"], li_day1=li_day[0], li_day2=li_day[1], li_day3=li_day[2], li_day4=li_day[3], li_day5=li_day[4], li_day6=li_day[5], li_day7=li_day[6], li_day8=li_day[7], li_day9=li_day[8], li_day10=li_day[9])


@app.route('/hateruma')
def hateruma():
    return render_template('hateruma.html', hateruma_day1=dic["LDB_haterumajima_hateruma0"], hateruma_day2=dic["LDB_haterumajima_hateruma1"], hateruma_day3=dic["LDB_haterumajima_hateruma2"], hateruma_day4=dic["LDB_haterumajima_hateruma3"], hateruma_day5=dic["LDB_haterumajima_hateruma4"], hateruma_day6=dic["LDB_haterumajima_hateruma5"], hateruma_day7=dic["LDB_haterumajima_hateruma6"], hateruma_day8=dic["LDB_haterumajima_hateruma7"], hateruma_day9=dic["LDB_haterumajima_hateruma8"], hateruma_day10=dic["LDB_haterumajima_hateruma9"], isigaki_day1=dic["LDB_haterumajima_ishigaki0"], isigaki_day2=dic["LDB_haterumajima_ishigaki1"], isigaki_day3=dic["LDB_haterumajima_ishigaki2"], isigaki_day4=dic["LDB_haterumajima_ishigaki3"], isigaki_day5=dic["LDB_haterumajima_ishigaki4"], isigaki_day6=dic["LDB_haterumajima_ishigaki5"], isigaki_day7=dic["LDB_haterumajima_ishigaki6"], isigaki_day8=dic["LDB_haterumajima_ishigaki7"], isigaki_day9=dic["LDB_haterumajima_ishigaki8"], isigaki_day10=dic["LDB_haterumajima_ishigaki9"], haterumagosa_day1=dic_dif["LDB_haterumajima_hateruma0"], haterumagosa_day2=dic_dif["LDB_haterumajima_hateruma1"], haterumagosa_day3=dic_dif["LDB_haterumajima_hateruma2"], haterumagosa_day4=dic_dif["LDB_haterumajima_hateruma3"], haterumagosa_day5=dic_dif["LDB_haterumajima_hateruma4"], haterumagosa_day6=dic_dif["LDB_haterumajima_hateruma5"], haterumagosa_day7=dic_dif["LDB_haterumajima_hateruma6"], haterumagosa_day8=dic_dif["LDB_haterumajima_hateruma7"], haterumagosa_day9=dic_dif["LDB_haterumajima_hateruma8"], haterumagosa_day10=dic_dif["LDB_haterumajima_hateruma9"], haterumagosa_ishigakiday1=dic_dif["LDB_haterumajima_ishigaki0"], haterumagosa_ishigakiday2=dic_dif["LDB_haterumajima_ishigaki1"], haterumagosa_ishigakiday3=dic_dif["LDB_haterumajima_ishigaki2"], haterumagosa_ishigakiday4=dic_dif["LDB_haterumajima_ishigaki3"], haterumagosa_ishigakiday5=dic_dif["LDB_haterumajima_ishigaki4"], haterumagosa_ishigakiday6=dic_dif["LDB_haterumajima_ishigaki5"], haterumagosa_ishigakiday7=dic_dif["LDB_haterumajima_ishigaki6"], haterumagosa_ishigakiday8=dic_dif["LDB_haterumajima_ishigaki7"], haterumagosa_ishigakiday9=dic_dif["LDB_haterumajima_ishigaki8"], haterumagosa_ishigakiday10=dic_dif["LDB_haterumajima_ishigaki9"], li_day1=li_day[0], li_day2=li_day[1], li_day3=li_day[2], li_day4=li_day[3], li_day5=li_day[4], li_day6=li_day[5], li_day7=li_day[6], li_day8=li_day[7], li_day9=li_day[8], li_day10=li_day[9])


@app.route('/ohara')
def ohara():
    return render_template('ohara.html', ohara_isigaki1=dic["LDB_iriomotejima_ohara_ishigaki0"], ohara_isigaki2=dic["LDB_iriomotejima_ohara_ishigaki1"], ohara_isigaki3=dic["LDB_iriomotejima_ohara_ishigaki2"], ohara_isigaki4=dic["LDB_iriomotejima_ohara_ishigaki3"], ohara_isigaki5=dic["LDB_iriomotejima_ohara_ishigaki4"], ohara_isigaki6=dic["LDB_iriomotejima_ohara_ishigaki5"], ohara_isigaki7=dic["LDB_iriomotejima_ohara_ishigaki6"], ohara_isigaki8=dic["LDB_iriomotejima_ohara_ishigaki7"], ohara_isigaki9=dic["LDB_iriomotejima_ohara_ishigaki8"], ohara_isigaki10=dic["LDB_iriomotejima_ohara_ishigaki9"], ohara_day1=dic["LDB_iriomotejima_ohara_ohara0"], ohara_day2=dic["LDB_iriomotejima_ohara_ohara1"], ohara_day3=dic["LDB_iriomotejima_ohara_ohara2"], ohara_day4=dic["LDB_iriomotejima_ohara_ohara3"], ohara_day5=dic["LDB_iriomotejima_ohara_ohara4"], ohara_day6=dic["LDB_iriomotejima_ohara_ohara5"], ohara_day7=dic["LDB_iriomotejima_ohara_ohara6"], ohara_day8=dic["LDB_iriomotejima_ohara_ohara7"], ohara_day9=dic["LDB_iriomotejima_ohara_ohara8"], ohara_day10=dic["LDB_iriomotejima_ohara_ohara9"], oharagosa_ishigaki1=dic_dif["LDB_iriomotejima_ohara_ishigaki0"], oharagosa_ishigaki2=dic_dif["LDB_iriomotejima_ohara_ishigaki1"], oharagosa_ishigaki3=dic_dif["LDB_iriomotejima_ohara_ishigaki2"], oharagosa_ishigaki4=dic_dif["LDB_iriomotejima_ohara_ishigaki3"], oharagosa_ishigaki5=dic_dif["LDB_iriomotejima_ohara_ishigaki4"], oharagosa_ishigaki6=dic_dif["LDB_iriomotejima_ohara_ishigaki5"], oharagosa_ishigaki7=dic_dif["LDB_iriomotejima_ohara_ishigaki6"], oharagosa_ishigaki8=dic_dif["LDB_iriomotejima_ohara_ishigaki7"], oharagosa_ishigaki9=dic_dif["LDB_iriomotejima_ohara_ishigaki8"], oharagosa_ishigaki10=dic_dif["LDB_iriomotejima_ohara_ishigaki9"], oharagosa_day1=dic_dif["LDB_iriomotejima_ohara_ohara0"], oharagosa_day2=dic_dif["LDB_iriomotejima_ohara_ohara1"], oharagosa_day3=dic_dif["LDB_iriomotejima_ohara_ohara2"], oharagosa_day4=dic_dif["LDB_iriomotejima_ohara_ohara3"], oharagosa_day5=dic_dif["LDB_iriomotejima_ohara_ohara4"], oharagosa_day6=dic_dif["LDB_iriomotejima_ohara_ohara5"], oharagosa_day7=dic_dif["LDB_iriomotejima_ohara_ohara6"], oharagosa_day8=dic_dif["LDB_iriomotejima_ohara_ohara7"], oharagosa_day9=dic_dif["LDB_iriomotejima_ohara_ohara8"], oharagosa_day10=dic_dif["LDB_iriomotejima_ohara_ohara9"], li_day1=li_day[0], li_day2=li_day[1], li_day3=li_day[2], li_day4=li_day[3], li_day5=li_day[4], li_day6=li_day[5], li_day7=li_day[6], li_day8=li_day[7], li_day9=li_day[8], li_day10=li_day[9])


@app.route('/uehara')
def uehara():
    return render_template('uehara.html', uehara_isigaki1=dic["LDB_iriomotejima_uehara_ishigaki0"], uehara_isigaki2=dic["LDB_iriomotejima_uehara_ishigaki1"], uehara_isigaki3=dic["LDB_iriomotejima_uehara_ishigaki2"], uehara_isigaki4=dic["LDB_iriomotejima_uehara_ishigaki3"], uehara_isigaki5=dic["LDB_iriomotejima_uehara_ishigaki4"], uehara_isigaki6=dic["LDB_iriomotejima_uehara_ishigaki5"], uehara_isigaki7=dic["LDB_iriomotejima_uehara_ishigaki6"], uehara_isigaki8=dic["LDB_iriomotejima_uehara_ishigaki7"], uehara_isigaki9=dic["LDB_iriomotejima_uehara_ishigaki8"], uehara_isigaki10=dic["LDB_iriomotejima_uehara_ishigaki9"], uehara_day1=dic["LDB_iriomotejima_uehara_uehara0"], uehara_day2=dic["LDB_iriomotejima_uehara_uehara1"], uehara_day3=dic["LDB_iriomotejima_uehara_uehara2"], uehara_day4=dic["LDB_iriomotejima_uehara_uehara3"], uehara_day5=dic["LDB_iriomotejima_uehara_uehara4"], uehara_day6=dic["LDB_iriomotejima_uehara_uehara5"], uehara_day7=dic["LDB_iriomotejima_uehara_uehara6"], uehara_day8=dic["LDB_iriomotejima_uehara_uehara7"], uehara_day9=dic["LDB_iriomotejima_uehara_uehara8"], uehara_day10=dic["LDB_iriomotejima_uehara_uehara9"], ueharagosa_isigaki1=dic_dif["LDB_iriomotejima_uehara_ishigaki0"], ueharagosa_isigaki2=dic_dif["LDB_iriomotejima_uehara_ishigaki1"], ueharagosa_isigaki3=dic_dif["LDB_iriomotejima_uehara_ishigaki2"], ueharagosa_isigaki4=dic_dif["LDB_iriomotejima_uehara_ishigaki3"], ueharagosa_isigaki5=dic_dif["LDB_iriomotejima_uehara_ishigaki4"], ueharagosa_isigaki6=dic_dif["LDB_iriomotejima_uehara_ishigaki5"], ueharagosa_isigaki7=dic_dif["LDB_iriomotejima_uehara_ishigaki6"], ueharagosa_isigaki8=dic_dif["LDB_iriomotejima_uehara_ishigaki7"], ueharagosa_isigaki9=dic_dif["LDB_iriomotejima_uehara_ishigaki8"], ueharagosa_isigaki10=dic_dif["LDB_iriomotejima_uehara_ishigaki9"], ueharagosa_1=dic_dif["LDB_iriomotejima_uehara_uehara0"], ueharagosa_2=dic_dif["LDB_iriomotejima_uehara_uehara1"], ueharagosa_3=dic_dif["LDB_iriomotejima_uehara_uehara2"], ueharagosa_4=dic_dif["LDB_iriomotejima_uehara_uehara3"], ueharagosa_5=dic_dif["LDB_iriomotejima_uehara_uehara4"], ueharagosa_6=dic_dif["LDB_iriomotejima_uehara_uehara5"], ueharagosa_7=dic_dif["LDB_iriomotejima_uehara_uehara6"], ueharagosa_8=dic_dif["LDB_iriomotejima_uehara_uehara7"], ueharagosa_9=dic_dif["LDB_iriomotejima_uehara_uehara8"], ueharagosa_10=dic_dif["LDB_iriomotejima_uehara_uehara9"], li_day1=li_day[0], li_day2=li_day[1], li_day3=li_day[2], li_day4=li_day[3], li_day5=li_day[4], li_day6=li_day[5], li_day7=li_day[6], li_day8=li_day[7], li_day9=li_day[8], li_day10=li_day[9])


@app.route('/kohama')
def kohama():
    return render_template('kohama.html', kohama_ishigaki1=dic["LDB_kohamajima_ishigaki0"], kohama_isigaki2=dic["LDB_kohamajima_ishigaki1"], kohama_isigaki3=dic["LDB_kohamajima_ishigaki2"], kohama_isigaki4=dic["LDB_kohamajima_ishigaki3"], kohama_isigaki5=dic["LDB_kohamajima_ishigaki4"], kohama_isigaki6=dic["LDB_kohamajima_ishigaki5"], kohama_isigaki7=dic["LDB_kohamajima_ishigaki6"], kohama_isigaki8=dic["LDB_kohamajima_ishigaki7"], kohama_isigaki9=dic["LDB_kohamajima_ishigaki8"], kohama_isigaki10=dic["LDB_kohamajima_ishigaki9"], kohama_day1=dic["LDB_kohamajima_kohama0"], kohama_day2=dic["LDB_kohamajima_kohama1"], kohama_day3=dic["LDB_kohamajima_kohama2"], kohama_day4=dic["LDB_kohamajima_kohama3"], kohama_day5=dic["LDB_kohamajima_kohama4"], kohama_day6=dic["LDB_kohamajima_kohama5"], kohama_day7=dic["LDB_kohamajima_kohama6"], kohama_day8=dic["LDB_kohamajima_kohama7"], kohama_day9=dic["LDB_kohamajima_kohama8"], kohama_day10=dic["LDB_kohamajima_kohama9"], kohamagosa_ishigaki1=dic_dif["LDB_kohamajima_ishigaki0"], kohamagosa_ishigaki2=dic_dif["LDB_kohamajima_ishigaki1"], kohamagosa_ishigaki3=dic_dif["LDB_kohamajima_ishigaki2"], kohamagosa_ishigaki4=dic_dif["LDB_kohamajima_ishigaki3"], kohamagosa_ishigaki5=dic_dif["LDB_kohamajima_ishigaki4"], kohamagosa_ishigaki6=dic_dif["LDB_kohamajima_ishigaki5"], kohamagosa_ishigaki7=dic_dif["LDB_kohamajima_ishigaki6"], kohamagosa_ishigaki8=dic_dif["LDB_kohamajima_ishigaki7"], kohamagosa_ishigaki9=dic_dif["LDB_kohamajima_ishigaki8"], kohamagosa_ishigaki10=dic_dif["LDB_kohamajima_ishigaki9"], kohamagosa_1=dic_dif["LDB_kohamajima_kohama0"], kohamagosa_2=dic_dif["LDB_kohamajima_kohama1"], kohamagosa_3=dic_dif["LDB_kohamajima_kohama2"], kohamagosa_4=dic_dif["LDB_kohamajima_kohama3"], kohamagosa_5=dic_dif["LDB_kohamajima_kohama4"], kohamagosa_6=dic_dif["LDB_kohamajima_kohama5"], kohamagosa_7=dic_dif["LDB_kohamajima_kohama6"], kohamagosa_8=dic_dif["LDB_kohamajima_kohama7"], kohamagosa_9=dic_dif["LDB_kohamajima_kohama8"], kohamagosa_10=dic_dif["LDB_kohamajima_kohama9"], li_day1=li_day[0], li_day2=li_day[1], li_day3=li_day[2], li_day4=li_day[3], li_day5=li_day[4], li_day6=li_day[5], li_day7=li_day[6], li_day8=li_day[7], li_day9=li_day[8], li_day10=li_day[9])


@app.route('/kurosima')
def kurosima():
    return render_template('kurosima.html', kurosima_isigaki1=dic["LDB_kuroshima_ishigaki0"], kurosima_isigaki2=dic["LDB_kuroshima_ishigaki1"], kurosima_isigaki3=dic["LDB_kuroshima_ishigaki2"], kurosima_isigaki4=dic["LDB_kuroshima_ishigaki3"], kurosima_isigaki5=dic["LDB_kuroshima_ishigaki4"], kurosima_isigaki6=dic["LDB_kuroshima_ishigaki5"], kurosima_isigaki7=dic["LDB_kuroshima_ishigaki6"], kurosima_isigaki8=dic["LDB_kuroshima_ishigaki7"], kurosima_isigaki9=dic["LDB_kuroshima_ishigaki8"], kurosima_isigaki10=dic["LDB_kuroshima_ishigaki9"], kurosima_day1=dic["LDB_kuroshima_kuroshima0"], kurosima_day2=dic["LDB_kuroshima_kuroshima1"], kurosima_day3=dic["LDB_kuroshima_kuroshima2"], kurosima_day4=dic["LDB_kuroshima_kuroshima3"], kurosima_day5=dic["LDB_kuroshima_kuroshima4"], kurosima_day6=dic["LDB_kuroshima_kuroshima5"], kurosima_day7=dic["LDB_kuroshima_kuroshima6"], kurosima_day8=dic["LDB_kuroshima_kuroshima7"], kurosima_day9=dic["LDB_kuroshima_kuroshima8"], kurosima_day10=dic["LDB_kuroshima_kuroshima9"], kuroshimagosa_ishigaki1=dic_dif["LDB_kuroshima_ishigaki0"], kuroshimagosa_ishigaki2=dic_dif["LDB_kuroshima_ishigaki1"], kuroshimagosa_ishigaki3=dic_dif["LDB_kuroshima_ishigaki2"], kuroshimagosa_ishigaki4=dic_dif["LDB_kuroshima_ishigaki3"], kuroshimagosa_ishigaki5=dic_dif["LDB_kuroshima_ishigaki4"], kuroshimagosa_ishigaki6=dic_dif["LDB_kuroshima_ishigaki5"], kuroshimagosa_ishigaki7=dic_dif["LDB_kuroshima_ishigaki6"], kuroshimagosa_ishigaki8=dic_dif["LDB_kuroshima_ishigaki7"], kuroshimagosa_ishigaki9=dic_dif["LDB_kuroshima_ishigaki8"], kuroshimagosa_ishigaki10=dic_dif["LDB_kuroshima_ishigaki9"], kuroshimagosa_1=dic_dif["LDB_kuroshima_kuroshima0"], kuroshimagosa_2=dic_dif["LDB_kuroshima_kuroshima1"], kuroshimagosa_3=dic_dif["LDB_kuroshima_kuroshima2"], kuroshimagosa_4=dic_dif["LDB_kuroshima_kuroshima3"], kuroshimagosa_5=dic_dif["LDB_kuroshima_kuroshima4"], kuroshimagosa_6=dic_dif["LDB_kuroshima_kuroshima5"], kuroshimagosa_7=dic_dif["LDB_kuroshima_kuroshima6"], kuroshimagosa_8=dic_dif["LDB_kuroshima_kuroshima7"], kuroshimagosa_9=dic_dif["LDB_kuroshima_kuroshima8"], kuroshimagosa_10=dic_dif["LDB_kuroshima_kuroshima9"], li_day1=li_day[0], li_day2=li_day[1], li_day3=li_day[2], li_day4=li_day[3], li_day5=li_day[4], li_day6=li_day[5], li_day7=li_day[6], li_day8=li_day[7], li_day9=li_day[8], li_day10=li_day[9])


@app.route('/taketomi')
def taketomi():
    return render_template('taketomi.html', taketomi_isigaki1=dic["LDB_taketomi_isigaki0"], taketomi_isigaki2=dic["LDB_taketomi_isigaki1"], taketomi_isigaki3=dic["LDB_taketomi_isigaki2"], taketomi_isigaki4=dic["LDB_taketomi_isigaki3"], taketomi_isigaki5=dic["LDB_taketomi_isigaki4"], taketomi_isigaki6=dic["LDB_taketomi_isigaki5"], taketomi_isigaki7=dic["LDB_taketomi_isigaki6"], taketomi_isigaki8=dic["LDB_taketomi_isigaki7"], taketomi_isigaki9=dic["LDB_taketomi_isigaki8"], taketomi_isigaki10=dic["LDB_taketomi_isigaki9"], taketomi_day1=dic["LDB_taketomi_taketomi0"], taketomi_day2=dic["LDB_taketomi_isigaki1"], taketomi_day3=dic["LDB_taketomi_isigaki2"], taketomi_day4=dic["LDB_taketomi_isigaki3"], taketomi_day5=dic["LDB_taketomi_isigaki4"], taketomi_day6=dic["LDB_taketomi_isigaki5"], taketomi_day7=dic["LDB_taketomi_isigaki6"], taketomi_day8=dic["LDB_taketomi_isigaki7"], taketomi_day9=dic["LDB_taketomi_isigaki8"], taketomi_day10=dic["LDB_taketomi_isigaki9"], taketomigosa_isigaki1=dic_dif["LDB_taketomi_isigaki0"], taketomigosa_isigaki2=dic_dif["LDB_taketomi_isigaki1"], taketomigosa_isigaki3=dic_dif["LDB_taketomi_isigaki2"], taketomigosa_isigaki4=dic_dif["LDB_taketomi_isigaki3"], taketomigosa_isigaki5=dic_dif["LDB_taketomi_isigaki4"], taketomigosaisigaki6=dic_dif["LDB_taketomi_isigaki5"], taketomigosa_isigaki7=dic_dif["LDB_taketomi_isigaki6"], taketomisosa_isigaki8=dic_dif["LDB_taketomi_isigaki7"], taketomigosa_isigaki9=dic_dif["LDB_taketomi_isigaki8"], taketomigosa_isigaki10=dic_dif["LDB_taketomi_isigaki9"], taketomigosa_day1=dic_dif["LDB_taketomi_taketomi0"], taketomigosa_day2=dic_dif["LDB_taketomi_isigaki1"], taketomigosa_day3=dic_dif["LDB_taketomi_isigaki2"], taketomigosa_day4=dic_dif["LDB_taketomi_isigaki3"], taketomigosa_day5=dic_dif["LDB_taketomi_isigaki4"], taketomigosa_day6=dic_dif["LDB_taketomi_isigaki5"], taketomigosa_day7=dic_dif["LDB_taketomi_isigaki6"], taketomigosa_day8=dic_dif["LDB_taketomi_isigaki7"], taketomigosa_day9=dic_dif["LDB_taketomi_isigaki8"], taketomigosa_day10=dic_dif["LDB_taketomi_isigaki9"], li_day1=li_day[0], li_day2=li_day[1], li_day3=li_day[2], li_day4=li_day[3], li_day5=li_day[4], li_day6=li_day[5], li_day7=li_day[6], li_day8=li_day[7], li_day9=li_day[8], li_day10=li_day[9])

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=8080, threaded=True)

