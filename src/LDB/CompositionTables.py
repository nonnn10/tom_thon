# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import os

# pandasのデータフレームを表示するときに見やすくするための処理
pd.options.display.max_columns = None
pd.options.display.max_rows = 100

# 書き込み用として用意
dir_db = 'databases2'
db_name = os.path.join(dir_db,"learningData.db")
# データベースに接続
connection = sqlite3.connect(db_name)
cursor = connection.cursor()
# 参照する安永テーブル(石垣発以外)のリスト
table_list_al = ["haterumajima_hateruma","hatomajima_hatoma","iriomotejima_ohara_ohara","iriomotejima_uehara_uehara",
                 "kohamajima_kohama","kuroshima_kuroshima","taketomi_taketomi"]
# 参照する安永テーブル(石垣発)のリスト
table_list_al_ishigaki = ["haterumajima_ishigaki","hatomajima_ishigaki","iriomotejima_ohara_ishigaki","iriomotejima_uehara_ishigaki",
                          "kohamajima_ishigaki","kuroshima_ishigaki","taketomi_isigaki"]
# 参照するwindyテーブルのリスト(石垣以外)
table_list_windy = ["hateruma","hatoma","ohara","uehara","kohama","kuroshima","taketomi"]
# テーブル結合に使う列名のリスト
dateList = ["Date0","Date1","Date2","Date3","Date4","Date5","Date6","Date7","Date8","Date9"]

try:
    # windyテーブル(ishigaki)の読み込み
    df_weather_ishigaki = pd.read_sql("select * from ishigaki", connection)
    for i in range(0, len(table_list_al)):
        # windyテーブル(ishigaki以外)の読み込み
        df_weather = pd.read_sql("select * from " + table_list_windy[i], connection)
        for j in range(0, 10):
            # 学習用テーブルの名前を作る処理
            accessName1 = str(table_list_al[i]) + str(j)
            tableName1 = "LDB_" + accessName1
            accessName2 = str(table_list_al_ishigaki[i]) + str(j)
            tableName2 = "LDB_" + accessName2
            # 安永テーブル(石垣発以外)の読み込み
            df_al = pd.read_sql('SELECT * FROM ' + accessName1, connection)
            # 安永テーブル(石垣発)の読み込み
            df_al_ishigaki = pd.read_sql('SELECT * FROM ' + accessName2, connection)
            # 安永テーブルの列名Dateとwindyテーブルの列名Date0~9で同じ日付のものがあれば結合する処理
            df_LDB1 = pd.merge(df_al, df_weather, left_on='Date', right_on=str(dateList[j]))
            df_LDB2 = pd.merge(df_al_ishigaki, df_weather_ishigaki, left_on='Date', right_on=str(dateList[j]))
            # テーブルの書き換え
            df_LDB1.to_sql(tableName1, connection, if_exists='replace', index=None)
            df_LDB2.to_sql(tableName2, connection, if_exists='replace', index=None)
except sqlite3.Error as e:
    print(e)

connection.commit()
connection.close()