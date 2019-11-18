# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

#読み込み用ディレクトリ
dir_csv_alert = Path('./alertData_windy/')
list_csv_alert = sorted(dir_csv_alert.glob('*.csv'))

#書き込み用ディレクトリ
dir_db = 'databases2'
if not os.path.exists(dir_db):
    os.makedirs(dir_db)
db_name = os.path.join(dir_db,"learningData.db")

connection = sqlite3.connect(db_name)
cursor = connection.cursor()

for files in list_csv_alert:
    #安永観光の欠航データのデータベース作成
    df_al = pd.read_csv(files,header=None,encoding='utf-8')
    df_al.columns = ["Date","Time","Alert"]
    df_al["Date"] = pd.to_datetime(df_al["Date"])

    table_name_alert = []
    for i in range(0,10):
        data1 = str(files).split("/")
        data2 = data1[1].split(".")
        data3 = data2[0] + str(i)
        table_name_alert.append(data3)

    date = "2019/10/"

    for i in range(0,10):
        date2 = date + str(22+i)
        if i==0:
            try:
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[0], connection, if_exists='append', index=None)
            except sqlite3.Error as e:
                print(e)
        elif i==1:
            try:
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[0], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[1], connection, if_exists='append', index=None)
            except sqlite3.Error as e:
                print(e)
        elif i==2:
            try:
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[0], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[1], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[2], connection, if_exists='append', index=None)
            except sqlite3.Error as e:
                print(e)
        elif i==3:
            try:
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[0], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[1], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[2], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[3], connection, if_exists='append', index=None)
            except sqlite3.Error as e:
                print(e)
        elif i==4:
            try:
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[0], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[1], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[2], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[3], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[4], connection, if_exists='append', index=None)
            except sqlite3.Error as e:
                print(e)
        elif i==5:
            try:
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[0], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[1], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[2], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[3], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[4], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[5], connection, if_exists='append', index=None)
            except sqlite3.Error as e:
                print(e)
        elif i==6:
            try:
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[0], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[1], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[2], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[3], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[4], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[5], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[6], connection, if_exists='append', index=None)
            except sqlite3.Error as e:
                print(e)
        elif i==7:
            try:
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[0], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[1], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[2], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[3], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[4], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[5], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[6], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[7], connection, if_exists='append', index=None)
            except sqlite3.Error as e:
                print(e)
        elif i==8:
            try:
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[0], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[1], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[2], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[3], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[4], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[5], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[6], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[7], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[8], connection, if_exists='append', index=None)
            except sqlite3.Error as e:
                print(e)
        elif i==9:
            try:
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[0], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[1], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[2], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[3], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[4], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[5], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[6], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[7], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[8], connection, if_exists='append', index=None)
                df_al[df_al['Date'].isin([date2])].to_sql(table_name_alert[9], connection, if_exists='append', index=None)
            except sqlite3.Error as e:
                print(e)

    df_al2 = df_al[df_al["Date"] > datetime(2019, 10, 31)]
    for i in range(0,10):
        df_al2.to_sql(table_name_alert[i],connection,if_exists='append',index=None)


#変更を保存．
connection.commit()
connection.close()
