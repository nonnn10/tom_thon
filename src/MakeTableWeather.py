# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import os
from pathlib import Path

# 読み込み用ディレクトリ
dir_csv_weather = Path('./windy_data/')
list_csv_weather = sorted(dir_csv_weather.glob('*.csv'))

# 書き込み用ディレクトリ
dir_db = 'databases2'
if not os.path.exists(dir_db):
    os.makedirs(dir_db)
db_name = os.path.join(dir_db, "learningData.db")

connection = sqlite3.connect(db_name)
cursor = connection.cursor()

for data in list_csv_weather:
    t_name1 = str(data).split("/")
    t_name2 = t_name1[1].split(".")
    table_name = t_name2[0]

    df_weather = pd.read_csv(data, header=None, encoding='utf-8')
    df_weather.columns = [
        "Date0", "Wave_swell0", "Temperature0", "Wave_height0", "Wind_speed0", "Wind_Max_Speed0",
        "Date1", "Wave_swell1", "Temperature1", "Wave_height1", "Wind_speed1", "Wind_Max_Speed1",
        "Date2", "Wave_swell2", "Temperature2", "Wave_height2", "Wind_speed2", "Wind_Max_Speed2",
        "Date3", "Wave_swell3", "Temperature3", "Wave_height3", "Wind_speed3", "Wind_Max_Speed3",
        "Date4", "Wave_swell4", "Temperature4", "Wave_height4", "Wind_speed4", "Wind_Max_Speed4",
        "Date5", "Wave_swell5", "Temperature5", "Wave_height5", "Wind_speed5", "Wind_Max_Speed5",
        "Date6", "Wave_swell6", "Temperature6", "Wave_height6", "Wind_speed6", "Wind_Max_Speed6",
        "Date7", "Wave_swell7", "Temperature7", "Wave_height7", "Wind_speed7", "Wind_Max_Speed7",
        "Date8", "Wave_swell8", "Temperature8", "Wave_height8", "Wind_speed8", "Wind_Max_Speed8",
        "Date9", "Wave_swell9", "Temperature9", "Wave_height9", "Wind_speed9", "Wind_Max_Speed9"
    ]
    df_weather["Date0"] = pd.to_datetime(df_weather["Date0"])
    df_weather["Date1"] = pd.to_datetime(df_weather["Date1"])
    df_weather["Date2"] = pd.to_datetime(df_weather["Date2"])
    df_weather["Date3"] = pd.to_datetime(df_weather["Date3"])
    df_weather["Date4"] = pd.to_datetime(df_weather["Date4"])
    df_weather["Date5"] = pd.to_datetime(df_weather["Date5"])
    df_weather["Date6"] = pd.to_datetime(df_weather["Date6"])
    df_weather["Date7"] = pd.to_datetime(df_weather["Date7"])
    df_weather["Date8"] = pd.to_datetime(df_weather["Date8"])
    df_weather["Date9"] = pd.to_datetime(df_weather["Date9"])

    try:
        df_weather.to_sql(table_name, connection, if_exists='append', index=None)
    except sqlite3.Error as e:
        print(e)

# 変更を保存．
connection.commit()
connection.close()
