# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import datetime
import os
import pandas as pd
import sqlite3

# 書き込み用ディレクトリ
dir_db = '/Users/moriken/sdn/web/Anei/src/db_windy'
if not os.path.exists(dir_db):
    os.makedirs(dir_db)
db_name = os.path.join(dir_db, "learningData.db")
connection = sqlite3.connect(db_name)
cursor = connection.cursor()

base_url = "http://www.aneikankou.co.jp/timetables?date=20%s-%s-%s"

table_name = ["taketomi_ishigaki",
              "taketomi_taketomi",
              "kuroshima_ishigaki",
              "kuroshima_kuroshima",
              "kohamajima_ishigaki",
              "kohamajima_kohama",
              "iriomotejima_uehara_ishigaki",
              "iriomotejima_uehara_uehara",
              "hatomajima_ishigaki",
              "hatomajima_hatoma",
              "iriomotejima_ohara_ishigaki",
              "iriomotejima_ohara_ohara",
              "haterumajima_ishigaki",
              "haterumajima_hateruma"]


def make_csv(table, date, tb1, tb2):
    rows = table.findAll('tr')
    for row in rows:
        table_data = []
        for cell in row.findAll('td'):
            table_data.append(cell.get_text())
        if len(table_data) > 0:
            # 石垣発のテーブル
            if table_data[1] != "-":
                if table_data[1] == "欠航":
                    AlertNumber1 = 1
                else:
                    AlertNumber1 = 0
                # DB書き込み処理
                writeData = [date, table_data[0], AlertNumber1]
                df = pd.DataFrame([writeData])
                df.columns = ["Date", "Time", "Alert"]
                df["Date"] = pd.to_datetime(df["Date"])
                for tbNumber in range(0, 10):
                    tbName = tb1 + str(tbNumber)
                    try:
                        df.to_sql(tbName, connection, if_exists='append', index=None)
                    except sqlite3.Error as e:
                        print(e)
            # 各港発のテーブル
            if table_data[3] != "-":
                if table_data[3] == "欠航":
                    AlertNumber2 = 1
                else:
                    AlertNumber2 = 0
                # DB書き込み処理
                writeData = [date, table_data[2], AlertNumber2]
                df = pd.DataFrame([writeData])
                df.columns = ["Date", "Time", "Alert"]
                df["Date"] = pd.to_datetime(df["Date"])
                for tbNumber in range(0, 10):
                    tbName = tb2 + str(tbNumber)
                    try:
                        df.to_sql(tbName, connection, if_exists='append', index=None)
                    except sqlite3.Error as e:
                        print(e)


def action_anei():
    count = 0
    time = str(datetime.datetime.now())
    print("Start Anei")
    print(time)
    data = time.split(' ')
    ymd = data[0].split('-')
    y = str(int(ymd[0]) - 2000)
    m = ymd[1]
    d = str(29)
    # d = ymd[2]
    html = requests.get(base_url % (y, m, d))
    soup = BeautifulSoup(html.text, 'html.parser')
    for table_num in range(7):
        start = soup.find_all("table")[table_num]
        make_csv(start, data[0], table_name[count], table_name[count + 1])
        count = count + 2
    print("Finish Anei")


action_anei()
