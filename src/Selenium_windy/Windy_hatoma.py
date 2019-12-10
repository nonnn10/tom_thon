# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
import datetime
import pandas as pd
import sqlite3
import os

uri = 'https://www.windy.com/24.418/123.805'

table_name = 'hatoma'

print("Start update " + table_name + ".")

Day = []
time_list = []
temp_list = []
ws_list = []
mws_list = []
WH_list = []
swell_list = []
try:
    # ブラウザのオプションを格納する変数をもらってくる
    options = Options()
    # ヘッドレスブラウザ化
    options.add_argument('--headless')

    # ブラウザを起動する
    PATH = '/Users/moriken/SDN/web/Anei/src/Selenium/chromedriver'
    driver = webdriver.Chrome(executable_path=PATH, chrome_options=options)
    # ブラウザでアクセスする
    driver.get(uri)
    print(uri)
    # HTMLを文字コードをUTF-8に変換してから取得
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupで扱えるようにパースします
    soup = BeautifulSoup(html, "html.parser")

    sleep(10)
    # 要素が見つかるまで指定時間繰り返し実行
    driver.implicitly_wait(10)
    # 10日後まで増やすボタンをクリック
    driver.find_element_by_css_selector(
        "#detail > div.table-wrapper.show.noselect.notap > div.data-table.noselect.flex-container > div.forecast-table.progress-bar > div.fg-gray.size-xs.inlined.clickable").click()
    sleep(10)

    # Days
    all_day = driver.find_element_by_css_selector("#detail-data-table > tbody > tr.td-days.height-days").text
    day = str(all_day.replace('\n', ' ')).split(" ")
    for i in range(0, len(day)):
        if i % 2 == 1:
            Day.append(day[i])

    # Times
    time = driver.find_element_by_css_selector(
        "#detail-data-table > tbody > tr.td-hour.height-hour.d-display-table").text
    time_list = str(time).split(" ")

    # Temperature
    temp = driver.find_element_by_css_selector(
        "#detail-data-table > tbody > tr.td-temp.height-temp.d-display-table").text
    temp_list = str(temp.replace('°', '')).split(" ")

    # Wind speed
    ws = driver.find_element_by_css_selector("#detail-data-table > tbody > tr.td-wind.height-wind.d-display-table").text
    ws_list = str(ws).split(" ")

    # Max wind speed
    mws = driver.find_element_by_css_selector(
        "#detail-data-table > tbody > tr.td-gust.height-gust.d-display-table").text
    mws_list = str(mws).split(" ")

    # 波のページを開くボタンをクリック
    driver.find_element_by_css_selector(
        "#detail-box > div.dbitem.transparent-switch.compact.size-s.display-as > div:nth-child(5)").click()

    # Wave height
    WH = driver.find_element_by_css_selector(
        "#detail-data-table > tbody > tr.td-waves.height-waves.d-display-waves").text
    WH_list = str(WH).translate(str.maketrans({'\n': None, '#': ' '})).split(" ")

    # Swell
    swell = driver.find_element_by_css_selector(
        "#detail-data-table > tbody > tr.td-swell1.height-swell1.d-display-waves").text
    swell_list = str(swell).translate(str.maketrans({'\n': None, '#': ' '})).split(" ")

    # ページを閉じる
    driver.close()
    # ブラウザを閉じる
    driver.quit()
except:
    # ブラウザを閉じる
    driver.quit()
    print("Your try is a failure.")

print("Day:" + str(len(Day)))
print("Time:" + str(len(time_list)))
print("Temp:" + str(len(temp_list)))
print("Wind speed:" + str(len(ws_list)))
print("Max wind speed:" + str(len(mws_list)))
print("Wave height:" + str(len(WH_list)))
print("Swell:" + str(len(swell_list)))

d_today = datetime.date.today()
Days = [str(d_today)]
for t in range(1, 10):
    td = datetime.timedelta(days=t)
    Days.append(str(d_today + td))

del WH_list[0]
del swell_list[0]
temp_list2 = list(map(float, temp_list))
ws_list2 = list(map(float, ws_list))
mws_list2 = list(map(float, mws_list))
WH_list2 = list(map(float, WH_list))
swell_list2 = list(map(float, swell_list))

data = []

s = 0
e = 8
for i in range(0, 10):
    if i <= 5:
        data.append(Days[i])
        data.append(sum(swell_list2[s:e]) / 8)
        data.append(sum(temp_list2[s:e]) / 8)
        data.append(sum(WH_list2[s:e]) / 8)
        data.append(sum(ws_list2[s:e]) / 8)
        data.append(sum(mws_list2[s:e]) / 8)
        if i != 5:
            s += 8
            e += 8
        else:
            s += 8
            e += 4
    else:
        data.append(Days[i])
        data.append(sum(swell_list2[s:e]) / 4)
        data.append(sum(temp_list2[s:e]) / 4)
        data.append(sum(WH_list2[s:e]) / 4)
        data.append(sum(ws_list2[s:e]) / 4)
        data.append(sum(mws_list2[s:e]) / 4)
        s += 4
        e += 4

df = pd.DataFrame([data])
df.columns = [
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

# 書き込み用ディレクトリ
dir_db = '/Users/moriken/sdn/web/Anei/src/db_windy'
if not os.path.exists(dir_db):
    os.makedirs(dir_db)
db_name = os.path.join(dir_db, "learningData.db")
connection = sqlite3.connect(db_name)
cursor = connection.cursor()

for t in range(0, 10):
    date = "Date" + str(t)
    df[date] = pd.to_datetime(df[date])

try:
    df.to_sql(table_name, connection, if_exists='append', index=None)
except sqlite3.Error as e:
    print(e)

# 変更を保存．
connection.commit()
connection.close()

print("Successful!")
print("Updated " + table_name + " DB.")

