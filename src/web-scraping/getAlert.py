import requests
from bs4 import BeautifulSoup
import csv
import datetime


base_url = "http://www.aneikankou.co.jp/timetables?date=20%s-%s-%s"

file_name = ["testdata3/taketomi_isigaki.csv",
             "testdata3/taketomi_taketomi.csv",
             "testdata3/kuroshima_ishigaki.csv",
             "testdata3/kuroshima_kuroshima.csv",
             "testdata3/kohamajima_ishigaki.csv",
             "testdata3/kohamajima_kohama.csv",
             "testdata3/iriomotejima_uehara_ishigaki.csv",
             "testdata3/iriomotejima_uehara_uehara.csv",
             "testdata3/hatomajima_ishigaki.csv",
             "testdata3/hatomajima_hatoma.csv",
             "testdata3/iriomotejima_ohara_ishigaki.csv",
             "testdata3/iriomotejima_ohara_ohara.csv",
             "testdata3/haterumajima_ishigaki.csv",
             "testdata3/haterumajima_hateruma.csv"]


def make_csv(table, date, file1, file2):
    rows = table.findAll('tr')
    for row in rows:
        table_data = []
        for cell in row.findAll('td'):
            table_data.append(cell.get_text())
        if len(table_data) > 0:
            if table_data[1] != "-":
                if table_data[1] == "欠航":
                    AlertNumber1 = 1
                else:
                    AlertNumber1 = 0
                with open(file1, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    return_data1 = [date, table_data[0], AlertNumber1]
                    writer.writerow(return_data1)
            if table_data[3] != "-":
                if table_data[3] == "欠航":
                    AlertNumber2 = 1
                else:
                    AlertNumber2 = 0
                with open(file2, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    return_data2 = [date,table_data[2],AlertNumber2]
                    writer.writerow(return_data2)


def action_anei():
    count1 = 0
    count2 = 0
    time = str(datetime.datetime.now())
    print("Start Anei")
    print(time)
    data = time.split(' ')
    ymd = data[0].split('-')
    y = str(int(ymd[0]) - 2000)
    m = ymd[1]
    d = ymd[2]
    html = requests.get(base_url % (y, m, d))
    soup = BeautifulSoup(html.text, 'html.parser')
    for table_num in range(7):
        start = soup.find_all("table")[count1]
        make_csv(start, data[0], file_name[count2], file_name[count2+1])
        count1 = count1 + 1
        count2 = count2 + 2
    print("Finish Anei")


action_anei()
