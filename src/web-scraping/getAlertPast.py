import requests
from bs4 import BeautifulSoup
import csv

base_url = "http://www.aneikankou.co.jp/timetables?date=20%s-%s-%s"

file_name = ["alertData_windy/taketomi_isigaki.csv",
             "alertData_windy/taketomi_taketomi.csv",
             "alertData_windy/kuroshima_ishigaki.csv",
             "alertData_windy/kuroshima_kuroshima.csv",
             "alertData_windy/kohamajima_ishigaki.csv",
             "alertData_windy/kohamajima_kohama.csv",
             "alertData_windy/iriomotejima_uehara_ishigaki.csv",
             "alertData_windy/iriomotejima_uehara_uehara.csv",
             "alertData_windy/hatomajima_ishigaki.csv",
             "alertData_windy/hatomajima_hatoma.csv",
             "alertData_windy/iriomotejima_ohara_ishigaki.csv",
             "alertData_windy/iriomotejima_ohara_ohara.csv",
             "alertData_windy/haterumajima_ishigaki.csv",
             "alertData_windy/haterumajima_hateruma.csv"]


def make_csv(table, date, file1, file2):
    rows = table.findAll('tr')
    for row in rows:
        table_data = []
        for cell in row.findAll('td'):
            table_data.append(cell.get_text())
        if len(table_data) > 0:
            if table_data[0] != "-":
                if table_data[1] == "欠航":
                    AlertNumber1 = 1
                else:
                    AlertNumber1 = 0
                with open(file1, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    return_data1 = [date, table_data[0], AlertNumber1]
                    writer.writerow(return_data1)
            if table_data[2] != "-":
                if table_data[3] == "欠航":
                    AlertNumber2 = 1
                else:
                    AlertNumber2 = 0
                with open(file2, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    return_data2 = [date,table_data[2],AlertNumber2]
                    writer.writerow(return_data2)

def action_Anei():
    for year in range(19, 20):
        for month in range(11, 12):
            for day in range(1, 6):
                try:
                    count1 = 0
                    count2 = 0
                    y = str(year)
                    if month < 10:
                        m = '0' + str(month)
                        if day < 10:
                            d = '0' + str(day)
                        else:
                            d = str(day)
                        html = requests.get(base_url % (y, m, d))
                        soup = BeautifulSoup(html.text, 'html.parser')
                    else:
                        m = str(month)
                        if day < 10:
                            d = '0' + str(day)
                        else:
                            d = str(day)
                        html = requests.get(base_url % (y, m, d))
                        soup = BeautifulSoup(html.text, 'html.parser')
                    print("Start Anei")
                    date = '20' + y + '/' + m + '/' + d
                    print(date)
                    for table_num in range(7):
                        start = soup.find_all("table")[count1]
                        make_csv(start, date, file_name[count2], file_name[count2 + 1])
                        count1 = count1 + 1
                        count2 = count2 + 2
                    print("Finish Anei of 1day")
                except:
                    print("Error")

action_Anei()