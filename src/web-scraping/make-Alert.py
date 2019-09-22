import requests
from bs4 import BeautifulSoup
import csv

base_url = "http://www.aneikankou.co.jp/timetables?date=20%s-%s-%s"
#ファイル名を指定
file_name = "test.csv"
#取得したい表番号を指定
#竹富航路(0)，黒島航路(1)，小浜島航路(2)，西表島上原航路(3)，鳩間島航路(4)，西表島大原航路(5)，波照間島航路(6)
table_number = 0

#csvファイルを作成する関数．出力ファイルは[日付,0 or 1]の形式
def make_csv(table,date):
   
    rows = table.findAll('tr')
    #欠航したかどうかの基準値を指定
    #竹富航路(0)：18，黒島航路(1)：5，小浜島航路(2)：11，西表島上原航路(3)：7，鳩間島航路(4)：2，西表島大原航路(5)：10，波照間島航路(6)：3
    standard = 18
    
    #mode='a'は追記モード
    with open(file_name, mode='a', newline = '') as f:
        
        writer = csv.writer(f)
        count_cancel = 0
        result = ''
        return_data = []
        
        for row in rows:
            
            table_data = []
            real_data = 0
            
            for cell in row.findAll('td'):
                table_data.append(cell.get_text())
            count_cancel = count_cancel + table_data.count('欠航')

        if count_cancel >= standard:
            result = '1'
        else:
            result = '0'

        return_data = [date,result]
        writer.writerow(return_data)

#取得したい期間をrange()で指定
for year in range(18,19):
    
    for month in range(9,11):
        
        for day in range(1,32):
            try:
                y = str(year)
                if(month < 10):
                    m = '0' + str(month)
                    if(day < 10):
                        d = '0' + str(day)
                    else:
                        d = str(day)
                    html = requests.get(base_url%(y,m,d))
                    soup = BeautifulSoup(html.text,'html.parser')
                else:
                    m = str(month)
                    if(day < 10):
                        d = '0' + str(day)
                    else:
                        d = str(day)
                    html = requests.get(base_url%(y,m,d))
                    soup = BeautifulSoup(html.text,'html.parser')
                date = '20' + y + '/' + m + '/' + d
                start = soup.find_all("table")[table_number]
                make_csv(start,date)
            except:
                pass
