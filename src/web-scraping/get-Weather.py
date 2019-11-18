import requests
import json
import datetime
import csv
import schedule
import time
from collections import Counter


def makeWeatherData():
    API_KEY = "e93cca674cda5628e91680c48caf5317"
    city_name = "ishigaki"
    api = "http://api.openweathermap.org/data/2.5/forecast?q={city}&APPID={key}&lang=ja&units=metric"
    url = api.format(city=city_name, key=API_KEY)
    response = requests.get(url)
    data = json.loads(response.text)
    today = datetime.datetime.now()
    file_name = "weather/" + today.strftime("%Y-%m-%d-%H-%M") + ".csv"
    print(file_name)
    count = 0
    num = 8
    temp = 0
    temp_min = 0
    temp_max = 0
    pressure = 0
    sea_level = 0
    grnd_level = 0
    humidity = 0
    weatherMainList = []
    weatherDescriptionList = []
    cloudsAll = 0
    windSpeed = 0
    windDeg = 0
    rainfall = 0

    for item in data['list']:
        temp = float(item['main']['temp']) + temp
        temp_min = float(item['main']['temp_min']) + temp_min
        temp_max = float(item['main']['temp_max']) + temp_max
        pressure = float(item['main']['pressure']) + pressure
        sea_level = float(item['main']['sea_level']) + sea_level
        grnd_level = float(item['main']['grnd_level']) + grnd_level
        humidity = float(item['main']['humidity']) + humidity
        weatherMainList.append(item['weather'][0]['main'])
        weatherDescriptionList.append(item['weather'][0]['description'])
        cloudsAll = float(item['clouds']['all']) + cloudsAll
        windSpeed = float(item['wind']['speed']) + windSpeed
        windDeg = float(item['wind']['deg']) + windDeg
        if 'rain' in item and '3h' in item['rain']:
            rainfall = float(item['rain']['3h']) + rainfall
        getTime = item['dt_txt'].split(' ')
        if count == 7:
            temp = temp / num
            temp_min = temp_min / num
            temp_max = temp_max / num
            pressure = pressure / num
            sea_level = sea_level / num
            grnd_level = grnd_level / num
            humidity = humidity / num
            cloudsAll = cloudsAll / num
            windSpeed = windSpeed / num
            windDeg = windDeg / num
            rainfall = rainfall / num
            counter1 = Counter(weatherMainList)
            for word1 in counter1.most_common(1):
                weatherMain = list(word1)
            counter2 = Counter(weatherDescriptionList)
            for word2 in counter2.most_common(1):
                weatherDescription = list(word2)
            listData = [getTime[0], round(temp,3), round(temp_min,3), round(temp_max,3), round(pressure,3), round(sea_level,3), round(grnd_level,3), round(humidity,3), weatherMain[0],
                    weatherDescription[0], round(cloudsAll,3), round(windSpeed,3), round(windDeg,3), round(rainfall,3)]
            with open(file_name, "a", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(listData)
            temp = 0
            temp_min = 0
            temp_max = 0
            pressure = 0
            sea_level = 0
            grnd_level = 0
            humidity = 0
            cloudsAll = 0
            windSpeed = 0
            windDeg = 0
            rainfall = 0
            weatherMainList.clear()
            weatherDescriptionList.clear()
            count = 0
        else:
            count += 1

makeWeatherData()
"""
schedule.every().day.at('06:00').do(makeWeatherData)

while True:
  schedule.run_pending()
  time.sleep(60)
"""