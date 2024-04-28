import pandas as pd
from datetime import datetime, timedelta
from joblib import load

model_temperature = load('modeltemperature.joblib')
model_temperature_max = load('modeltemperature max.joblib')
model_temperature_min = load('modeltemperature min.joblib')
model_dewpoint = load('modeldewpoint.joblib')
model_humidity = load('modelhumidity.joblib')

#adapter cette ligne pour qu'il prenne le bon csv (données METEO FRANCE)
df = pd.read_csv("H_20_latest-2023-2024.csv", sep=";")


df['AAAAMMJJHH'] = pd.to_datetime(df['AAAAMMJJHH'], format='%Y%m%d%H')
df = df[['AAAAMMJJHH', 'TD', 'U', ' T', 'TX', 'TN', 'RR1']]
df.rename(columns={
    'AAAAMMJJHH': 'date',
    'TD': 'dewpoint',
    'U': 'humidity',
    ' T': 'temperature',
    'TX': 'temperature max',
    'TN': 'temperature min',
    'RR1': 'rain'
}, inplace=True)

df.interpolate(method='linear', inplace=True)

today = datetime.now()
week_ago = today - timedelta(days=7)


data_last_week = df[(df['date'] >= week_ago) & (df['date'] <= today)]
last_entry = data_last_week.iloc[-1]

dates = pd.date_range(start=today + timedelta(days=1), periods=7, freq='D')
future_predictions = pd.DataFrame({
    'date': dates,
    'temperature': [model_temperature.predict([[last_entry['dewpoint'], last_entry['humidity'], last_entry['temperature max'], last_entry['temperature min'], last_entry['rain']]])[0] for _ in range(7)],
    'temperature max': [model_temperature_max.predict([[last_entry['dewpoint'], last_entry['humidity'], last_entry['temperature'], last_entry['temperature min'], last_entry['rain']]])[0] for _ in range(7)],
    'temperature min': [model_temperature_min.predict([[last_entry['dewpoint'], last_entry['humidity'], last_entry['temperature max'], last_entry['temperature'], last_entry['rain']]])[0] for _ in range(7)],
    'dewpoint': [model_dewpoint.predict([[last_entry['humidity'], last_entry['temperature'], last_entry['temperature max'], last_entry['temperature min'], last_entry['rain']]])[0] for _ in range(7)],
    'humidity': [model_humidity.predict([[last_entry['dewpoint'], last_entry['temperature'], last_entry['temperature max'], last_entry['temperature min'], last_entry['rain']]])[0] for _ in range(7)],
    'rain': [last_entry['rain']] * 7  
})

results = pd.concat([data_last_week, future_predictions])

#print si nécessaire
#print(results[['date', 'temperature', 'temperature max', 'temperature min', 'dewpoint', 'humidity', 'rain']])

#fichier crée avec les prévisions pour les 7 prochains jours et les données de la semaine précédente
results.to_csv("forecasted_weather_data.csv", index=False)
