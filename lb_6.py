import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Загрузка данных о пассажирских авиаперевозках из библиотеки statsmodels
# "AirPassengers" содержит информацию о ежемесячном количестве пассажиров, перевезённых международными авиалиниями с 1949 по 1960 год
data_loader = sm.datasets.get_rdataset("AirPassengers")
data = data_loader.data

# Числовые индексы данных заменяются индексами, соответствующими датам
data.index = pd.date_range(start='1949-01', periods=len(data), freq='ME')

# Извлечение данных за последние 3 года (36 месяцев)
ts_monthly = data['value'][-36:]

# Данные суммируются в кварталы
ts_quarterly = ts_monthly.resample('QE').sum()

# Преобразование данных в DataFrame 
df_quarterly = ts_quarterly.to_frame(name='Value')
df_quarterly['Date'] = df_quarterly.index
df_quarterly['Quarter'] = df_quarterly.index.quarter

print("Исходная таблица данных:")
print(df_quarterly[['Quarter', 'Value']])

# Извлекаем данные в виде временного ряда
ts = df_quarterly['Value'].values

# Функция для реализации модели Хольта-Уинтерса
def holt_winters(ts, alpha, beta, gamma, season_length, forecast_periods):
    # Инициализация
    n = len(ts)
    level = np.zeros(n)
    trend = np.zeros(n)
    seasonal = np.zeros(season_length)
    forecast = np.zeros(forecast_periods)

    # Первоначальные значения уровня и тренда
    level[0] = ts[0]  # Начальный уровень
    trend[0] = ts[1] - ts[0]  # Начальный тренд (разница между первым и вторым значением)

    # Инициализация сезонности
    seasonal[:season_length] = ts[:season_length] / np.mean(ts[:season_length])

    # Применение алгоритма Хольта-Уинтерса
    for t in range(1, n):
        level[t] = alpha * (ts[t] / seasonal[t % season_length]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        seasonal[t % season_length] = gamma * (ts[t] / level[t]) + (1 - gamma) * seasonal[t % season_length]

    # Прогноз на будущие периоды
    for t in range(n, n + forecast_periods):
        forecast[t - n] = (level[n - 1] + (t - n + 1) * trend[n - 1]) * seasonal[t % season_length]

    return level, trend, seasonal, forecast

# Параметры сглаживания
alpha = 0.5  # Сглаживание уровня
beta = 0.5   # Сглаживание тренда
gamma = 0.5  # Сглаживание сезонности
season_length = 4  # Длительность сезона (квартал)
forecast_periods = 4  # Количество прогнозных периодов (1 год = 4 квартала)

# Применение модели Хольта-Уинтерса
level, trend, seasonal, forecast = holt_winters(ts, alpha, beta, gamma, season_length, forecast_periods)

# Вывод результатов
print("\nКоэффициенты парной регрессии (уровень, тренд):")
print(f"Уровень: {level[-1]}, Тренд: {trend[-1]}")
print("Коэффициенты сезонности:")
print(seasonal)

print("\nПрогноз на следующие 4 квартала:")
print(forecast)

