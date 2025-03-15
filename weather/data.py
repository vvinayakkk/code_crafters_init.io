import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats

np.random.seed(42)

def generate_weather_data(n_samples=10000):
    # Base weather parameters
    timestamps = [datetime(2025, 1, 1) + timedelta(minutes=i*15) for i in range(n_samples)]
    temperature = np.random.normal(20, 10, n_samples)  # Celsius
    humidity = np.clip(np.random.normal(60, 20, n_samples), 10, 100)  # Percentage
    wind_speed = np.random.exponential(5, n_samples)  # m/s
    precipitation = np.random.exponential(2, n_samples)  # mm/hour
    pressure = np.random.normal(1013, 10, n_samples)  # hPa
    solar_radiation = np.clip(np.random.normal(300, 200, n_samples), 0, 1000)  # W/m²
    cloud_cover = np.clip(np.random.normal(50, 30, n_samples), 0, 100)  # Percentage
    dew_point = temperature - ((100 - humidity) / 5)  # Simplified dew point calculation
    
    # Major weather events and subcategories
    major_events = ['Normal', 'Storm', 'Heatwave', 'Coldwave', 'Fog', 'Extreme']
    detailed_events = {
        'Normal': ['Clear', 'Cloudy', 'Light Rain'],
        'Storm': ['Thunderstorm', 'Hail', 'Tornado'],
        'Heatwave': ['Dry Heat', 'Humid Heat', 'Extreme Heat'],
        'Coldwave': ['Frost', 'Snow', 'Blizzard'],
        'Fog': ['Mist', 'Dense Fog', 'Freezing Fog'],
        'Extreme': ['Hurricane', 'Monsoon', 'Dust Storm']
    }
    
    # Flatten detailed events for indexing
    all_detailed_events = sum([v for v in detailed_events.values()], [])
    
    # Assign events based on conditions
    major_labels = []
    detailed_labels = []
    for i in range(n_samples):
        if temperature[i] > 35 and humidity[i] < 40:
            major = 'Heatwave'
            sub = np.random.choice(detailed_events['Heatwave'])
        elif temperature[i] < 0 and wind_speed[i] > 10:
            major = 'Coldwave'
            sub = np.random.choice(detailed_events['Coldwave'])
        elif wind_speed[i] > 20 and precipitation[i] > 10:
            major = 'Storm'
            sub = np.random.choice(detailed_events['Storm'])
        elif humidity[i] > 90 and (visibility := (1000 / (1 + cloud_cover[i] + precipitation[i]))) < 1000:
            major = 'Fog'
            sub = np.random.choice(detailed_events['Fog'])
        elif wind_speed[i] > 30 or precipitation[i] > 50:
            major = 'Extreme'
            sub = np.random.choice(detailed_events['Extreme'])
        else:
            major = 'Normal'
            sub = np.random.choice(detailed_events['Normal'])
        major_labels.append(major_events.index(major))
        detailed_labels.append(all_detailed_events.index(sub))
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'precipitation': precipitation,
        'pressure': pressure,
        'solar_radiation': solar_radiation,
        'cloud_cover': cloud_cover,
        'dew_point': dew_point,
        'major_label': major_labels,
        'detailed_label': detailed_labels
    })
    
    return data

# Generate and save data
weather_data = generate_weather_data()
weather_data.to_csv('weather_data.csv', index=False)
print("Synthetic weather data generated and saved as 'weather_data.csv'")