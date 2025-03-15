import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
data = pd.read_csv('weather_data.csv')

# Feature engineering
def engineer_features(df):
    df = df.copy()
    df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
    df['wind_precip_interaction'] = df['wind_speed'] * df['precipitation']
    df['heat_index'] = df['temperature'] + 0.33 * (df['humidity'] - 10)
    df['pressure_anomaly'] = df['pressure'] - 1013
    df['solar_cloud_ratio'] = df['solar_radiation'] / (df['cloud_cover'] + 1)
    df['dew_point_diff'] = df['temperature'] - df['dew_point']
    df['wind_chill'] = 13.12 + 0.6215*df['temperature'] - 11.37*(df['wind_speed']**0.16) + 0.3965*df['temperature']*(df['wind_speed']**0.16)
    df['precip_intensity'] = df['precipitation'] / (df['cloud_cover'] + 1)
    df['temp_squared'] = df['temperature'] ** 2
    df['humidity_exp'] = np.exp(df['humidity'] / 50) - 1
    return df

features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure', 'solar_radiation', 
            'cloud_cover', 'dew_point', 'temp_humidity_ratio', 'wind_precip_interaction', 'heat_index', 
            'pressure_anomaly', 'solar_cloud_ratio', 'dew_point_diff', 'wind_chill', 'precip_intensity', 
            'temp_squared', 'humidity_exp']

data = engineer_features(data)
X = data[features]
y_major = data['major_label']
y_detailed = data['detailed_label']

# Split data
X_train, X_test, y_major_train, y_major_test, y_detailed_train, y_detailed_test = train_test_split(
    X, y_major, y_detailed, test_size=0.2, random_state=42
)

# Level 1: Major event classifier (Random Forest)
level1_model = RandomForestClassifier(n_estimators=200, random_state=42)
level1_model.fit(X_train, y_major_train)

# Level 2: Detailed event classifiers (Gradient Boosting per major class)
level2_models = {}
label_mappings = {}
major_class_names = ['Normal', 'Storm', 'Heatwave', 'Coldwave', 'Fog', 'Extreme']
class_names = sum([[f"{k}: {v}" for v in vs] for k, vs in {
    'Normal': ['Clear', 'Cloudy', 'Light Rain'],
    'Storm': ['Thunderstorm', 'Hail', 'Tornado'],
    'Heatwave': ['Dry Heat', 'Humid Heat', 'Extreme Heat'],
    'Coldwave': ['Frost', 'Snow', 'Blizzard'],
    'Fog': ['Mist', 'Dense Fog', 'Freezing Fog'],
    'Extreme': ['Hurricane', 'Monsoon', 'Dust Storm']
}.items()], [])

for major_class in range(len(major_class_names)):
    mask = y_major_train == major_class
    if mask.sum() > 0:
        X_sub = X_train[mask]
        y_sub = y_detailed_train[mask]
        unique_labels = np.unique(y_sub)
        label_mappings[major_class] = {i: label for i, label in enumerate(unique_labels)}
        model = GradientBoostingClassifier(n_estimators=150, random_state=42)
        model.fit(X_sub, [np.where(unique_labels == y)[0][0] for y in y_sub])
        level2_models[major_class] = model

# Save model
model_package = {
    'level1_model': level1_model,
    'level2_models': level2_models,
    'label_mappings': label_mappings,
    'major_class_names': major_class_names,
    'class_names': class_names,
    'features': features
}
joblib.dump(model_package, 'weather_event_model.joblib')
print("Model trained and saved as 'weather_event_model.joblib'")