from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import warnings
import scipy.stats as stats
from datetime import datetime
from flask_cors import CORS
import threading
import time
import random

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# Load model
print("Loading model from file...")
try:
    model = joblib.load('weather_event_model.joblib')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Prediction function with detailed insights
def predict_weather_event(model, input_data):
    level1_model = model['level1_model']
    level2_models = model['level2_models']
    label_mappings = model['label_mappings']
    major_class_names = model['major_class_names']
    class_names = model['class_names']
    features = model['features']

    required_raw = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure', 
                    'solar_radiation', 'cloud_cover', 'dew_point']
    
    if not all(feat in input_data.columns for feat in features):
        for feat in required_raw:
            if feat not in input_data.columns:
                raise ValueError(f"Missing required feature: {feat}")

        input_data = input_data.copy()
        # Feature engineering
        input_data['temp_humidity_ratio'] = input_data['temperature'] / (input_data['humidity'] + 1)
        input_data['wind_precip_interaction'] = input_data['wind_speed'] * input_data['precipitation']
        input_data['heat_index'] = input_data['temperature'] + 0.33 * (input_data['humidity'] - 10)
        input_data['pressure_anomaly'] = input_data['pressure'] - 1013
        input_data['solar_cloud_ratio'] = input_data['solar_radiation'] / (input_data['cloud_cover'] + 1)
        input_data['dew_point_diff'] = input_data['temperature'] - input_data['dew_point']
        input_data['wind_chill'] = 13.12 + 0.6215*input_data['temperature'] - 11.37*(input_data['wind_speed']**0.16) + 0.3965*input_data['temperature']*(input_data['wind_speed']**0.16)
        input_data['precip_intensity'] = input_data['precipitation'] / (input_data['cloud_cover'] + 1)
        input_data['temp_squared'] = input_data['temperature'] ** 2
        input_data['humidity_exp'] = np.exp(input_data['humidity'] / 50) - 1

        # Severity thresholds
        thresholds = {
            "temperature": {"Low": 10, "Moderate": 25, "High": 35, "Critical": 45},
            "wind_speed": {"Low": 5, "Moderate": 15, "High": 25, "Critical": 40},
            "precipitation": {"Low": 5, "Moderate": 15, "High": 25, "Critical": 40},
            "humidity": {"Low": 30, "Moderate": 60, "High": 80, "Critical": 95},
            "pressure": {"Low": 1000, "Moderate": 1010, "High": 1020, "Critical": 1030}
        }
        severity_colors = {"Low": "green", "Moderate": "yellow", "High": "red", "Critical": "magenta"}

        def assign_severity(value, thresholds):
            for level, thresh in thresholds.items():
                if value <= thresh:
                    return level, severity_colors[level]
            return "Critical", severity_colors["Critical"]

        # Raw sensor values with severity
        raw_values = {}
        for sensor in required_raw:
            value = input_data[sensor].iloc[0]
            if sensor in thresholds:
                severity, color = assign_severity(value, thresholds[sensor])
                raw_values[sensor] = {"value": float(value), "severity": severity, "color": color}
            else:
                raw_values[sensor] = {"value": float(value), "severity": "N/A", "color": "white"}

        # Advanced physical metrics
        convective_energy = float((input_data['temperature'] * input_data['humidity'] / 100 * 1000).iloc[0])
        wind_energy = float((0.5 * 1.225 * (input_data['wind_speed'] ** 3)).iloc[0])  # Air density ~1.225 kg/m³
        precip_energy = float((input_data['precipitation'] * 1000 * 9.81).iloc[0])  # Potential energy
        solar_flux = float((input_data['solar_radiation'] * (1 - input_data['cloud_cover'] / 100)).iloc[0])
        atmospheric_stability = float((input_data['temperature'] - input_data['dew_point']).iloc[0])
        storm_potential = float((input_data['wind_speed'] * input_data['precipitation'] * input_data['humidity'] / 100).iloc[0])
        thermal_gradient = float((input_data['temperature'] / (input_data['pressure'] / 1000)).iloc[0])
        moisture_flux = float((input_data['humidity'] * input_data['precipitation'] / 10).iloc[0])
        pressure_wave = float(((input_data['wind_speed'] * input_data['pressure']) / 1000).iloc[0])
        visibility_index = float((1000 / (1 + input_data['cloud_cover'] + input_data['precipitation'])).iloc[0])

        # Risk scores
        storm_risk = float((input_data['wind_speed'] * input_data['precipitation'] / 10).iloc[0])
        storm_risk_severity, storm_color = assign_severity(storm_risk, {"Low": 10, "Moderate": 30, "High": 50, "Critical": 80})
        heat_risk = float((input_data['temperature'] * (1 - input_data['humidity'] / 100)).iloc[0])
        heat_risk_severity, heat_color = assign_severity(heat_risk, {"Low": 20, "Moderate": 30, "High": 40, "Critical": 50})
        cold_risk = float(((input_data['wind_chill'] - input_data['temperature']) * -1).iloc[0])
        cold_risk_severity, cold_color = assign_severity(cold_risk, {"Low": 10, "Moderate": 20, "High": 30, "Critical": 40})
        fog_risk = float((input_data['humidity'] * (100 - visibility_index) / 100).iloc[0])
        fog_risk_severity, fog_color = assign_severity(fog_risk, {"Low": 50, "Moderate": 70, "High": 85, "Critical": 95})
        extreme_risk = float((input_data['wind_speed'] + input_data['precipitation'] + input_data['temp_squared'] / 100).iloc[0])

        # Statistical measures
        sensor_array = input_data[required_raw].values[0]
        mean_sensors = float(np.mean(sensor_array))
        std_sensors = float(np.std(sensor_array))

        # Time-based metrics
        event_duration = float(np.random.uniform(15, 360))  # minutes
        rate_of_change_temp = float((input_data['temperature'] / event_duration).iloc[0])

    X = input_data[features]
    major_class = int(level1_model.predict(X)[0])

    if major_class in level2_models:
        pred_subclass = level2_models[major_class].predict(X)[0]
        detailed_class = int(label_mappings[major_class][int(pred_subclass)])
    else:
        detailed_class = int(major_class * 3)  # Fallback

    # Compile insights
    insights = {
        "timestamp": datetime.now().isoformat(),
        "raw_sensors": raw_values,
        "physical_metrics": {
            "convective_energy": convective_energy,
            "wind_energy": wind_energy,
            "precip_energy": precip_energy,
            "solar_flux": solar_flux,
            "atmospheric_stability": atmospheric_stability,
            "storm_potential": storm_potential,
            "thermal_gradient": thermal_gradient,
            "moisture_flux": moisture_flux,
            "pressure_wave": pressure_wave,
            "visibility_index": visibility_index
        },
        "risk_scores": {
            "storm_risk": {"value": storm_risk, "severity": storm_risk_severity, "color": storm_color},
            "heat_risk": {"value": heat_risk, "severity": heat_risk_severity, "color": heat_color},
            "cold_risk": {"value": cold_risk, "severity": cold_risk_severity, "color": cold_color},
            "fog_risk": {"value": fog_risk, "severity": fog_risk_severity, "color": fog_color},
            "extreme_risk": extreme_risk
        },
        "statistical_measures": {"mean_sensors": mean_sensors, "std_sensors": std_sensors},
        "time_based_metrics": {"event_duration": event_duration, "rate_of_change_temp": rate_of_change_temp}
    }

    return major_class, detailed_class, major_class_names[major_class], class_names[detailed_class], insights

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        major_class, detailed_class, major_name, detailed_name, insights = predict_weather_event(model, input_data)
        
        response = {
            'major_class': major_class,
            'detailed_class': detailed_class,
            'major_class_name': major_name,
            'detailed_class_name': detailed_name,
            'insights': insights
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Real-time simulation (Crazy Use Case: Weather Control Center)
real_time_data = []
def simulate_real_time_weather():
    global real_time_data
    while True:
        data = {
            'temperature': random.normalvariate(20, 10),
            'humidity': random.uniform(20, 100),
            'wind_speed': random.expovariate(0.2),
            'precipitation': random.expovariate(0.5),
            'pressure': random.normalvariate(1013, 10),
            'solar_radiation': random.uniform(0, 1000),
            'cloud_cover': random.uniform(0, 100),
            'dew_point': random.normalvariate(15, 5)
        }
        input_data = pd.DataFrame([data])
        major_class, detailed_class, major_name, detailed_name, insights = predict_weather_event(model, input_data)
        real_time_data.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': {'major': major_name, 'detailed': detailed_name},
            'insights': insights
        })
        if len(real_time_data) > 100:  # Keep last 100 entries
            real_time_data.pop(0)
        time.sleep(5)  # Update every 5 seconds

@app.route('/realtime', methods=['GET'])
def get_realtime():
    return jsonify({'real_time_data': real_time_data[-10:]})  # Return last 10 entries

if __name__ == '__main__':
    threading.Thread(target=simulate_real_time_weather, daemon=True).start()
    app.run(debug=True, host='0.0.0.0', port=3900)