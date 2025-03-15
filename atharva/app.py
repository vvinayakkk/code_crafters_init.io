from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import warnings
import scipy.stats as stats
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load model once when the app starts
print("Loading model from file...")
try:
    model = joblib.load('hierarchical_event_model.joblib')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Prediction function with all insights
def predict_event(model, input_data):
    level1_model = model['level1_model']
    level2_models = model['level2_models']
    label_mappings = model['label_mappings']
    major_class_names = model['major_class_names']
    class_names = model['class_names']
    features = model['features']

    required_raw = ['lidar_distance', 'fiber_vibration', 'ugs_motion', 'temperature', 
                    'humidity', 'rainfall', 'noise']
    
    if not all(feat in input_data.columns for feat in features):
        for feat in required_raw:
            if feat not in input_data.columns:
                raise ValueError(f"Missing required feature: {feat}")

        input_data = input_data.copy()
        # Original feature engineering
        input_data['temp_humid_ratio'] = input_data['temperature'] / (input_data['humidity'] + 1)
        input_data['motion_indicator'] = (input_data['lidar_distance'] > 1) & (input_data['fiber_vibration'] > 0.3)
        input_data['water_indicator'] = (input_data['humidity'] > 80) & (input_data['rainfall'] > 10)
        input_data['fire_indicator'] = (input_data['temperature'] > 50) & (input_data['humidity'] < 40)
        input_data['vibration_distance_ratio'] = input_data['fiber_vibration'] / (input_data['lidar_distance'] + 0.01)
        input_data['intrusion_speed'] = input_data['fiber_vibration'] * input_data['lidar_distance']
        input_data['noise_to_vibration'] = input_data['noise'] / (input_data['fiber_vibration'] + 0.01)
        input_data['temperature_change'] = input_data['temperature'] * input_data['humidity'] / 100
        input_data['moisture_index'] = input_data['humidity'] * input_data['rainfall'] / 10
        input_data['fire_risk'] = input_data['temperature'] / (input_data['humidity'] + 1) * 10
        input_data['vibration_intensity'] = input_data['fiber_vibration'] * input_data['ugs_motion'] * 10
        input_data['temp_rainfall_interaction'] = input_data['temperature'] * input_data['rainfall'] / 100
        input_data['exp_temp'] = np.exp(input_data['temperature'] / 50) - 1
        input_data['exp_humidity'] = np.exp(input_data['humidity'] / 50) - 1
        input_data['log_rainfall'] = np.log1p(input_data['rainfall'])
        input_data['vibration_temp_ratio'] = input_data['fiber_vibration'] / (input_data['temperature'] + 1)
        input_data['humidity_noise_interaction'] = input_data['humidity'] * input_data['noise']
        input_data['temp_noise_ratio'] = input_data['temperature'] / (input_data['noise'] + 0.01)
        input_data['vibration_squared'] = input_data['fiber_vibration'] ** 2
        input_data['lidar_vibration_interaction'] = input_data['lidar_distance'] * input_data['fiber_vibration']

        # Additional thresholds for severity
        thresholds = {
            "temperature": {"Low": 20, "Moderate": 35, "High": 50, "Critical": 70},
            "fiber_vibration": {"Low": 0.5, "Moderate": 1.5, "High": 2.5, "Critical": 4.0},
            "rainfall": {"Low": 5, "Moderate": 15, "High": 25, "Critical": 40},
            "noise": {"Low": 0.5, "Moderate": 1.0, "High": 1.5, "Critical": 2.5},
            "lidar_distance": {"Low": 1.5, "Moderate": 1.0, "High": 0.5, "Critical": 0.2}
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

        # Advanced physical metrics (scalars)
        energy_dissipation = float((input_data['fiber_vibration'] * input_data['noise'] * input_data['ugs_motion']).iloc[0])
        thermal_energy = float((input_data['temperature'] * (input_data['humidity'] / 100) * 4184).iloc[0])
        acoustic_power = float(((input_data['noise'] ** 2) / 1000).iloc[0])
        seismic_intensity = float((input_data['fiber_vibration'] * input_data['ugs_motion'] * input_data['lidar_distance']).iloc[0])
        moisture_load = float((input_data['humidity'] * input_data['rainfall'] / 10).iloc[0])
        heat_index = float((input_data['temperature'] + 0.33 * (input_data['humidity'] - 10)).iloc[0])
        wind_equivalent = float((input_data['fiber_vibration'] * 3.6).iloc[0])
        pressure_wave = float(((input_data['noise'] * input_data['fiber_vibration']) / 10).iloc[0])
        kinetic_energy = float((0.5 * input_data['fiber_vibration'] * (input_data['lidar_distance'] ** 2)).iloc[0])
        water_vapor_density = float(((input_data['humidity'] / 100) * (input_data['temperature'] / 273.15) * 0.804).iloc[0])

        # Additional transformations (scalars)
        exp_vibration = float((np.exp(input_data['fiber_vibration'] / 5) - 1).iloc[0])
        log_vibration = float(np.log1p(input_data['fiber_vibration']).iloc[0])
        log_noise = float(np.log1p(input_data['noise']).iloc[0])
        humidity_cubed = float((input_data['humidity'] ** 3).iloc[0])

        # Additional interaction terms (scalars)
        temp_noise_interaction = float((input_data['temperature'] * input_data['noise']).iloc[0])
        humidity_vibration_interaction = float((input_data['humidity'] * input_data['fiber_vibration']).iloc[0])
        rainfall_temp_product = float((input_data['rainfall'] * input_data['temperature']).iloc[0])
        noise_humidity_product = float((input_data['noise'] * input_data['humidity']).iloc[0])
        temp_vib_interaction = float((input_data['temperature'] * input_data['fiber_vibration']).iloc[0])
        rain_noise_interaction = float((input_data['rainfall'] * input_data['noise']).iloc[0])
        lidar_humidity_interaction = float((input_data['lidar_distance'] * input_data['humidity']).iloc[0])

        # Risk scores with severity (scalars)
        flood_risk = float((input_data['rainfall'] * input_data['humidity'] / 100).iloc[0])
        flood_risk_severity, flood_color = assign_severity(flood_risk, {"Low": 10, "Moderate": 20, "High": 40, "Critical": 60})
        intrusion_risk = float((input_data['fiber_vibration'] * input_data['lidar_distance'] * input_data['ugs_motion'] / 5).iloc[0])
        intrusion_risk_severity, intrusion_color = assign_severity(intrusion_risk, {"Low": 0.5, "Moderate": 1.0, "High": 2.0, "Critical": 3.0})
        seismic_risk = float((input_data['fiber_vibration'] * input_data['ugs_motion'] / 2).iloc[0])
        seismic_risk_severity, seismic_color = assign_severity(seismic_risk, {"Low": 0.5, "Moderate": 1.0, "High": 2.0, "Critical": 3.0})
        noise_risk = float((input_data['noise'] * 10).iloc[0])
        noise_risk_severity, noise_color = assign_severity(noise_risk, {"Low": 5, "Moderate": 10, "High": 15, "Critical": 25})
        thermal_risk = float((input_data['temperature'] * (1 - input_data['humidity'] / 100)).iloc[0])
        thermal_risk_severity, thermal_color = assign_severity(thermal_risk, {"Low": 10, "Moderate": 20, "High": 30, "Critical": 50})
        environmental_stress = float(((input_data['temperature'] ** 2 + input_data['humidity'] ** 2 + input_data['rainfall'] ** 2) ** 0.5).iloc[0])

        # Statistical measures (scalars)
        sensor_array = input_data[required_raw].values[0]
        mean_sensors = float(np.mean(sensor_array))
        median_sensors = float(np.median(sensor_array))
        std_sensors = float(np.std(sensor_array))
        skew_sensors = float(stats.skew(sensor_array))
        kurtosis_sensors = float(stats.kurtosis(sensor_array))
        max_sensor = float(np.max(sensor_array))
        min_sensor = float(np.min(sensor_array))
        range_sensors = float(max_sensor - min_sensor)
        variance_sensors = float(np.var(sensor_array))
        coeff_variation = float(std_sensors / mean_sensors if mean_sensors != 0 else 0)

        # Time-based metrics (scalars)
        event_duration = float(np.random.uniform(5, 180))
        rate_of_change_temp = float((input_data['temperature'] / event_duration).iloc[0])
        rate_of_change_vib = float((input_data['fiber_vibration'] / event_duration).iloc[0])
        rate_of_change_noise = float((input_data['noise'] / event_duration).iloc[0])
        time_to_peak = float(event_duration * 0.35)
        decay_rate = float(1 / event_duration)
        peak_energy_time = float(energy_dissipation / time_to_peak)

        # Probabilistic metrics (scalars)
        anomaly_prob = float((1 / (1 + np.exp(-(input_data['fiber_vibration'] + input_data['noise'] + input_data['temperature']/50)))).iloc[0])
        critical_threshold_exceedance = float(sum([
            input_data['temperature'].iloc[0] > 50,
            input_data['fiber_vibration'].iloc[0] > 2,
            input_data['rainfall'].iloc[0] > 15,
            input_data['noise'].iloc[0] > 1.5
        ]) / 4)
        signal_to_noise_ratio = float((input_data['fiber_vibration'] / (input_data['noise'] + 0.01)).iloc[0])

        # Composite indices (scalars)
        event_severity_index = float((input_data['fire_risk'].iloc[0] + flood_risk + intrusion_risk + seismic_risk + noise_risk + thermal_risk) / 6)
        environmental_impact = float((moisture_load + thermal_energy + seismic_intensity + acoustic_power) / 1000)
        detection_reliability = float(1 - std_sensors / (mean_sensors + 1))
        structural_risk_index = float((input_data['vibration_squared'].iloc[0] + seismic_risk + intrusion_risk) / 10)
        weather_hazard_index = float((flood_risk + thermal_risk + noise_risk) / 3)

        # Differential metrics (scalars)
        temp_diff_from_normal = float((input_data['temperature'] - 20).iloc[0])
        humidity_diff_from_normal = float((input_data['humidity'] - 50).iloc[0])
        vibration_anomaly = float((input_data['fiber_vibration'] - 0.5).iloc[0])
        noise_anomaly = float((input_data['noise'] - 0.5).iloc[0])
        rainfall_deviation = float((input_data['rainfall'] - 5).iloc[0])
        lidar_proximity_alert = float((1 - input_data['lidar_distance']).iloc[0])

        # Additional insightful metrics (scalars)
        vibration_energy_density = float((input_data['vibration_squared'] / (input_data['lidar_distance'] + 0.01)).iloc[0])
        thermal_gradient = float((input_data['temperature'] / (input_data['lidar_distance'] + 0.01)).iloc[0])
        humidity_stability = float((1 / (1 + input_data['exp_humidity'])).iloc[0])
        rain_intensity = float(input_data['rainfall'].iloc[0] / (event_duration / 3600) if input_data['rainfall'].iloc[0] > 0 else 0)
        noise_vibration_correlation = float(((input_data['noise'] * input_data['fiber_vibration']) / (std_sensors + 0.01)).iloc[0])
        temp_humidity_stress = float(((input_data['temperature'] ** 2 + humidity_cubed) ** 0.33).iloc[0])
        seismic_energy_release = float(seismic_intensity * event_duration)
        acoustic_energy_flux = float(acoustic_power * event_duration)

    X = input_data[features]
    major_class = int(level1_model.predict(X)[0])

    if major_class in level2_models:
        pred_subclass = level2_models[major_class].predict(X)[0]
        detailed_class = int(label_mappings[major_class][int(pred_subclass.item())])
    else:
        detailed_class = int([k for k, v in {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 
                            7: 2, 8: 3, 9: 3, 10: 3, 11: 4, 12: 4, 13: 4}.items() if v == major_class][0])

    # Compile insights dictionary with scalar values
    insights = {
        "timestamp": datetime.now().isoformat(),
        "raw_sensors": raw_values,
        "basic_metrics": {
            "temp_humidity_ratio": float(input_data['temp_humid_ratio'].iloc[0]),
            "motion_indicator": int(input_data['motion_indicator'].iloc[0]),
            "water_indicator": int(input_data['water_indicator'].iloc[0]),
            "fire_indicator": int(input_data['fire_indicator'].iloc[0]),
            "vibration_distance_ratio": float(input_data['vibration_distance_ratio'].iloc[0]),
            "intrusion_speed": float(input_data['intrusion_speed'].iloc[0]),
            "noise_to_vibration": float(input_data['noise_to_vibration'].iloc[0]),
            "temperature_change": float(input_data['temperature_change'].iloc[0]),
            "moisture_index": float(input_data['moisture_index'].iloc[0]),
            "vibration_intensity": float(input_data['vibration_intensity'].iloc[0]),
            "temp_rainfall_interaction": float(input_data['temp_rainfall_interaction'].iloc[0]),
            "vibration_temp_ratio": float(input_data['vibration_temp_ratio'].iloc[0]),
            "temp_noise_ratio": float(input_data['temp_noise_ratio'].iloc[0]),
        },
        "physical_metrics": {
            "energy_dissipation": energy_dissipation,
            "thermal_energy": thermal_energy,
            "acoustic_power": acoustic_power,
            "seismic_intensity": seismic_intensity,
            "moisture_load": moisture_load,
            "heat_index": heat_index,
            "wind_equivalent": wind_equivalent,
            "pressure_wave": pressure_wave,
            "kinetic_energy": kinetic_energy,
            "water_vapor_density": water_vapor_density,
        },
        "transformations": {
            "exp_temperature": float(input_data['exp_temp'].iloc[0]),
            "exp_humidity": float(input_data['exp_humidity'].iloc[0]),
            "exp_vibration": exp_vibration,
            "log_rainfall": float(input_data['log_rainfall'].iloc[0]),
            "log_vibration": log_vibration,
            "log_noise": log_noise,
            "vibration_squared": float(input_data['vibration_squared'].iloc[0]),
            "temp_squared": float((input_data['temperature'] ** 2).iloc[0]),
            "humidity_cubed": humidity_cubed,
        },
        "interaction_terms": {
            "temp_noise_interaction": temp_noise_interaction,
            "humidity_vibration_interaction": humidity_vibration_interaction,
            "lidar_vibration_product": float(input_data['lidar_vibration_interaction'].iloc[0]),
            "rainfall_temp_product": rainfall_temp_product,
            "noise_humidity_product": noise_humidity_product,
            "temp_vib_interaction": temp_vib_interaction,
            "rain_noise_interaction": rain_noise_interaction,
            "lidar_humidity_interaction": lidar_humidity_interaction,
        },
        "risk_scores": {
            "fire_risk": {"value": float(input_data['fire_risk'].iloc[0]), "severity": assign_severity(input_data['fire_risk'].iloc[0], {"Low": 5, "Moderate": 10, "High": 20, "Critical": 30})[0], "color": assign_severity(input_data['fire_risk'].iloc[0], {"Low": 5, "Moderate": 10, "High": 20, "Critical": 30})[1]},
            "flood_risk": {"value": flood_risk, "severity": flood_risk_severity, "color": flood_color},
            "intrusion_risk": {"value": intrusion_risk, "severity": intrusion_risk_severity, "color": intrusion_color},
            "seismic_risk": {"value": seismic_risk, "severity": seismic_risk_severity, "color": seismic_color},
            "noise_risk": {"value": noise_risk, "severity": noise_risk_severity, "color": noise_color},
            "thermal_risk": {"value": thermal_risk, "severity": thermal_risk_severity, "color": thermal_color},
            "environmental_stress": environmental_stress,
        },
        "statistical_measures": {
            "mean_sensors": mean_sensors,
            "median_sensors": median_sensors,
            "std_sensors": std_sensors,
            "skew_sensors": skew_sensors,
            "kurtosis_sensors": kurtosis_sensors,
            "max_sensor": max_sensor,
            "min_sensor": min_sensor,
            "range_sensors": range_sensors,
            "variance_sensors": variance_sensors,
            "coeff_variation": coeff_variation,
        },
        "time_based_metrics": {
            "event_duration": event_duration,
            "rate_of_change_temp": rate_of_change_temp,
            "rate_of_change_vib": rate_of_change_vib,
            "rate_of_change_noise": rate_of_change_noise,
            "time_to_peak": time_to_peak,
            "decay_rate": decay_rate,
            "peak_energy_time": peak_energy_time,
        },
        "probabilistic_metrics": {
            "anomaly_prob": anomaly_prob,
            "critical_threshold_exceedance": critical_threshold_exceedance,
            "signal_to_noise_ratio": signal_to_noise_ratio,
        },
        "composite_indices": {
            "event_severity_index": event_severity_index,
            "environmental_impact": environmental_impact,
            "detection_reliability": detection_reliability,
            "structural_risk_index": structural_risk_index,
            "weather_hazard_index": weather_hazard_index,
        },
        "differential_metrics": {
            "temp_diff_from_normal": temp_diff_from_normal,
            "humidity_diff_from_normal": humidity_diff_from_normal,
            "vibration_anomaly": vibration_anomaly,
            "noise_anomaly": noise_anomaly,
            "rainfall_deviation": rainfall_deviation,
            "lidar_proximity_alert": lidar_proximity_alert,
        },
        "additional_insights": {
            "vibration_energy_density": vibration_energy_density,
            "thermal_gradient": thermal_gradient,
            "humidity_stability": humidity_stability,
            "rain_intensity": rain_intensity,
            "noise_vibration_correlation": noise_vibration_correlation,
            "temp_humidity_stress": temp_humidity_stress,
            "seismic_energy_release": seismic_energy_release,
            "acoustic_energy_flux": acoustic_energy_flux,
        }
    }

    return major_class, detailed_class, major_class_names[major_class], class_names[detailed_class], insights

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction with insights
        major_class, detailed_class, major_name, detailed_name, insights = predict_event(model, input_data)
        
        # Return response with all insights
        response = {
            'major_class': major_class,
            'detailed_class': detailed_class,
            'major_class_name': major_name,
            'detailed_class_name': detailed_name,
            'insights': insights
        }
        return jsonify(response), 200
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# Test endpoint
@app.route('/test', methods=['GET'])
def test_model_endpoint():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    test_cases = [
        {'lidar_distance': 1.0, 'fiber_vibration': 0.1, 'ugs_motion': 0, 'temperature': 22.0, 
         'humidity': 50.0, 'rainfall': 1.0, 'noise': 0.2, 'expected_major': 'Normal', 
         'expected_detail': 'Normal Operation'},
        {'lidar_distance': 1.1, 'fiber_vibration': 0.2, 'ugs_motion': 0, 'temperature': 18.0, 
         'humidity': 60.0, 'rainfall': 3.0, 'noise': 0.3, 'expected_major': 'Normal', 
         'expected_detail': 'Power Outage'},
        {'lidar_distance': 0.6, 'fiber_vibration': 1.7, 'ugs_motion': 1, 'temperature': 25.0, 
         'humidity': 45.0, 'rainfall': 0.5, 'noise': 0.7, 'expected_major': 'Intrusion', 
         'expected_detail': 'Human Intrusion'},
        {'lidar_distance': 1.2, 'fiber_vibration': 1.8, 'ugs_motion': 1, 'temperature': 70.0, 
         'humidity': 25.0, 'rainfall': 0.0, 'noise': 1.2, 'expected_major': 'Fire', 
         'expected_detail': 'Building Fire'},
        {'lidar_distance': 1.0, 'fiber_vibration': 0.5, 'ugs_motion': 0, 'temperature': 20.0, 
         'humidity': 90.0, 'rainfall': 30.0, 'noise': 0.8, 'expected_major': 'Flood', 
         'expected_detail': 'Flood'},
        {'lidar_distance': 0.7, 'fiber_vibration': 2.7, 'ugs_motion': 1, 'temperature': 25.0, 
         'humidity': 50.0, 'rainfall': 0.0, 'noise': 1.8, 'expected_major': 'Extreme', 
         'expected_detail': 'Earthquake'}
    ]
    
    results = []
    for i, test_case in enumerate(test_cases):
        input_data = pd.DataFrame([{k: v for k, v in test_case.items() 
                                  if k not in ['expected_major', 'expected_detail']}])
        major_class, detailed_class, major_name, detailed_name, insights = predict_event(model, input_data)
        
        result = {
            'test_case': i + 1,
            'input': input_data.to_dict(orient='records')[0],
            'predicted_major_class': major_name,
            'expected_major_class': test_case['expected_major'],
            'predicted_detailed_class': detailed_name,
            'expected_detailed_class': test_case['expected_detail'],
            'correct_major': major_name == test_case['expected_major'],
            'correct_detail': detailed_name == test_case['expected_detail'],
            'insights': insights
        }
        results.append(result)
    
    return jsonify({'results': results}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6900)