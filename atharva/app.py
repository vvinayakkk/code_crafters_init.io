from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import warnings
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

# Prediction function (same as original)
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
    
    X = input_data[features]
    major_class = level1_model.predict(X)[0]
    
    if major_class in level2_models:
        pred_subclass = level2_models[major_class].predict(X)[0]
        detailed_class = label_mappings[major_class][int(pred_subclass)]
    else:
        detailed_class = [k for k, v in {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 
                        7: 2, 8: 3, 9: 3, 10: 3, 11: 4, 12: 4, 13: 4}.items() if v == major_class][0]
    
    return major_class, detailed_class, major_class_names[major_class], class_names[detailed_class]

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
        
        # Make prediction
        major_class, detailed_class, major_name, detailed_name = predict_event(model, input_data)
        
        # Return response
        response = {
            'major_class': int(major_class),
            'detailed_class': int(detailed_class),
            'major_class_name': major_name,
            'detailed_class_name': detailed_name
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
        major_class, detailed_class, major_name, detailed_name = predict_event(model, input_data)
        
        result = {
            'test_case': i + 1,
            'input': input_data.to_dict(orient='records')[0],
            'predicted_major_class': major_name,
            'expected_major_class': test_case['expected_major'],
            'predicted_detailed_class': detailed_name,
            'expected_detailed_class': test_case['expected_detail'],
            'correct_major': major_name == test_case['expected_major'],
            'correct_detail': detailed_name == test_case['expected_detail']
        }
        results.append(result)
    
    return jsonify({'results': results}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)